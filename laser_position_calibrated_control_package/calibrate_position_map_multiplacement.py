#!/usr/bin/env python3
"""Sparse multi-placement position-map calibration.

At each placement of the existing A4 sheet, record only its four outer grid
crosses: 1 (top-left), 6 (top-right), 30 (bottom-right), and 25 (bottom-left).
Move the sheet around the SAME physical target plane to spread samples across
the camera workspace. The samples are stored as arbitrary pixel/motor pairs,
not as one 5x6 grid.

The current fine actuator jog settings are preserved; F toggles to the existing
coarse/fast jog mode. Every recorded point is checkpointed immediately. Q
exits and restores the initial servo and actuator positions.
"""

import json
import math
import os
import time
import threading

import cv2
import Jetson.GPIO as GPIO
from st3215 import ST3215

from position_map import PositionMap, PositionMapError


# ============================================================
# CAMERA - MUST MATCH THE RUNTIME CONTROLLER
# ============================================================
CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30


# ============================================================
# ST3215 SERVO - EXISTING VALUES PRESERVED
# ============================================================
ST_SERIAL_PORT = "/dev/ttyUSB0"
ST_BAUDRATE = 115200
ST_SERVO_ID = 1
ST_MIN_POSITION = 1740
ST_MAX_POSITION = 2500
ST_VISIBLE_MIN_POSITION = ST_MIN_POSITION
ST_VISIBLE_MAX_POSITION = ST_MAX_POSITION
ST_SPEED_NEAR = 100
ST_ACCEL = 200


# ============================================================
# ACTUATOR - EXISTING VALUES PRESERVED
# ============================================================
ACT_STEP_PIN = 29
ACT_DIR_PIN = 31
ACT_EN_PIN = 33
ACT_MAX_SPEED_STEPS_S = 15000
ACT_ACCELERATION_STEPS_S2 = 5000
ACT_JOG_SPEED_STEPS_S = 1400
ACTUATOR_JOG_PULSE_S = 0.03
ACT_ENDPOINT_TOLERANCE_STEPS = 5
ACT_STOP_SPEED_THRESHOLD = 10

CALIBRATION_FILE = "servo_calibration.json"
POSITION_MAP_FILE = "position_map.json"
CHECKPOINT_FILE = "position_map_sparse_checkpoint.json"


# ============================================================
# SPARSE SHEET CALIBRATION
# ============================================================
# Follow this order by default: top-left #1, top-right #6, bottom-right #30,
# bottom-left #25. P lets you choose any of those corners in any order.
SHEET_CORNERS = [
    ("1", "top-left cross #1"),
    ("6", "top-right cross #6"),
    ("30", "bottom-right cross #30"),
    ("25", "bottom-left cross #25"),
]
MIN_SAMPLES_TO_SAVE = 12  # quadratic fit minimum is 6; 12 is a practical floor
RECOMMENDED_SAMPLES = 16  # 4 placements x 4 corners; 20 is better with center placement


# ============================================================
# MANUAL CALIBRATION JOGS
# ============================================================
CAL_SERVO_JOG_STEP = 5
CAL_SERVO_JOG_SPEED = 500
CAL_ACTUATOR_JOG_PULSE_S = 0.03
# New coarse mode; existing fine values above are not changed.
CAL_ACTUATOR_JOG_FAST_SPEED_STEPS_S = 10000
CAL_ACTUATOR_JOG_FAST_PULSE_S = 0.15

servo = None
_actuator_position = 0
_actuator_speed = 0.0
_actuator_target_speed = 0.0
_actuator_referenced = False
_actuator_running = True
ACTUATOR_MAX_POSITION = 0


def clamp(value, low, high):
    return max(low, min(high, value))


def load_limits():
    global ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION, ACTUATOR_MAX_POSITION
    if not os.path.exists(CALIBRATION_FILE):
        raise RuntimeError(f"{CALIBRATION_FILE} not found; restore the normal actuator calibration first.")
    try:
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {CALIBRATION_FILE}: {exc}") from exc

    ST_VISIBLE_MIN_POSITION = int(cal.get("servo_min", ST_MIN_POSITION))
    ST_VISIBLE_MAX_POSITION = int(cal.get("servo_max", ST_MAX_POSITION))
    if "actuator_travel" in cal:
        ACTUATOR_MAX_POSITION = int(cal["actuator_travel"])
    else:
        ACTUATOR_MAX_POSITION = int(cal.get("actuator_max", 0)) - int(cal.get("actuator_min", 0))
    if ACTUATOR_MAX_POSITION <= 0:
        raise RuntimeError("No valid actuator travel in servo_calibration.json")


def setup_servo():
    global servo
    servo = ST3215(ST_SERIAL_PORT)
    servo.portHandler.ser.baudrate = ST_BAUDRATE
    if not servo.PingServo(ST_SERVO_ID):
        raise RuntimeError("ST3215 did not respond to ping")
    print(f"[ST3215] Servo ID {ST_SERVO_ID} responded")


def set_servo_position(position, speed=ST_SPEED_NEAR, acc=ST_ACCEL):
    position = int(clamp(position, ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION))
    servo.MoveTo(ST_SERVO_ID, position, speed=int(speed), acc=int(acc), wait=False)
    return position


def read_servo_position(fallback):
    try:
        p = servo.ReadPosition(ST_SERVO_ID)
    except Exception:
        p = None
    return float(p) if p is not None else float(fallback)


def precise_wait(seconds):
    seconds = max(0.000002, seconds)
    guard = min(0.00004, seconds)
    if seconds > guard:
        time.sleep(seconds - guard)
    deadline = time.perf_counter() + guard
    while time.perf_counter() < deadline:
        pass


def actuator_step(half_period):
    GPIO.output(ACT_STEP_PIN, GPIO.HIGH)
    precise_wait(half_period)
    GPIO.output(ACT_STEP_PIN, GPIO.LOW)
    precise_wait(half_period)


def set_actuator_enabled(enabled):
    GPIO.output(ACT_EN_PIN, GPIO.LOW if enabled else GPIO.HIGH)


def set_actuator_direction(direction):
    GPIO.output(ACT_DIR_PIN, GPIO.HIGH if direction > 0 else GPIO.LOW)
    time.sleep(0.00001)


def set_actuator_speed(speed):
    global _actuator_target_speed
    _actuator_target_speed = float(speed)


def actuator_thread():
    global _actuator_position, _actuator_speed, _actuator_target_speed
    current_speed = 0.0
    direction = 0
    last_time = time.monotonic()
    while _actuator_running:
        now = time.monotonic()
        dt = min(now - last_time, 0.03)
        last_time = now
        target_speed = _actuator_target_speed
        max_change = ACT_ACCELERATION_STEPS_S2 * dt
        delta = target_speed - current_speed
        if abs(delta) <= max_change:
            current_speed = target_speed
        else:
            current_speed += math.copysign(max_change, delta)

        if _actuator_referenced and ACTUATOR_MAX_POSITION > 0:
            if current_speed > 0 and _actuator_position >= ACTUATOR_MAX_POSITION:
                current_speed = 0.0
                _actuator_target_speed = 0.0
            elif current_speed < 0 and _actuator_position <= 0:
                current_speed = 0.0
                _actuator_target_speed = 0.0

        _actuator_speed = current_speed
        if abs(current_speed) < 1.0:
            direction = 0
            time.sleep(0.001)
            continue

        new_direction = 1 if current_speed > 0 else -1
        if new_direction != direction:
            set_actuator_direction(new_direction)
            set_actuator_enabled(True)
            direction = new_direction
        actuator_step(0.5 / abs(current_speed))
        _actuator_position += direction
        if _actuator_referenced:
            _actuator_position = int(clamp(_actuator_position, 0, ACTUATOR_MAX_POSITION))


def move_actuator_jog(direction, fast=False):
    if _actuator_referenced:
        if direction < 0 and _actuator_position <= 0:
            return
        if direction > 0 and _actuator_position >= ACTUATOR_MAX_POSITION:
            return
    if fast:
        speed = CAL_ACTUATOR_JOG_FAST_SPEED_STEPS_S
        pulse = CAL_ACTUATOR_JOG_FAST_PULSE_S
    else:
        speed = ACT_JOG_SPEED_STEPS_S
        pulse = CAL_ACTUATOR_JOG_PULSE_S
    set_actuator_speed(direction * speed)
    time.sleep(pulse)
    set_actuator_speed(0.0)


def actuator_middle():
    return ACTUATOR_MAX_POSITION / 2.0


def initialize_middle():
    global _actuator_position, _actuator_referenced
    answer = input(f"Is the actuator physically at saved middle ({actuator_middle():.0f})? [Y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError("Put the actuator physically at saved middle and rerun calibration.")
    _actuator_position = int(round(actuator_middle()))
    _actuator_referenced = True
    print(f"[STARTUP] Accepted actuator middle={_actuator_position}")


def save_checkpoint(samples, placement, selected_corner):
    data = {
        "version": 2,
        "map_type": "SCATTERED",
        "camera_width": FRAME_WIDTH,
        "camera_height": FRAME_HEIGHT,
        "samples": samples,
        "placement": int(placement),
        "selected_corner": str(selected_corner),
    }
    temp_path = CHECKPOINT_FILE + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, CHECKPOINT_FILE)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return [], 1, "1"
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        if data.get("map_type") != "SCATTERED":
            raise ValueError("checkpoint is not a scattered multi-placement map")
        if int(data.get("camera_width", -1)) != FRAME_WIDTH or int(data.get("camera_height", -1)) != FRAME_HEIGHT:
            raise ValueError("camera resolution differs from checkpoint")
        samples = data.get("samples", [])
        for sample in samples:
            for key in ("x", "y", "servo", "actuator", "placement", "corner"):
                if key not in sample:
                    raise ValueError(f"checkpoint sample is missing {key}")
        return samples, int(data.get("placement", 1)), str(data.get("selected_corner", "1"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"[CAL] WARNING: checkpoint not loaded: {exc}")
        return [], 1, "1"


def restore_initial_position(initial_servo_position, initial_actuator_position):
    print(
        f"[CAL] Returning to initial position: servo={initial_servo_position:.0f}, "
        f"actuator={initial_actuator_position}"
    )
    try:
        set_servo_position(initial_servo_position, speed=ST_SPEED_NEAR, acc=ST_ACCEL)
    except Exception as exc:
        print(f"[CAL] WARNING: could not restore servo: {exc}")

    deadline = time.monotonic() + 10.0
    set_actuator_speed(0.0)
    while time.monotonic() < deadline:
        remaining = initial_actuator_position - _actuator_position
        if abs(remaining) <= ACT_ENDPOINT_TOLERANCE_STEPS:
            set_actuator_speed(0.0)
            break
        speed = min(ACT_MAX_SPEED_STEPS_S, math.sqrt(2.0 * ACT_ACCELERATION_STEPS_S2 * abs(remaining)))
        set_actuator_speed(math.copysign(speed, remaining))
        time.sleep(0.005)
    set_actuator_speed(0.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and abs(_actuator_speed) > ACT_STOP_SPEED_THRESHOLD:
        time.sleep(0.005)
    print(
        f"[CAL] Restored: servo={read_servo_position(initial_servo_position):.0f}, "
        f"actuator={_actuator_position}"
    )


def draw_cross(frame, x, y, label, active=False):
    radius = 14 if active else 10
    thickness = 3 if active else 2
    cv2.drawMarker(frame, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, radius * 2, thickness)
    cv2.putText(frame, label, (int(x) + 12, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def _next_corner_for_placement(samples, placement):
    completed = {str(s.get("corner")) for s in samples if int(s.get("placement", -1)) == int(placement)}
    for corner, _ in SHEET_CORNERS:
        if corner not in completed:
            return corner
    return "DONE"


def main():
    global _actuator_running, _actuator_position
    load_limits()
    setup_servo()
    initialize_middle()

    initial_servo_position = read_servo_position(ST_VISIBLE_MIN_POSITION)
    initial_actuator_position = _actuator_position

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(ACT_STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ACT_DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ACT_EN_PIN, GPIO.OUT, initial=GPIO.HIGH)
    set_actuator_enabled(True)

    worker = threading.Thread(target=actuator_thread, daemon=True)
    worker.start()
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        _actuator_running = False
        set_actuator_enabled(False)
        GPIO.cleanup()
        raise RuntimeError(f"Could not open /dev/video{CAMERA_DEVICE}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    window = "Sparse Multi-Placement Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)

    samples = []
    placement = 1
    selected_corner = "1"
    calibration_saved = False
    if os.path.exists(CHECKPOINT_FILE):
        answer = input(f"[CAL] Found {CHECKPOINT_FILE}. Resume previous calibration? [Y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            samples, placement, selected_corner = load_checkpoint()
            selected_corner = _next_corner_for_placement(samples, placement) if selected_corner not in {c for c, _ in SHEET_CORNERS} else selected_corner
            print(f"[CAL] Resumed {len(samples)} samples; placement={placement}, next corner={selected_corner}")
        else:
            try:
                os.remove(CHECKPOINT_FILE)
            except OSError:
                pass

    state = {
        "clicked": None,
        "samples": samples,
        "placement": placement,
        "selected_corner": selected_corner,
        "current_servo": initial_servo_position,
        "actuator_jog_fast": False,
    }

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            userdata["clicked"] = (x, y)
            print(
                f"[CAL] Placement {userdata['placement']}, paper cross #{userdata['selected_corner']} "
                f"clicked at pixel ({x},{y})"
            )

    cv2.setMouseCallback(window, on_mouse, state)
    print("\n============================================================")
    print(" SPARSE MULTI-PLACEMENT POSITION MAP CALIBRATION")
    print("============================================================")
    print("Use the existing A4 sheet, but record only its FOUR outer crosses per placement:")
    print("  #1 top-left, #6 top-right, #30 bottom-right, #25 bottom-left")
    print("At each placement: click a corner cross, jog until the laser is centered on it, SPACE to record.")
    print("After all four corners, move the sheet to another location in the SAME target plane and press N.")
    print("Recommended: 4 placements around the workspace (16 samples); add a center placement (20) for better accuracy.")
    print("Spread the recorded points so their convex hull covers the intended operating area.")
    print(f"Fine actuator jog: {ACT_JOG_SPEED_STEPS_S} steps/s for {CAL_ACTUATOR_JOG_PULSE_S:.2f}s")
    print(f"Fast actuator jog: {CAL_ACTUATOR_JOG_FAST_SPEED_STEPS_S} steps/s for {CAL_ACTUATOR_JOG_FAST_PULSE_S:.2f}s (F toggles mode)")
    print("Controls: I/K servo; U/O actuator; F fine/fast; P select paper corner; SPACE record; N next sheet placement; Z undo latest sample; S save map; Q checkpoint+exit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            output = frame.copy()

            # Draw every previously recorded sample, regardless of which paper placement it came from.
            for sample in state["samples"]:
                draw_cross(
                    output, sample["x"], sample["y"],
                    f"{sample['id']}:{sample['corner']}@{sample['placement']}"
                )

            completed_this_placement = {
                str(s["corner"]) for s in state["samples"]
                if int(s["placement"]) == int(state["placement"])
            }
            if len(completed_this_placement) >= 4:
                instruction = f"PLACEMENT {state['placement']} COMPLETE - MOVE A4 AND PRESS N"
            else:
                instruction = (
                    f"PLACEMENT {state['placement']}  CORNER #{state['selected_corner']}  "
                    f"SAMPLES {len(state['samples'])}"
                )
                if state["clicked"] is not None:
                    x, y = state["clicked"]
                    draw_cross(output, x, y, f"NEXT #{state['selected_corner']}", active=True)

            cv2.putText(output, instruction, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(
                output,
                f"Servo={state['current_servo']:.0f}  Act={_actuator_position}  Jog={'FAST' if state['actuator_jog_fast'] else 'FINE'}",
                (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            cv2.putText(
                output,
                "I/K servo  U/O actuator  F speed  P corner  SPACE record  N next placement  Z undo  S save  Q exit",
                (20, FRAME_HEIGHT - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2,
            )
            cv2.imshow(window, output)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                save_checkpoint(state["samples"], state["placement"], state["selected_corner"])
                print(f"[CAL] Checkpoint saved: {len(state['samples'])} samples. Exiting and restoring initial position.")
                break

            if key == ord("p"):
                raw = input("[CAL] Select paper corner number (1, 6, 25, or 30): ").strip()
                if raw not in {c for c, _ in SHEET_CORNERS}:
                    print("[CAL] Enter one of: 1, 6, 25, 30")
                    continue
                already = {str(s["corner"]) for s in state["samples"] if int(s["placement"]) == int(state["placement"])}
                if raw in already:
                    print(f"[CAL] Corner #{raw} is already recorded for this placement. Press N to start another placement.")
                    continue
                state["selected_corner"] = raw
                state["clicked"] = None
                print(f"[CAL] Selected paper corner #{raw} for placement {state['placement']}")
                continue

            if key == ord("n"):
                already = {str(s["corner"]) for s in state["samples"] if int(s["placement"]) == int(state["placement"])}
                if already and len(already) < 4:
                    print(f"[CAL] Warning: current placement has only {len(already)}/4 corners. Moving on anyway.")
                state["placement"] += 1
                state["selected_corner"] = _next_corner_for_placement(state["samples"], state["placement"])
                if state["selected_corner"] == "DONE":
                    state["selected_corner"] = "1"
                state["clicked"] = None
                save_checkpoint(state["samples"], state["placement"], state["selected_corner"])
                print(f"[CAL] New sheet placement {state['placement']}. Record corners #1, #6, #30, #25.")
                continue

            if key == ord("f"):
                state["actuator_jog_fast"] = not state["actuator_jog_fast"]
                mode = "FAST" if state["actuator_jog_fast"] else "FINE"
                speed = CAL_ACTUATOR_JOG_FAST_SPEED_STEPS_S if state["actuator_jog_fast"] else ACT_JOG_SPEED_STEPS_S
                pulse = CAL_ACTUATOR_JOG_FAST_PULSE_S if state["actuator_jog_fast"] else CAL_ACTUATOR_JOG_PULSE_S
                print(f"[CAL] Actuator jog {mode}: target={speed} steps/s, pulse={pulse:.2f}s")
                continue

            if key == ord("r"):
                state["clicked"] = None
                continue

            if key == ord("i"):
                state["current_servo"] = set_servo_position(
                    state["current_servo"] - CAL_SERVO_JOG_STEP,
                    speed=CAL_SERVO_JOG_SPEED, acc=ST_ACCEL,
                )
                continue

            if key == ord("k"):
                state["current_servo"] = set_servo_position(
                    state["current_servo"] + CAL_SERVO_JOG_STEP,
                    speed=CAL_SERVO_JOG_SPEED, acc=ST_ACCEL,
                )
                continue

            if key == ord("u"):
                move_actuator_jog(-1, fast=state["actuator_jog_fast"])
                continue

            if key == ord("o"):
                move_actuator_jog(+1, fast=state["actuator_jog_fast"])
                continue

            if key == ord("z"):
                if not state["samples"]:
                    print("[CAL] No sample to undo")
                    continue
                last = state["samples"].pop()
                state["placement"] = int(last["placement"])
                state["selected_corner"] = str(last["corner"])
                state["clicked"] = None
                save_checkpoint(state["samples"], state["placement"], state["selected_corner"])
                print(f"[CAL] Undid sample {last['id']} at placement {last['placement']} corner #{last['corner']}")
                continue

            if key == 32:  # SPACE
                completed = {str(s["corner"]) for s in state["samples"] if int(s["placement"]) == int(state["placement"])}
                if len(completed) >= 4:
                    print("[CAL] This placement already has four corners; move the sheet and press N.")
                    continue
                if state["selected_corner"] == "DONE":
                    state["selected_corner"] = _next_corner_for_placement(state["samples"], state["placement"])
                if state["selected_corner"] in completed:
                    print("[CAL] This corner is already recorded. Use P to choose another corner.")
                    continue
                if state["clicked"] is None:
                    print(f"[CAL] Click the center of printed cross #{state['selected_corner']} first.")
                    continue
                x, y = state["clicked"]
                for old in state["samples"]:
                    if math.hypot(float(old["x"]) - x, float(old["y"]) - y) < 2.0:
                        print(f"[CAL] Pixel point too close to sample {old['id']}; click a distinct cross or use Z to undo.")
                        break
                else:
                    actual_servo = read_servo_position(state["current_servo"])
                    sample = {
                        "id": len(state["samples"]) + 1,
                        "placement": int(state["placement"]),
                        "corner": str(state["selected_corner"]),
                        "x": float(x),
                        "y": float(y),
                        "servo": float(actual_servo),
                        "actuator": float(_actuator_position),
                    }
                    state["samples"].append(sample)
                    print(
                        f"[CAL] SAVED sample {sample['id']}: placement={sample['placement']} "
                        f"corner=#{sample['corner']} pixel=({x},{y}) servo={actual_servo:.1f} "
                        f"actuator={_actuator_position}"
                    )
                    state["clicked"] = None
                    state["selected_corner"] = _next_corner_for_placement(state["samples"], state["placement"])
                    save_checkpoint(state["samples"], state["placement"], state["selected_corner"])
                    if state["selected_corner"] == "DONE":
                        print(f"[CAL] Placement {state['placement']} complete. Move A4 and press N.")
                        state["selected_corner"] = "DONE"
                    continue

            if key == ord("s"):
                if len(state["samples"]) < MIN_SAMPLES_TO_SAVE:
                    print(
                        f"[CAL] Need at least {MIN_SAMPLES_TO_SAVE} samples to save a map; "
                        f"currently have {len(state['samples'])}. Recommended: {RECOMMENDED_SAMPLES}+ samples."
                    )
                    continue
                try:
                    position_map = PositionMap.from_scattered_samples(
                        state["samples"], FRAME_WIDTH, FRAME_HEIGHT
                    )
                except PositionMapError as exc:
                    print(f"[CAL] Cannot build map yet: {exc}")
                    continue
                position_map.save(POSITION_MAP_FILE)
                print(f"[CAL] Saved {POSITION_MAP_FILE} with {position_map.sample_count} scattered samples")
                print(
                    f"[CAL] Global quadratic fit RMSE (training points): "
                    f"servo={position_map.model_rmse_servo:.1f} positions, "
                    f"actuator={position_map.model_rmse_actuator:.1f} steps"
                )
                print("[CAL] Within sample convex hull: triangle interpolation; outside: quadratic extrapolation.")
                snapshot_dir = os.path.join(SCRIPT_DIR, "calibration_snapshots")
                os.makedirs(snapshot_dir, exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snapshot_path = os.path.join(
                    snapshot_dir,
                    f"position_map_calibration_{timestamp}.png"
                )

                if cv2.imwrite(snapshot_path, output):
                    print(f"[CAL] Annotated calibration screenshot saved: {snapshot_path}")
                else:
                    print(f"[CAL] WARNING: Could not save screenshot: {snapshot_path}")
                if len(state["samples"]) < RECOMMENDED_SAMPLES:
                    print("[CAL] WARNING: map saved with fewer than 16 points; wider coverage may be inaccurate.")
                try:
                    os.remove(CHECKPOINT_FILE)
                except OSError:
                    pass
                calibration_saved = True
                break

    finally:
        if not calibration_saved and state.get("samples") is not None:
            save_checkpoint(state["samples"], state["placement"], state["selected_corner"])
        set_actuator_speed(0.0)
        try:
            restore_initial_position(initial_servo_position, initial_actuator_position)
        except Exception as exc:
            print(f"[CAL] WARNING: restore failed: {exc}")
        _actuator_running = False
        worker.join(timeout=1.0)
        set_actuator_enabled(False)
        GPIO.cleanup()
        try:
            servo.portHandler.closePort()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
