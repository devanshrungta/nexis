#!/usr/bin/env python3

import json
import math
import os
import time

import cv2
import Jetson.GPIO as GPIO
from st3215 import ST3215

from position_map import PositionMap


# ============================================================
# CAMERA - MUST MATCH THE RUNTIME CONTROLLER
# ============================================================

CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 60


# ============================================================
# ST3215 SERVO - KEEP THESE VALUES FROM THE CURRENT CONTROLLER
# ============================================================

ST_SERIAL_PORT = "/dev/ttyUSB0"
ST_BAUDRATE = 115200
ST_SERVO_ID = 1

ST_MIN_POSITION = 1981
ST_MAX_POSITION = 2541
ST_VISIBLE_MIN_POSITION = ST_MIN_POSITION
ST_VISIBLE_MAX_POSITION = ST_MAX_POSITION

ST_SPEED_NEAR = 100
ST_ACCEL = 200


# ============================================================
# ACTUATOR - KEEP THESE VALUES FROM THE CURRENT CONTROLLER
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


# ============================================================
# PRINTED A4 GRID
# ============================================================

CAL_ROWS = 5
CAL_COLS = 6
CAL_POINTS = CAL_ROWS * CAL_COLS


# ============================================================
# SMALL CALIBRATION JOG CONTROLS
# These are ONLY for one-time manual calibration.
# ============================================================

CAL_SERVO_JOG_STEP = 5
CAL_SERVO_JOG_SPEED = 500
CAL_ACTUATOR_JOG_PULSE_S = 0.03


servo = None

_actuator_position = 0
_actuator_speed = 0.0
_actuator_target_speed = 0.0
_actuator_referenced = False
_actuator_running = True


def clamp(value, low, high):
    return max(low, min(high, value))


def load_limits():
    global ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION
    global ACTUATOR_MAX_POSITION

    if not os.path.exists(CALIBRATION_FILE):
        raise RuntimeError(
            f"{CALIBRATION_FILE} not found. Run the normal controller calibration first."
        )

    with open(CALIBRATION_FILE) as f:
        cal = json.load(f)

    ST_VISIBLE_MIN_POSITION = int(cal.get("servo_min", ST_MIN_POSITION))
    ST_VISIBLE_MAX_POSITION = int(cal.get("servo_max", ST_MAX_POSITION))

    if "actuator_travel" in cal:
        ACTUATOR_MAX_POSITION = int(cal["actuator_travel"])
    else:
        ACTUATOR_MAX_POSITION = int(cal.get("actuator_max", 0)) - int(
            cal.get("actuator_min", 0)
        )

    if ACTUATOR_MAX_POSITION <= 0:
        raise RuntimeError("No valid actuator travel is saved in servo_calibration.json")


def setup_servo():
    global servo
    servo = ST3215(ST_SERIAL_PORT)
    servo.portHandler.ser.baudrate = ST_BAUDRATE
    if not servo.PingServo(ST_SERVO_ID):
        raise RuntimeError("ST3215 did not respond to ping")
    print(f"[ST3215] Servo ID {ST_SERVO_ID} responded")


def set_servo_position(position, speed=ST_SPEED_NEAR, acc=ST_ACCEL):
    position = int(
        clamp(position, ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION)
    )
    servo.MoveTo(
        ST_SERVO_ID,
        position,
        speed=int(speed),
        acc=int(acc),
        wait=False,
    )
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
            _actuator_position = int(
                clamp(_actuator_position, 0, ACTUATOR_MAX_POSITION)
            )


def move_actuator_jog(direction):
    global _actuator_position

    if _actuator_referenced:
        if direction < 0 and _actuator_position <= 0:
            return
        if direction > 0 and _actuator_position >= ACTUATOR_MAX_POSITION:
            return

    set_actuator_speed(direction * ACT_JOG_SPEED_STEPS_S)
    time.sleep(CAL_ACTUATOR_JOG_PULSE_S)
    set_actuator_speed(0.0)


def actuator_middle():
    return ACTUATOR_MAX_POSITION / 2.0


def initialize_middle():
    global _actuator_position, _actuator_referenced

    answer = input(
        f"Is the actuator physically at saved middle ({actuator_middle():.0f})? [Y/N]: "
    ).strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError(
            "Put the actuator physically at saved middle and run calibration again."
        )

    _actuator_position = int(round(actuator_middle()))
    _actuator_referenced = True
    print(f"[STARTUP] Accepted actuator middle={_actuator_position}")


def draw_cross(frame, x, y, label, active=False):
    radius = 14 if active else 10
    thickness = 3 if active else 2
    cv2.drawMarker(
        frame,
        (int(x), int(y)),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        radius * 2,
        thickness,
    )
    cv2.putText(
        frame,
        label,
        (int(x) + 12, int(y) - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )


def main():
    global _actuator_running
    global _actuator_position

    load_limits()
    setup_servo()
    initialize_middle()

    servo_position = read_servo_position(ST_VISIBLE_MIN_POSITION)

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(ACT_STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ACT_DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ACT_EN_PIN, GPIO.OUT, initial=GPIO.HIGH)
    set_actuator_enabled(True)

    worker = __import__("threading").Thread(target=actuator_thread, daemon=True)
    worker.start()

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open /dev/video{CAMERA_DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    window = "Position Map Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)

    state = {
        "clicked": None,
        "recorded": [],
        "current_servo": servo_position,
    }

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            userdata["clicked"] = (x, y)
            print(
                f"[CAL] Point {len(userdata['recorded']) + 1}/{CAL_POINTS} "
                f"selected at pixel ({x}, {y})"
            )

    cv2.setMouseCallback(window, on_mouse, state)

    print()
    print("============================================================")
    print(" POSITION MAP CALIBRATION")
    print("============================================================")
    print("Place the printed A4 calibration sheet in the actual target plane.")
    print("Use the numbered grid in ROW-MAJOR order: 1,2,3,...,30.")
    print()
    print("For each point:")
    print("  1. Click the CENTER of the printed cross.")
    print("  2. Use I/K to move the servo until the laser is centered on it.")
    print("  3. Use U/O to move the actuator until the laser is centered on it.")
    print("  4. Press SPACE to record the servo ID + actuator position.")
    print("  R = reselect the current point")
    print("  Q = abort")
    print()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            output = frame.copy()

            for i, sample in enumerate(state["recorded"]):
                draw_cross(output, sample["x"], sample["y"], str(i + 1))

            next_index = len(state["recorded"])
            if next_index < CAL_POINTS:
                if state["clicked"] is not None:
                    x, y = state["clicked"]
                    draw_cross(output, x, y, f"NEXT {next_index + 1}", active=True)
                cv2.putText(
                    output,
                    f"CALIBRATION {next_index + 1}/{CAL_POINTS}",
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    output,
                    "CALIBRATION COMPLETE - PRESS S TO SAVE",
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            cv2.putText(
                output,
                f"Servo={state['current_servo']:.0f}  Act={_actuator_position}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                output,
                "I/K servo   U/O actuator   SPACE record   R reset click   S save   Q quit",
                (25, FRAME_HEIGHT - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window, output)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                state["clicked"] = None
                continue

            if key == ord("i"):
                state["current_servo"] = set_servo_position(
                    state["current_servo"] - CAL_SERVO_JOG_STEP,
                    speed=CAL_SERVO_JOG_SPEED,
                    acc=ST_ACCEL,
                )
                continue

            if key == ord("k"):
                state["current_servo"] = set_servo_position(
                    state["current_servo"] + CAL_SERVO_JOG_STEP,
                    speed=CAL_SERVO_JOG_SPEED,
                    acc=ST_ACCEL,
                )
                continue

            if key == ord("u"):
                move_actuator_jog(-1)
                continue

            if key == ord("o"):
                move_actuator_jog(+1)
                continue

            if key == 32:
                if next_index >= CAL_POINTS:
                    continue
                if state["clicked"] is None:
                    print("[CAL] Click the current printed cross first")
                    continue

                x, y = state["clicked"]
                actual_servo = read_servo_position(state["current_servo"])
                state["current_servo"] = actual_servo
                state["recorded"].append(
                    {
                        "row": next_index // CAL_COLS,
                        "col": next_index % CAL_COLS,
                        "x": float(x),
                        "y": float(y),
                        "servo": float(actual_servo),
                        "actuator": float(_actuator_position),
                    }
                )
                print(
                    f"[CAL] SAVED {next_index + 1}/{CAL_POINTS}: "
                    f"pixel=({x},{y}) servo={actual_servo:.1f} "
                    f"actuator={_actuator_position}"
                )
                state["clicked"] = None
                continue

            if key == ord("s") and len(state["recorded"]) == CAL_POINTS:
                position_map = PositionMap(
                    rows=CAL_ROWS,
                    cols=CAL_COLS,
                    samples=state["recorded"],
                    camera_width=FRAME_WIDTH,
                    camera_height=FRAME_HEIGHT,
                )
                position_map.save(POSITION_MAP_FILE)
                print(f"[CAL] Saved {POSITION_MAP_FILE}")
                break

    finally:
        _actuator_running = False
        set_actuator_speed(0.0)
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
