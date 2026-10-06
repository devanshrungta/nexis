#!/usr/bin/env python3

import cv2
import math
import os
import json
import time
import threading

import Jetson.GPIO as GPIO
from st3215 import ST3215


# ============================================================
# CAMERA
# ============================================================

CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 60


# ============================================================
# ST3215 SERVO
# ============================================================

ST_SERIAL_PORT = "/dev/ttyUSB0"
ST_BAUDRATE = 115200
ST_SERVO_ID = 1

# Absolute electrical range of the servo.
ST_MIN_POSITION = 1981
ST_MAX_POSITION = 2541
ST_MIDDLE_POSITION = 2150
# Fallback middle before calibration is loaded. After calibration, the
# actual safe middle is computed from ST_VISIBLE_MIN/MAX_POSITION.

# Calibrated range in which the laser remains usable in the camera view.
ST_VISIBLE_MIN_POSITION = ST_MIN_POSITION
ST_VISIBLE_MAX_POSITION = ST_MAX_POSITION

ST_SPEED_MAX = 6000
ST_SPEED_NEAR = 100
ST_SPEED_EASE_PX = 60
ST_ACCEL = 200


# ============================================================
# LASER DETECTION
# ============================================================

BRIGHTNESS_MIN = 240
BRIGHTNESS_MAX = 255
MORPH_KERNEL_SIZE = 5
MIN_BLOB_AREA = 6


# ============================================================
# SIMPLE 2-DOF CONTROLLER
#
# Fixed mapping for stationary testing:
#   servo    -> image Y
#   actuator -> image X
#
# Each DOF has exactly three control concepts:
#   error -> gain -> sign
#
# No controller memory, no integral term, no optimizer.
# ============================================================

SERVO_GAIN_POS_PER_PX = 0.25
SERVO_SIGN = -1

ACTUATOR_GAIN_STEPS_PER_PX = 0.7
ACTUATOR_SIGN = 1

CONTROL_UPDATE_INTERVAL = 0.01
ALPHA = 0.4

SERVO_SETTLE_DEADBAND_PX = 20
ACTUATOR_SETTLE_DEADBAND_PX = 7
SETTLE_DWELL_S = 1

MAX_SERVO_STEP_PER_UPDATE = 30
MAX_ACTUATOR_STEP_PER_UPDATE = 5000


# ============================================================
# JITTER -- SERVO ONLY
#
# Jitter uses two alternating servo endpoints rather than asking the
# servo to reproduce an unrealistically fast sine wave. The endpoint
# commands themselves use the servo's onboard acceleration.
# ============================================================

AUTO_JITTER = True
JITTER_DURATION_S = 3.0
JITTER_AMPLITUDE_POS = 20
JITTER_FREQUENCY_HZ = 5.0
# Mechanical command-reversal frequency in Hz. One full +A -> -A -> +A
# cycle takes 1 / JITTER_FREQUENCY_HZ seconds.
#
# At high frequency the servo may not physically reach both endpoints
# before the next reversal. In that case the commanded frequency still
# changes, but the actual mechanical amplitude will shrink.
JITTER_SPEED = 3400
# Documented maximum operation speed for this STS register in the
# python-st3215 reference implementation.
JITTER_ACCEL = 254
# ST3215 acceleration field is one byte: valid range is 0..254.
JITTER_DWELL_S = 0.0
# No dwell: dwell would change the actual jitter frequency.


# ============================================================
# ACTUATOR
# ============================================================

ACT_STEP_PIN = 29
ACT_DIR_PIN = 31
ACT_EN_PIN = 33

ACT_MAX_SPEED_STEPS_S = 15000
ACT_ACCELERATION_STEPS_S2 = 5000

ACT_JOG_SPEED_STEPS_S = 1400
ACTUATOR_JOG_PULSE_S = 0.15

ACT_UNREFERENCED_JOG_LIMIT = 20000
ACTUATOR_CALIBRATION_EXTENSION_LIMIT = 30000

ACT_ENDPOINT_TOLERANCE_STEPS = 5
ACT_STOP_SPEED_THRESHOLD = 10

CALIBRATION_FILE = "servo_calibration.json"


# ============================================================
# STATE
# ============================================================

servo = None

_actuator_lock = threading.Lock()
_actuator_target_speed = 0.0
_actuator_position = 0
_actuator_speed = 0.0
_actuator_referenced = False
_actuator_min_reference_done = False
_actuator_extension_mode = False
_actuator_running = True

_jitter_lock = threading.Lock()
_jitter_active = False
_jitter_start = None
_jitter_base = None
_jitter_running = True


# ============================================================
# GENERIC HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def scaled_servo_speed(error_px):
    ratio = min(1.0, abs(error_px) / ST_SPEED_EASE_PX)
    return int(SPEED_NEAR + ratio * (ST_SPEED_MAX - ST_SPEED_NEAR))


# Keep the public name simple in the rest of the file.
SPEED_NEAR = ST_SPEED_NEAR


# ============================================================
# SERVO
# ============================================================

def setup_servo():
    global servo

    servo = ST3215(ST_SERIAL_PORT)
    try:
        servo.portHandler.ser.baudrate = ST_BAUDRATE
    except AttributeError:
        raise RuntimeError("Could not access the ST3215 serial port handler")

    if not servo.PingServo(ST_SERVO_ID):
        print("[ST3215] WARNING: servo did not respond to ping")
    else:
        print(f"[ST3215] Servo ID {ST_SERVO_ID} responded")


def set_servo_position(position, speed=ST_SPEED_MAX, acc=ST_ACCEL):
    position = int(clamp(position, ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION))
    position = int(clamp(position, ST_MIN_POSITION, ST_MAX_POSITION))
    servo.MoveTo(ST_SERVO_ID, position, speed=int(speed), acc=int(acc), wait=False)
    return position


def configure_jitter_servo():
    """Set jitter speed/acceleration once, before rapid position writes."""
    speed = int(clamp(JITTER_SPEED, 1, 3400))
    acc = int(clamp(JITTER_ACCEL, 0, 254))

    if speed != int(JITTER_SPEED):
        print(f"[JITTER] Clamped speed {JITTER_SPEED} -> {speed}")
    if acc != int(JITTER_ACCEL):
        print(f"[JITTER] Clamped acceleration {JITTER_ACCEL} -> {acc}")

    servo.SetAcceleration(ST_SERVO_ID, acc)
    servo.SetSpeed(ST_SERVO_ID, speed)
    return speed, acc


def set_jitter_position(position):
    """Rapid jitter target write; avoids MoveTo() re-sending speed/acc each time."""
    position = int(clamp(position, ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION))
    position = int(clamp(position, ST_MIN_POSITION, ST_MAX_POSITION))
    servo.WritePosition(ST_SERVO_ID, position)
    return position


def servo_middle_position():
    """Return the midpoint of the currently active safe servo range."""
    return 0.5 * (ST_VISIBLE_MIN_POSITION + ST_VISIBLE_MAX_POSITION)


def read_servo_position(fallback):
    try:
        position = servo.ReadPosition(ST_SERVO_ID)
    except Exception:
        position = None
    return float(position) if position is not None else float(fallback)


def servo_command_position(current_position, error_px):
    correction = clamp(
        error_px * SERVO_GAIN_POS_PER_PX,
        -MAX_SERVO_STEP_PER_UPDATE,
        MAX_SERVO_STEP_PER_UPDATE,
    )
    return current_position + SERVO_SIGN * correction


def actuator_command_position(current_position, error_px):
    correction = clamp(
        error_px * ACTUATOR_GAIN_STEPS_PER_PX,
        -MAX_ACTUATOR_STEP_PER_UPDATE,
        MAX_ACTUATOR_STEP_PER_UPDATE,
    )
    return clamp(
        current_position + ACTUATOR_SIGN * correction,
        ACTUATOR_MIN_POSITION,
        ACTUATOR_MAX_POSITION,
    )


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration():
    global ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION
    global ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION

    if not os.path.exists(CALIBRATION_FILE):
        ACTUATOR_MIN_POSITION = 0
        ACTUATOR_MAX_POSITION = 0
        print("[CAL] No saved calibration")
        return False

    try:
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)

        ST_VISIBLE_MIN_POSITION = int(cal.get("servo_min", ST_MIN_POSITION))
        ST_VISIBLE_MAX_POSITION = int(cal.get("servo_max", ST_MAX_POSITION))

        if "actuator_travel" in cal:
            travel = int(cal["actuator_travel"])
        else:
            travel = int(cal.get("actuator_max", 0)) - int(cal.get("actuator_min", 0))

        ACTUATOR_MIN_POSITION = 0
        ACTUATOR_MAX_POSITION = max(0, travel)

        print(
            f"[CAL] Loaded: servo={ST_VISIBLE_MIN_POSITION}..{ST_VISIBLE_MAX_POSITION}, "
            f"actuator travel={ACTUATOR_MAX_POSITION} steps"
        )
        return ACTUATOR_MAX_POSITION > 0

    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[CAL] WARNING: could not load calibration: {exc}")
        ACTUATOR_MIN_POSITION = 0
        ACTUATOR_MAX_POSITION = 0
        return False


def save_calibration():
    data = {
        "servo_min": int(ST_VISIBLE_MIN_POSITION),
        "servo_max": int(ST_VISIBLE_MAX_POSITION),
        "actuator_min": 0,
        "actuator_max": int(ACTUATOR_MAX_POSITION),
        "actuator_travel": int(ACTUATOR_MAX_POSITION),
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[CAL] Saved actuator travel={ACTUATOR_MAX_POSITION} steps")


def actuator_middle():
    if ACTUATOR_MAX_POSITION <= 0:
        return None
    return ACTUATOR_MAX_POSITION / 2.0


def actuator_is_referenced():
    with _actuator_lock:
        return _actuator_referenced and ACTUATOR_MAX_POSITION > 0


def accept_saved_middle():
    global _actuator_position, _actuator_referenced, _actuator_min_reference_done

    middle = actuator_middle()
    if middle is None:
        return False

    set_actuator_speed(0.0)
    wait_for_actuator_stop(1.0)

    with _actuator_lock:
        _actuator_position = int(round(middle))
        _actuator_referenced = True
        _actuator_min_reference_done = False

    return True


def mark_actuator_min():
    global _actuator_position, _actuator_referenced
    global _actuator_min_reference_done, ACTUATOR_MIN_POSITION

    set_actuator_speed(0.0)
    wait_for_actuator_stop(1.0)

    with _actuator_lock:
        _actuator_position = 0
        ACTUATOR_MIN_POSITION = 0
        _actuator_min_reference_done = True
        _actuator_referenced = ACTUATOR_MAX_POSITION > 0

    print("[CAL] Actuator MIN referenced as position 0")


def mark_actuator_max():
    global ACTUATOR_MAX_POSITION, _actuator_referenced, _actuator_extension_mode

    with _actuator_lock:
        if not _actuator_min_reference_done:
            return False
        if _actuator_position <= 0:
            return False
        ACTUATOR_MAX_POSITION = int(_actuator_position)
        _actuator_referenced = True
        _actuator_extension_mode = False

    save_calibration()
    print(f"[CAL] Actuator MAX marked at {ACTUATOR_MAX_POSITION}")
    return True


# ============================================================
# ACTUATOR LOW-LEVEL
# ============================================================

GPIO.setmode(GPIO.BOARD)
GPIO.setup(ACT_STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ACT_DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ACT_EN_PIN, GPIO.OUT, initial=GPIO.HIGH)


def set_actuator_enabled(enabled):
    GPIO.output(ACT_EN_PIN, GPIO.LOW if enabled else GPIO.HIGH)


def set_actuator_direction(direction):
    GPIO.output(ACT_DIR_PIN, GPIO.HIGH if direction > 0 else GPIO.LOW)
    time.sleep(0.00001)


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


def set_actuator_speed(speed):
    global _actuator_target_speed
    with _actuator_lock:
        _actuator_target_speed = float(speed)


def get_actuator_position():
    with _actuator_lock:
        return _actuator_position


def get_actuator_speed():
    with _actuator_lock:
        return _actuator_speed


def wait_for_actuator_stop(timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if abs(get_actuator_speed()) <= ACT_STOP_SPEED_THRESHOLD:
            return True
        time.sleep(0.005)
    return abs(get_actuator_speed()) <= ACT_STOP_SPEED_THRESHOLD


def actuator_thread():
    global _actuator_position, _actuator_speed, _actuator_target_speed

    current_speed = 0.0
    current_direction = 0
    last_time = time.monotonic()

    while _actuator_running:
        now = time.monotonic()
        dt = min(now - last_time, 0.03)
        last_time = now

        with _actuator_lock:
            target_speed = _actuator_target_speed
            position = _actuator_position
            referenced = _actuator_referenced
            extension_mode = _actuator_extension_mode
            max_position = ACTUATOR_MAX_POSITION

        target_direction = 1 if target_speed > 0 else -1 if target_speed < 0 else 0

        if current_direction and target_direction and target_direction != current_direction:
            target_speed = 0.0

        max_change = ACT_ACCELERATION_STEPS_S2 * dt
        delta = target_speed - current_speed
        if abs(delta) <= max_change:
            current_speed = target_speed
        else:
            current_speed += math.copysign(max_change, delta)

        # Software travel protection.
        if extension_mode:
            if current_speed > 0 and position >= ACTUATOR_CALIBRATION_EXTENSION_LIMIT:
                current_speed = 0.0
                set_actuator_speed(0.0)
            elif current_speed < 0 and position <= -ACTUATOR_CALIBRATION_EXTENSION_LIMIT:
                current_speed = 0.0
                set_actuator_speed(0.0)
        elif referenced and max_position > 0:
            if current_speed > 0:
                distance_to_end = max_position - position
            else:
                distance_to_end = position

            if distance_to_end <= 0:
                current_speed = 0.0
                set_actuator_speed(0.0)
            else:
                endpoint_speed = math.sqrt(
                    2.0 * ACT_ACCELERATION_STEPS_S2 * distance_to_end
                )
                if abs(current_speed) > endpoint_speed:
                    current_speed = math.copysign(endpoint_speed, current_speed)

        with _actuator_lock:
            _actuator_speed = current_speed

        if abs(current_speed) < 1.0:
            current_direction = 0
            time.sleep(0.001)
            continue

        direction = 1 if current_speed > 0 else -1
        if direction != current_direction:
            set_actuator_direction(direction)
            set_actuator_enabled(True)
            current_direction = direction

        actuator_step(0.5 / abs(current_speed))

        with _actuator_lock:
            _actuator_position += current_direction

            if _actuator_referenced and not _actuator_extension_mode:
                _actuator_position = int(
                    clamp(_actuator_position, ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION)
                )

    with _actuator_lock:
        _actuator_speed = 0.0
        _actuator_target_speed = 0.0


# ============================================================
# JITTER THREAD
# ============================================================

def jitter_thread():
    global _jitter_active

    while _jitter_running:
        with _jitter_lock:
            active = _jitter_active
            base = _jitter_base

        if not active or base is None:
            time.sleep(0.005)
            continue

        amplitude = abs(float(JITTER_AMPLITUDE_POS))
        frequency = max(0.1, float(JITTER_FREQUENCY_HZ))

        if amplitude < 1.0:
            time.sleep(0.01)
            continue

        try:
            speed, acc = configure_jitter_servo()
        except Exception as exc:
            print(f"[JITTER] Configuration failed: {exc}")
            with _jitter_lock:
                _jitter_active = False
            time.sleep(0.1)
            continue

        # Frequency-driven endpoint jitter. The thread does NOT wait for
        # IsMoving(). It changes the target at the requested half-period.
        # This is what makes JITTER_FREQUENCY_HZ actually affect the
        # commanded waveform.
        half_period = 1.0 / (2.0 * frequency)
        side = 1
        next_switch = time.monotonic()

        # -- Diagnostic instrumentation --
        # Measures the REAL time between consecutive position writes,
        # not just the requested half_period. If achieved frequency
        # comes out far below whatever JITTER_FREQUENCY_HZ is set to,
        # and stays roughly the same across different requested
        # values, that's direct evidence something per-call (serial
        # round-trip, servo firmware processing, Python/OS syscall
        # overhead) is the real ceiling -- not the software setting.
        last_switch_time = None
        interval_count = 0
        interval_sum = 0.0
        interval_min = float("inf")
        write_time_sum = 0.0

        while True:
            with _jitter_lock:
                if not _jitter_active:
                    break
                current_base = _jitter_base

            now = time.monotonic()

            if now >= next_switch:
                target = current_base + side * amplitude
                target = clamp(
                    target,
                    ST_VISIBLE_MIN_POSITION,
                    ST_VISIBLE_MAX_POSITION,
                )

                write_start = time.monotonic()
                try:
                    set_jitter_position(target)
                except Exception as exc:
                    print(f"[JITTER] Position write failed: {exc}")
                    with _jitter_lock:
                        _jitter_active = False
                    break
                write_time_sum += time.monotonic() - write_start

                if last_switch_time is not None:
                    interval = now - last_switch_time
                    interval_count += 1
                    interval_sum += interval
                    interval_min = min(interval_min, interval)
                last_switch_time = now

                side *= -1

                # Keep the requested frequency even if one serial write
                # takes slightly longer than the nominal half-period.
                next_switch += half_period
                if next_switch < now - half_period:
                    next_switch = now + half_period
            else:
                time.sleep(min(0.001, max(0.0001, next_switch - now)))

        if interval_count > 0:
            avg_interval = interval_sum / interval_count
            avg_write_ms = 1000 * write_time_sum / interval_count
            achieved_hz = 1.0 / (2.0 * avg_interval) if avg_interval > 0 else 0.0
            ceiling_hz = 1.0 / (2.0 * interval_min) if interval_min > 0 else 0.0
            print()
            print(
                f"[JITTER] requested={frequency:.1f} Hz  "
                f"achieved={achieved_hz:.1f} Hz (avg)  "
                f"ceiling={ceiling_hz:.1f} Hz (best single switch)  "
                f"avg write call={avg_write_ms:.2f} ms"
            )
            if achieved_hz < frequency * 0.7:
                print(
                    "[JITTER]   Achieved is well below requested -- the "
                    "bottleneck is per-write overhead (serial round-trip / "
                    "servo processing / Python+OS latency), not the "
                    "JITTER_FREQUENCY_HZ setting. Raising the setting "
                    "further will not exceed the ceiling above. See the "
                    "baud-rate note for the most likely lever to move it."
                )

        time.sleep(0.005)


# ============================================================
# LASER
# ============================================================

def detect_laser(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, BRIGHTNESS_MIN, BRIGHTNESS_MAX)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_BLOB_AREA and area > best_area:
            best = contour
            best_area = area

    if best is None:
        return None, None, best_area

    moments = cv2.moments(best)
    if moments["m00"] == 0:
        return None, None, best_area

    return (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
        best_area,
    )


# ============================================================
# UI
# ============================================================

def draw_cross(frame, x, y, color, label):
    if x is None or y is None:
        return
    point = (int(x), int(y))
    cv2.drawMarker(frame, point, color, cv2.MARKER_CROSS, 25, 2)
    cv2.putText(
        frame,
        label,
        (point[0] + 10, point[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["target"] = (x, y)

        # A new click is a NEW target, even when tracking is already active.
        # Reset the per-target state so automatic jitter can run again after
        # this new target is reached.
        state["new_target"] = True

        print(f"[TARGET] ({x}, {y})")


# ============================================================
# MAIN
# ============================================================

def main():
    global ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION
    global ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION
    global _actuator_referenced, _actuator_running
    global _actuator_min_reference_done, _actuator_extension_mode
    global _jitter_active, _jitter_base, _jitter_start, _jitter_running

    load_calibration()
    setup_servo()

    # Use the midpoint of the saved safe/visible range. This avoids an
    # inconsistent situation such as saved 1981..2052 with a fixed middle
    # of 2150 (which is outside the saved range and would be clamped).
    servo_mid = servo_middle_position()
    servo_position = set_servo_position(
        servo_mid,
        speed=ST_SPEED_NEAR,
        acc=ST_ACCEL,
    )
    print(
        f"[STARTUP] Servo safe range={ST_VISIBLE_MIN_POSITION}.."
        f"{ST_VISIBLE_MAX_POSITION}, middle={servo_mid:.1f}"
    )

    if ACTUATOR_MAX_POSITION > 0:
        middle = actuator_middle()
        answer = input(
            f"[STARTUP] Actuator saved travel={ACTUATOR_MAX_POSITION} steps, "
            f"middle={middle:.0f}. Is the actuator physically at middle? [Y/N]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            accept_saved_middle()
            print(f"[STARTUP] Accepted actuator middle={middle:.0f}")
        else:
            print("[STARTUP] Actuator unreferenced. Use M -> U/O -> 3.")
    else:
        print("[STARTUP] No actuator travel saved. Use M -> physical MIN -> 3 -> physical MAX -> 4.")

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open /dev/video{CAMERA_DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    window = "Laser Tracker - Stationary"
    raw_window = "Raw Camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(raw_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)

    ui_state = {"target": None, "new_target": False}
    cv2.setMouseCallback(window, mouse_callback, ui_state)

    actuator_worker = threading.Thread(target=actuator_thread, daemon=True)
    jitter_worker = threading.Thread(target=jitter_thread, daemon=True)
    actuator_worker.start()
    jitter_worker.start()
    set_actuator_enabled(True)

    filtered_x = None
    filtered_y = None
    last_update = 0.0
    servo_settle_since = None
    actuator_settle_since = None
    target_active = False
    target_reached = False
    jitter_done = False
    jitter_manual = False

    jog_mode = False
    actuator_jog_until = 0.0

    sweep_active = False
    sweep_selection_pending = False
    sweep_mode = None
    sweep_leg = None
    sweep_servo_commanded = False
    sweep_target_actuator = None

    frame_count = 0
    fps = 0.0
    fps_start = time.monotonic()
    fps_count = 0

    print()
    print("CONTROLS")
    print("  Left click = set target")
    print("  G = start target tracking")
    print("  R = return servo + actuator to MIDDLE")
    print("  C = cancel tracking / sweep / jitter")
    print("  M = JOG mode")
    print("      I/K = servo -/+   U/O = actuator -/+")
    print("      1/2 = servo MIN/MAX")
    print("      3   = actuator MIN/reference")
    print("      4   = actuator MAX/save travel")
    print("      E   = actuator calibration-extension mode")
    print("  T = sweep selection: 1 servo / 2 actuator / 3 both")
    print("  J = servo-only jitter")
    print("  Q = quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_count += 1
            laser_x, laser_y, blob_area = detect_laser(frame)

            now = time.monotonic()
            if now - fps_start >= 1.0:
                fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now
            fps_count += 1

            servo_error = None
            actuator_error = None

            # A left-click while tracking is active creates a fresh target.
            # Cancel any current jitter and reset all per-target state.
            if ui_state.get("new_target", False):
                ui_state["new_target"] = False

                target_reached = False
                jitter_done = False
                jitter_manual = False
                filtered_x = None
                filtered_y = None
                servo_settle_since = None
                actuator_settle_since = None

                with _jitter_lock:
                    _jitter_active = False
                    _jitter_start = None
                    _jitter_base = None

                if target_active:
                    set_actuator_speed(0.0)
                    print(
                        f"[TARGET] New target armed: "
                        f"({ui_state['target'][0]}, {ui_state['target'][1]})"
                    )

            if target_active and ui_state["target"] is not None and not sweep_active:
                target_x, target_y = ui_state["target"]

                # Jitter owns the servo while active. The actuator is
                # explicitly stopped and normal tracking is skipped.
                if _jitter_active:
                    set_actuator_speed(0.0)
                    with _jitter_lock:
                        jitter_started = _jitter_start

                    if (
                        jitter_started is not None
                        and now - jitter_started >= JITTER_DURATION_S
                    ):
                        with _jitter_lock:
                            _jitter_active = False
                            base_position = _jitter_base

                        if base_position is not None:
                            servo_position = set_servo_position(
                                base_position,
                                speed=ST_SPEED_NEAR,
                                acc=ST_ACCEL,
                            )

                        if laser_x is not None:
                            filtered_x = float(laser_x)
                        if laser_y is not None:
                            filtered_y = float(laser_y)

                        target_reached = True
                        if not jitter_manual:
                            jitter_done = True
                        jitter_manual = False

                else:
                    if laser_x is not None:
                        filtered_x = (
                            float(laser_x)
                            if filtered_x is None
                            else ALPHA * laser_x + (1.0 - ALPHA) * filtered_x
                        )
                    if laser_y is not None:
                        filtered_y = (
                            float(laser_y)
                            if filtered_y is None
                            else ALPHA * laser_y + (1.0 - ALPHA) * filtered_y
                        )

                    if now - last_update >= CONTROL_UPDATE_INTERVAL:
                        if filtered_y is not None:
                            servo_error = target_y - filtered_y
                        if filtered_x is not None:
                            actuator_error = target_x - filtered_x

                        servo_in_deadband = (
                            servo_error is not None
                            and abs(servo_error) <= SERVO_SETTLE_DEADBAND_PX
                        )

                        if servo_error is not None and not servo_in_deadband:
                            current_servo = read_servo_position(servo_position)
                            commanded_servo = servo_command_position(
                                current_servo, servo_error
                            )
                            servo_position = set_servo_position(
                                commanded_servo,
                                speed=scaled_servo_speed(servo_error),
                                acc=ST_ACCEL,
                            )
                            print(
                                f"[SERVO CTRL] target_y={target_y:.1f} "
                                f"laser_y={filtered_y:.1f} "
                                f"error={servo_error:+.1f} "
                                f"current={current_servo:.1f} "
                                f"command={servo_position:.1f} "
                                f"sign={SERVO_SIGN:+d}"
                            )
                            servo_settle_since = None
                        elif servo_error is not None:
                            if servo_settle_since is None:
                                servo_settle_since = now

                        actuator_in_deadband = (
                            actuator_error is not None
                            and abs(actuator_error) <= ACTUATOR_SETTLE_DEADBAND_PX
                        )

                        if not actuator_is_referenced():
                            set_actuator_speed(0.0)
                            actuator_in_deadband = True
                        elif actuator_error is not None and not actuator_in_deadband:
                            current_position = get_actuator_position()
                            desired_position = actuator_command_position(
                                current_position, actuator_error
                            )

                            remaining = desired_position - current_position
                            if abs(remaining) <= ACT_ENDPOINT_TOLERANCE_STEPS:
                                set_actuator_speed(0.0)
                            else:
                                speed = min(
                                    ACT_MAX_SPEED_STEPS_S,
                                    math.sqrt(
                                        2.0
                                        * ACT_ACCELERATION_STEPS_S2
                                        * abs(remaining)
                                    ),
                                )
                                set_actuator_speed(math.copysign(speed, remaining))
                            actuator_settle_since = None
                        elif actuator_error is not None:
                            set_actuator_speed(0.0)
                            if actuator_settle_since is None:
                                actuator_settle_since = now

                        servo_settled = (
                            servo_error is not None
                            and abs(servo_error) <= SERVO_SETTLE_DEADBAND_PX
                            and servo_settle_since is not None
                            and now - servo_settle_since >= SETTLE_DWELL_S
                        )
                        actuator_settled = (
                            not actuator_is_referenced()
                            or (
                                actuator_error is not None
                                and abs(actuator_error) <= ACTUATOR_SETTLE_DEADBAND_PX
                                and actuator_settle_since is not None
                                and now - actuator_settle_since >= SETTLE_DWELL_S
                            )
                        )

                        target_reached = servo_settled and actuator_settled

                        # Automatic jitter starts ONLY after the complete 2-DOF
                        # target has been reached. Both the servo (Y) and actuator
                        # (X) must satisfy their settling conditions first.
                        # Jitter itself remains servo-only.
                        if target_reached and AUTO_JITTER and not jitter_done:
                            set_actuator_speed(0.0)
                            try:
                                servo_stopped = not servo.IsMoving(ST_SERVO_ID)
                            except Exception:
                                servo_stopped = False

                            if servo_stopped and wait_for_actuator_stop(2.0):
                                with _jitter_lock:
                                    _jitter_base = read_servo_position(servo_position)
                                    _jitter_start = time.monotonic()
                                    try:
                                        configure_jitter_servo()
                                    except Exception as exc:
                                        print(f"[JITTER] Cannot start: {exc}")
                                        _jitter_active = False
                                        continue
                                    _jitter_active = True
                                    jitter_manual = False

                        last_update = now

            # Manual jitter also runs when no target is active.
            if _jitter_active and not target_active and not sweep_active:
                set_actuator_speed(0.0)
                with _jitter_lock:
                    jitter_started = _jitter_start

                if (
                    jitter_started is not None
                    and now - jitter_started >= JITTER_DURATION_S
                ):
                    with _jitter_lock:
                        _jitter_active = False
                        base_position = _jitter_base
                    if base_position is not None:
                        servo_position = set_servo_position(
                            base_position,
                            speed=ST_SPEED_NEAR,
                            acc=ST_ACCEL,
                        )

            # Return to middle outside tracking.
            if not target_active and not sweep_active and not jog_mode:
                set_actuator_speed(0.0)

            # Manual actuator jog timeout.
            if jog_mode and actuator_jog_until and now >= actuator_jog_until:
                set_actuator_speed(0.0)
                actuator_jog_until = 0.0

            # ---------------- SWEEP ----------------
            if sweep_active:
                if not actuator_is_referenced():
                    sweep_active = False
                    sweep_leg = None
                    sweep_mode = None
                    set_actuator_speed(0.0)
                    print("[SWEEP] Aborted: actuator must be referenced")
                else:
                    middle = actuator_middle()
                    if sweep_leg == "to_middle_start":
                        servo_target = servo_middle_position()
                        actuator_target = middle
                    elif sweep_leg == "to_max":
                        servo_target = ST_VISIBLE_MAX_POSITION
                        actuator_target = ACTUATOR_MAX_POSITION
                    elif sweep_leg == "to_min":
                        servo_target = ST_VISIBLE_MIN_POSITION
                        actuator_target = ACTUATOR_MIN_POSITION
                    elif sweep_leg == "to_middle_end":
                        servo_target = servo_middle_position()
                        actuator_target = middle
                    else:
                        sweep_active = False
                        set_actuator_speed(0.0)
                        continue

                    # Start/end always move BOTH DOF to MIDDLE. During MAX/MIN,
                    # only the selected DOF moves; the other remains at MIDDLE.
                    move_servo = (
                        sweep_leg in ("to_middle_start", "to_middle_end")
                        or sweep_mode in ("servo", "both")
                    )
                    move_actuator = (
                        sweep_leg in ("to_middle_start", "to_middle_end")
                        or sweep_mode in ("actuator", "both")
                    )

                    servo_arrived = not move_servo
                    if move_servo:
                        if not sweep_servo_commanded:
                            set_servo_position(
                                servo_target,
                                speed=ST_SPEED_MAX,
                                acc=ST_ACCEL,
                            )
                            sweep_servo_commanded = True
                        actual_servo = servo.ReadPosition(ST_SERVO_ID)
                        if actual_servo is not None:
                            servo_position = float(actual_servo)
                            servo_arrived = abs(actual_servo - servo_target) <= 8

                    actuator_arrived = not move_actuator
                    if move_actuator:
                        current = get_actuator_position()
                        remaining = actuator_target - current
                        if (
                            abs(remaining) <= ACT_ENDPOINT_TOLERANCE_STEPS
                            and abs(get_actuator_speed()) <= ACT_STOP_SPEED_THRESHOLD
                        ):
                            set_actuator_speed(0.0)
                            actuator_arrived = True
                        else:
                            speed = min(
                                ACT_MAX_SPEED_STEPS_S,
                                math.sqrt(
                                    2.0 * ACT_ACCELERATION_STEPS_S2 * abs(remaining)
                                ),
                            )
                            set_actuator_speed(
                                math.copysign(speed, remaining) if remaining else 0.0
                            )
                            actuator_arrived = False
                    else:
                        set_actuator_speed(0.0)

                    if servo_arrived and actuator_arrived:
                        set_actuator_speed(0.0)
                        sweep_servo_commanded = False

                        if sweep_leg == "to_middle_start":
                            sweep_leg = "to_max"
                            print("[SWEEP] MIDDLE -> MAX")
                        elif sweep_leg == "to_max":
                            sweep_leg = "to_min"
                            print("[SWEEP] MAX -> MIN")
                        elif sweep_leg == "to_min":
                            sweep_leg = "to_middle_end"
                            print("[SWEEP] MIN -> MIDDLE")
                        else:
                            print(f"[SWEEP] Complete: {sweep_mode.upper()}")
                            sweep_active = False
                            sweep_leg = None
                            sweep_mode = None

            # Draw UI.
            output = frame.copy()
            draw_cross(output, laser_x, laser_y, (0, 0, 255), f"Laser {laser_x},{laser_y}")

            if ui_state["target"] is not None:
                tx, ty = ui_state["target"]
                draw_cross(output, tx, ty, (255, 0, 0), f"Target {tx},{ty}")

            if target_active and ui_state["target"] is not None:
                status = "JITTERING" if _jitter_active else "TRACKING"
            elif sweep_active:
                status = f"SWEEP {sweep_mode.upper()} {sweep_leg}"
            elif jog_mode:
                status = "JOG"
            else:
                status = "IDLE"

            cv2.putText(output, status, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(output, f"FPS {fps:.1f}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(output, f"Servo {servo_position:.0f}  eY={servo_error if servo_error is not None else 0:+.1f}", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Act {get_actuator_position()}  v={get_actuator_speed():+.0f}  eX={actuator_error if actuator_error is not None else 0:+.1f}", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Act ref={'YES' if actuator_is_referenced() else 'NO'}  travel={ACTUATOR_MAX_POSITION}", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Signs: servo={SERVO_SIGN:+d} actuator={ACTUATOR_SIGN:+d}", (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow(window, output)
            cv2.imshow(raw_window, frame)

            key = cv2.waitKey(1) & 0xFF

            if sweep_selection_pending:
                if key == ord("1"):
                    sweep_mode = "servo"
                elif key == ord("2"):
                    sweep_mode = "actuator"
                elif key == ord("3"):
                    sweep_mode = "both"
                elif key in (ord("c"), 27):
                    sweep_selection_pending = False
                    continue
                else:
                    continue

                sweep_selection_pending = False
                if sweep_mode in ("actuator", "both") and not actuator_is_referenced():
                    print("[SWEEP] Actuator sweep requires a reference")
                    sweep_mode = None
                    continue

                # Every sweep starts with both DOF at middle.
                sweep_active = True
                sweep_leg = "to_middle_start"
                sweep_servo_commanded = False
                set_actuator_speed(0.0)
                print(f"[SWEEP] Starting {sweep_mode.upper()} -- MIDDLE -> MAX -> MIN -> MIDDLE")
                continue

            if key == ord("q"):
                break

            if key == ord("g"):
                if ui_state["target"] is None:
                    print("[TARGET] Set a target with left click first")
                elif not actuator_is_referenced():
                    print("[TARGET] Actuator is not referenced")
                else:
                    target_active = True
                    target_reached = False
                    jitter_done = False
                    filtered_x = None
                    filtered_y = None
                    servo_settle_since = None
                    actuator_settle_since = None
                    with _jitter_lock:
                        _jitter_active = False
                        _jitter_start = None
                        _jitter_base = None
                    print(f"[TARGET] Tracking ({ui_state['target'][0]}, {ui_state['target'][1]})")
                continue

            if key == ord("r"):
                target_active = False
                target_reached = False
                jitter_done = False
                jitter_manual = False
                ui_state["target"] = None
                filtered_x = None
                filtered_y = None
                servo_settle_since = None
                actuator_settle_since = None
                with _jitter_lock:
                    _jitter_active = False
                    _jitter_start = None
                    _jitter_base = None
                jog_mode = False
                sweep_active = False
                sweep_mode = None
                sweep_leg = None
                set_actuator_speed(0.0)

                servo_position = set_servo_position(
                    servo_middle_position(),
                    speed=ST_SPEED_NEAR,
                    acc=ST_ACCEL,
                )
                if actuator_is_referenced():
                    middle = actuator_middle()
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        current = get_actuator_position()
                        remaining = middle - current
                        if abs(remaining) <= ACT_ENDPOINT_TOLERANCE_STEPS:
                            set_actuator_speed(0.0)
                            break
                        speed = min(
                            ACT_MAX_SPEED_STEPS_S,
                            math.sqrt(2.0 * ACT_ACCELERATION_STEPS_S2 * abs(remaining)),
                        )
                        set_actuator_speed(math.copysign(speed, remaining))
                        time.sleep(0.01)
                    wait_for_actuator_stop(1.0)
                print("[SYSTEM] Return to MIDDLE commanded")
                continue

            if key == ord("c"):
                target_active = False
                target_reached = False
                jitter_done = False
                jitter_manual = False
                ui_state["target"] = None
                filtered_x = None
                filtered_y = None
                servo_settle_since = None
                actuator_settle_since = None
                with _jitter_lock:
                    _jitter_active = False
                    _jitter_start = None
                    _jitter_base = None
                set_actuator_speed(0.0)
                sweep_active = False
                sweep_selection_pending = False
                sweep_mode = None
                sweep_leg = None
                print("[SYSTEM] Cancelled")
                continue

            if key == ord("m"):
                jog_mode = not jog_mode
                target_active = False
                target_reached = False
                jitter_done = False
                jitter_manual = False
                ui_state["target"] = None
                with _jitter_lock:
                    _jitter_active = False
                    _jitter_start = None
                    _jitter_base = None
                set_actuator_speed(0.0)
                actuator_jog_until = 0.0
                print(f"[JOG] {'ON' if jog_mode else 'OFF'}")
                continue

            if key == ord("i") and jog_mode:
                servo_position = set_servo_position(
                    servo_position - JOG_STEP_SERVO,
                    speed=JOG_SPEED_SERVO,
                    acc=ST_ACCEL,
                )
                print(f"[JOG] Servo -> {servo_position:.0f}")
                continue

            if key == ord("k") and jog_mode:
                servo_position = set_servo_position(
                    servo_position + JOG_STEP_SERVO,
                    speed=JOG_SPEED_SERVO,
                    acc=ST_ACCEL,
                )
                print(f"[JOG] Servo -> {servo_position:.0f}")
                continue

            if key == ord("u") and jog_mode:
                with _actuator_lock:
                    position = _actuator_position
                    extension = _actuator_extension_mode
                    referenced = _actuator_referenced
                if extension:
                    allowed = position > -ACTUATOR_CALIBRATION_EXTENSION_LIMIT
                elif referenced:
                    allowed = position > ACTUATOR_MIN_POSITION
                else:
                    allowed = position > -ACT_UNREFERENCED_JOG_LIMIT
                if allowed:
                    set_actuator_speed(-ACT_JOG_SPEED_STEPS_S)
                    actuator_jog_until = now + ACTUATOR_JOG_PULSE_S
                else:
                    set_actuator_speed(0.0)
                continue

            if key == ord("o") and jog_mode:
                with _actuator_lock:
                    position = _actuator_position
                    extension = _actuator_extension_mode
                    referenced = _actuator_referenced
                if extension:
                    allowed = position < ACTUATOR_CALIBRATION_EXTENSION_LIMIT
                elif referenced:
                    allowed = position < ACTUATOR_MAX_POSITION
                else:
                    allowed = position < ACT_UNREFERENCED_JOG_LIMIT
                if allowed:
                    set_actuator_speed(ACT_JOG_SPEED_STEPS_S)
                    actuator_jog_until = now + ACTUATOR_JOG_PULSE_S
                else:
                    set_actuator_speed(0.0)
                continue

            if key == ord("e") and jog_mode:
                _actuator_extension_mode = not _actuator_extension_mode
                set_actuator_speed(0.0)
                print(
                    f"[CAL] Extension mode {'ON' if _actuator_extension_mode else 'OFF'}"
                )
                continue

            if key == ord("1") and jog_mode:
                servo_position = read_servo_position(servo_position)
                ST_VISIBLE_MIN_POSITION = int(round(servo_position))
                save_calibration()
                print(f"[CAL] Servo MIN={ST_VISIBLE_MIN_POSITION}")
                continue

            if key == ord("2") and jog_mode:
                servo_position = read_servo_position(servo_position)
                ST_VISIBLE_MAX_POSITION = int(round(servo_position))
                save_calibration()
                print(f"[CAL] Servo MAX={ST_VISIBLE_MAX_POSITION}")
                continue

            if key == ord("3") and jog_mode:
                mark_actuator_min()
                if ACTUATOR_MAX_POSITION > 0:
                    save_calibration()
                continue

            if key == ord("4") and jog_mode:
                if not mark_actuator_max():
                    print("[CAL] Mark MIN with 3 before MAX with 4")
                continue

            if key == ord("t") and not jog_mode:
                if target_active:
                    print("[SWEEP] Cancel target first with C")
                else:
                    sweep_selection_pending = True
                    print("[SWEEP] 1=SERVO  2=ACTUATOR  3=BOTH  C=cancel")
                continue

            if key == ord("j"):
                with _jitter_lock:
                    if _jitter_active:
                        print("[JITTER] Already active")
                    else:
                        set_actuator_speed(0.0)
                        wait_for_actuator_stop(2.0)
                        _jitter_base = read_servo_position(servo_position)
                        servo_position = _jitter_base
                        try:
                            configure_jitter_servo()
                        except Exception as exc:
                            print(f"[JITTER] Cannot start: {exc}")
                            continue
                        _jitter_start = time.monotonic()
                        _jitter_active = True
                        jitter_manual = True
                        print(
                            f"[JITTER] Servo only, amplitude={JITTER_AMPLITUDE_POS}, "
                            f"requested={JITTER_FREQUENCY_HZ} Hz"
                        )
                continue

    finally:
        with _jitter_lock:
            _jitter_active = False
        _jitter_running = False
        set_actuator_speed(0.0)

        if actuator_is_referenced():
            middle = actuator_middle()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                current = get_actuator_position()
                remaining = middle - current
                if abs(remaining) <= ACT_ENDPOINT_TOLERANCE_STEPS:
                    set_actuator_speed(0.0)
                    break
                speed = min(
                    ACT_MAX_SPEED_STEPS_S,
                    math.sqrt(2.0 * ACT_ACCELERATION_STEPS_S2 * abs(remaining)),
                )
                set_actuator_speed(math.copysign(speed, remaining))
                time.sleep(0.01)
            wait_for_actuator_stop(1.0)
        else:
            set_actuator_speed(0.0)

        try:
            set_servo_position(servo_middle_position(), speed=ST_SPEED_NEAR, acc=ST_ACCEL)
        except Exception:
            pass

        time.sleep(0.15)

        _actuator_running = False
        actuator_worker.join(timeout=1.0)
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