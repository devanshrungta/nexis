#!/usr/bin/env python3

import cv2
import math
import os
import json
import time
import threading

import numpy as np
import Jetson.GPIO as GPIO
from st3215 import ST3215

from position_map import PositionMap, PositionMapError


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
MAX_ACTUATOR_STEP_PER_UPDATE = 15000


# ============================================================
# VISION-ONLY MOVING TARGET TRACKING
#
# The clicked point is NOT kept as a fixed image coordinate.
# Instead, the tracker follows visual features around that point.
#
# Two complementary visual estimates are used:
#   1. LOCAL target optical flow: follows a target that moves in
#      the image, including an independently moving object.
#   2. GLOBAL background homography: follows a world-fixed point
#      when the camera/setup moves relative to the scene.
#
# LOCAL is preferred when it has good target features. GLOBAL is
# the fallback when the local target texture is weak/lost.
# No IMU, GPS, encoder, or world-position information is used.
# ============================================================

VISION_TRACKING_ENABLED = True

VISION_TARGET_ROI_PX = 70
# Half-width of the feature-search area around the selected target.

VISION_TARGET_MAX_FEATURES = 40
VISION_BACKGROUND_MAX_FEATURES = 180
VISION_FEATURE_QUALITY = 0.01
VISION_FEATURE_MIN_DISTANCE = 7

VISION_LK_WIN_SIZE = (21, 21)
VISION_LK_MAX_LEVEL = 3
VISION_LK_ERROR_MAX = 30.0

VISION_MIN_TARGET_INLIERS = 4
VISION_MIN_BACKGROUND_INLIERS = 8
VISION_MIN_INLIER_RATIO = 0.25

VISION_AFFINE_RANSAC_PX = 4.0
VISION_HOMOGRAPHY_RANSAC_PX = 4.0

VISION_FUSION_MAX_DIFF_PX = 25.0
# If local and global predictions agree this closely, fuse them.
# If they disagree substantially, the local target estimate wins
# because that can represent independent target motion.

VISION_TARGET_SMOOTH_ALPHA = 0.80
# Higher = less lag, lower = more smoothing.

VISION_MAX_TARGET_JUMP_PX = 160.0
# Reject impossible single-frame jumps, which can occur after a bad
# optical-flow match or feature mismatch.

VISION_REDETECT_TARGET_BELOW = 6
VISION_REDETECT_BACKGROUND_BELOW = 40
VISION_MAX_LOST_FRAMES = 8


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
POSITION_MAP_FILE = "position_map.json"

# Runtime uses absolute calibrated positions. These thresholds only avoid
# redundant commands when the tracked target moves by a negligible amount.
POSITION_SERVO_COMMAND_MIN_DELTA = 1.0
POSITION_ACTUATOR_COMMAND_MIN_DELTA = 1.0
POSITION_SERVO_STATUS_CHECK_INTERVAL = 0.05


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
_actuator_position_target = None
_actuator_position_mode = False

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
    """Legacy velocity command used by jog/sweep/control stop paths."""
    global _actuator_target_speed, _actuator_position_target, _actuator_position_mode
    with _actuator_lock:
        _actuator_target_speed = float(speed)
        _actuator_position_target = None
        _actuator_position_mode = False


def set_actuator_position_target(position):
    """Command an absolute calibrated actuator position."""
    global _actuator_target_speed, _actuator_position_target, _actuator_position_mode
    position = float(
        clamp(position, ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION)
    )
    with _actuator_lock:
        _actuator_position_target = position
        _actuator_position_mode = True
        _actuator_target_speed = 0.0
    return position


def get_actuator_position():
    with _actuator_lock:
        return _actuator_position


def get_actuator_speed():
    with _actuator_lock:
        return _actuator_speed


def get_actuator_position_target():
    with _actuator_lock:
        return _actuator_position_target


def wait_for_actuator_stop(timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if abs(get_actuator_speed()) <= ACT_STOP_SPEED_THRESHOLD:
            return True
        time.sleep(0.005)
    return abs(get_actuator_speed()) <= ACT_STOP_SPEED_THRESHOLD


def actuator_thread():
    global _actuator_position, _actuator_speed, _actuator_target_speed
    global _actuator_position_target, _actuator_position_mode

    current_speed = 0.0
    current_direction = 0
    last_time = time.monotonic()

    while _actuator_running:
        now = time.monotonic()
        dt = min(now - last_time, 0.03)
        last_time = now

        with _actuator_lock:
            velocity_target = _actuator_target_speed
            position_target = _actuator_position_target
            position_mode = _actuator_position_mode
            position = _actuator_position
            referenced = _actuator_referenced
            extension_mode = _actuator_extension_mode
            max_position = ACTUATOR_MAX_POSITION

        if position_mode and position_target is not None:
            remaining = position_target - position
            if abs(remaining) <= ACT_ENDPOINT_TOLERANCE_STEPS:
                target_speed = 0.0
            else:
                # Fastest commanded velocity that can still decelerate to the
                # target under the existing tested acceleration limit.
                braking_speed = math.sqrt(
                    2.0 * ACT_ACCELERATION_STEPS_S2 * abs(remaining)
                )
                target_speed = math.copysign(
                    min(ACT_MAX_SPEED_STEPS_S, braking_speed),
                    remaining,
                )
        else:
            target_speed = velocity_target
            target_direction = (
                1 if target_speed > 0 else -1 if target_speed < 0 else 0
            )
            if (
                current_direction
                and target_direction
                and target_direction != current_direction
            ):
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
                with _actuator_lock:
                    _actuator_target_speed = 0.0
                    _actuator_position_target = None
                    _actuator_position_mode = False
            elif current_speed < 0 and position <= -ACTUATOR_CALIBRATION_EXTENSION_LIMIT:
                current_speed = 0.0
                with _actuator_lock:
                    _actuator_target_speed = 0.0
                    _actuator_position_target = None
                    _actuator_position_mode = False
        elif referenced and max_position > 0:
            if current_speed > 0:
                distance_to_end = max_position - position
            else:
                distance_to_end = position

            if distance_to_end <= 0:
                current_speed = 0.0
                with _actuator_lock:
                    _actuator_target_speed = 0.0
                    if not position_mode:
                        _actuator_position_target = None
                        _actuator_position_mode = False
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
                    clamp(
                        _actuator_position,
                        ACTUATOR_MIN_POSITION,
                        ACTUATOR_MAX_POSITION,
                    )
                )

    with _actuator_lock:
        _actuator_speed = 0.0
        _actuator_target_speed = 0.0
        _actuator_position_target = None
        _actuator_position_mode = False


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


class VisionTargetTracker:
    """Vision-only tracker for a clicked image point."""

    def __init__(self):
        self.active = False
        self.point = None
        self.prev_gray = None
        self.target_points = None
        self.background_points = None
        self.status = "OFF"
        self.target_inliers = 0
        self.background_inliers = 0
        self.global_dx = 0.0
        self.global_dy = 0.0
        self.target_dx = 0.0
        self.target_dy = 0.0
        self.relative_dx = 0.0
        self.relative_dy = 0.0
        self.lost_frames = 0

    def reset(self):
        self.active = False
        self.point = None
        self.prev_gray = None
        self.target_points = None
        self.background_points = None
        self.status = "OFF"
        self.target_inliers = 0
        self.background_inliers = 0
        self.global_dx = 0.0
        self.global_dy = 0.0
        self.target_dx = 0.0
        self.target_dy = 0.0
        self.relative_dx = 0.0
        self.relative_dy = 0.0
        self.lost_frames = 0

    @staticmethod
    def _inside(point, width, height, margin=2):
        x, y = float(point[0]), float(point[1])
        return (
            margin <= x < width - margin
            and margin <= y < height - margin
        )

    def _target_mask(self, gray, point, laser_point=None):
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y = int(round(point[0])), int(round(point[1]))
        r = int(VISION_TARGET_ROI_PX)
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        mask[y0:y1, x0:x1] = 255

        if laser_point[0] is not None and laser_point[1] is not None:
            lx, ly = int(laser_point[0]), int(laser_point[1])
            cv2.circle(mask, (lx, ly), 18, 0, -1)

        return mask

    def _background_mask(self, gray, target_point, laser_point=None):
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)

        x, y = int(round(target_point[0])), int(round(target_point[1]))
        r = int(VISION_TARGET_ROI_PX + 15)
        cv2.rectangle(
            mask,
            (max(0, x - r), max(0, y - r)),
            (min(w - 1, x + r), min(h - 1, y + r)),
            0,
            -1,
        )

        if laser_point[0] is not None and laser_point[1] is not None:
            lx, ly = int(laser_point[0]), int(laser_point[1])
            cv2.circle(mask, (lx, ly), 25, 0, -1)

        return mask

    def _detect_target_features(self, gray, point, laser_point=None):
        mask = self._target_mask(gray, point, laser_point)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=VISION_TARGET_MAX_FEATURES,
            qualityLevel=VISION_FEATURE_QUALITY,
            minDistance=VISION_FEATURE_MIN_DISTANCE,
            blockSize=7,
            mask=mask,
        )
        if pts is None:
            return None
        return pts.reshape(-1, 2).astype(np.float32)

    def _detect_background_features(self, gray, point, laser_point=None):
        mask = self._background_mask(gray, point, laser_point)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=VISION_BACKGROUND_MAX_FEATURES,
            qualityLevel=VISION_FEATURE_QUALITY,
            minDistance=VISION_FEATURE_MIN_DISTANCE,
            blockSize=7,
            mask=mask,
        )
        if pts is None:
            return None
        return pts.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _track_points(prev_gray, gray, points):
        if points is None or len(points) < 3:
            return None, None, None

        p0 = points.reshape(-1, 1, 2)
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            p0,
            None,
            winSize=VISION_LK_WIN_SIZE,
            maxLevel=VISION_LK_MAX_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )

        if p1 is None or status is None:
            return None, None, None

        status = status.reshape(-1).astype(bool)
        error = (
            error.reshape(-1)
            if error is not None
            else np.full(len(status), 0.0, dtype=np.float32)
        )
        valid = status & np.isfinite(error) & (error <= VISION_LK_ERROR_MAX)

        old_valid = points[valid]
        new_valid = p1.reshape(-1, 2)[valid]

        if len(old_valid) == 0:
            return None, None, None
        return old_valid.astype(np.float32), new_valid.astype(np.float32), error[valid]

    @staticmethod
    def _affine_prediction(old_points, new_points, reference_point):
        if len(old_points) < VISION_MIN_TARGET_INLIERS:
            return None, 0, 0.0

        matrix, inliers = cv2.estimateAffinePartial2D(
            old_points,
            new_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=VISION_AFFINE_RANSAC_PX,
            maxIters=200,
            confidence=0.99,
            refineIters=10,
        )

        if matrix is None or inliers is None:
            return None, 0, 0.0

        inlier_mask = inliers.reshape(-1).astype(bool)
        count = int(np.count_nonzero(inlier_mask))
        ratio = count / max(1, len(old_points))
        if count < VISION_MIN_TARGET_INLIERS or ratio < VISION_MIN_INLIER_RATIO:
            return None, count, ratio

        src = np.array([[reference_point]], dtype=np.float32)
        predicted = cv2.transform(src, matrix)[0, 0]
        return predicted.astype(np.float32), count, ratio

    @staticmethod
    def _homography_prediction(old_points, new_points, reference_point):
        if len(old_points) < VISION_MIN_BACKGROUND_INLIERS:
            return None, 0, 0.0

        matrix, inliers = cv2.findHomography(
            old_points,
            new_points,
            cv2.RANSAC,
            VISION_HOMOGRAPHY_RANSAC_PX,
        )

        if matrix is None or inliers is None:
            return None, 0, 0.0

        inlier_mask = inliers.reshape(-1).astype(bool)
        count = int(np.count_nonzero(inlier_mask))
        ratio = count / max(1, len(old_points))
        if count < VISION_MIN_BACKGROUND_INLIERS or ratio < VISION_MIN_INLIER_RATIO:
            return None, count, ratio

        src = np.array([[[reference_point[0], reference_point[1]]]], dtype=np.float32)
        predicted = cv2.perspectiveTransform(src, matrix)[0, 0]
        return predicted.astype(np.float32), count, ratio

    def initialize(self, gray, point, laser_point=(None, None)):
        self.reset()

        point = np.array([float(point[0]), float(point[1])], dtype=np.float32)
        h, w = gray.shape[:2]
        if not self._inside(point, w, h):
            return False

        self.point = point
        self.prev_gray = gray.copy()
        self.target_points = self._detect_target_features(
            gray, self.point, laser_point
        )
        self.background_points = self._detect_background_features(
            gray, self.point, laser_point
        )
        self.active = True
        self.status = (
            "LOCAL"
            if self.target_points is not None
            and len(self.target_points) >= VISION_MIN_TARGET_INLIERS
            else "GLOBAL"
            if self.background_points is not None
            and len(self.background_points) >= VISION_MIN_BACKGROUND_INLIERS
            else "WEAK"
        )
        return True

    def update(self, gray, laser_point=(None, None)):
        if not self.active or self.point is None or self.prev_gray is None:
            return None

        h, w = gray.shape[:2]
        previous_point = self.point.copy()

        local_pred = None
        local_inliers = 0
        local_ratio = 0.0
        local_old = local_new = None

        if self.target_points is not None and len(self.target_points) >= 3:
            local_old, local_new, _ = self._track_points(
                self.prev_gray, gray, self.target_points
            )
            if local_old is not None:
                local_pred, local_inliers, local_ratio = self._affine_prediction(
                    local_old, local_new, previous_point
                )
                if local_pred is None and len(local_old) >= 3:
                    displacement = np.median(local_new - local_old, axis=0)
                    local_pred = previous_point + displacement
                    local_inliers = len(local_old)
                    local_ratio = 0.20

        global_pred = None
        global_inliers = 0
        global_ratio = 0.0
        bg_old = bg_new = None

        if self.background_points is not None and len(self.background_points) >= 4:
            bg_old, bg_new, _ = self._track_points(
                self.prev_gray, gray, self.background_points
            )
            if bg_old is not None:
                global_pred, global_inliers, global_ratio = self._homography_prediction(
                    bg_old, bg_new, previous_point
                )
                if global_pred is None and len(bg_old) >= VISION_MIN_BACKGROUND_INLIERS:
                    affine_pred, count, ratio = self._affine_prediction(
                        bg_old, bg_new, previous_point
                    )
                    global_pred = affine_pred
                    global_inliers = count
                    global_ratio = ratio

        self.target_inliers = local_inliers
        self.background_inliers = global_inliers

        candidate = None
        status = "LOST"

        local_good = (
            local_pred is not None
            and local_inliers >= VISION_MIN_TARGET_INLIERS
            and local_ratio >= VISION_MIN_INLIER_RATIO
        )
        global_good = (
            global_pred is not None
            and global_inliers >= VISION_MIN_BACKGROUND_INLIERS
            and global_ratio >= VISION_MIN_INLIER_RATIO
        )

        if local_good and global_good:
            local_global_gap = float(np.linalg.norm(local_pred - global_pred))
            if local_global_gap <= VISION_FUSION_MAX_DIFF_PX:
                candidate = (
                    0.75 * local_pred + 0.25 * global_pred
                ).astype(np.float32)
                status = "HYBRID"
            else:
                candidate = local_pred
                status = "LOCAL"
        elif local_good:
            candidate = local_pred
            status = "LOCAL"
        elif global_good:
            candidate = global_pred
            status = "GLOBAL"
        elif local_pred is not None:
            candidate = local_pred
            status = "LOCAL-WEAK"
        elif global_pred is not None:
            candidate = global_pred
            status = "GLOBAL-WEAK"

        if candidate is not None:
            jump = float(np.linalg.norm(candidate - previous_point))
            if jump <= VISION_MAX_TARGET_JUMP_PX and self._inside(candidate, w, h):
                smoothed = (
                    VISION_TARGET_SMOOTH_ALPHA * candidate
                    + (1.0 - VISION_TARGET_SMOOTH_ALPHA) * previous_point
                ).astype(np.float32)
                self.point = smoothed
                self.status = status
                self.lost_frames = 0

                self.target_dx = float(self.point[0] - previous_point[0])
                self.target_dy = float(self.point[1] - previous_point[1])

                if global_pred is not None:
                    self.global_dx = float(global_pred[0] - previous_point[0])
                    self.global_dy = float(global_pred[1] - previous_point[1])
                else:
                    self.global_dx = 0.0
                    self.global_dy = 0.0

                self.relative_dx = self.target_dx - self.global_dx
                self.relative_dy = self.target_dy - self.global_dy
            else:
                candidate = None

        if candidate is None:
            self.lost_frames += 1
            self.status = "LOST"

        # Refresh target features around the newest estimate when the
        # existing tracks are getting sparse.
        if self.point is not None:
            if (
                self.target_points is None
                or len(self.target_points) < VISION_REDETECT_TARGET_BELOW
            ):
                refreshed = self._detect_target_features(
                    gray, self.point, laser_point
                )
                if refreshed is not None and len(refreshed) > 0:
                    self.target_points = refreshed
                else:
                    self.target_points = (
                        local_new if local_new is not None else None
                    )
            elif local_new is not None:
                self.target_points = local_new

            if (
                self.background_points is None
                or len(self.background_points) < VISION_REDETECT_BACKGROUND_BELOW
            ):
                refreshed_bg = self._detect_background_features(
                    gray, self.point, laser_point
                )
                if refreshed_bg is not None and len(refreshed_bg) > 0:
                    self.background_points = refreshed_bg
            elif bg_new is not None:
                self.background_points = bg_new

        self.prev_gray = gray.copy()

        if self.lost_frames > VISION_MAX_LOST_FRAMES:
            return None
        return (float(self.point[0]), float(self.point[1]))


def mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["target"] = (x, y)
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

    try:
        position_map = PositionMap.load(POSITION_MAP_FILE)
    except PositionMapError as exc:
        raise RuntimeError(str(exc)) from exc

    if (
        position_map.camera_width != FRAME_WIDTH
        or position_map.camera_height != FRAME_HEIGHT
    ):
        print(
            f"[MAP] WARNING: map calibrated at "
            f"{position_map.camera_width}x{position_map.camera_height}; "
            f"current camera is {FRAME_WIDTH}x{FRAME_HEIGHT}"
        )

    print(
        f"[MAP] Loaded {POSITION_MAP_FILE}: "
        f"{position_map.rows}x{position_map.cols} grid"
    )

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

    vision_tracker = VisionTargetTracker()
    vision_enabled = VISION_TRACKING_ENABLED

    actuator_worker = threading.Thread(target=actuator_thread, daemon=True)
    jitter_worker = threading.Thread(target=jitter_thread, daemon=True)
    actuator_worker.start()
    jitter_worker.start()
    set_actuator_enabled(True)
    last_update = 0.0
    servo_settle_since = None
    actuator_settle_since = None
    target_active = False
    target_reached = False
    jitter_done = False
    jitter_manual = False

    last_servo_status_check = 0.0
    servo_is_moving = False

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
    print(f"  Position map = {POSITION_MAP_FILE}")
    print("  V = toggle vision target tracking ON/OFF")
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            now = time.monotonic()
            if now - fps_start >= 1.0:
                fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now
            fps_count += 1

            servo_error = None
            actuator_error = None

            # --------------------------------------------------------
            # VISION TARGET TRACKER + CALIBRATED POSITION CONTROL
            # --------------------------------------------------------
            # Vision supplies the CURRENT target pixel. The one-time position
            # calibration converts that pixel directly into absolute servo and
            # actuator positions. Runtime laser detection is not used.
            # --------------------------------------------------------
            if ui_state.get("new_target", False):
                ui_state["new_target"] = False
                target_reached = False
                jitter_done = False
                jitter_manual = False
                servo_settle_since = None
                actuator_settle_since = None

                with _jitter_lock:
                    _jitter_active = False
                    _jitter_start = None
                    _jitter_base = None

                vision_tracker.reset()
                if target_active and ui_state["target"] is not None:
                    vision_tracker.initialize(
                        gray,
                        ui_state["target"],
                        (None, None),
                    )
                    set_actuator_speed(0.0)
                    print(
                        f"[TARGET] New target armed: "
                        f"({ui_state['target'][0]}, {ui_state['target'][1]})"
                    )

            tracked_target = None
            map_status = "-"
            mapped_servo_target = None
            mapped_actuator_target = None

            if target_active and ui_state["target"] is not None and not sweep_active:
                if vision_enabled:
                    if not vision_tracker.active:
                        vision_tracker.initialize(
                            gray,
                            ui_state["target"],
                            (None, None),
                        )
                    tracked_target = vision_tracker.update(gray, (None, None))
                    if tracked_target is None:
                        target_x = target_y = None
                    else:
                        target_x, target_y = tracked_target
                else:
                    target_x, target_y = ui_state["target"]
                    tracked_target = (float(target_x), float(target_y))

                if tracked_target is None and not _jitter_active:
                    # Never drive toward stale coordinates after vision loss.
                    set_actuator_speed(0.0)
                elif _jitter_active:
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

                        target_reached = True
                        if not jitter_manual:
                            jitter_done = True
                        jitter_manual = False

                else:
                    if tracked_target is not None and now - last_update >= CONTROL_UPDATE_INTERVAL:
                        mapped_servo_target, mapped_actuator_target, map_status = (
                            position_map.pixel_to_position(target_x, target_y)
                        )

                        mapped_servo_target = clamp(
                            mapped_servo_target,
                            ST_VISIBLE_MIN_POSITION,
                            ST_VISIBLE_MAX_POSITION,
                        )
                        mapped_actuator_target = clamp(
                            mapped_actuator_target,
                            ACTUATOR_MIN_POSITION,
                            ACTUATOR_MAX_POSITION,
                        )

                        # Absolute servo target. No ReadPosition() on every
                        # frame - the servo controller already receives the
                        # final calibrated position directly.
                        if (
                            abs(mapped_servo_target - servo_position)
                            >= POSITION_SERVO_COMMAND_MIN_DELTA
                        ):
                            servo_position = set_servo_position(
                                mapped_servo_target,
                                speed=ST_SPEED_MAX,
                                acc=ST_ACCEL,
                            )
                            servo_settle_since = None

                        # Absolute actuator target. The actuator worker now
                        # uses the full remaining distance to brake correctly
                        # instead of only commanding gain*pixel_error steps.
                        if actuator_is_referenced():
                            previous_target = get_actuator_position_target()
                            if (
                                previous_target is None
                                or abs(mapped_actuator_target - previous_target)
                                >= POSITION_ACTUATOR_COMMAND_MIN_DELTA
                            ):
                                set_actuator_position_target(mapped_actuator_target)
                                actuator_settle_since = None
                        else:
                            set_actuator_speed(0.0)
                            actuator_settle_since = None

                        if (
                            now - last_servo_status_check
                            >= POSITION_SERVO_STATUS_CHECK_INTERVAL
                        ):
                            try:
                                servo_is_moving = servo.IsMoving(ST_SERVO_ID)
                            except Exception:
                                servo_is_moving = True
                            last_servo_status_check = now

                        if servo_is_moving:
                            servo_settle_since = None
                        elif servo_settle_since is None:
                            servo_settle_since = now

                        if actuator_is_referenced():
                            current_actuator = get_actuator_position()
                            actuator_remaining = (
                                mapped_actuator_target - current_actuator
                            )
                            if (
                                abs(actuator_remaining) <= ACT_ENDPOINT_TOLERANCE_STEPS
                                and abs(get_actuator_speed()) <= ACT_STOP_SPEED_THRESHOLD
                            ):
                                if actuator_settle_since is None:
                                    actuator_settle_since = now
                            else:
                                actuator_settle_since = None

                        servo_settled = (
                            servo_settle_since is not None
                            and now - servo_settle_since >= SETTLE_DWELL_S
                        )
                        actuator_settled = (
                            not actuator_is_referenced()
                            or (
                                actuator_settle_since is not None
                                and now - actuator_settle_since >= SETTLE_DWELL_S
                            )
                        )

                        target_reached = servo_settled and actuator_settled

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
            if ui_state["target"] is not None:
                sx, sy = ui_state["target"]
                draw_cross(output, sx, sy, (255, 0, 0), f"Seed {sx},{sy}")

            if target_active and tracked_target is not None:
                tx, ty = tracked_target
                draw_cross(output, tx, ty, (0, 255, 0), f"Vision {tx:.0f},{ty:.0f}")

            if target_active and ui_state["target"] is not None:
                if _jitter_active:
                    status = "JITTERING"
                elif vision_enabled and tracked_target is None:
                    status = "VISION LOST"
                else:
                    status = "TRACKING"
            elif sweep_active:
                status = f"SWEEP {sweep_mode.upper()} {sweep_leg}"
            elif jog_mode:
                status = "JOG"
            else:
                status = "IDLE"

            cv2.putText(output, status, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(output, f"FPS {fps:.1f}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(output, f"Servo cmd {servo_position:.0f}", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Act pos {get_actuator_position()}  v={get_actuator_speed():+.0f}  target={get_actuator_position_target() if get_actuator_position_target() is not None else 0:.0f}", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Act ref={'YES' if actuator_is_referenced() else 'NO'}  travel={ACTUATOR_MAX_POSITION}", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Signs: servo={SERVO_SIGN:+d} actuator={ACTUATOR_SIGN:+d}", (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            if target_active and vision_enabled:
                cv2.putText(
                    output,
                    f"Vision {vision_tracker.status}  Tfeat={vision_tracker.target_inliers}  Bfeat={vision_tracker.background_inliers}",
                    (25, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    output,
                    f"dTarget=({vision_tracker.target_dx:+.1f},{vision_tracker.target_dy:+.1f})  dBG=({vision_tracker.global_dx:+.1f},{vision_tracker.global_dy:+.1f})",
                    (25, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )

            if target_active and tracked_target is not None:
                cv2.putText(
                    output,
                    f"Map {map_status}  servo={mapped_servo_target:.0f}  act={mapped_actuator_target:.0f}",
                    (25, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

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
                    servo_settle_since = None
                    actuator_settle_since = None
                    with _jitter_lock:
                        _jitter_active = False
                        _jitter_start = None
                        _jitter_base = None
                    vision_tracker.reset()
                    if vision_enabled:
                        vision_tracker.initialize(
                            gray,
                            ui_state["target"],
                            (None, None),
                        )
                    print(f"[TARGET] Tracking ({ui_state['target'][0]}, {ui_state['target'][1]})")
                continue

            if key == ord("v"):
                vision_enabled = not vision_enabled
                vision_tracker.reset()
                if vision_enabled and target_active and ui_state["target"] is not None:
                    vision_tracker.initialize(
                        gray,
                        ui_state["target"],
                        (None, None),
                    )
                print(f"[VISION] Target tracking {'ON' if vision_enabled else 'OFF'}")
                continue

            if key == ord("r"):
                target_active = False
                vision_tracker.reset()
                target_reached = False
                jitter_done = False
                jitter_manual = False
                ui_state["target"] = None
                servo_settle_since = None
                actuator_settle_since = None
                with _jitter_lock:
                    _jitter_active = False
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
                vision_tracker.reset()
                target_reached = False
                jitter_done = False
                jitter_manual = False
                ui_state["target"] = None
                servo_settle_since = None
                actuator_settle_since = None
                with _jitter_lock:
                    _jitter_active = False
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
                vision_tracker.reset()
                target_reached = False
                jitter_done = False
                jitter_manual = False
                ui_state["target"] = None
                with _jitter_lock:
                    _jitter_active = False
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
        vision_tracker.reset()
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