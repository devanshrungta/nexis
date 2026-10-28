#!/usr/bin/env python3

import cv2
import math
import os
import json
import time
import threading
from collections import deque

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
ST_MIN_POSITION = 1740
ST_MAX_POSITION = 2440
ST_MIDDLE_POSITION = 2048
# Fallback middle before calibration is loaded. After calibration, the
# actual safe middle is computed from ST_VISIBLE_MIN/MAX_POSITION.

# Calibrated range in which the laser remains usable in the camera view.
ST_VISIBLE_MIN_POSITION = ST_MIN_POSITION
ST_VISIBLE_MAX_POSITION = ST_MAX_POSITION

ST_SPEED_MAX = 6000
ST_SPEED_NEAR = 100
ST_SPEED_EASE_PX = 60
ST_ACCEL = 200

JOG_STEP_SERVO = 20
JOG_SPEED_SERVO = 200

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
# Require a short stable dwell before declaring the target reached and
# starting automatic servo jitter. Tiny target-map fluctuations are filtered
# by the command deadbands below, so they should not keep restarting this timer.
SETTLE_DWELL_S = 0.35

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
JITTER_FREQUENCY_HZ = 6.0
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

ACT_MAX_SPEED_STEPS_S = 2500
ACT_ACCELERATION_STEPS_S2 = 1250

ACT_JOG_SPEED_STEPS_S = 1400  # Retained for reference; U/O now use position targets.
ACTUATOR_JOG_PULSE_S = 0.15  # Retained for reference; fixed-step jog no longer uses timed pulses.
ACT_JOG_STEP_STEPS = 100
ACT_JOG_KEY_DEBOUNCE_S = 0.18

ACT_UNREFERENCED_JOG_LIMIT = 50000
ACTUATOR_CALIBRATION_EXTENSION_LIMIT = 50000

ACT_ENDPOINT_TOLERANCE_STEPS = 5
ACT_STOP_SPEED_THRESHOLD = 10

CALIBRATION_FILE = "servo_calibration.json"
POSITION_MAP_FILE = "position_map.json"

# ============================================================
# TARGET SOURCE: MANUAL POINT OR YOLO WEED DETECTION
# ============================================================
# Press P for the existing click-a-point mode, or Y for YOLO mode.
# YOLO is imported and the model is loaded only when you switch to YOLO mode.
YOLO_MODEL_PATH = "/home/sunoj-orin/devansh/camera/best 1.pt"

# None = all model classes. Example: [0, 2] = only class IDs 0 and 2.
# The model's class IDs/names are printed when YOLO mode is enabled.
YOLO_ALLOWED_CLASS_IDS = None

# Use a lower raw threshold so short confidence dips do not immediately remove
# a weed from the tracker. Stable tracks are confirmed across multiple results.
YOLO_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 640
YOLO_MAX_DETECTIONS = 50
YOLO_INPUT_SUBMIT_INTERVAL_S = 1.0 / 30.0
YOLO_DEVICE = None  # selected at model load: CUDA 0 when available, otherwise CPU
YOLO_HALF = True  # use FP16 inference when CUDA is available
YOLO_REQUIRE_CUDA = True  # fail fast rather than silently running very slowly on CPU

# Temporal detection stabilization.
YOLO_TRACK_CONFIRM_HITS = 2
YOLO_TRACK_MAX_MISSES = 5       # inference results, not camera frames
YOLO_BBOX_SMOOTH_ALPHA = 0.60   # higher follows moving targets faster
YOLO_CONF_SMOOTH_ALPHA = 0.35
YOLO_UNTRACKED_ASSOCIATION_PX = 90.0
YOLO_TRACK_REASSOCIATE_PX = 100.0

# Bbox target choice: "center" or "lower_center". Start with center.
YOLO_TARGET_POINT_MODE = "center"
YOLO_LOWER_CENTER_FRACTION = 0.65
YOLO_TARGET_SMOOTH_ALPHA = 0.60  # applied to the detector's smoothed target
YOLO_OPTICAL_FLOW_GRACE_S = 1.5  # continue visual tracking between detections
YOLO_FLOW_HISTORY_FRAMES = 90  # lets a detector result catch up through per-frame optical flow

# Autonomous visit mode: A starts a queue of stable weed tracks. Each weed is
# approached, allowed to settle, jittered, then the next queued weed is visited.
AUTO_VISIT_ENABLED = True
AUTO_VISIT_LOST_TIMEOUT_S = 2.5
AUTO_VISIT_QUEUE_STALE_S = 5.0
AUTO_VISIT_POST_JITTER_PAUSE_S = 0.20
AUTO_VISIT_ORDER = "nearest_motor"  # reduce unnecessary actuator/servo travel
AUTO_VISIT_ALLOW_EXTRAPOLATION = False  # do not auto-aim outside the measured calibration region by default
AUTO_VISIT_ACTUATOR_END_MARGIN_STEPS = 250  # keep auto targets away from software travel endpoints

# Runtime uses absolute calibrated positions. Ignore very small map/vision
# fluctuations so the motors do not receive repeated tiny retargets near rest.
POSITION_SERVO_COMMAND_MIN_DELTA = 2.0
POSITION_ACTUATOR_COMMAND_MIN_DELTA = 10.0
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
    """Wait against an absolute deadline to reduce STEP-pulse timing jitter.

    time.sleep() is too coarse for the ~50 us half-period used at 10,000
    steps/s, so short waits busy-spin instead of sleeping for a few us and
    potentially overshooting by much more. Longer waits sleep until the last
    100 us, then spin to the deadline. This improves timing consistency but
    cannot make Linux/Python GPIO a hard-real-time pulse generator.
    """
    seconds = max(0.000002, float(seconds))
    deadline = time.perf_counter() + seconds

    if seconds > 0.001:
        sleep_time = deadline - time.perf_counter() - 0.0001
        if sleep_time > 0.0:
            time.sleep(sleep_time)

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


def set_actuator_jog_target(position):
    """Set a fixed-step jog target with bounds appropriate to the current mode."""
    global _actuator_target_speed, _actuator_position_target, _actuator_position_mode

    with _actuator_lock:
        referenced = _actuator_referenced and ACTUATOR_MAX_POSITION > 0
        extension = _actuator_extension_mode
        if referenced:
            low, high = ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION
        elif extension:
            low, high = -ACTUATOR_CALIBRATION_EXTENSION_LIMIT, ACTUATOR_CALIBRATION_EXTENSION_LIMIT
        else:
            # This is only a software-count limit, not a physical end-stop.
            low, high = -ACT_UNREFERENCED_JOG_LIMIT, ACT_UNREFERENCED_JOG_LIMIT

        position = float(clamp(position, low, high))
        _actuator_position_target = position
        _actuator_position_mode = True
        _actuator_target_speed = 0.0

    return position


def command_actuator_jog_step(direction):
    """Advance the outstanding jog target by one fixed number of steps."""
    with _actuator_lock:
        if _actuator_position_mode and _actuator_position_target is not None:
            base_position = float(_actuator_position_target)
        else:
            base_position = float(_actuator_position)

    requested = base_position + int(direction) * ACT_JOG_STEP_STEPS
    target = set_actuator_jog_target(requested)
    if abs(target - base_position) < 0.5:
        print("[JOG] Software travel limit reached; no jog command issued")
    return target


def move_actuator_to_position_and_wait(target, timeout=15.0, label="move"):
    """Move once to an absolute target; avoid repeatedly reversing velocity near it."""
    if not actuator_is_referenced():
        set_actuator_speed(0.0)
        print(f"[ACTUATOR] {label}: cannot move to saved position because actuator is unreferenced")
        return False

    target = float(clamp(target, ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION))

    # Cancel any jog/sweep velocity request and let the actuator stop before
    # issuing one absolute position target.
    set_actuator_speed(0.0)
    if not wait_for_actuator_stop(4.0):
        print(f"[ACTUATOR] {label}: warning -- actuator did not stop promptly before repositioning")

    set_actuator_position_target(target)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = get_actuator_position()
        speed = get_actuator_speed()
        if (
            abs(target - current) <= ACT_ENDPOINT_TOLERANCE_STEPS
            and abs(speed) <= ACT_STOP_SPEED_THRESHOLD
        ):
            set_actuator_speed(0.0)
            wait_for_actuator_stop(1.0)
            print(f"[ACTUATOR] {label}: reached target {target:.0f} (position count={get_actuator_position()})")
            return True
        time.sleep(0.01)

    set_actuator_speed(0.0)
    wait_for_actuator_stop(4.0)
    print(
        f"[ACTUATOR] WARNING: {label} timed out; target={target:.0f}, "
        f"position count={get_actuator_position()}, speed={get_actuator_speed():.1f}"
    )
    return False


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
        elif not referenced:
            # A conservative software-count boundary for unreferenced motion.
            # It cannot protect against a physical end-stop if the initial count is unknown.
            unreferenced_limit = (
                ACTUATOR_CALIBRATION_EXTENSION_LIMIT
                if extension_mode
                else ACT_UNREFERENCED_JOG_LIMIT
            )
            if current_speed > 0 and position >= unreferenced_limit:
                current_speed = 0.0
                with _actuator_lock:
                    _actuator_target_speed = 0.0
                    _actuator_position_target = None
                    _actuator_position_mode = False
            elif current_speed < 0 and position <= -unreferenced_limit:
                current_speed = 0.0
                with _actuator_lock:
                    _actuator_target_speed = 0.0
                    _actuator_position_target = None
                    _actuator_position_mode = False

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



def load_yolo_model():
    """Load the optional Ultralytics .pt model; manual-point mode needs no YOLO install."""
    if not os.path.isfile(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO_MODEL_PATH does not exist: {YOLO_MODEL_PATH!r}. "
            "Edit YOLO_MODEL_PATH near the top of this script."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO mode requires the 'ultralytics' Python package in the same Python "
            "environment used to run this script. Manual-point mode is unaffected."
        ) from exc

    global YOLO_DEVICE
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available:
        YOLO_DEVICE = 0
        # Frames have a consistent camera/input size, so benchmark CUDA kernels once
        # to let cuDNN choose efficient implementations for this fixed workload.
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        print(f"[YOLO] GPU inference enabled: {torch.cuda.get_device_name(0)}")
    else:
        YOLO_DEVICE = "cpu"
        print("[YOLO] CUDA is unavailable in this Python environment; CPU inference will be much slower.")
        print("[YOLO] For GPU + hardware access, launch with:")
        print("       sudo /home/sunoj-orin/venvs/weed-yolo/bin/python laser_servo_tracker_yolo_stable_auto_visit.py")
        if YOLO_REQUIRE_CUDA:
            raise RuntimeError(
                "CUDA is required for YOLO in this controller. Use the weed-yolo virtualenv "
                "command above, or set YOLO_REQUIRE_CUDA=False only if slow CPU inference is intentional."
            )

    model = YOLO(YOLO_MODEL_PATH)
    names = getattr(model, "names", {})
    print(f"[YOLO] Loaded model: {YOLO_MODEL_PATH}")
    print(f"[YOLO] Model classes: {names}")
    if YOLO_ALLOWED_CLASS_IDS is None:
        print("[YOLO] Class filter: ALL classes")
    else:
        print(f"[YOLO] Class filter IDs: {YOLO_ALLOWED_CLASS_IDS}")
    precision = "FP16" if (YOLO_HALF and cuda_available) else "FP32"
    print(f"[YOLO] imgsz={YOLO_IMAGE_SIZE}, confidence={YOLO_CONFIDENCE:.2f}, device={YOLO_DEVICE}, precision={precision}")
    return model


def yolo_target_point(bbox):
    """Return a pixel target point from (x1, y1, x2, y2) bbox coordinates."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x = 0.5 * (x1 + x2)
    mode = str(YOLO_TARGET_POINT_MODE).strip().lower()
    if mode == "lower_center":
        fraction = clamp(float(YOLO_LOWER_CENTER_FRACTION), 0.0, 1.0)
        y = y1 + fraction * (y2 - y1)
    else:
        y = 0.5 * (y1 + y2)
    return float(x), float(y)


def seed_yolo_flow_tracker(
    tracker, source_frame, target, current_gray,
    source_frame_number=None, current_frame_number=None, gray_history=None,
):
    """Initialize in the inference frame and follow each intervening camera frame.

    Stepping through intervening frames avoids rejecting legitimate large motion
    simply because inference runs much slower than the camera.
    """
    if target is None:
        tracker.reset()
        return None
    if source_frame is None:
        tracker.initialize(current_gray, target, (None, None))
        return (float(target[0]), float(target[1]))

    source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    if not tracker.initialize(source_gray, target, (None, None)):
        return None

    history = list(gray_history) if gray_history is not None else []
    can_step = (
        source_frame_number is not None
        and current_frame_number is not None
        and history
        and any(seq == source_frame_number for seq, _ in history)
    )
    if can_step:
        current_point = (float(target[0]), float(target[1]))
        for seq, gray_frame in history:
            if source_frame_number < seq <= current_frame_number:
                propagated = tracker.update(gray_frame)
                if propagated is not None:
                    current_point = propagated
        if getattr(tracker, "lost_frames", 0) > VISION_MAX_LOST_FRAMES:
            return None
        if getattr(tracker, "point", None) is not None:
            return (float(tracker.point[0]), float(tracker.point[1]))
        return current_point

    # If the source frame fell out of the ring buffer, try direct propagation.
    return tracker.update(current_gray)


def extract_yolo_detections(model, frame):
    """Infer one frame, returning detections and Ultralytics' plotted output."""
    use_half = bool(YOLO_HALF and YOLO_DEVICE not in (None, "cpu"))
    kwargs = {
        "persist": True,
        "conf": float(YOLO_CONFIDENCE),
        "classes": YOLO_ALLOWED_CLASS_IDS,
        "imgsz": int(YOLO_IMAGE_SIZE),
        "max_det": int(YOLO_MAX_DETECTIONS),
        "verbose": False,
        "device": YOLO_DEVICE,
        # Ultralytics 8.4+ uses quantize=16 for FP16 inference.
        # Avoid the deprecated half= argument and its per-frame warning.
        "quantize": 16 if use_half else None,
    }

    results = model.track(frame, **kwargs)
    if not results:
        return [], frame.copy()

    result = results[0]
    # This is the same annotated-image API used by the user's standalone script.
    annotated = result.plot()
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], annotated

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    ids_tensor = getattr(boxes, "id", None)
    ids = ids_tensor.detach().cpu().numpy().astype(int) if ids_tensor is not None else None
    names = getattr(model, "names", {})

    detections = []
    for i, bbox in enumerate(xyxy):
        class_id = int(classes[i])
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            class_name = str(names[class_id])
        else:
            class_name = str(class_id)

        track_id = int(ids[i]) if ids is not None and i < len(ids) else None
        bbox_tuple = tuple(float(v) for v in bbox)
        tx, ty = yolo_target_point(bbox_tuple)
        detections.append({
            "bbox": bbox_tuple,
            "confidence": float(confs[i]),
            "class_id": class_id,
            "class_name": class_name,
            "track_id": track_id,
            "target": (tx, ty),
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, annotated


class YOLOTemporalStabilizer:
    """Smooth detector boxes/confidence and retain confirmed tracks briefly on misses.

    Ultralytics' persistent tracker supplies IDs when possible. If an output has
    no ID, a conservative proximity/IoU association assigns a temporary negative
    ID. Tracks are displayable/targetable only after they have been seen twice.
    """
    def __init__(self):
        self.tracks = {}
        self.next_synthetic_id = -1

    @staticmethod
    def _center(bbox):
        return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    def _synthetic_match(self, det, used):
        candidates = []
        cx, cy = self._center(det["bbox"])
        for tid, state in self.tracks.items():
            if tid >= 0 or tid in used or state["class_id"] != det["class_id"]:
                continue
            if state["missed"] > 2:
                continue
            iou = _bbox_iou(state["bbox"], det["bbox"])
            sx, sy = self._center(state["bbox"])
            distance = math.hypot(cx - sx, cy - sy)
            if iou >= 0.05 or distance <= YOLO_UNTRACKED_ASSOCIATION_PX:
                candidates.append((iou, -distance, tid))
        return max(candidates)[2] if candidates else None

    def _provisional_to_real_id(self, det, used, real_id):
        if real_id in self.tracks:
            return
        cx, cy = self._center(det["bbox"])
        candidates = []
        for tid, state in self.tracks.items():
            if tid >= 0 or tid in used or state["class_id"] != det["class_id"]:
                continue
            if state["missed"] > 2:
                continue
            sx, sy = self._center(state["bbox"])
            distance = math.hypot(cx - sx, cy - sy)
            if distance <= YOLO_UNTRACKED_ASSOCIATION_PX:
                candidates.append((distance, tid))
        if candidates:
            _, old_id = min(candidates)
            self.tracks[real_id] = self.tracks.pop(old_id)
            self.tracks[real_id]["track_id"] = real_id

    def update(self, detections):
        used = set()
        alpha = clamp(float(YOLO_BBOX_SMOOTH_ALPHA), 0.05, 1.0)
        conf_alpha = clamp(float(YOLO_CONF_SMOOTH_ALPHA), 0.05, 1.0)

        for det in detections:
            raw_id = det.get("track_id")
            if raw_id is None:
                track_id = self._synthetic_match(det, used)
                if track_id is None:
                    track_id = self.next_synthetic_id
                    self.next_synthetic_id -= 1
            else:
                track_id = int(raw_id)
                self._provisional_to_real_id(det, used, track_id)

            raw_bbox = tuple(float(v) for v in det["bbox"])
            state = self.tracks.get(track_id)
            if state is None:
                state = {
                    "track_id": track_id,
                    "class_id": int(det["class_id"]),
                    "class_name": str(det["class_name"]),
                    "bbox": raw_bbox,
                    "velocity": (0.0, 0.0),
                    "confidence": float(det["confidence"]),
                    "hits": 1,
                    "missed": 0,
                    "confirmed": False,
                }
                self.tracks[track_id] = state
            else:
                old_bbox = state["bbox"]
                old_cx, old_cy = self._center(old_bbox)
                raw_cx, raw_cy = self._center(raw_bbox)
                dx = clamp(raw_cx - old_cx, -160.0, 160.0)
                dy = clamp(raw_cy - old_cy, -160.0, 160.0)
                ovx, ovy = state["velocity"]
                state["velocity"] = (0.55 * ovx + 0.45 * dx, 0.55 * ovy + 0.45 * dy)
                state["bbox"] = tuple((1.0 - alpha) * old_bbox[i] + alpha * raw_bbox[i] for i in range(4))
                state["confidence"] = (1.0 - conf_alpha) * state["confidence"] + conf_alpha * float(det["confidence"])
                state["hits"] += 1
                state["missed"] = 0
                state["class_name"] = str(det["class_name"])
                state["class_id"] = int(det["class_id"])

            state["missed"] = 0
            state["visible"] = True
            if state["hits"] >= int(YOLO_TRACK_CONFIRM_HITS):
                state["confirmed"] = True
            used.add(track_id)

        # Preserve recent missed tracks with a damped constant-velocity prediction.
        # The optical-flow tracker in the main loop provides frame-by-frame motion
        # for the actively targeted weed; this prediction bridges detector gaps.
        for track_id, state in list(self.tracks.items()):
            if track_id in used:
                continue
            state["missed"] += 1
            state["visible"] = False
            vx, vy = state["velocity"]
            decay = 0.70
            state["bbox"] = (state["bbox"][0] + vx, state["bbox"][1] + vy,
                              state["bbox"][2] + vx, state["bbox"][3] + vy)
            state["velocity"] = (vx * decay, vy * decay)
            state["confidence"] *= 0.92
            if state["missed"] > int(YOLO_TRACK_MAX_MISSES):
                del self.tracks[track_id]

        stable = []
        for state in self.tracks.values():
            if not state.get("confirmed") or state["missed"] > int(YOLO_TRACK_MAX_MISSES):
                continue
            bbox = tuple(float(v) for v in state["bbox"])
            tx, ty = yolo_target_point(bbox)
            stable.append({
                "bbox": bbox,
                "confidence": float(state["confidence"]),
                "class_id": int(state["class_id"]),
                "class_name": state["class_name"],
                "track_id": int(state["track_id"]),
                "target": (tx, ty),
                "visible": bool(state["visible"]),
                "missed_frames": int(state["missed"]),
                "stable_hits": int(state["hits"]),
            })
        stable.sort(key=lambda d: (not d["visible"], -d["confidence"]))
        return stable


class AsyncYOLOWorker:
    """Run YOLO on a background thread without blocking camera display/control.

    Incoming frames are latest-only: if inference is busy, intermediate frames are
    discarded rather than building an unbounded queue and adding display latency.
    """
    def __init__(self, model):
        self.model = model
        self.stabilizer = YOLOTemporalStabilizer()
        self.lock = threading.Lock()
        self.running = True
        self.input_frame = None
        self.input_frame_number = None
        self.source_frame_number = None
        self.input_seq = 0
        self.result_seq = 0
        self.detections = []
        self.annotated_frame = None
        self.source_frame = None
        self.error = None
        self.inference_fps = 0.0
        self.inference_ms = 0.0
        self.thread = threading.Thread(target=self._run, name="yolo-inference", daemon=True)
        self.thread.start()

    def submit(self, frame, frame_number=None):
        with self.lock:
            if not self.running:
                return
            self.input_frame = frame.copy()
            self.input_frame_number = frame_number
            self.input_seq += 1

    def snapshot(self):
        with self.lock:
            return (
                self.result_seq,
                self.detections,
                self.annotated_frame,
                self.error,
                self.inference_fps,
                self.inference_ms,
                self.source_frame,
                self.source_frame_number,
            )

    def clear_results(self):
        with self.lock:
            self.detections = []
            self.annotated_frame = None
            self.source_frame = None
            self.source_frame_number = None
            self.error = None
            self.inference_fps = 0.0
            self.inference_ms = 0.0
            self.result_seq += 1
            return self.result_seq

    def _run(self):
        processed_seq = 0
        last_finish = None
        while True:
            with self.lock:
                if not self.running:
                    break
                seq = self.input_seq
                frame = self.input_frame
                frame_number = self.input_frame_number
            if frame is None or seq == processed_seq:
                time.sleep(0.001)
                continue

            start = time.perf_counter()
            try:
                raw_detections, annotated = extract_yolo_detections(self.model, frame)
                detections = self.stabilizer.update(raw_detections)
                error = None
            except Exception as exc:
                # A failed inference counts as one missed result for each track;
                # don't immediately erase every confirmed weed because of one error.
                try:
                    detections = self.stabilizer.update([])
                except Exception:
                    detections = []
                annotated = frame.copy()
                error = f"{type(exc).__name__}: {exc}"
            finish = time.perf_counter()
            latency_ms = (finish - start) * 1000.0
            inference_fps = (1.0 / (finish - last_finish)) if last_finish is not None and finish > last_finish else 0.0
            last_finish = finish
            processed_seq = seq

            with self.lock:
                self.detections = detections
                self.annotated_frame = annotated
                self.source_frame = frame.copy()
                self.source_frame_number = frame_number
                self.error = error
                self.inference_fps = inference_fps
                self.inference_ms = latency_ms
                self.result_seq += 1

    def stop(self):
        with self.lock:
            self.running = False
        self.thread.join(timeout=4.0)


def find_yolo_detection_at_point(detections, point):
    """Find the tightest/highest-confidence detection containing a clicked point."""
    if point is None:
        return None
    px, py = float(point[0]), float(point[1])
    matches = []
    for det in detections:
        if not det.get("visible", True) or det.get("stable_hits", YOLO_TRACK_CONFIRM_HITS) < YOLO_TRACK_CONFIRM_HITS:
            continue
        x1, y1, x2, y2 = det["bbox"]
        if x1 <= px <= x2 and y1 <= py <= y2:
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            matches.append((area, -det["confidence"], det))
    return min(matches, key=lambda item: (item[0], item[1]))[2] if matches else None


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def find_locked_yolo_detection(detections, track_id, class_id, previous_bbox):
    """Keep identity through brief ID changes; reject ambiguous re-association."""
    if track_id is not None:
        exact = next((d for d in detections if d["track_id"] == track_id), None)
        if exact is not None:
            return exact

    if previous_bbox is None:
        return None
    same_class = [d for d in detections if class_id is None or d["class_id"] == class_id]
    if not same_class:
        return None

    by_iou = sorted(same_class, key=lambda d: _bbox_iou(previous_bbox, d["bbox"]), reverse=True)
    if by_iou and _bbox_iou(previous_bbox, by_iou[0]["bbox"]) >= 0.10:
        # If two detections overlap the previous box almost equally, don't guess.
        if len(by_iou) == 1 or (_bbox_iou(previous_bbox, by_iou[0]["bbox"])
                                - _bbox_iou(previous_bbox, by_iou[1]["bbox"])) >= 0.04:
            return by_iou[0]

    # If a tracker ID changes after missed detections, re-associate by nearest
    # same-class center only when the closest candidate is clearly unambiguous.
    px, py = 0.5 * (previous_bbox[0] + previous_bbox[2]), 0.5 * (previous_bbox[1] + previous_bbox[3])
    ranked = sorted(
        same_class,
        key=lambda d: math.hypot(
            0.5 * (d["bbox"][0] + d["bbox"][2]) - px,
            0.5 * (d["bbox"][1] + d["bbox"][3]) - py,
        ),
    )
    def center_distance(det):
        cx = 0.5 * (det["bbox"][0] + det["bbox"][2])
        cy = 0.5 * (det["bbox"][1] + det["bbox"][3])
        return math.hypot(cx - px, cy - py)
    if not ranked or center_distance(ranked[0]) > YOLO_TRACK_REASSOCIATE_PX:
        return None
    if len(ranked) > 1 and center_distance(ranked[1]) - center_distance(ranked[0]) < 18.0:
        return None
    return ranked[0]


def detection_is_selected(det, track_id, class_id, bbox):
    if det is None:
        return False
    if track_id is not None:
        return det["track_id"] == track_id
    if bbox is not None and (class_id is None or det["class_id"] == class_id):
        return _bbox_iou(det["bbox"], bbox) >= 0.10
    return False


def mouse_callback(event, x, y, flags, state):
    if state.get("ignore_clicks", False):
        return
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

    if position_map.map_type == "SCATTERED":
        print(
            f"[MAP] Loaded {POSITION_MAP_FILE}: "
            f"{position_map.sample_count} multi-placement samples"
        )
        print(
            "[MAP] Inside the measured sample hull: scattered-triangle interpolation; "
            "outside: global quadratic pixel-to-position extrapolation"
        )
        print(
            f"[MAP] Global quadratic fit RMSE (training points): "
            f"servo={position_map.model_rmse_servo:.1f} positions, "
            f"actuator={position_map.model_rmse_actuator:.1f} steps"
        )
    else:
        print(
            f"[MAP] Loaded {POSITION_MAP_FILE}: "
            f"{position_map.rows}x{position_map.cols} grid"
        )
        print(
            "[MAP] Outside the A4 grid: using planar projective mapping + "
            "global quadratic actuator-position model"
        )
        print(
            f"[MAP] Global model fit RMSE (training points): "
            f"servo={position_map.model_rmse_servo:.1f} positions, "
            f"actuator={position_map.model_rmse_actuator:.1f} steps; "
            f"grid homography reprojection RMSE={position_map.homography_rmse_px:.1f}px"
        )
        if position_map.homography_rmse_px > 5.0:
            print(
                "[MAP] WARNING: grid homography residual is high. Check the 30 "
                "clicked points, lens distortion, and that the calibration sheet "
                "was flat in the target plane."
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
    raw_window = "YOLO Model Output (or Raw Camera in P mode)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(raw_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.resizeWindow(raw_window, FRAME_WIDTH, FRAME_HEIGHT)

    ui_state = {"target": None, "new_target": False, "ignore_clicks": False}
    # Allow selecting detections whether the user clicks the controller overlay
    # or the separate native Ultralytics annotated-output window.
    cv2.setMouseCallback(window, mouse_callback, ui_state)
    cv2.setMouseCallback(raw_window, mouse_callback, ui_state)

    vision_tracker = VisionTargetTracker()
    vision_enabled = VISION_TRACKING_ENABLED

    # POINT preserves the existing click-and-track behavior. YOLO is optional.
    target_source_mode = "POINT"
    yolo_model = None
    yolo_worker = None
    yolo_detections = []
    yolo_annotated_frame = None
    last_yolo_submit = -1e9
    last_yolo_result_seq = 0
    yolo_last_error_message = None
    yolo_inference_fps = 0.0
    yolo_inference_ms = 0.0
    selected_yolo_track_id = None
    selected_yolo_class_id = None
    selected_yolo_bbox = None
    yolo_smoothed_target = None
    yolo_current_target = None
    yolo_source_frame = None
    yolo_source_frame_number = None
    gray_history = deque(maxlen=YOLO_FLOW_HISTORY_FRAMES)
    yolo_flow_updated_this_frame = False

    auto_visit_active = False
    auto_visit_queue = []
    auto_visit_seen_ids = set()
    auto_visit_done_ids = set()
    auto_visit_queued_at = {}
    auto_visit_current_id = None
    auto_visit_lost_since = None
    auto_visit_wait_message_time = 0.0
    auto_visit_rejected_ids = set()

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
    actuator_jog_until = 0.0  # No longer used for U/O; retained for compatibility with the loop state.
    last_actuator_jog_key_time = -1e9

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
    print("  P = existing manual-point mode (left click target, then G)")
    print("  Y = YOLO preview / single-weed mode")
    print("  In YOLO mode: click a stabilized box to select a weed; G starts aiming")
    print("  In YOLO mode: G selects the highest-confidence stable weed if none selected")
    print("  A = AUTO VISIT: aim at each confirmed weed, settle, jitter, then advance")
    print("  C = cancel tracking, auto-visit, sweep or jitter")
    print("  Position map = pixel-to-servo/actuator map shared by both modes")
    print("  V = toggle vision target tracking ON/OFF (manual-point mode only)")
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
            gray_history.append((frame_count, gray.copy()))

            now = time.monotonic()
            yolo_flow_updated_this_frame = False
            if now - fps_start >= 1.0:
                fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now
            fps_count += 1

            # Submit frames to a background YOLO worker. Camera display, keyboard
            # controls and actuator logic no longer wait for model inference.
            if target_source_mode == "YOLO" and yolo_worker is not None:
                if now - last_yolo_submit >= YOLO_INPUT_SUBMIT_INTERVAL_S:
                    yolo_worker.submit(frame, frame_count)
                    last_yolo_submit = now

                (
                    result_seq,
                    worker_detections,
                    worker_annotated,
                    worker_error,
                    worker_fps,
                    worker_ms,
                    worker_source_frame,
                    worker_source_frame_number,
                ) = yolo_worker.snapshot()
                if result_seq != last_yolo_result_seq:
                    last_yolo_result_seq = result_seq
                    yolo_detections = worker_detections
                    yolo_annotated_frame = worker_annotated
                    yolo_source_frame = worker_source_frame
                    yolo_source_frame_number = worker_source_frame_number
                    yolo_inference_fps = worker_fps
                    yolo_inference_ms = worker_ms

                    if worker_error is not None:
                        if worker_error != yolo_last_error_message:
                            print(f"[YOLO] Inference failed: {worker_error}")
                            yolo_last_error_message = worker_error
                    else:
                        yolo_last_error_message = None

                        # Auto-visit discovers only confirmed, currently visible tracks.
                        if auto_visit_active:
                            for det in yolo_detections:
                                tid = det["track_id"]
                                if (det.get("visible", False)
                                    and det.get("stable_hits", 0) >= YOLO_TRACK_CONFIRM_HITS
                                    and tid is not None
                                    and tid not in auto_visit_seen_ids
                                    and tid not in auto_visit_done_ids):
                                    auto_visit_queue.append(tid)
                                    auto_visit_seen_ids.add(tid)
                                    auto_visit_queued_at[tid] = now
                                    print(f"[AUTO] Queued weed ID={tid} conf={det['confidence']:.2f}")

                        selected_det = find_locked_yolo_detection(
                            yolo_detections,
                            selected_yolo_track_id,
                            selected_yolo_class_id,
                            selected_yolo_bbox,
                        )
                        if selected_det is None:
                            # Keep optical flow alive during temporary detector gaps.
                            # The controller will stop if flow is also lost.
                            if not (target_active and vision_tracker.active):
                                yolo_current_target = None
                        else:
                            previous_selected_id = selected_yolo_track_id
                            selected_yolo_bbox = selected_det["bbox"]
                            selected_yolo_class_id = selected_det["class_id"]
                            selected_yolo_track_id = selected_det["track_id"]
                            if (auto_visit_active and auto_visit_current_id == previous_selected_id
                                and selected_yolo_track_id != previous_selected_id):
                                auto_visit_current_id = selected_yolo_track_id
                                auto_visit_seen_ids.add(selected_yolo_track_id)
                                print(f"[AUTO] Re-associated current weed track {previous_selected_id} -> {selected_yolo_track_id}")
                            raw_x, raw_y = selected_det["target"]
                            if selected_det.get("visible", True):
                                if yolo_smoothed_target is None:
                                    yolo_smoothed_target = (raw_x, raw_y)
                                else:
                                    alpha = clamp(float(YOLO_TARGET_SMOOTH_ALPHA), 0.0, 1.0)
                                    yolo_smoothed_target = (
                                        alpha * raw_x + (1.0 - alpha) * yolo_smoothed_target[0],
                                        alpha * raw_y + (1.0 - alpha) * yolo_smoothed_target[1],
                                    )
                                yolo_current_target = yolo_smoothed_target
                                # Seed the optical-flow tracker in the exact frame
                                # on which this detection was made, then propagate to
                                # the latest camera frame. This follows both scene/camera
                                # motion and independent weed motion between inferences.
                                if target_active and worker_source_frame is not None:
                                    try:
                                        yolo_current_target = seed_yolo_flow_tracker(
                                            vision_tracker, worker_source_frame, yolo_current_target, gray,
                                            worker_source_frame_number, frame_count, gray_history,
                                        )
                                        yolo_flow_updated_this_frame = True
                                    except Exception as flow_exc:
                                        print(f"[YOLO] Optical-flow seed failed: {flow_exc}")
                                        yolo_current_target = None
                            # For coasted/missed tracks, don't reseed flow from a
                            # prediction; continue tracking the actual target texture.

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
                if target_source_mode == "YOLO" and auto_visit_active:
                    print("[AUTO] Automatic visit is active; press C to stop it before manually selecting a weed.")
                elif target_source_mode == "YOLO":
                    clicked_det = find_yolo_detection_at_point(
                        yolo_detections, ui_state["target"]
                    )
                    if clicked_det is None:
                        print("[YOLO] Click inside a visible detection box to select that weed; G selects the highest-confidence box if none is selected.")
                    else:
                        selected_yolo_track_id = clicked_det["track_id"]
                        selected_yolo_class_id = clicked_det["class_id"]
                        selected_yolo_bbox = clicked_det["bbox"]
                        yolo_smoothed_target = clicked_det["target"]
                        yolo_current_target = clicked_det["target"]
                        print(
                            f"[YOLO] Selected {clicked_det['class_name']} "
                            f"conf={clicked_det['confidence']:.2f} "
                            f"track={clicked_det['track_id']} target="
                            f"({yolo_current_target[0]:.0f},{yolo_current_target[1]:.0f})"
                        )
                        if target_active:
                            set_actuator_speed(0.0)
                elif target_active and ui_state["target"] is not None:
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

            if (
                target_active
                and (
                    target_source_mode == "YOLO"
                    or (target_source_mode == "POINT" and ui_state["target"] is not None)
                )
                and not sweep_active
            ):
                if target_source_mode == "YOLO":
                    if vision_tracker.active and not yolo_flow_updated_this_frame:
                        flow_target = vision_tracker.update(gray)
                        yolo_current_target = flow_target
                    tracked_target = yolo_current_target
                    if tracked_target is None:
                        target_x = target_y = None
                    else:
                        target_x, target_y = tracked_target
                elif vision_enabled:
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

                # In auto-visit mode, stop before commanding an extrapolated map
                # position or an actuator target too close to a saved travel end.
                # The actuator is open-loop, so this margin reduces (but cannot
                # eliminate) hard-stop risk if steps have been missed.
                if (auto_visit_active and auto_visit_current_id is not None
                    and tracked_target is not None):
                    unsafe_reason = None
                    try:
                        _, guard_actuator_target, guard_map_status = position_map.pixel_to_position(*tracked_target)
                        if "EXTRAP" in str(guard_map_status).upper() and not AUTO_VISIT_ALLOW_EXTRAPOLATION:
                            unsafe_reason = "outside calibration sample region"
                        elif (guard_actuator_target <= ACTUATOR_MIN_POSITION + AUTO_VISIT_ACTUATOR_END_MARGIN_STEPS
                              or guard_actuator_target >= ACTUATOR_MAX_POSITION - AUTO_VISIT_ACTUATOR_END_MARGIN_STEPS):
                            unsafe_reason = "too close to actuator travel endpoint"
                    except Exception as guard_exc:
                        unsafe_reason = f"position-map error ({guard_exc})"
                    if unsafe_reason is not None:
                        bad_id = auto_visit_current_id
                        if bad_id not in auto_visit_rejected_ids:
                            print(f"[AUTO] Pausing weed ID={bad_id}: {unsafe_reason}; will retry only when its target is safe")
                        auto_visit_rejected_ids.add(bad_id)
                        auto_visit_seen_ids.discard(bad_id)
                        auto_visit_current_id = None
                        target_active = False
                        target_reached = False
                        jitter_done = False
                        selected_yolo_track_id = None
                        selected_yolo_class_id = None
                        selected_yolo_bbox = None
                        yolo_current_target = None
                        vision_tracker.reset()
                        set_actuator_speed(0.0)
                        tracked_target = None

                if tracked_target is None and not _jitter_active:
                    # Never drive toward stale coordinates after vision loss.
                    set_actuator_speed(0.0)
                    target_reached = False
                    servo_settle_since = None
                    actuator_settle_since = None
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
                        if auto_visit_active and auto_visit_current_id is not None:
                            # Freeze the just-completed visit while the servo finishes
                            # returning to its jitter base; auto-advance runs below.
                            target_active = False

                else:
                    if tracked_target is not None and now - last_update >= CONTROL_UPDATE_INTERVAL:
                        mapped_servo_target, mapped_actuator_target, map_status = (
                            position_map.pixel_to_position(target_x, target_y)
                        )

                        # Keep the unbounded model output only long enough to report
                        # when the safe mechanical limits clip a requested target.
                        raw_servo_target = mapped_servo_target
                        raw_actuator_target = mapped_actuator_target
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
                        if (
                            abs(raw_servo_target - mapped_servo_target) > 0.5
                            or abs(raw_actuator_target - mapped_actuator_target) > 0.5
                        ):
                            map_status = f"{map_status}-LIMIT"

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

                        if target_reached and (AUTO_JITTER or auto_visit_active) and not jitter_done:
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

            # AUTO VISIT lifecycle: skip a weed only if both detector tracking and
            # optical flow stay lost for a bounded period; after jitter completes,
            # wait for servo stop before selecting the next stable weed.
            if auto_visit_active and auto_visit_current_id is not None:
                if target_active and tracked_target is None and not _jitter_active:
                    if auto_visit_lost_since is None:
                        auto_visit_lost_since = now
                    elif now - auto_visit_lost_since >= AUTO_VISIT_LOST_TIMEOUT_S:
                        print(f"[AUTO] Skipping weed ID={auto_visit_current_id}: target lost for {AUTO_VISIT_LOST_TIMEOUT_S:.1f}s")
                        auto_visit_done_ids.add(auto_visit_current_id)
                        auto_visit_current_id = None
                        target_active = False
                        target_reached = False
                        jitter_done = False
                        selected_yolo_track_id = None
                        selected_yolo_class_id = None
                        selected_yolo_bbox = None
                        yolo_current_target = None
                        vision_tracker.reset()
                        set_actuator_speed(0.0)
                        auto_visit_lost_since = None
                else:
                    auto_visit_lost_since = None

                if (auto_visit_current_id is not None and not target_active
                    and target_reached and jitter_done and not _jitter_active):
                    try:
                        servo_still = not servo.IsMoving(ST_SERVO_ID)
                    except Exception:
                        servo_still = False
                    if servo_still:
                        print(f"[AUTO] Completed weed ID={auto_visit_current_id}")
                        auto_visit_done_ids.add(auto_visit_current_id)
                        auto_visit_current_id = None
                        target_reached = False
                        jitter_done = False
                        selected_yolo_track_id = None
                        selected_yolo_class_id = None
                        selected_yolo_bbox = None
                        yolo_smoothed_target = None
                        yolo_current_target = None
                        vision_tracker.reset()
                        auto_visit_lost_since = None

            # Select next queued weed when no target visit is in progress. Choose
            # the next target by estimated motor-space distance, not confidence.
            if auto_visit_active and auto_visit_current_id is None and not target_active:
                now_ids = {d["track_id"] for d in yolo_detections if d.get("track_id") is not None}
                # Expire queue entries whose tracks have disappeared for too long.
                for tid in list(auto_visit_queue):
                    if tid not in now_ids and now - auto_visit_queued_at.get(tid, now) > AUTO_VISIT_QUEUE_STALE_S:
                        auto_visit_queue.remove(tid)
                        auto_visit_seen_ids.discard(tid)
                        auto_visit_queued_at.pop(tid, None)
                        print(f"[AUTO] Dropped stale queued weed ID={tid}; it may be queued again if reacquired")

                available = [d for d in yolo_detections
                             if d.get("visible", False) and d.get("track_id") in auto_visit_queue]
                safe_available = []
                for det in available:
                    tid = det["track_id"]
                    try:
                        _, predicted_actuator, map_status_for_visit = position_map.pixel_to_position(*det["target"])
                    except Exception as exc:
                        if tid not in auto_visit_rejected_ids:
                            print(f"[AUTO] Holding weed ID={tid}: map prediction failed ({exc})")
                            auto_visit_rejected_ids.add(tid)
                        continue
                    extrapolated = "EXTRAP" in str(map_status_for_visit).upper()
                    near_actuator_end = (
                        predicted_actuator <= ACTUATOR_MIN_POSITION + AUTO_VISIT_ACTUATOR_END_MARGIN_STEPS
                        or predicted_actuator >= ACTUATOR_MAX_POSITION - AUTO_VISIT_ACTUATOR_END_MARGIN_STEPS
                    )
                    if (extrapolated and not AUTO_VISIT_ALLOW_EXTRAPOLATION) or near_actuator_end:
                        if tid not in auto_visit_rejected_ids:
                            reason = "outside calibration sample region" if extrapolated and not AUTO_VISIT_ALLOW_EXTRAPOLATION else "too close to actuator travel endpoint"
                            print(f"[AUTO] Holding weed ID={tid}: {reason}; queued until it becomes safe or is lost")
                            auto_visit_rejected_ids.add(tid)
                        continue
                    auto_visit_rejected_ids.discard(tid)
                    safe_available.append(det)
                available = safe_available
                if available:
                    if AUTO_VISIT_ORDER == "nearest_motor":
                        current_act = get_actuator_position()
                        servo_span = max(1.0, ST_VISIBLE_MAX_POSITION - ST_VISIBLE_MIN_POSITION)
                        act_span = max(1.0, ACTUATOR_MAX_POSITION - ACTUATOR_MIN_POSITION)
                        def visit_cost(det):
                            try:
                                ms, ma, _ = position_map.pixel_to_position(*det["target"])
                                ms = clamp(ms, ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION)
                                ma = clamp(ma, ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION)
                                return (abs(ms - servo_position) / servo_span
                                        + abs(ma - current_act) / act_span)
                            except Exception:
                                return float("inf")
                        next_det = min(available, key=visit_cost)
                    else:
                        next_det = max(available, key=lambda d: d["confidence"])

                    next_id = next_det["track_id"]
                    if next_id in auto_visit_queue:
                        auto_visit_queue.remove(next_id)
                    auto_visit_queued_at.pop(next_id, None)
                    auto_visit_current_id = next_id
                    selected_yolo_track_id = next_id
                    selected_yolo_class_id = next_det["class_id"]
                    selected_yolo_bbox = next_det["bbox"]
                    yolo_smoothed_target = next_det["target"]
                    yolo_current_target = next_det["target"]
                    target_active = True
                    target_reached = False
                    jitter_done = False
                    servo_settle_since = None
                    actuator_settle_since = None
                    auto_visit_lost_since = None
                    with _jitter_lock:
                        _jitter_active = False
                        _jitter_start = None
                        _jitter_base = None
                    # Seed in the detector's source frame and propagate through
                    # camera/weed motion to the current camera frame.
                    try:
                        yolo_current_target = seed_yolo_flow_tracker(
                            vision_tracker, yolo_source_frame, yolo_current_target, gray,
                            yolo_source_frame_number, frame_count, gray_history,
                        )
                    except Exception as flow_exc:
                        print(f"[AUTO] Optical-flow setup failed for ID={next_id}: {flow_exc}")
                        yolo_current_target = None
                    print(f"[AUTO] Visiting weed ID={next_id} class={next_det['class_name']} conf={next_det['confidence']:.2f}; queue remaining={len(auto_visit_queue)}")
                elif now - auto_visit_wait_message_time > 3.0:
                    print(f"[AUTO] Waiting for stable weed tracks. queued={len(auto_visit_queue)}, visited={len(auto_visit_done_ids)}")
                    auto_visit_wait_message_time = now

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

            # U/O jogs are finite position targets, so they do not need a time-based stop.

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
            if target_source_mode == "POINT" and ui_state["target"] is not None:
                sx, sy = ui_state["target"]
                draw_cross(output, sx, sy, (255, 0, 0), f"Seed {sx},{sy}")

            if target_source_mode == "YOLO":
                for det in yolo_detections:
                    x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
                    is_selected = detection_is_selected(
                        det, selected_yolo_track_id, selected_yolo_class_id, selected_yolo_bbox
                    )
                    box_color = (0, 255, 0) if is_selected else (0, 200, 255)
                    cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    if det["track_id"] is not None:
                        label += f" ID:{det['track_id']}"
                    label += f" H:{det.get('stable_hits', 0)}"
                    if not det.get("visible", True):
                        label += f" LOST:{det.get('missed_frames', 0)}"
                        box_color = (120, 120, 120)
                    cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    tx, ty = det["target"]
                    cv2.circle(output, (int(round(tx)), int(round(ty))), 4, box_color, -1)
                if yolo_current_target is not None:
                    draw_cross(output, yolo_current_target[0], yolo_current_target[1], (0, 255, 0), "YOLO AIM")

            if target_active and tracked_target is not None:
                tx, ty = tracked_target
                draw_cross(output, tx, ty, (0, 255, 0), f"Vision {tx:.0f},{ty:.0f}")

            if target_active and (
                target_source_mode == "YOLO" or ui_state["target"] is not None
            ):
                if _jitter_active:
                    status = "JITTERING"
                elif tracked_target is None:
                    status = "YOLO LOST" if target_source_mode == "YOLO" else "VISION LOST"
                else:
                    status = "YOLO AIMING" if target_source_mode == "YOLO" else "TRACKING"
            elif sweep_active:
                status = f"SWEEP {sweep_mode.upper()} {sweep_leg}"
            elif jog_mode:
                status = "JOG"
            elif auto_visit_active:
                status = "AUTO VISIT WAITING"
            elif target_source_mode == "YOLO":
                status = "YOLO PREVIEW"
            else:
                status = "POINT MODE IDLE"

            cv2.putText(output, status, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if target_source_mode == "YOLO":
                fps_text = f"UI {fps:.1f} FPS | YOLO {yolo_inference_fps:.1f} FPS | {yolo_inference_ms:.0f} ms | det={len(yolo_detections)}"
            else:
                fps_text = f"UI FPS {fps:.1f}"
            cv2.putText(output, fps_text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(output, f"Servo cmd {servo_position:.0f}", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Act pos {get_actuator_position()}  v={get_actuator_speed():+.0f}  target={get_actuator_position_target() if get_actuator_position_target() is not None else 0:.0f}", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Act ref={'YES' if actuator_is_referenced() else 'NO'}  travel={ACTUATOR_MAX_POSITION}", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"Signs: servo={SERVO_SIGN:+d} actuator={ACTUATOR_SIGN:+d}", (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            if target_active and target_source_mode == "POINT" and vision_enabled:
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

            mode_label = ("YOLO AUTO VISIT" if auto_visit_active else "YOLO") if target_source_mode == "YOLO" else "MANUAL POINT"
            class_filter_label = "all" if YOLO_ALLOWED_CLASS_IDS is None else str(YOLO_ALLOWED_CLASS_IDS)
            cv2.putText(
                output,
                f"Mode: {mode_label}" + (f"  classes={class_filter_label}" if target_source_mode == "YOLO" else ""),
                (25, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            if target_source_mode == "YOLO":
                if yolo_last_error_message:
                    yolo_state_text = "YOLO ERROR: " + yolo_last_error_message[:70]
                elif yolo_annotated_frame is None:
                    yolo_state_text = "YOLO: waiting for first inference result..."
                elif not yolo_detections:
                    yolo_state_text = f"YOLO: no detections (conf={YOLO_CONFIDENCE:.2f})"
                else:
                    visible_count = sum(1 for d in yolo_detections if d.get("visible", True))
                    stable_count = sum(1 for d in yolo_detections if d.get("stable_hits", 0) >= YOLO_TRACK_CONFIRM_HITS)
                    yolo_state_text = f"YOLO: stable={stable_count} visible={visible_count} queue={len(auto_visit_queue)} done={len(auto_visit_done_ids)}"
                cv2.putText(
                    output,
                    yolo_state_text,
                    (25, 370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255) if yolo_last_error_message else (0, 255, 255),
                    2,
                )

            if target_active and tracked_target is not None:
                if mapped_servo_target is not None and mapped_actuator_target is not None:
                    map_text = (
                        f"Map {map_status}  servo={mapped_servo_target:.0f}  "
                        f"act={mapped_actuator_target:.0f}"
                    )
                else:
                    map_text = "Map waiting..."
                cv2.putText(
                    output,
                    map_text,
                    (25, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            ui_state["ignore_clicks"] = bool(auto_visit_active)
            cv2.imshow(window, output)
            if target_source_mode == "YOLO":
                # Show the stabilized boxes on the live frame in both windows.
                # The raw Ultralytics plot belongs to an older inference frame and
                # can misalign click coordinates when the camera moves.
                cv2.imshow(raw_window, output)
            else:
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

            if key in (ord("p"), ord("P")):
                auto_visit_active = False
                auto_visit_queue.clear()
                auto_visit_current_id = None
                target_source_mode = "POINT"
                target_active = False
                target_reached = False
                jitter_done = False
                jitter_manual = False
                vision_tracker.reset()
                ui_state["target"] = None
                ui_state["new_target"] = False
                with _jitter_lock:
                    _jitter_active = False
                    _jitter_start = None
                    _jitter_base = None
                set_actuator_speed(0.0)
                selected_yolo_track_id = None
                selected_yolo_class_id = None
                selected_yolo_bbox = None
                yolo_smoothed_target = None
                yolo_current_target = None
                print("[MODE] Manual point mode. Left click a point, then press G.")
                continue

            if key in (ord("y"), ord("Y")):
                auto_visit_active = False
                auto_visit_queue.clear()
                auto_visit_current_id = None
                try:
                    if yolo_model is None:
                        yolo_model = load_yolo_model()
                    if yolo_worker is None:
                        yolo_worker = AsyncYOLOWorker(yolo_model)
                except Exception as exc:
                    print(f"[YOLO] Cannot enable YOLO mode: {exc}")
                    continue

                target_source_mode = "YOLO"
                target_active = False
                target_reached = False
                jitter_done = False
                jitter_manual = False
                vision_tracker.reset()
                ui_state["target"] = None
                ui_state["new_target"] = False
                with _jitter_lock:
                    _jitter_active = False
                    _jitter_start = None
                    _jitter_base = None
                set_actuator_speed(0.0)
                selected_yolo_track_id = None
                selected_yolo_class_id = None
                selected_yolo_bbox = None
                yolo_smoothed_target = None
                yolo_current_target = None
                yolo_detections = []
                yolo_annotated_frame = None
                yolo_inference_fps = 0.0
                yolo_inference_ms = 0.0
                last_yolo_submit = -1e9
                last_yolo_result_seq = yolo_worker.clear_results()
                print("[MODE] YOLO mode: detections will appear in both windows; click a weed box to select it, then press G.")
                continue

            if key in (ord("a"), ord("A")) and AUTO_VISIT_ENABLED:
                if auto_visit_active:
                    auto_visit_active = False
                    auto_visit_queue.clear()
                    auto_visit_current_id = None
                    target_active = False
                    target_reached = False
                    jitter_done = False
                    selected_yolo_track_id = None
                    selected_yolo_class_id = None
                    selected_yolo_bbox = None
                    yolo_current_target = None
                    vision_tracker.reset()
                    with _jitter_lock:
                        _jitter_active = False
                        _jitter_start = None
                        _jitter_base = None
                    set_actuator_speed(0.0)
                    print("[AUTO] Auto-visit stopped")
                    continue

                try:
                    if yolo_model is None:
                        yolo_model = load_yolo_model()
                    if yolo_worker is None:
                        yolo_worker = AsyncYOLOWorker(yolo_model)
                except Exception as exc:
                    print(f"[YOLO] Cannot start auto-visit: {exc}")
                    continue

                if target_source_mode != "YOLO":
                    target_source_mode = "YOLO"
                    target_active = False
                    target_reached = False
                    jitter_done = False
                    jitter_manual = False
                    vision_tracker.reset()
                    ui_state["target"] = None
                    ui_state["new_target"] = False
                    with _jitter_lock:
                        _jitter_active = False
                        _jitter_start = None
                        _jitter_base = None
                    selected_yolo_track_id = None
                    selected_yolo_class_id = None
                    selected_yolo_bbox = None
                    yolo_smoothed_target = None
                    yolo_current_target = None
                    yolo_detections = []
                    yolo_annotated_frame = None
                    yolo_inference_fps = 0.0
                    yolo_inference_ms = 0.0
                    last_yolo_submit = -1e9
                    last_yolo_result_seq = yolo_worker.clear_results()
                    set_actuator_speed(0.0)

                auto_visit_active = True
                auto_visit_queue = []
                auto_visit_seen_ids = set()
                auto_visit_done_ids = set()
                auto_visit_queued_at = {}
                auto_visit_current_id = None
                auto_visit_lost_since = None
                auto_visit_wait_message_time = 0.0
                auto_visit_rejected_ids = set()
                target_active = False
                target_reached = False
                jitter_done = False
                selected_yolo_track_id = None
                selected_yolo_class_id = None
                selected_yolo_bbox = None
                yolo_smoothed_target = None
                yolo_current_target = None
                vision_tracker.reset()
                print("[AUTO] Active. Stable confirmed weeds will be queued, aimed at, settled, jittered, then visited one by one. Press C to stop.")
                continue

            if key == ord("q"):
                auto_visit_active = False
                break

            if key == ord("g"):
                if auto_visit_active:
                    print("[AUTO] Auto-visit is already running; press C to cancel it.")
                    continue
                if not actuator_is_referenced():
                    print("[TARGET] Actuator is not referenced")
                    continue

                if target_source_mode == "POINT":
                    if ui_state["target"] is None:
                        print("[TARGET] Set a target with left click first")
                        continue
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
                        vision_tracker.initialize(gray, ui_state["target"], (None, None))
                    print(f"[TARGET] Point mode tracking ({ui_state['target'][0]}, {ui_state['target'][1]})")
                else:
                    had_explicit_selection = (
                        selected_yolo_track_id is not None or selected_yolo_bbox is not None
                    )
                    selected_det = find_locked_yolo_detection(
                        yolo_detections,
                        selected_yolo_track_id,
                        selected_yolo_class_id,
                        selected_yolo_bbox,
                    )
                    if selected_det is None and not had_explicit_selection and yolo_detections:
                        selected_det = yolo_detections[0]
                        print("[YOLO] No current selection; choosing highest-confidence detection")
                    if selected_det is None:
                        if had_explicit_selection:
                            print("[YOLO] Previously selected weed is not currently visible. Click its box again or wait for it to reappear; not switching automatically.")
                        else:
                            print("[YOLO] No detection available. Check model path, confidence, class filter, and camera view.")
                        continue

                    selected_yolo_track_id = selected_det["track_id"]
                    selected_yolo_class_id = selected_det["class_id"]
                    selected_yolo_bbox = selected_det["bbox"]
                    yolo_smoothed_target = selected_det["target"]
                    yolo_current_target = selected_det["target"]
                    target_active = True
                    target_reached = False
                    jitter_done = False
                    servo_settle_since = None
                    actuator_settle_since = None
                    try:
                        yolo_current_target = seed_yolo_flow_tracker(
                            vision_tracker, yolo_source_frame, yolo_current_target, gray,
                            yolo_source_frame_number, frame_count, gray_history,
                        )
                    except Exception as flow_exc:
                        print(f"[YOLO] Initial optical-flow setup failed: {flow_exc}")
                        yolo_current_target = None
                    with _jitter_lock:
                        _jitter_active = False
                        _jitter_start = None
                        _jitter_base = None
                    display_target = yolo_current_target if yolo_current_target is not None else selected_det["target"]
                    print(
                        f"[YOLO] Aiming at {selected_det['class_name']} "
                        f"class={selected_det['class_id']} conf={selected_det['confidence']:.2f} "
                        f"track={selected_det['track_id']} target="
                        f"({display_target[0]:.0f},{display_target[1]:.0f})"
                    )
                continue

            if key == ord("v"):
                if target_source_mode != "POINT":
                    print("[VISION] V applies only in manual-point mode; YOLO detections already update target pixels each inference.")
                    continue
                vision_enabled = not vision_enabled
                vision_tracker.reset()
                if vision_enabled and target_active and ui_state["target"] is not None:
                    vision_tracker.initialize(gray, ui_state["target"], (None, None))
                print(f"[VISION] Target tracking {'ON' if vision_enabled else 'OFF'}")
                continue

            if key == ord("r"):
                auto_visit_active = False
                auto_visit_queue.clear()
                auto_visit_current_id = None
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
                    move_actuator_to_position_and_wait(
                        actuator_middle(), timeout=15.0, label="R return to middle"
                    )
                else:
                    wait_for_actuator_stop(4.0)
                    print("[ACTUATOR] Not referenced; stopped without attempting a midpoint move")
                print("[SYSTEM] Return to MIDDLE completed or safely stopped")
                continue

            if key == ord("c"):
                auto_visit_active = False
                auto_visit_queue.clear()
                auto_visit_current_id = None
                auto_visit_lost_since = None
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
                auto_visit_active = False
                auto_visit_queue.clear()
                auto_visit_current_id = None
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
                if now - last_actuator_jog_key_time < ACT_JOG_KEY_DEBOUNCE_S:
                    continue
                last_actuator_jog_key_time = now
                target = command_actuator_jog_step(-1)
                print(f"[JOG] Actuator target -> {target:.0f} ({-ACT_JOG_STEP_STEPS:+d} steps requested)")
                continue

            if key == ord("o") and jog_mode:
                if now - last_actuator_jog_key_time < ACT_JOG_KEY_DEBOUNCE_S:
                    continue
                last_actuator_jog_key_time = now
                target = command_actuator_jog_step(+1)
                print(f"[JOG] Actuator target -> {target:.0f} ({ACT_JOG_STEP_STEPS:+d} steps requested)")
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
        if yolo_worker is not None:
            yolo_worker.stop()
        with _jitter_lock:
            _jitter_active = False
        _jitter_running = False
        set_actuator_speed(0.0)

        if actuator_is_referenced():
            move_actuator_to_position_and_wait(
                actuator_middle(), timeout=15.0, label="shutdown return to middle"
            )
        else:
            set_actuator_speed(0.0)
            wait_for_actuator_stop(4.0)
            print("[ACTUATOR] Shutdown: actuator unreferenced; stopped without a midpoint move")

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