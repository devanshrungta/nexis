#!/usr/bin/env python3
# [NOT TESTED (by devansh)] Tried to implement this but failed, since the laser detection is not working the best. So use at your own risk.
"""Automatic multi-placement laser/actuator position-map calibration.

Uses the existing 5 x 6 A4 calibration sheet and the current position_map.json
as a starting estimate. For each placement, click only the four outside grid
crosses (#1, #6, #30, #25). The script computes all 30 target pixels, moves the
servo and actuator using the existing map, detects the actual red laser spot,
and iteratively nudges the axes until the spot is centered on each printed cross.

Reflective areas can be masked by drawing exclusion polygons on the camera view.
Successful samples are checkpointed as they are collected. The existing active
map is never overwritten without explicit confirmation; a backup is made first.

Controls in the live window:
  Left-click  select sheet crosses #1, #6, #30, #25 in that order
  P           toggle exclusion-polygon drawing mode
  Enter       finish/save the polygon currently being drawn
  U           undo last polygon vertex (drawing) / last polygon / last corner
  C           clear ALL exclusion polygons (asks for confirmation in terminal)
  D           toggle live red-laser detection-mask preview
  B           clear the four corner clicks and start selecting them again
  G           start automatic calibration when four corners are selected
  F           finish collecting placements and build the candidate map
  Q           quit

While automatically aiming:
  M           manually mark the actual laser spot with the next mouse click
  S           skip the current calibration point
  R           retry the current target from the existing map estimate
  Q           stop calibration

Needs the current project's position_map.py (the version that supports
PositionMap.from_scattered_samples), servo_calibration.json, position_map.json,
Jetson.GPIO, st3215, OpenCV and NumPy. Run from the project directory.
"""

import copy
import datetime
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import Jetson.GPIO as GPIO
from st3215 import ST3215

from position_map import PositionMap, PositionMapError


# ---------------------------------------------------------------------------
# CAMERA / EXISTING HARDWARE SETTINGS (kept consistent with the current app)
# ---------------------------------------------------------------------------
CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 60

ST_SERIAL_PORT = "/dev/ttyUSB0"
ST_BAUDRATE = 115200
ST_SERVO_ID = 1
ST_MIN_POSITION = 1740
ST_MAX_POSITION = 2440
ST_VISIBLE_MIN_POSITION = ST_MIN_POSITION
ST_VISIBLE_MAX_POSITION = ST_MAX_POSITION
ST_SPEED_MAX = 6000
ST_SPEED_NEAR = 100
ST_ACCEL = 200

ACT_STEP_PIN = 29
ACT_DIR_PIN = 31
ACT_EN_PIN = 33
ACT_MAX_SPEED_STEPS_S = 10000
ACT_ACCELERATION_STEPS_S2 = 5000
ACT_ENDPOINT_TOLERANCE_STEPS = 5
ACT_STOP_SPEED_THRESHOLD = 10

CALIBRATION_FILE = "servo_calibration.json"
POSITION_MAP_FILE = "position_map.json"
CANDIDATE_MAP_FILE = "position_map_autocal_candidate.json"
CHECKPOINT_FILE = "position_map_autocal_checkpoint.json"
EXCLUSION_FILE = "laser_calibration_exclusions.json"

# Existing sheet is 5 rows x 6 columns, numbered left-to-right then top-to-bottom.
GRID_ROWS = 5
GRID_COLS = 6
CORNER_ORDER = [
    (1, "top-left #1", 0, 0),
    (6, "top-right #6", GRID_COLS - 1, 0),
    (30, "bottom-right #30", GRID_COLS - 1, GRID_ROWS - 1),
    (25, "bottom-left #25", 0, GRID_ROWS - 1),
]

# Accuracy and convergence controls. No motion/calibration ranges are changed.
LASER_TARGET_TOLERANCE_PX = 2.5
LASER_SEARCH_RADIUS_PX = 260.0
MAX_CORRECTION_ITERATIONS = 12
CORRECTION_GAIN = 0.75
MAX_SERVO_CORRECTION_PER_ITER = 45.0
MAX_ACTUATOR_CORRECTION_PER_ITER = 1800.0
MOTION_TIMEOUT_S = 15.0
MOTION_SETTLE_DWELL_S = 0.18
TARGET_POST_MOVE_PAUSE_S = 0.08

# Red-laser detection. Tune only if your laser is not strongly red in the image.
RED_HUE_LOW_MAX = 14
RED_HUE_HIGH_MIN = 166
RED_SATURATION_MIN = 45
RED_VALUE_MIN = 65
RED_DOMINANCE_MIN = 28
LASER_MIN_COMPONENT_AREA = 1
LASER_MAX_COMPONENT_AREA = 4500
LASER_MIN_RED_DOMINANCE = 20.0

# A close duplicate pixel replaces the older sample, rather than letting the map
# builder receive nearly coincident points.
DUPLICATE_PIXEL_DISTANCE_PX = 2.1
MIN_NEW_SAMPLES_TO_FINALIZE = 6


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
servo = None
cap = None
_servo_cached_speed = None
_servo_cached_acc = None
SERVO_COMMAND_RETRIES = 4
SERVO_RETRY_DELAY_S = 0.06

window = "Laser Position Auto-Calibration"

_actuator_lock = threading.Lock()
_actuator_position = 0
_actuator_speed = 0.0
_actuator_running = True
_actuator_referenced = False
_actuator_position_target = None
_actuator_position_mode = False
ACTUATOR_MIN_POSITION = 0
ACTUATOR_MAX_POSITION = 0

_ui = {
    "mode": "corners",          # corners | polygon | aiming | manual_spot
    "corner_clicks": [],
    "polygon_current": [],
    "exclusion_polygons": [],
    "manual_spot": None,
    "target_pixel": None,
    "laser_pixel": None,
    "ui_message": "",
    "current_point_number": None,
    "placement": 1,
    "quit_requested": False,
    "show_laser_mask": False,
    "skip_requested": False,
    "retry_requested": False,
    "last_key": -1,
}


def clamp(value, low, high):
    return max(low, min(high, value))


def _safe_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def load_limits():
    global ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION, ACTUATOR_MAX_POSITION
    path = Path(CALIBRATION_FILE)
    if not path.exists():
        raise RuntimeError(f"{CALIBRATION_FILE} not found. Restore the existing hardware calibration first.")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {CALIBRATION_FILE}: {exc}") from exc

    ST_VISIBLE_MIN_POSITION = int(data.get("servo_min", ST_MIN_POSITION))
    ST_VISIBLE_MAX_POSITION = int(data.get("servo_max", ST_MAX_POSITION))
    if ST_VISIBLE_MIN_POSITION >= ST_VISIBLE_MAX_POSITION:
        raise RuntimeError("Servo visible min/max in servo_calibration.json are invalid.")

    if "actuator_travel" in data:
        ACTUATOR_MAX_POSITION = int(data["actuator_travel"])
    else:
        ACTUATOR_MAX_POSITION = int(data.get("actuator_max", 0)) - int(data.get("actuator_min", 0))
    if ACTUATOR_MAX_POSITION <= 0:
        raise RuntimeError("No valid actuator travel in servo_calibration.json.")


def setup_servo():
    global servo
    servo = ST3215(ST_SERIAL_PORT)
    servo.portHandler.ser.baudrate = ST_BAUDRATE
    if not servo.PingServo(ST_SERVO_ID):
        raise RuntimeError("ST3215 did not respond to ping on /dev/ttyUSB0.")
    print(f"[ST3215] Servo ID {ST_SERVO_ID} responded")


def _clear_servo_input_buffer():
    """Discard a partial/stale servo reply after a serial communication error."""
    try:
        serial_port = servo.portHandler.ser
        if hasattr(serial_port, "reset_input_buffer"):
            serial_port.reset_input_buffer()
    except Exception:
        pass


def set_servo_position(position, speed=ST_SPEED_MAX, acc=ST_ACCEL):
    """Command the servo without MoveTo()'s internal ReadPosition() call.

    On this ST3215 package version, MoveTo() first calls ReadPosition(); a
    short/malformed serial reply can raise IndexError inside the library before
    the target is sent. Set speed/acceleration explicitly and write the target
    position directly, with retries for transient serial errors.
    """
    global _servo_cached_speed, _servo_cached_acc
    position = int(round(clamp(position, ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION)))
    speed, acc = int(speed), int(acc)
    last_error = None

    for attempt in range(SERVO_COMMAND_RETRIES):
        try:
            if _servo_cached_acc != acc:
                servo.SetAcceleration(ST_SERVO_ID, acc)
                _servo_cached_acc = acc
            if _servo_cached_speed != speed:
                servo.SetSpeed(ST_SERVO_ID, speed)
                _servo_cached_speed = speed
            # WritePosition avoids MoveTo()'s implicit ReadPosition() round-trip.
            servo.WritePosition(ST_SERVO_ID, position)
            return position
        except Exception as exc:
            last_error = exc
            _servo_cached_speed = None
            _servo_cached_acc = None
            _clear_servo_input_buffer()
            time.sleep(SERVO_RETRY_DELAY_S * (attempt + 1))

    raise RuntimeError(
        f"ST3215 command failed after {SERVO_COMMAND_RETRIES} attempts "
        f"(target={position}): {type(last_error).__name__}: {last_error}"
    ) from last_error


def read_servo_position(fallback):
    """Read the current servo position; tolerate transient short serial replies."""
    for attempt in range(3):
        try:
            value = servo.ReadPosition(ST_SERVO_ID)
            if value is not None and float(value) > 0:
                return float(value)
        except Exception:
            _clear_servo_input_buffer()
        if attempt < 2:
            time.sleep(0.025 * (attempt + 1))
    return float(fallback)


def precise_wait(seconds):
    """Wait against an absolute deadline; busy-wait the short STEP intervals."""
    deadline = time.perf_counter() + max(0.000002, float(seconds))
    remaining = deadline - time.perf_counter()
    if remaining > 0.0005:
        time.sleep(max(0.0, remaining - 0.00025))
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


def get_actuator_position():
    with _actuator_lock:
        return int(_actuator_position)


def get_actuator_speed():
    with _actuator_lock:
        return float(_actuator_speed)


def set_actuator_position_target(position):
    global _actuator_position_target, _actuator_position_mode
    position = float(clamp(position, ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION))
    with _actuator_lock:
        _actuator_position_target = position
        _actuator_position_mode = True
    return position


def actuator_thread():
    """Absolute-position actuator worker with acceleration and braking limits."""
    global _actuator_position, _actuator_speed, _actuator_position_target, _actuator_position_mode
    current_speed = 0.0
    current_direction = 0
    last_time = time.monotonic()

    while _actuator_running:
        now = time.monotonic()
        dt = min(max(now - last_time, 0.0001), 0.03)
        last_time = now

        with _actuator_lock:
            position = _actuator_position
            target = _actuator_position_target
            position_mode = _actuator_position_mode

        if position_mode and target is not None:
            remaining = target - position
            if abs(remaining) <= ACT_ENDPOINT_TOLERANCE_STEPS:
                target_speed = 0.0
            else:
                braking_speed = math.sqrt(2.0 * ACT_ACCELERATION_STEPS_S2 * abs(remaining))
                target_speed = math.copysign(min(ACT_MAX_SPEED_STEPS_S, braking_speed), remaining)
        else:
            target_speed = 0.0

        max_change = ACT_ACCELERATION_STEPS_S2 * dt
        delta = target_speed - current_speed
        if abs(delta) <= max_change:
            current_speed = target_speed
        else:
            current_speed += math.copysign(max_change, delta)

        # Hard software travel guard. This is not a substitute for a physical limit switch.
        with _actuator_lock:
            current_position_snapshot = _actuator_position
        if current_speed > 0 and current_position_snapshot >= ACTUATOR_MAX_POSITION:
            current_speed = 0.0
        elif current_speed < 0 and current_position_snapshot <= ACTUATOR_MIN_POSITION:
            current_speed = 0.0

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
            _actuator_position = int(clamp(_actuator_position, ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION))

    with _actuator_lock:
        _actuator_speed = 0.0
        _actuator_position_target = None
        _actuator_position_mode = False


def actuator_is_stopped_near(target):
    return (
        abs(get_actuator_position() - float(target)) <= ACT_ENDPOINT_TOLERANCE_STEPS
        and abs(get_actuator_speed()) <= ACT_STOP_SPEED_THRESHOLD
    )


def load_polygons():
    path = Path(EXCLUSION_FILE)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        polygons = data.get("polygons", [])
        out = []
        for polygon in polygons:
            points = [(int(p[0]), int(p[1])) for p in polygon if len(p) == 2]
            if len(points) >= 3:
                out.append(points)
        print(f"[LASER] Loaded {len(out)} exclusion polygons from {EXCLUSION_FILE}")
        return out
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"[LASER] WARNING: could not load exclusion polygons: {exc}")
        return []


def save_polygons():
    data = {"camera_width": FRAME_WIDTH, "camera_height": FRAME_HEIGHT,
            "polygons": _ui["exclusion_polygons"]}
    temp = EXCLUSION_FILE + ".tmp"
    Path(temp).write_text(json.dumps(data, indent=2))
    os.replace(temp, EXCLUSION_FILE)
    print(f"[LASER] Saved {len(_ui['exclusion_polygons'])} exclusion polygons to {EXCLUSION_FILE}")


def mouse_callback(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    mode = _ui["mode"]
    if mode == "polygon":
        _ui["polygon_current"].append((int(x), int(y)))
    elif mode == "manual_spot":
        _ui["manual_spot"] = (float(x), float(y))
        _ui["mode"] = "aiming"
    elif mode == "corners":
        if len(_ui["corner_clicks"]) < 4:
            _ui["corner_clicks"].append((float(x), float(y)))
            idx = len(_ui["corner_clicks"]) - 1
            point_num, label, _, _ = CORNER_ORDER[idx]
            print(f"[SHEET] Clicked cross #{point_num} ({label}) at ({x}, {y})")


def exclusion_mask():
    mask = np.full((FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=np.uint8)
    for polygon in _ui["exclusion_polygons"]:
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 0)
    return mask


def detect_laser(frame, target_pixel):
    """Detect a red spot near the current target, excluding user-drawn polygons.

    Returns (x, y, confidence, candidates). If no plausible candidate exists,
    returns None for x/y and leaves manual click as a safe recovery path.
    """
    if target_pixel is None:
        return None, None, 0.0, []
    tx, ty = float(target_pixel[0]), float(target_pixel[1])
    b, g, r = cv2.split(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    hue_red = ((h <= RED_HUE_LOW_MAX) | (h >= RED_HUE_HIGH_MIN)) & (s >= RED_SATURATION_MIN) & (v >= RED_VALUE_MIN)
    red_dom = (r.astype(np.int16) - g.astype(np.int16) >= RED_DOMINANCE_MIN) & (
        r.astype(np.int16) - b.astype(np.int16) >= 8
    ) & (r >= RED_VALUE_MIN)
    raw = (hue_red | red_dom).astype(np.uint8) * 255

    mask = exclusion_mask()
    raw = cv2.bitwise_and(raw, mask)
    # A light close joins tiny saturated pixels, but avoid opening away a small laser core.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    candidates = []
    radius2 = float(LASER_SEARCH_RADIUS_PX) ** 2
    red_i16 = r.astype(np.int16)
    g_i16 = g.astype(np.int16)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < LASER_MIN_COMPONENT_AREA or area > LASER_MAX_COMPONENT_AREA:
            continue
        x, y = float(centroids[label][0]), float(centroids[label][1])
        dist2 = (x - tx) ** 2 + (y - ty) ** 2
        if dist2 > radius2:
            continue
        component = labels == label
        dominance = float(np.mean(red_i16[component] - g_i16[component]))
        if dominance < LASER_MIN_RED_DOMINANCE:
            continue
        mean_red = float(np.mean(r[component]))
        # Distance dominates selection; intensity is only a weak tiebreaker.
        score = math.sqrt(dist2) - min(10.0, max(0.0, dominance - 30.0) * 0.08) - min(3.0, mean_red / 100.0)
        candidates.append({"x": x, "y": y, "area": area, "dominance": dominance,
                           "mean_red": mean_red, "distance": math.sqrt(dist2), "score": score})

    candidates.sort(key=lambda c: c["score"])
    if not candidates:
        return None, None, 0.0, []
    best = candidates[0]
    # Confidence is diagnostic only; the visible marker and polygon tools let the user
    # correct a scene where a red reflection is closer than the true laser spot.
    confidence = clamp(1.0 - best["distance"] / LASER_SEARCH_RADIUS_PX, 0.0, 1.0)
    return best["x"], best["y"], confidence, candidates


def detect_laser_stable(target_pixel, sample_count=5):
    """Use a temporal median to reduce single-frame centroid jitter.

    A result is accepted only when at least three frames detect a spot and at
    least three detections agree within 4 px of their median. Ambiguous results
    return None and go through the existing manual-confirmation flow.
    """
    observations = []
    last_frame = None
    last_candidates = []
    for _ in range(max(3, int(sample_count))):
        frame = get_frame()
        if frame is None:
            continue
        last_frame = frame
        x, y, confidence, candidates = detect_laser(frame, target_pixel)
        last_candidates = candidates
        if x is not None and y is not None:
            observations.append((float(x), float(y), float(confidence)))

    if last_frame is None or len(observations) < 3:
        return last_frame, None, None, 0.0, last_candidates

    coordinates = np.asarray([(item[0], item[1]) for item in observations], dtype=np.float32)
    median_xy = np.median(coordinates, axis=0)
    distances = np.linalg.norm(coordinates - median_xy, axis=1)
    inliers = coordinates[distances <= 4.0]
    if len(inliers) < 3:
        return last_frame, None, None, 0.0, last_candidates

    stable_xy = np.median(inliers, axis=0)
    stable_confidence = float(np.median([item[2] for item in observations]))
    return last_frame, float(stable_xy[0]), float(stable_xy[1]), stable_confidence, last_candidates


def draw_polygon_overlay(frame):
    output = frame.copy()
    if _ui["exclusion_polygons"]:
        overlay = output.copy()
        for polygon in _ui["exclusion_polygons"]:
            pts = np.asarray(polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (30, 30, 180))
            cv2.polylines(output, [pts], True, (0, 0, 255), 2)
        output = cv2.addWeighted(overlay, 0.20, output, 0.80, 0)
        for polygon in _ui["exclusion_polygons"]:
            cv2.polylines(output, [np.asarray(polygon, dtype=np.int32)], True, (0, 0, 255), 2)
    current = _ui["polygon_current"]
    if current:
        for p in current:
            cv2.circle(output, p, 4, (255, 0, 255), -1)
        if len(current) > 1:
            cv2.polylines(output, [np.asarray(current, dtype=np.int32)], False, (255, 0, 255), 2)
    return output


def draw_target_overlay(frame, target=None, spot=None, status=""):
    output = draw_polygon_overlay(frame)
    if target is not None:
        tx, ty = int(round(target[0])), int(round(target[1]))
        cv2.drawMarker(output, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 25, 2)
        cv2.circle(output, (tx, ty), int(round(LASER_TARGET_TOLERANCE_PX)), (0, 255, 255), 1)
        cv2.putText(output, "A4 target", (tx + 12, ty - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if spot is not None:
        sx, sy = int(round(spot[0])), int(round(spot[1]))
        cv2.drawMarker(output, (sx, sy), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 25, 2)
        cv2.putText(output, "laser", (sx + 12, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    lines = [
        f"Placement { _ui['placement'] }   Samples {len(_ui.get('new_samples', []))}",
        status or _ui.get("ui_message", ""),
    ]
    y = 28
    for line in lines:
        if line:
            cv2.putText(output, line[:150], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            y += 27
    return output


def get_frame():
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    return frame


def display_and_key(frame, target=None, spot=None, status=""):
    output = draw_target_overlay(frame, target, spot, status)
    cv2.imshow(window, output)
    key = cv2.waitKey(1) & 0xFF
    _ui["last_key"] = key
    if key == ord("q"):
        _ui["quit_requested"] = True
    elif key == ord("s"):
        _ui["skip_requested"] = True
    elif key == ord("r") and ("retry" in (status or "").lower() or "not converged" in (status or "").lower()):
        _ui["retry_requested"] = True
    elif key == ord("m") and _ui["mode"] == "aiming":
        _ui["mode"] = "manual_spot"
        _ui["manual_spot"] = None
        print("[LASER] Click the actual laser spot in the camera window.")
    return key


def make_grid_targets(corners):
    """Return 30 perspective-correct target pixel positions from four clicked crosses."""
    if len(corners) != 4:
        raise ValueError("Need four sheet corner clicks (#1, #6, #30, #25).")
    src = np.asarray([
        [0.0, 0.0],
        [float(GRID_COLS - 1), 0.0],
        [float(GRID_COLS - 1), float(GRID_ROWS - 1)],
        [0.0, float(GRID_ROWS - 1)],
    ], dtype=np.float32)
    dst = np.asarray(corners, dtype=np.float32)
    if abs(cv2.contourArea(dst.reshape(-1, 1, 2))) < 1000:
        raise ValueError("The four clicked crosses do not define a large enough sheet region.")
    hmat = cv2.getPerspectiveTransform(src, dst)
    grid_src = np.asarray([[[float(c), float(r)] for c in range(GRID_COLS)] for r in range(GRID_ROWS)], dtype=np.float32)
    flat = grid_src.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(flat, hmat).reshape(-1, 2)
    targets = []
    for idx, (x, y) in enumerate(projected, start=1):
        if not (0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT):
            raise ValueError(f"Computed grid point #{idx} is outside the camera frame. Re-click the four crosses.")
        targets.append({"point": idx, "x": float(x), "y": float(y),
                        "row": (idx - 1) // GRID_COLS, "col": (idx - 1) % GRID_COLS})
    # Reject bad corner order or crossed quadrilaterals.
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    if max(xs) - min(xs) < 100 or max(ys) - min(ys) < 100:
        raise ValueError("Clicked grid crosses are too close together. Re-click the four outer crosses.")
    return targets


def command_motor_positions(servo_target, actuator_target):
    servo_target = clamp(servo_target, ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION)
    actuator_target = clamp(actuator_target, ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION)
    set_servo_position(servo_target, speed=ST_SPEED_MAX, acc=ST_ACCEL)
    set_actuator_position_target(actuator_target)
    return float(servo_target), float(actuator_target)


def wait_for_motion(target, timeout=MOTION_TIMEOUT_S):
    deadline = time.monotonic() + timeout
    still_since = None
    fallback_servo = ST_VISIBLE_MIN_POSITION
    while time.monotonic() < deadline:
        frame = get_frame()
        if frame is None:
            continue
        display_and_key(frame, target, _ui.get("laser_pixel"), "Moving to target; waiting for both axes to settle")
        if _ui["quit_requested"] or _ui["skip_requested"]:
            return False
        try:
            servo_moving = bool(servo.IsMoving(ST_SERVO_ID))
        except Exception:
            servo_moving = True
        target_act = get_actuator_position_target_value()
        actuator_stopped = actuator_is_stopped_near(target_act)
        if not servo_moving and actuator_stopped:
            if still_since is None:
                still_since = time.monotonic()
            elif time.monotonic() - still_since >= MOTION_SETTLE_DWELL_S:
                time.sleep(TARGET_POST_MOVE_PAUSE_S)
                return True
        else:
            still_since = None
    print("[MOTION] WARNING: motion settle timeout; point can be retried or skipped.")
    return False


def get_actuator_position_target_value():
    with _actuator_lock:
        target = _actuator_position_target
        position = _actuator_position
    return target if target is not None else position


def read_stable_motor_positions(servo_fallback):
    return read_servo_position(servo_fallback), float(get_actuator_position())


def build_scattered_map(samples):
    if not hasattr(PositionMap, "from_scattered_samples"):
        raise PositionMapError(
            "This project has an older position_map.py. Use the current multi-placement "
            "version that defines PositionMap.from_scattered_samples()."
        )
    return PositionMap.from_scattered_samples(samples, FRAME_WIDTH, FRAME_HEIGHT)


def merge_sample(samples, new_sample):
    """Replace an old sample at the same image location, otherwise append."""
    updated = list(samples)
    keep = []
    for old in updated:
        d = math.hypot(float(old["x"]) - float(new_sample["x"]),
                       float(old["y"]) - float(new_sample["y"]))
        if d < DUPLICATE_PIXEL_DISTANCE_PX:
            continue
        keep.append(old)
    keep.append(new_sample)
    return keep


def save_checkpoint(samples, new_samples, next_placement):
    data = {
        "version": 1,
        "camera_width": FRAME_WIDTH,
        "camera_height": FRAME_HEIGHT,
        "samples": samples,
        "new_samples": new_samples,
        "next_placement": int(next_placement),
        "exclusion_file": EXCLUSION_FILE,
    }
    temp = CHECKPOINT_FILE + ".tmp"
    Path(temp).write_text(json.dumps(data, indent=2))
    os.replace(temp, CHECKPOINT_FILE)


def load_checkpoint():
    try:
        data = json.loads(Path(CHECKPOINT_FILE).read_text())
        if int(data.get("camera_width", -1)) != FRAME_WIDTH or int(data.get("camera_height", -1)) != FRAME_HEIGHT:
            raise ValueError("camera resolution differs from checkpoint")
        samples = data.get("samples", [])
        new_samples = data.get("new_samples", [])
        for sample in samples:
            for key in ("x", "y", "servo", "actuator"):
                if key not in sample:
                    raise ValueError(f"checkpoint sample is missing {key}")
        return samples, new_samples, max(1, int(data.get("next_placement", 1)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"[CAL] WARNING: checkpoint could not be loaded: {exc}")
        return None


def initial_samples_from_map(position_map):
    samples = []
    for idx, source in enumerate(position_map.samples, start=1):
        samples.append({
            "id": idx,
            "placement": 0,
            "corner": f"existing-{idx}",
            "x": float(source["x"]),
            "y": float(source["y"]),
            "servo": float(source["servo"]),
            "actuator": float(source["actuator"]),
        })
    return samples


def make_laser_mask_preview(frame):
    b, g, r = cv2.split(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_red = ((h <= RED_HUE_LOW_MAX) | (h >= RED_HUE_HIGH_MIN)) & (s >= RED_SATURATION_MIN) & (v >= RED_VALUE_MIN)
    red_dom = (r.astype(np.int16) - g.astype(np.int16) >= RED_DOMINANCE_MIN) & (r.astype(np.int16) - b.astype(np.int16) >= 8) & (r >= RED_VALUE_MIN)
    raw = ((hue_red | red_dom).astype(np.uint8) * 255)
    raw = cv2.bitwise_and(raw, exclusion_mask())
    return raw


def handle_laser_not_found(target, point_number, attempt):
    """Wait for manual spot click, retry, skip, or quit when automatic detection fails."""
    _ui["mode"] = "aiming"
    _ui["manual_spot"] = None
    print(f"[LASER] No confident candidate near point #{point_number}. Press M and click the actual spot, R to retry, S to skip, Q to quit.")
    while not _ui["quit_requested"] and not _ui["skip_requested"] and not _ui["retry_requested"]:
        frame = get_frame()
        if frame is None:
            continue
        if _ui.get("manual_spot") is not None:
            spot = _ui["manual_spot"]
            _ui["manual_spot"] = None
            return "manual", spot
        display_and_key(frame, target, None,
                        f"Point #{point_number}: laser not found. M=click real spot  R=retry  S=skip  Q=quit")
        time.sleep(0.005)
    if _ui["quit_requested"]:
        return "quit", None
    if _ui["skip_requested"]:
        _ui["skip_requested"] = False
        return "skip", None
    if _ui["retry_requested"]:
        _ui["retry_requested"] = False
        return "retry", None
    return "skip", None


def aim_and_record(target, placement, working_map, samples, new_samples, last_servo_position):
    """Automatically center the laser over a target cross; then record motor values."""
    point_no = int(target["point"])
    target_xy = (float(target["x"]), float(target["y"]))
    _ui["target_pixel"] = target_xy
    _ui["current_point_number"] = point_no
    _ui["laser_pixel"] = None
    _ui["skip_requested"] = False
    _ui["retry_requested"] = False

    while True:  # outer loop supports an explicit R/retry from the user
        try:
            mapped_servo, mapped_actuator, map_status = working_map.pixel_to_position(*target_xy)
        except Exception as exc:
            print(f"[CAL] Point #{point_no}: mapping failed: {exc}")
            return "skip", last_servo_position, working_map
        commanded_servo, commanded_actuator = command_motor_positions(mapped_servo, mapped_actuator)
        print(f"[CAL] Placement {placement}, point #{point_no}: map={map_status}, moving servo={commanded_servo:.1f}, actuator={commanded_actuator:.0f}")
        if not wait_for_motion(target_xy):
            if _ui["quit_requested"]:
                return "quit", last_servo_position, working_map
            if _ui["skip_requested"]:
                _ui["skip_requested"] = False
                return "skip", last_servo_position, working_map
            action, _ = handle_laser_not_found(target_xy, point_no, 0)
            if action == "quit":
                return "quit", last_servo_position, working_map
            if action == "skip":
                return "skip", last_servo_position, working_map
            continue

        converged = False
        retry_target_now = False
        final_error = None
        last_spot = None
        for attempt in range(MAX_CORRECTION_ITERATIONS + 1):
            if _ui["quit_requested"]:
                return "quit", last_servo_position, working_map
            if _ui["skip_requested"]:
                _ui["skip_requested"] = False
                return "skip", last_servo_position, working_map

            frame, lx, ly, confidence, candidates = detect_laser_stable(target_xy, sample_count=5)
            if frame is None:
                continue
            if _ui.get("manual_spot") is not None:
                lx, ly = _ui["manual_spot"]
                _ui["manual_spot"] = None
                confidence = 1.0
            if lx is None or ly is None:
                action, manual_spot = handle_laser_not_found(target_xy, point_no, attempt)
                if action == "quit":
                    return "quit", last_servo_position, working_map
                if action == "skip":
                    return "skip", last_servo_position, working_map
                if action == "retry":
                    retry_target_now = True
                    break
                lx, ly = manual_spot

            last_spot = (float(lx), float(ly))
            _ui["laser_pixel"] = last_spot
            error_px = math.hypot(last_spot[0] - target_xy[0], last_spot[1] - target_xy[1])
            final_error = error_px
            status = f"Point #{point_no}: error={error_px:.1f}px / target {LASER_TARGET_TOLERANCE_PX:.1f}px, attempt {attempt}/{MAX_CORRECTION_ITERATIONS}"
            display_and_key(frame, target_xy, last_spot, status)
            if _ui["quit_requested"]:
                return "quit", last_servo_position, working_map
            if _ui["skip_requested"]:
                _ui["skip_requested"] = False
                return "skip", last_servo_position, working_map

            if error_px <= LASER_TARGET_TOLERANCE_PX:
                converged = True
                break
            if attempt >= MAX_CORRECTION_ITERATIONS:
                break

            try:
                target_s, target_a, _ = working_map.pixel_to_position(*target_xy)
                spot_s, spot_a, _ = working_map.pixel_to_position(*last_spot)
            except Exception as exc:
                print(f"[CAL] Could not compute correction for point #{point_no}: {exc}")
                break

            ds = clamp((target_s - spot_s) * CORRECTION_GAIN,
                       -MAX_SERVO_CORRECTION_PER_ITER, MAX_SERVO_CORRECTION_PER_ITER)
            da = clamp((target_a - spot_a) * CORRECTION_GAIN,
                       -MAX_ACTUATOR_CORRECTION_PER_ITER, MAX_ACTUATOR_CORRECTION_PER_ITER)
            current_s, current_a = read_stable_motor_positions(last_servo_position)
            next_s = clamp(current_s + ds, ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION)
            next_a = clamp(current_a + da, ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION)
            if abs(next_s - current_s) < 0.5 and abs(next_a - current_a) < 1.0:
                print(f"[CAL] Correction hit a software limit for point #{point_no}; cannot reduce error further.")
                break
            set_servo_position(next_s, speed=ST_SPEED_MAX, acc=ST_ACCEL)
            set_actuator_position_target(next_a)
            last_servo_position = next_s
            if not wait_for_motion(target_xy):
                if _ui["quit_requested"]:
                    return "quit", last_servo_position, working_map
                if _ui["skip_requested"]:
                    _ui["skip_requested"] = False
                    return "skip", last_servo_position, working_map

        if retry_target_now:
            _ui["retry_requested"] = False
            continue

        if converged:
            actual_servo = read_servo_position(last_servo_position)
            actual_actuator = get_actuator_position()
            sample = {
                "id": max([int(s.get("id", 0)) for s in samples] + [0]) + 1,
                "placement": int(placement),
                "corner": str(point_no),
                "x": float(target_xy[0]),
                "y": float(target_xy[1]),
                "servo": float(actual_servo),
                "actuator": float(actual_actuator),
                "final_error_px": float(final_error if final_error is not None else 0.0),
            }
            samples[:] = merge_sample(samples, sample)
            new_samples.append(sample)
            try:
                working_map = build_scattered_map(samples)
                working_map.save(CANDIDATE_MAP_FILE)
            except PositionMapError as exc:
                print(f"[CAL] Sample saved; candidate fit not refreshed yet: {exc}")
            save_checkpoint(samples, new_samples, placement + 1)
            print(f"[CAL] SAVED placement={placement} point=#{point_no} error={final_error:.1f}px servo={actual_servo:.1f} actuator={actual_actuator}")
            return "saved", actual_servo, working_map

        # Failed to converge within the automatic correction budget.
        print(f"[CAL] Point #{point_no} did not converge (last error {final_error if final_error is not None else float('nan'):.1f}px). R=retry, M=click actual laser spot, S=skip, Q=quit.")
        _ui["retry_requested"] = False
        manual_correction_done = False
        while (not _ui["quit_requested"] and not _ui["skip_requested"]
               and not _ui["retry_requested"]):
            frame = get_frame()
            if frame is None:
                continue
            display_and_key(frame, target_xy, last_spot,
                            f"Point #{point_no} not converged: R retry  M click spot  S skip  Q quit")
            if _ui.get("manual_spot") is not None:
                manual_xy = _ui["manual_spot"]
                _ui["manual_spot"] = None
                _ui["laser_pixel"] = manual_xy
                # Manual mark becomes a one-shot correction: re-measure after the move.
                last_servo_position = read_servo_position(last_servo_position)
                try:
                    target_s, target_a, _ = working_map.pixel_to_position(*target_xy)
                    spot_s, spot_a, _ = working_map.pixel_to_position(*manual_xy)
                    ds = clamp((target_s - spot_s) * CORRECTION_GAIN,
                               -MAX_SERVO_CORRECTION_PER_ITER, MAX_SERVO_CORRECTION_PER_ITER)
                    da = clamp((target_a - spot_a) * CORRECTION_GAIN,
                               -MAX_ACTUATOR_CORRECTION_PER_ITER, MAX_ACTUATOR_CORRECTION_PER_ITER)
                    set_servo_position(read_servo_position(last_servo_position) + ds, speed=ST_SPEED_MAX, acc=ST_ACCEL)
                    set_actuator_position_target(get_actuator_position() + da)
                    wait_for_motion(target_xy)
                    manual_correction_done = True
                except Exception as exc:
                    print(f"[CAL] Manual correction failed: {exc}")
                break
        if _ui["quit_requested"]:
            return "quit", last_servo_position, working_map
        if _ui["skip_requested"]:
            _ui["skip_requested"] = False
            return "skip", last_servo_position, working_map
        if _ui["retry_requested"]:
            _ui["retry_requested"] = False
            continue
        # After manual correction, restart the loop with current motor position and fresh measurement.
        if manual_correction_done:
            continue


def draw_setup_screen(frame):
    output = draw_polygon_overlay(frame)
    if _ui.get("show_laser_mask", False):
        mask = make_laser_mask_preview(frame)
        tint = np.zeros_like(output)
        tint[:, :, 0] = mask
        tint[:, :, 2] = mask
        output = cv2.addWeighted(output, 0.78, tint, 0.22, 0)
    for idx, point in enumerate(_ui["corner_clicks"]):
        point_num, label, _, _ = CORNER_ORDER[idx]
        p = (int(point[0]), int(point[1]))
        cv2.circle(output, p, 8, (0, 255, 0), 2)
        cv2.putText(output, f"#{point_num}", (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if _ui["mode"] == "polygon":
        prompt = "POLYGON MODE: click vertices; ENTER closes; U undo vertex; P exits"
    elif len(_ui["corner_clicks"]) < 4:
        next_num, next_label, _, _ = CORNER_ORDER[len(_ui["corner_clicks"])]
        prompt = f"Click outer A4 cross #{next_num} ({next_label}); {len(_ui['corner_clicks'])}/4 selected"
    else:
        prompt = "Corners ready. Press G to calibrate these 30 points, B to reselect, F to finish."
    mask_status = "DETECTION MASK ON (highlighted pixels are candidates)" if _ui.get("show_laser_mask", False) else "Detection-mask preview OFF"
    lines = [prompt,
             f"Placement { _ui['placement'] } | new samples={len(_ui.get('new_samples', []))} | excluded polygons={len(_ui['exclusion_polygons'])} | {mask_status}",
             "P polygons | Enter finish polygon | D mask preview | U undo | C clear masks | S save | G calibrate | F finish | Q quit"]
    y = 28
    for line in lines:
        cv2.putText(output, line[:150], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 27
    return output


def restore_position(initial_servo, initial_actuator):
    print(f"[EXIT] Returning to startup position: servo={initial_servo:.0f}, actuator={initial_actuator}")
    try:
        set_servo_position(initial_servo, speed=ST_SPEED_NEAR, acc=ST_ACCEL)
    except Exception as exc:
        print(f"[EXIT] Servo restore command failed: {exc}")
    try:
        set_actuator_position_target(initial_actuator)
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            frame = get_frame() if cap is not None else None
            if frame is not None:
                display_and_key(frame, None, None, "Returning to startup position")
            try:
                servo_moving = bool(servo.IsMoving(ST_SERVO_ID))
            except Exception:
                servo_moving = False
            if not servo_moving and actuator_is_stopped_near(initial_actuator):
                break
            time.sleep(0.01)
    except Exception as exc:
        print(f"[EXIT] Actuator restore warning: {exc}")


def finalize_map(samples, new_samples):
    if len(new_samples) < MIN_NEW_SAMPLES_TO_FINALIZE:
        print(f"[CAL] Need at least {MIN_NEW_SAMPLES_TO_FINALIZE} new successful points before finalizing; have {len(new_samples)}.")
        return False
    try:
        final_map = build_scattered_map(samples)
    except PositionMapError as exc:
        print(f"[CAL] Cannot create candidate map: {exc}")
        return False

    final_map.save(CANDIDATE_MAP_FILE)
    errors = [float(s["final_error_px"]) for s in new_samples if "final_error_px" in s]
    print("\n========== AUTO-CALIBRATION SUMMARY ==========")
    print(f"Successful new samples: {len(new_samples)}")
    print(f"Unique map samples after merging existing map: {final_map.sample_count}")
    if errors:
        rms_pixel_error = math.sqrt(sum(e * e for e in errors) / len(errors))
        print(f"Final centering residuals: mean={sum(errors) / len(errors):.2f}px, RMS={rms_pixel_error:.2f}px, max={max(errors):.2f}px")
    print(f"Global quadratic fit RMSE: servo={final_map.model_rmse_servo:.2f} positions, actuator={final_map.model_rmse_actuator:.1f} steps")
    print(f"Candidate map written to: {CANDIDATE_MAP_FILE}")
    print("Use placements around the whole intended workspace; training-point fit error alone does not guarantee accuracy between points.")
    print("==============================================\n")

    answer = input(f"Replace active {POSITION_MAP_FILE} with this candidate? A timestamped backup will be made. [Y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print(f"[CAL] Keeping active {POSITION_MAP_FILE} unchanged. Candidate remains at {CANDIDATE_MAP_FILE}.")
        return True

    if os.path.exists(POSITION_MAP_FILE):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"position_map_before_autocalibration_{stamp}.json"
        shutil.copy2(POSITION_MAP_FILE, backup)
        print(f"[CAL] Active map backup: {backup}")
    shutil.copy2(CANDIDATE_MAP_FILE, POSITION_MAP_FILE)
    print(f"[CAL] Updated active map: {POSITION_MAP_FILE}")
    try:
        os.remove(CHECKPOINT_FILE)
    except OSError:
        pass
    return True


def main():
    global _actuator_running, _actuator_position, _actuator_referenced, cap

    if not hasattr(PositionMap, "from_scattered_samples"):
        raise RuntimeError(
            "The position_map.py beside this script is an older version. Use the same "
            "multi-placement position_map.py as the current calibrated tracker."
        )
    load_limits()
    position_map = PositionMap.load(POSITION_MAP_FILE)
    if position_map.camera_width != FRAME_WIDTH or position_map.camera_height != FRAME_HEIGHT:
        print(f"[MAP] WARNING: active map is {position_map.camera_width}x{position_map.camera_height}; camera is {FRAME_WIDTH}x{FRAME_HEIGHT}.")

    setup_servo()
    middle = ACTUATOR_MAX_POSITION / 2.0
    answer = input(f"[STARTUP] Is the actuator physically at saved middle ({middle:.0f})? [Y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError("Move the actuator physically to its saved midpoint, then run this script again.")
    _actuator_position = int(round(middle))
    _actuator_referenced = True

    existing_samples = initial_samples_from_map(position_map)
    samples = existing_samples
    new_samples = []
    next_placement = 1
    if os.path.exists(CHECKPOINT_FILE):
        answer = input(f"[CAL] Found checkpoint {CHECKPOINT_FILE}. Resume collected samples? [Y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            loaded = load_checkpoint()
            if loaded is not None:
                samples, new_samples, next_placement = loaded
                try:
                    position_map = build_scattered_map(samples)
                except PositionMapError as exc:
                    print(f"[CAL] Checkpoint model invalid; starting from active map instead: {exc}")
                    samples, new_samples, next_placement = existing_samples, [], 1
                    position_map = PositionMap.load(POSITION_MAP_FILE)
        else:
            try:
                os.remove(CHECKPOINT_FILE)
            except OSError:
                pass

    _ui["exclusion_polygons"] = load_polygons()
    _ui["new_samples"] = new_samples
    _ui["placement"] = next_placement

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        try:
            servo.portHandler.closePort()
        except Exception:
            pass
        cap.release()
        raise RuntimeError(f"Could not open /dev/video{CAMERA_DEVICE}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(ACT_STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ACT_DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ACT_EN_PIN, GPIO.OUT, initial=GPIO.HIGH)
    set_actuator_enabled(True)

    worker = threading.Thread(target=actuator_thread, daemon=True)
    worker.start()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(window, mouse_callback)

    initial_servo = read_servo_position(ST_VISIBLE_MIN_POSITION)
    initial_actuator = _actuator_position
    samples = list(samples)
    calibration_finished = False
    print("\n========================================================")
    print(" AUTOMATIC A4 LASER / SERVO / ACTUATOR CALIBRATION")
    print("========================================================")
    print("This uses the current position map as the starting estimate.")
    print("For each sheet placement, click crosses #1, #6, #30, #25, then press G.")
    print("The script aims at all 30 crosses, detects the laser, corrects the axes, and saves successful points.")
    print("P draws exclusion polygons around reflections; click vertices and press ENTER to finish each polygon.")
    print("Place the A4 sheet at several regions of the SAME target plane. Press F when coverage is adequate.")
    print(f"Current safe servo range={ST_VISIBLE_MIN_POSITION}..{ST_VISIBLE_MAX_POSITION}; actuator travel={ACTUATOR_MAX_POSITION} steps")

    try:
        while not _ui["quit_requested"] and not calibration_finished:
            frame = get_frame()
            if frame is None:
                continue
            display = draw_setup_screen(frame)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            _ui["last_key"] = key

            if key == ord("q"):
                _ui["quit_requested"] = True
                break
            if key == ord("p"):
                if _ui["mode"] == "polygon":
                    print("[LASER] Finish current polygon with Enter before leaving polygon mode.")
                else:
                    _ui["mode"] = "polygon"
                    print("[LASER] Polygon mode ON: click at least 3 vertices around a reflective region, then press Enter.")
                continue
            if key in (10, 13) and _ui["mode"] == "polygon":
                if len(_ui["polygon_current"]) >= 3:
                    _ui["exclusion_polygons"].append(list(_ui["polygon_current"]))
                    _ui["polygon_current"] = []
                    _ui["mode"] = "corners"
                    save_polygons()
                    print(f"[LASER] Polygon added. There are now {len(_ui['exclusion_polygons'])} exclusions.")
                else:
                    print("[LASER] A polygon needs at least 3 vertices.")
                continue
            if key == ord("u"):
                if _ui["mode"] == "polygon" and _ui["polygon_current"]:
                    _ui["polygon_current"].pop()
                elif _ui["mode"] == "polygon" and not _ui["polygon_current"] and _ui["exclusion_polygons"]:
                    _ui["exclusion_polygons"].pop()
                    save_polygons()
                elif _ui["mode"] == "corners" and _ui["corner_clicks"]:
                    _ui["corner_clicks"].pop()
                continue
            if key == ord("c"):
                answer = input("Clear every laser exclusion polygon? Type YES to confirm: ").strip()
                if answer == "YES":
                    _ui["exclusion_polygons"] = []
                    _ui["polygon_current"] = []
                    save_polygons()
                continue
            if key == ord("s"):
                save_polygons()
                continue
            if key == ord("d"):
                _ui["show_laser_mask"] = not _ui.get("show_laser_mask", False)
                print(f"[LASER] Detection-mask preview {'ON' if _ui['show_laser_mask'] else 'OFF'}")
                continue
            if key == ord("b"):
                _ui["corner_clicks"] = []
                _ui["mode"] = "corners"
                continue
            if key == ord("g"):
                if _ui["mode"] == "polygon":
                    print("[LASER] Finish the exclusion polygon with Enter first.")
                    continue
                if len(_ui["corner_clicks"]) != 4:
                    print("[SHEET] Click crosses #1, #6, #30, #25 first.")
                    continue
                try:
                    targets = make_grid_targets(_ui["corner_clicks"])
                except ValueError as exc:
                    print(f"[SHEET] {exc}")
                    _ui["corner_clicks"] = []
                    continue
                _ui["mode"] = "aiming"
                placement = int(_ui["placement"])
                print(f"\n[PLACEMENT {placement}] Starting automatic calibration of {len(targets)} grid points.")
                placement_results = []
                last_servo = read_servo_position(initial_servo)
                for target in targets:
                    if _ui["quit_requested"]:
                        break
                    outcome, last_servo, position_map = aim_and_record(
                        target, placement, position_map, samples, new_samples, last_servo
                    )
                    if outcome == "saved":
                        err = math.hypot(float(_ui["laser_pixel"][0]) - target["x"],
                                         float(_ui["laser_pixel"][1]) - target["y"]) if _ui.get("laser_pixel") else float("nan")
                        placement_results.append(err)
                        _ui["new_samples"] = new_samples
                    elif outcome == "quit":
                        _ui["quit_requested"] = True
                        break
                    # skipped points are intentionally not saved
                if _ui["quit_requested"]:
                    break
                passed = len(placement_results)
                mean_error = float(np.mean(placement_results)) if placement_results else float("nan")
                max_error = float(np.max(placement_results)) if placement_results else float("nan")
                print(f"[PLACEMENT {placement}] complete: saved {passed}/30 points; final measured miss mean={mean_error:.1f}px, max={max_error:.1f}px")
                save_checkpoint(samples, new_samples, placement + 1)
                _ui["placement"] = placement + 1
                _ui["corner_clicks"] = []
                _ui["mode"] = "corners"
                _ui["ui_message"] = "Placement complete. Move A4 to another area of the SAME target plane and click the four outside crosses."
                print("[NEXT] Move the A4 sheet to another region of the same target plane, then click #1, #6, #30, #25 and press G again.")
                continue
            if key == ord("f"):
                calibration_finished = True
                break

    except Exception as exc:
        print(f"[CAL] FATAL ERROR: {type(exc).__name__}: {exc}")
        print("[CAL] Stopping calibration safely; successful points remain checkpointed.")
        _ui["quit_requested"] = True
    finally:
        try:
            save_polygons()
            if new_samples:
                save_checkpoint(samples, new_samples, int(_ui["placement"]))
        except Exception as exc:
            print(f"[CAL] Checkpoint save warning: {exc}")
        try:
            restore_position(initial_servo, initial_actuator)
        except Exception as exc:
            print(f"[EXIT] Position restore warning: {exc}")
        _actuator_running = False
        if "worker" in locals():
            worker.join(timeout=1.0)
        try:
            set_actuator_enabled(False)
        except Exception:
            pass
        GPIO.cleanup()
        try:
            if servo is not None:
                servo.portHandler.closePort()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    if calibration_finished:
        finalize_map(samples, new_samples)
    elif _ui["quit_requested"]:
        print("[CAL] Stopped. All successful points remain checkpointed; the active map was not automatically replaced.")


if __name__ == "__main__":
    main()
