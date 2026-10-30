#!/usr/bin/env python3
"""Moving green-target tracker for the Jetson laser-weeding prototype.

Uses HSV green segmentation instead of YOLO. Reuses the existing tested
actuator/servo primitives and position_map.py from this project folder.

IMPORTANT:
- This test version contains no laser GPIO commands; the laser is not installed.
- The actuator is open-loop: reported position is estimated from step pulses.
- Motion requires an explicit GO confirmation and a manually accepted actuator reference.
"""

import os
import time
import math
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import Jetson.GPIO as GPIO

# Prefer the newer controller if present; fall back to the October async-display version.
try:
    import laser_servo_tracker_yolo_stable_auto_visit as hw
except ImportError:
    import laser_servo_tracker_point_yolo_modes_async_display as hw

from position_map import PositionMap, PositionMapError


# Resolve calibration resources relative to THIS script, not the terminal's
# current working directory. The controller module otherwise uses relative
# paths and can silently load a different position_map.json/servo_calibration.json.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_PATH = os.path.join(SCRIPT_DIR, "servo_calibration.json")
POSITION_MAP_PATH = os.path.join(SCRIPT_DIR, "position_map.json")

# Conservative motion profile from the pre-README controller version. The
# actuator has no encoder; high rate/acceleration can cause missed physical
# steps while the software counter still assumes each commanded pulse moved it.
TRACKER_ACTUATOR_MAX_SPEED_STEPS_S = 2500
TRACKER_ACTUATOR_ACCELERATION_STEPS_S2 = 1250


# --------------------------- CAMERA ----------------------------------------
CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30

# ------------------------ GREEN SEGMENTATION -------------------------------
# OpenCV HSV hue is 0..179. Tune these values for the dummy weed and lighting.
GREEN_H_MIN = 30
GREEN_H_MAX = 95
GREEN_S_MIN = 36
GREEN_S_MAX = 255
GREEN_V_MIN = 128
GREEN_V_MAX = 255
MIN_GREEN_CONTOUR_AREA = 2000       # pixels; reduce if the weed is small
MORPH_OPEN_SIZE = 3
MORPH_CLOSE_SIZE = 7
BBOX_SMOOTH_ALPHA = 0.65           # higher follows motion faster
MAX_TRACK_JUMP_PX = 240            # association gate for a moving target
# Temporal tracking / box grouping.
MERGE_IOU_THRESHOLD = 0.30         # group substantially overlapping leaf/fragment boxes
MERGE_GAP_PX = 10                   # also group contour boxes separated by a very small gap
CONFIRM_SECONDS = 0.50             # candidate must persist this long before it can drive motion
MIN_HIT_RATIO = 0.50               # tolerate brief per-frame segmentation gaps before confirmation
HISTORY_FRAMES = 20                # hit-ratio window (20 frames at 30 FPS ~= 0.67 s)
MIN_CONFIRM_HITS = 8               # minimum detections in history before confirmation
MAX_MISSED_SECONDS = 0.30          # only predict/drive through a very brief detection gap
TRACK_RETENTION_SECONDS = 1.00    # retain confirmed identity longer, but stop motion after MAX_MISSED_SECONDS
TRACK_MATCH_DISTANCE_PX = 80       # maximum centroid distance for matching detections to tracks
TRACK_MATCH_IOU_MIN = 0.01         # matching can also happen by IoU, even at low centroid movement
# Detection thresholds below are intentionally hard-coded; no JSON settings are loaded.

# ----------------------- MOTION / ENGAGEMENT -------------------------------
SERVO_CENTER_ID = 2056
SERVO_WINDOW_MIN_ID = 1930
SERVO_WINDOW_MAX_ID = 2171
JITTER_AMPLITUDE_ID = 12            # existing controller's servo-only jitter amplitude
JITTER_FREQUENCY_HZ = 6.0           # existing controller's jitter thread setting

# Only use positions from the measured calibration region, not extrapolation.
VALID_MAP_STATUSES = {"INTERPOLATED", "GRID"}
ACTUATOR_RETARGET_DELTA_STEPS = 25
ACTUATOR_ALIGNMENT_TOLERANCE_STEPS = 60
ACTUATOR_MAX_SPEED_FOR_ENGAGEMENT = 75
SERVO_SETTLE_TOLERANCE_ID = 8
SERVO_COMMAND_DEADBAND_ID = 4
TARGET_POINT_SMOOTH_ALPHA = 0.25   # restored pre-README target-point smoothing
TARGET_STABLE_WINDOW_S = 0.20
TARGET_STABLE_DISTANCE_PX = 12
APPROACH_MAX_SECONDS = 1.50
SIMULATED_SHOOT_DWELL_S = 0.15
JITTER_DURATION_S = 5
NEXT_TARGET_PAUSE_S = 0.05

# ----------------------- RECORDING / UI -----------------------------------
RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
RECORD_ON_START = True
WINDOW_NAME = "Moving Green Weed Tracker (annotated recording)"


def load_detection_settings():
    """Report the hard-coded detector values; intentionally do not load JSON."""
    print("[DETECTION] Using ONLY the hard-coded values in this script; tuner and JSON files are untouched/ignored.")
    print(f"[DETECTION] HSV H={GREEN_H_MIN}-{GREEN_H_MAX}, S={GREEN_S_MIN}-{GREEN_S_MAX}, "
          f"V={GREEN_V_MIN}-{GREEN_V_MAX}; min area={MIN_GREEN_CONTOUR_AREA}; "
          f"open={MORPH_OPEN_SIZE}; close={MORPH_CLOSE_SIZE}")


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bbox_gap(a, b):
    """Euclidean distance between rectangle edges; zero when they overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0.0, bx - (ax + aw), ax - (bx + bw))
    dy = max(0.0, by - (ay + ah), ay - (by + bh))
    return math.hypot(dx, dy)


def merge_boxes(boxes):
    """Group overlapping or immediately adjacent boxes belonging to the same weed."""
    merged = [tuple(map(float, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        result = []
        while merged:
            current = merged.pop(0)
            i = 0
            while i < len(merged):
                other = merged[i]
                if (bbox_iou(current, other) >= MERGE_IOU_THRESHOLD or
                        bbox_gap(current, other) <= MERGE_GAP_PX):
                    x1 = min(current[0], other[0])
                    y1 = min(current[1], other[1])
                    x2 = max(current[0] + current[2], other[0] + other[2])
                    y2 = max(current[1] + current[3], other[1] + other[3])
                    current = (x1, y1, x2 - x1, y2 - y1)
                    merged.pop(i)
                    changed = True
                    # Re-check after expansion so all adjacent fragments are included.
                    i = 0
                else:
                    i += 1
            result.append(current)
        merged = result
    return merged


class GreenTargetTracker:
    """Track green weeds while grouping overlapping leaf/fragment boxes."""
    def __init__(self):
        self.tracks = {}
        self.next_track_id = 1
        self.debug_tracks = []
        self.raw_detections = []
        self.frame_index = 0
        self.max_missed_frames = max(1, int(round(MAX_MISSED_SECONDS * FRAME_FPS)))
        open_size = self._odd_kernel(MORPH_OPEN_SIZE)
        close_size = self._odd_kernel(MORPH_CLOSE_SIZE)
        self.kernel_open = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
                            if open_size > 0 else None)
        self.kernel_close = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
                             if close_size > 0 else None)

    @staticmethod
    def _odd_kernel(value):
        value = int(value)
        if value <= 0:
            return 0
        return value if value % 2 else value + 1

    def segment(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([GREEN_H_MIN, GREEN_S_MIN, GREEN_V_MIN], dtype=np.uint8)
        upper = np.array([GREEN_H_MAX, GREEN_S_MAX, GREEN_V_MAX], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if self.kernel_open is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        if self.kernel_close is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        return mask

    @staticmethod
    def _center(b):
        return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0

    def update(self, mask):
        '''Track accepted contours using the earlier, box-center target logic.'''
        self.frame_index += 1
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MIN_GREEN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 3 and h >= 3:
                candidates.append({
                    "bbox": (float(x), float(y), float(w), float(h)),
                    "area": area,
                })

        # Pre-README logic: merge overlapping/near-touching boxes and use the center
        # of the merged, smoothed bounding box—not the green-pixel centroid.
        self.raw_detections = candidates
        boxes = merge_boxes([candidate["bbox"] for candidate in candidates])

        # Match by direct overlap and centroid distance. Do not use velocity to project
        # a different target center when the visible leaf outline shifts.
        pairs = []
        for tid, track in self.tracks.items():
            old = track["bbox"]
            old_cx, old_cy = self._center(old)
            for bi, box in enumerate(boxes):
                cx, cy = self._center(box)
                dist = math.hypot(cx - old_cx, cy - old_cy)
                iou = bbox_iou(old, box)
                if (dist <= TRACK_MATCH_DISTANCE_PX or
                        (iou >= TRACK_MATCH_IOU_MIN and dist <= 1.5 * TRACK_MATCH_DISTANCE_PX)):
                    score = (1.0 - iou) + dist / max(1.0, TRACK_MATCH_DISTANCE_PX)
                    pairs.append((score, tid, bi))
        pairs.sort(key=lambda item: item[0])

        used_tracks, used_boxes = set(), set()
        matches = {}
        for _, tid, bi in pairs:
            if tid not in used_tracks and bi not in used_boxes:
                used_tracks.add(tid)
                used_boxes.add(bi)
                matches[tid] = bi

        # Keep the original box smoothing (alpha 0.65). The main loop then applies
        # the original point smoothing (alpha 0.25) to the resulting box center.
        for tid, track in list(self.tracks.items()):
            if tid in matches:
                box = boxes[matches[tid]]
                old_box = track["bbox"]
                a = BBOX_SMOOTH_ALPHA
                track["bbox"] = tuple(a * box[i] + (1.0 - a) * old_box[i] for i in range(4))
                track["last_seen"] = time.monotonic()
                track["misses"] = 0
                track["hits"] += 1
                track["history"].append(1)
                track["matched"] = True
                track["area"] = track["bbox"][2] * track["bbox"][3]
            else:
                track["misses"] += 1
                track["history"].append(0)
                track["matched"] = False

        # Start a tentative track for each unmatched merged box.
        for bi, box in enumerate(boxes):
            if bi in used_boxes:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            now = time.monotonic()
            self.tracks[tid] = {
                "bbox": box,
                "first_seen": now,
                "last_seen": now,
                "misses": 0,
                "hits": 1,
                "history": deque([1], maxlen=HISTORY_FRAMES),
                "matched": True,
                "area": box[2] * box[3],
                "confirmed": False,
            }

        # Use elapsed time for retention, but keep confirmation latched to avoid
        # reverting a stable track to tentative due to a brief segmentation gap.
        now = time.monotonic()
        for tid in list(self.tracks):
            if now - self.tracks[tid]["last_seen"] > TRACK_RETENTION_SECONDS:
                del self.tracks[tid]

        self.debug_tracks = []
        confirmed = []
        for tid, track in self.tracks.items():
            age = now - track["first_seen"]
            history = track["history"]
            hit_ratio = sum(history) / max(1, len(history))
            meets_confirmation = (
                age >= CONFIRM_SECONDS
                and track["hits"] >= MIN_CONFIRM_HITS
                and len(history) >= MIN_CONFIRM_HITS
                and hit_ratio >= MIN_HIT_RATIO
            )
            if meets_confirmation:
                track["confirmed"] = True
            is_confirmed = bool(track.get("confirmed", False))
            b = tuple(int(round(v)) for v in track["bbox"])
            if track["matched"]:
                status = "CONFIRMED" if is_confirmed else "TENTATIVE"
            else:
                gap_s = now - track["last_seen"]
                if is_confirmed and gap_s <= MAX_MISSED_SECONDS:
                    status = "CONFIRMED-HOLD"
                elif is_confirmed and gap_s <= TRACK_RETENTION_SECONDS:
                    status = "CONFIRMED-RECOVERY"
                else:
                    status = "LOST"
            self.debug_tracks.append({
                "id": tid,
                "bbox": b,
                # Hold the last measured box through a short gap; do not extrapolate it.
                "predicted_bbox": b,
                "age": age,
                "hit_ratio": hit_ratio,
                "status": status,
                "matched": track["matched"],
                "confirmed": is_confirmed,
                "area": track["area"],
                "hits": track["hits"],
                "misses": track["misses"],
            })
            if is_confirmed and track["matched"]:
                confirmed.append((track["area"], age, tid, b))

        if not confirmed:
            return None
        _, _, tid, bbox = max(confirmed, key=lambda item: (item[0], item[1]))
        x, y, w, h = bbox
        midpoint = (int(round(x + w / 2)), int(round(y + h / 2)))
        return bbox, midpoint, tid


def main():
    """Identify -> lock -> move actuator and servo -> settle -> simulated shoot -> jitter -> next target.

    Laser GPIO is deliberately absent: this test setup has no laser installed.
    """
    import threading

    camera = None
    writer = None
    actuator_worker = None
    jitter_worker = None
    actuator_ready = False
    motion_started = False
    tracking_enabled = True
    recording = RECORD_ON_START
    last_commanded_actuator = None
    last_commanded_servo = None
    last_servo_command_time = 0.0
    last_servo_command_attempt = 0.0
    servo_command_failed = False
    last_map_status = "NO TARGET"
    last_waiting_map_status = None
    fps = 0.0
    fps_count = 0
    fps_start = time.monotonic()
    recorded_frames = 0
    last_candidate_report = 0.0

    # Explicit state machine.
    state = "IDENTIFY"
    locked_track_id = None
    locked_last_seen_time = None
    processed_track_ids = set()
    rejected_track_ids = set()  # Map/safety rejection is NOT a completed target.
    smoothed_point = None
    point_history = deque(maxlen=12)
    state_started = time.monotonic()
    state_note = "Waiting for confirmed weed"

    def change_state(new_state, note):
        nonlocal state, state_started, state_note
        state = new_state
        state_started = time.monotonic()
        state_note = note
        print(f"[STATE] {state}: {note}")

    def send_servo_position_direct(target):
        """Command position through python-st3215's documented WritePosition API."""
        target = int(round(max(SERVO_WINDOW_MIN_ID, min(SERVO_WINDOW_MAX_ID, target))))
        if hw.servo is None:
            raise RuntimeError("ST3215 servo object is not initialized")
        # Do not call ReadPosition or rewrite speed/acceleration for every target.
        # WritePosition is the library's direct target-position command.
        result = hw.servo.WritePosition(hw.ST_SERVO_ID, target)
        if result is False:
            raise RuntimeError("python-st3215 WritePosition reported failure")
        return target

    def command_servo(target):
        nonlocal last_commanded_servo, last_servo_command_time
        nonlocal last_servo_command_attempt, servo_command_failed
        target = int(round(max(SERVO_WINDOW_MIN_ID, min(SERVO_WINDOW_MAX_ID, target))))
        needs_command = (last_commanded_servo is None or
                         abs(target - last_commanded_servo) >= SERVO_COMMAND_DEADBAND_ID)
        if not needs_command:
            return last_commanded_servo

        now = time.monotonic()
        # A missing serial response can raise IndexError inside the ST3215 SDK.
        # Throttle retries so one bad response does not flood the servo bus.
        if now - last_servo_command_attempt < 0.25:
            return None if servo_command_failed else last_commanded_servo
        last_servo_command_attempt = now
        try:
            result = send_servo_position_direct(target)
            last_commanded_servo = int(result)
            last_servo_command_time = time.monotonic()
            servo_command_failed = False
            print(f"[SERVO] Aim command -> {last_commanded_servo}")
            return last_commanded_servo
        except Exception as exc:
            servo_command_failed = True
            print(f"[SERVO] WARNING: command/read failed ({type(exc).__name__}: {exc}); will retry. "
                  "Tracking continues; simulated shoot is blocked until servo command succeeds.")
            return None

    def servo_is_settled(target):
        # Never accept alignment after a failed servo command.
        if servo_command_failed:
            return False
        # Prefer real feedback when available; otherwise use a conservative settle delay.
        try:
            actual = hw.servo.ReadPosition(hw.ST_SERVO_ID) if hw.servo is not None else None
            if actual is not None and float(actual) >= 0:
                return abs(float(actual) - float(target)) <= SERVO_SETTLE_TOLERANCE_ID
        except Exception:
            pass
        return time.monotonic() - last_servo_command_time >= 0.30

    def start_jitter(base):
        # Keep both jitter endpoints inside the user's updated safe range.
        amplitude = min(JITTER_AMPLITUDE_ID,
                        max(0, (SERVO_WINDOW_MAX_ID - SERVO_WINDOW_MIN_ID) // 2))
        safe_base = int(round(max(SERVO_WINDOW_MIN_ID + amplitude,
                                  min(SERVO_WINDOW_MAX_ID - amplitude, base))))
        hw.JITTER_AMPLITUDE_POS = amplitude
        hw.JITTER_FREQUENCY_HZ = JITTER_FREQUENCY_HZ
        with hw._jitter_lock:
            hw._jitter_base = safe_base
            hw._jitter_start = time.monotonic()
            hw._jitter_active = True
        print(f"[JITTER] Enabled: base={safe_base}, amplitude=+/-{amplitude} IDs, "
              f"requested frequency={JITTER_FREQUENCY_HZ:.1f} Hz, duration={JITTER_DURATION_S:.1f} s")
        return safe_base

    def stop_jitter():
        with hw._jitter_lock:
            was_active = hw._jitter_active
            hw._jitter_active = False
            hw._jitter_start = None
            hw._jitter_base = None
        if was_active:
            print("[JITTER] Stop requested.")

    try:
        load_detection_settings()

        # Pin both files to the package directory. This avoids using a stale map
        # or travel calibration based on whichever folder launched Python.
        hw.CALIBRATION_FILE = CALIBRATION_PATH
        hw.POSITION_MAP_FILE = POSITION_MAP_PATH

        # Restore the conservative actuator speed/acceleration used in the older
        # controller version. Position is open-loop estimated, so don't trade
        # possible missed steps for speed while verifying aiming accuracy.
        if hasattr(hw, "ACT_MAX_SPEED_STEPS_S"):
            hw.ACT_MAX_SPEED_STEPS_S = TRACKER_ACTUATOR_MAX_SPEED_STEPS_S
        if hasattr(hw, "ACT_ACCELERATION_STEPS_S2"):
            hw.ACT_ACCELERATION_STEPS_S2 = TRACKER_ACTUATOR_ACCELERATION_STEPS_S2
        print(f"[FILES] Travel calibration: {CALIBRATION_PATH}")
        print(f"[FILES] Position map: {POSITION_MAP_PATH}")
        print(f"[ACTUATOR PROFILE] max speed={hw.ACT_MAX_SPEED_STEPS_S} steps/s; "
              f"acceleration={hw.ACT_ACCELERATION_STEPS_S2} steps/s^2 "
              "(conservative accuracy-test profile)")

        hw.load_calibration()
        # Override the controller's older visible limits with the user's current calibrated limits.
        hw.ST_VISIBLE_MIN_POSITION = SERVO_WINDOW_MIN_ID
        hw.ST_VISIBLE_MAX_POSITION = SERVO_WINDOW_MAX_ID
        hw.setup_servo()
        # Configure motion parameters once using python-st3215. Keep position
        # updates on WritePosition so an unavailable telemetry read cannot block aiming.
        try:
            hw.servo.SetAcceleration(hw.ST_SERVO_ID, int(hw.ST_ACCEL))
            hw.servo.SetSpeed(hw.ST_SERVO_ID, int(hw.ST_SPEED_MAX))
        except Exception as exc:
            print(f"[ST3215] WARNING: could not set speed/acceleration ({type(exc).__name__}: {exc}); "
                  "continuing with servo's stored settings.")
        position_map = PositionMap.load(POSITION_MAP_PATH)
        sample_x = [float(p["x"]) for p in position_map.samples]
        sample_y = [float(p["y"]) for p in position_map.samples]
        sample_a = [float(p["actuator"]) for p in position_map.samples]
        sample_s = [float(p["servo"]) for p in position_map.samples]
        print(f"[MAP] Loaded {position_map.map_type}; {position_map.camera_width}x{position_map.camera_height}; "
              f"{len(position_map.samples)} samples from {POSITION_MAP_PATH}")
        print(f"[MAP COVERAGE] pixel x={min(sample_x):.0f}..{max(sample_x):.0f}, "
              f"y={min(sample_y):.0f}..{max(sample_y):.0f}; "
              f"servo samples={min(sample_s):.0f}..{max(sample_s):.0f}; "
              f"actuator samples={min(sample_a):.0f}..{max(sample_a):.0f} steps")
        if (min(sample_a) < float(hw.ACTUATOR_MIN_POSITION) - 1.0 or
                max(sample_a) > float(hw.ACTUATOR_MAX_POSITION) + 1.0):
            raise RuntimeError(
                f"Position map actuator samples span {min(sample_a):.0f}..{max(sample_a):.0f} steps, "
                f"but the loaded travel is {hw.ACTUATOR_MIN_POSITION}..{hw.ACTUATOR_MAX_POSITION}. "
                "The map and actuator calibration do not match; motion is stopped for safety."
            )
        print(f"[SERVO] Safe IDs {SERVO_WINDOW_MIN_ID}..{SERVO_WINDOW_MAX_ID}; center={SERVO_CENTER_ID}")

        if hw.ACTUATOR_MAX_POSITION > 0:
            middle = hw.actuator_middle()
            answer = input(f"Is the actuator PHYSICALLY at its saved middle ({middle:.0f} steps)? [y/N]: ").strip().lower()
            if answer in ("y", "yes"):
                hw.accept_saved_middle()
                actuator_ready = True
                print("[ACTUATOR] Saved midpoint accepted; position is open-loop estimated.")
            else:
                print("[ACTUATOR] Motion disabled until a valid reference is accepted.")
        else:
            print("[ACTUATOR] No saved travel calibration; motion disabled.")

        print("\nSAFETY: motion remains disabled until you type GO.")
        if input("Type GO to start: ").strip() != "GO":
            print("[START] GO not entered; no motion started.")
            return
        motion_started = True
        send_servo_position_direct(SERVO_CENTER_ID)
        last_commanded_servo = SERVO_CENTER_ID
        last_servo_command_time = time.monotonic()
        print(f"[SERVO] Sent center command {SERVO_CENTER_ID}; servo will aim using calibrated map targets.")
        print("[LASER] No laser control in this version. SHOOT is a timed software test stage only.")

        if position_map.camera_width != FRAME_WIDTH or position_map.camera_height != FRAME_HEIGHT:
            raise RuntimeError(f"Position map expects {position_map.camera_width}x{position_map.camera_height}, "
                               f"but configured camera is {FRAME_WIDTH}x{FRAME_HEIGHT}.")
        camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(CAMERA_DEVICE)
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera device {CAMERA_DEVICE}")
        # Match the tuner: request resolution only. Do not force FPS/MJPG here,
        # because that can change the camera stream used by HSV segmentation.
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"[CAMERA] Requested {FRAME_WIDTH}x{FRAME_HEIGHT}; actual "
              f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}; "
              f"reported FPS={camera.get(cv2.CAP_PROP_FPS):.1f}")

        os.makedirs(RECORD_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(RECORD_DIR, f"green_weed_cycle_{timestamp}.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(FRAME_FPS),
                                 (FRAME_WIDTH, FRAME_HEIGHT))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create annotated MP4 recording: {video_path}")
        print(f"[VIDEO] Recording started: {os.path.abspath(video_path)}")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
        print("[UI] R recording toggle | C stop motion | T resume | Q quit")
        print("[TRACK METRICS] hit=recent detection percentage; miss=consecutive missed frames.")
        print(f"[TRACK STABILITY] predict/control gap={MAX_MISSED_SECONDS:.2f}s; confirmed ID retention={TRACK_RETENTION_SECONDS:.2f}s. Motion stops after the short control grace, but the confirmed ID is kept for reacquisition.")
        print("[BOX COLORS] GREEN=locked, BLUE/CYAN=confirmed candidate, YELLOW=tentative, RED=DONE/processed track.")
        print("[FLOW] IDENTIFY -> LOCK -> ACTUATE -> SETTLE -> SIMULATED_SHOOT -> JITTER -> NEXT")

        actuator_worker = threading.Thread(target=hw.actuator_thread, daemon=True)
        jitter_worker = threading.Thread(target=hw.jitter_thread, daemon=True)
        actuator_worker.start()
        jitter_worker.start()
        if actuator_ready:
            hw.set_actuator_enabled(True)

        tracker = GreenTargetTracker()
        running = True
        while running:
            ok, frame = camera.read()
            if not ok:
                if actuator_ready:
                    hw.set_actuator_speed(0.0)
                stop_jitter()
                change_state("IDENTIFY", "Camera read failed; waiting for a valid frame")
                time.sleep(0.01)
                continue
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                raise RuntimeError(f"Camera returned {frame.shape[1]}x{frame.shape[0]}, expected {FRAME_WIDTH}x{FRAME_HEIGHT}.")

            now = time.monotonic()
            fps_count += 1
            if now - fps_start >= 1.0:
                fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now

            mask = tracker.segment(frame)
            detection = tracker.update(mask)  # (bbox, midpoint, track_id) or None
            debug_by_id = {item["id"]: item for item in tracker.debug_tracks}
            # A processed target can be selected again only after its track disappears.
            live_ids = set(debug_by_id)
            processed_track_ids.intersection_update(live_ids)
            rejected_track_ids.intersection_update(live_ids)

            selected = None
            selected_is_predicted = False
            if locked_track_id is None and state == "NEXT" and now - state_started >= NEXT_TARGET_PAUSE_S:
                change_state("IDENTIFY", "Searching for next unprocessed target")
            if locked_track_id is not None:
                item = debug_by_id.get(locked_track_id)
                if item and item.get("confirmed") and item["matched"]:
                    # Refresh the real observation timestamp only when the camera has
                    # actually detected this target in the current frame.
                    locked_last_seen_time = now
                    x, y, w, h = item["bbox"]
                    selected = ((x, y, w, h), (int(x + w / 2), int(y + h / 2)), locked_track_id)
                elif (item and item.get("confirmed") and locked_last_seen_time is not None
                      and now - locked_last_seen_time <= MAX_MISSED_SECONDS):
                    # Keep the same confirmed ID selected through a brief segmentation
                    # gap. The last measured box center is held through a brief gap (no velocity prediction).
                    x, y, w, h = item.get("predicted_bbox", item["bbox"])
                    selected = ((x, y, w, h), (int(round(x + w / 2)), int(round(y + h / 2))), locked_track_id)
                    selected_is_predicted = True
                elif (locked_last_seen_time is not None
                      and now - locked_last_seen_time <= TRACK_RETENTION_SECONDS):
                    # Keep the confirmed ID available for reacquisition, but selected
                    # remains None after the short control grace expires. The fail-safe
                    # below stops actuator motion and jitter until a real detection returns.
                    pass
                elif locked_last_seen_time is not None and now - locked_last_seen_time > TRACK_RETENTION_SECONDS:
                    gap_s = now - locked_last_seen_time
                    print(f"[LOCK] Track {locked_track_id} lost; releasing after {gap_s:.2f}s without a detection.")
                    stop_jitter()
                    if actuator_ready:
                        hw.set_actuator_speed(0.0)
                    locked_track_id = None
                    locked_last_seen_time = None
                    smoothed_point = None
                    point_history.clear()
                    change_state("IDENTIFY", "Previous target lost")
            elif state == "IDENTIFY":
                # Only acquire a new target in IDENTIFY; never lock one during NEXT.
                available = [item for item in tracker.debug_tracks
                             if item["status"] == "CONFIRMED" and item["matched"]
                             and item["id"] not in processed_track_ids
                             and item["id"] not in rejected_track_ids]
                if available:
                    # Prefer the confirmed weed whose mapped servo position is closest
                    # to the safe window; use area only as a tie-breaker. Extrapolated
                    # map values are for ranking only, never for commanding motion.
                    def target_priority(candidate):
                        x, y, w, h = candidate["bbox"]
                        px, py = x + w / 2.0, y + h / 2.0
                        try:
                            servo_est, _, _ = position_map.pixel_to_position(px, py)
                            distance = max(SERVO_WINDOW_MIN_ID - servo_est, 0.0,
                                           servo_est - SERVO_WINDOW_MAX_ID)
                        except (PositionMapError, TypeError, ValueError):
                            distance = float("inf")
                        return (distance, -candidate["area"])
                    item = min(available, key=target_priority)
                    bbox = item["bbox"]
                    x, y, w, h = bbox
                    point = (int(x + w / 2), int(y + h / 2))
                    tid = item["id"]
                    locked_track_id = tid
                    locked_last_seen_time = now
                    smoothed_point = (float(point[0]), float(point[1]))
                    point_history.clear()
                    point_history.append((now, smoothed_point[0], smoothed_point[1]))
                    change_state("LOCK", f"Locked confirmed target ID {tid}")
                    print(f"[TARGET] Acquired ID={tid}, bbox={bbox}, midpoint={point}, area={item['area']:.0f}")
                    selected = (bbox, point, tid)
                elif now - last_candidate_report >= 1.0:
                    last_candidate_report = now
                    summary = [f"ID {c['id']}:{c['status']}:{'DONE' if c['id'] in processed_track_ids else ('MAP_REJECTED' if c['id'] in rejected_track_ids else ('READY' if c['matched'] and c['status'] == 'CONFIRMED' else 'WAIT'))}"
                               for c in tracker.debug_tracks]
                    print(f"[TARGET] IDENTIFY found no selectable target. candidates={summary or 'none'}; processed={sorted(processed_track_ids)}; map_rejected={sorted(rejected_track_ids)}")

            # Keep visual diagnosis of all candidates, but select only the locked target.
            output = frame.copy()
            for candidate in tracker.raw_detections:
                x, y, w, h = (int(round(v)) for v in candidate["bbox"])
                cv2.rectangle(output, (x, y), (x + w, y + h), (185, 185, 185), 1)
            for item in tracker.debug_tracks:
                x, y, w, h = item["bbox"]
                is_locked = item["id"] == locked_track_id
                is_done = item["id"] in processed_track_ids
                is_rejected = item["id"] in rejected_track_ids
                is_ready = item["status"] == "CONFIRMED" and item["matched"] and not is_done and not is_rejected
                color = (0, 255, 0) if is_locked else ((0, 0, 255) if is_done else ((255, 0, 255) if is_rejected else ((0, 220, 255) if item["status"] == "TENTATIVE" else (255, 180, 0))))
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 4 if is_locked else 3)
                tag = "LOCKED" if is_locked else ("DONE" if is_done else ("MAP REJECT" if is_rejected else ("READY" if is_ready else item["status"])))
                label = f"ID {item['id']} {tag} | {item['age']:.1f}s | hit {item['hit_ratio']:.0%} | miss {item['misses']}"
                label_y = max(24, y - 7)
                cv2.putText(output, label, (max(8, x), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)

            mapped_servo = mapped_actuator = None
            map_status = "NO LOCKED TARGET"
            map_valid = False
            aligned = False
            servo_settled = False
            stable_target = False
            target_point = None

            if selected is not None and locked_track_id is not None:
                bbox, raw_point, tid = selected
                if smoothed_point is None:
                    smoothed_point = (float(raw_point[0]), float(raw_point[1]))
                else:
                    a = TARGET_POINT_SMOOTH_ALPHA
                    smoothed_point = (a * raw_point[0] + (1.0 - a) * smoothed_point[0],
                                      a * raw_point[1] + (1.0 - a) * smoothed_point[1])
                target_point = (int(round(smoothed_point[0])), int(round(smoothed_point[1])))
                point_history.append((now, smoothed_point[0], smoothed_point[1]))
                recent = [(t, x, y) for t, x, y in point_history if now - t <= TARGET_STABLE_WINDOW_S]
                if len(recent) >= 3:
                    xs = [p[1] for p in recent]
                    ys = [p[2] for p in recent]
                    stable_target = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) <= TARGET_STABLE_DISTANCE_PX

                x, y, w, h = bbox
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Mark the exact image pixel passed to position_map.pixel_to_position().
                # This changes only the viewport annotation; target and motion logic stay unchanged.
                cross_size = 12
                cx, cy = target_point
                cv2.drawMarker(
                    output, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS,
                    markerSize=cross_size * 2, thickness=2, line_type=cv2.LINE_AA
                )

                try:
                    mapped_servo, mapped_actuator, map_status = position_map.pixel_to_position(*target_point)
                    # Map-region validity and servo-window validity are separate. A weed can be
                    # outside the servo window while the actuator moves underneath its path.
                    map_valid = map_status in VALID_MAP_STATUSES
                    last_map_status = map_status
                except PositionMapError as exc:
                    map_status = f"MAP ERROR: {exc}"
                    map_valid = False

                servo_in_window = (mapped_servo is not None and
                                   SERVO_WINDOW_MIN_ID <= mapped_servo <= SERVO_WINDOW_MAX_ID)
                actuator_in_range = (mapped_actuator is not None and
                                     hw.ACTUATOR_MIN_POSITION <= mapped_actuator <= hw.ACTUATOR_MAX_POSITION)
                if not actuator_in_range:
                    map_valid = False
                    map_status = "ACTUATOR TARGET OUT OF RANGE"

                if tracking_enabled and actuator_ready and map_valid and state in ("LOCK", "ACTUATE", "SETTLE"):
                    # Always follow the mapped actuator line while the weed is tracked, even when
                    # the servo target is outside its safe range. Only command the servo inside range.
                    if last_commanded_actuator is None or abs(mapped_actuator - last_commanded_actuator) >= ACTUATOR_RETARGET_DELTA_STEPS:
                        previous_target = last_commanded_actuator
                        last_commanded_actuator = hw.set_actuator_position_target(mapped_actuator)
                        print(f"[ACTUATOR] Target={last_commanded_actuator:.1f} steps; "
                              f"estimated position={hw.get_actuator_position():.1f}; "
                              f"pixel={target_point}; previous target={previous_target}")
                    if servo_in_window:
                        command_servo(mapped_servo)
                    elif state == "LOCK":
                        print(f"[AIM] ID={tid}: servo target {mapped_servo:.1f} outside safe window "
                              f"{SERVO_WINDOW_MIN_ID}..{SERVO_WINDOW_MAX_ID}; holding servo and following "
                              f"actuator target {mapped_actuator:.0f} until weed enters window.")
                    if state == "LOCK":
                        change_state("ACTUATE", "Following weed path; waiting for servo-safe window")

                if actuator_ready and map_valid:
                    estimated_position = float(hw.get_actuator_position())
                    estimated_speed = abs(float(hw.get_actuator_speed()))
                    desired_position = hw.get_actuator_position_target()
                    if desired_position is None:
                        desired_position = mapped_actuator
                    aligned = (abs(estimated_position - float(desired_position)) <= ACTUATOR_ALIGNMENT_TOLERANCE_STEPS
                               and estimated_speed <= ACTUATOR_MAX_SPEED_FOR_ENGAGEMENT)
                servo_settled = (servo_in_window and last_commanded_servo is not None
                                 and servo_is_settled(last_commanded_servo))

                # EXTRAPOLATED means the pixel is outside the measured calibration hull.
                # Keep tracking the weed (do not mark it rejected/DONE), but do not move
                # using an unvalidated motor estimate. When it enters the measured map,
                # actuator following resumes automatically. Servo-window misses alone
                # never reject a weed.
                if map_status == "EXTRAPOLATED":
                    stop_jitter()
                    if actuator_ready:
                        hw.set_actuator_speed(0.0)
                    last_commanded_actuator = None
                    if last_waiting_map_status != "EXTRAPOLATED":
                        print(f"[MAP WAIT] ID={tid}; pixel={target_point} is outside measured calibration hull. "
                              "Keeping track; motion paused until the weed enters the calibrated map.")
                    last_waiting_map_status = "EXTRAPOLATED"
                elif map_status == "ACTUATOR TARGET OUT OF RANGE":
                    stop_jitter()
                    if actuator_ready:
                        hw.set_actuator_speed(0.0)
                    last_commanded_actuator = None
                    if last_waiting_map_status != map_status:
                        print(f"[MAP WAIT] ID={tid}; mapped actuator target {mapped_actuator} is outside "
                              f"software travel {hw.ACTUATOR_MIN_POSITION}..{hw.ACTUATOR_MAX_POSITION}. "
                              "Keeping the target locked; motion pauses until the mapped target is in range.")
                    last_waiting_map_status = map_status
                elif not map_valid:
                    stop_jitter()
                    if actuator_ready:
                        hw.set_actuator_speed(0.0)
                    rejected_id = locked_track_id
                    rejected_track_ids.add(rejected_id)
                    print(f"[TARGET] MAP REJECTED ID={rejected_id}; pixel={target_point}; servo_target={mapped_servo}; actuator_target={mapped_actuator}; reason={map_status}. Not marked DONE.")
                    locked_track_id = None
                    locked_last_seen_time = None
                    smoothed_point = None
                    point_history.clear()
                    last_commanded_actuator = None
                    last_waiting_map_status = None
                    change_state("IDENTIFY", f"Target rejected: {map_status}")
                else:
                    last_waiting_map_status = None
                    if state == "ACTUATE" and not servo_in_window:
                        # Keep moving the actuator with the moving weed; never shoot/jitter outside window.
                        pass
                    elif state == "ACTUATE" and servo_in_window and aligned and servo_settled:
                        change_state("SETTLE", "Actuator aligned and servo reached aim")
                    elif state == "SETTLE" and not servo_in_window:
                        change_state("ACTUATE", "Weed moved outside servo window; continue following with actuator")
                    elif state == "SETTLE":
                        if aligned and servo_settled and (stable_target or now - state_started >= APPROACH_MAX_SECONDS):
                            change_state("SIMULATED_SHOOT", "Aim accepted; no laser installed, timing simulated shot")
                    elif state == "SIMULATED_SHOOT" and now - state_started >= SIMULATED_SHOOT_DWELL_S:
                        base = start_jitter(last_commanded_servo if last_commanded_servo is not None else mapped_servo)
                        change_state("JITTER", f"Servo-only jitter around ID {base}")
                    elif state == "JITTER" and now - state_started >= JITTER_DURATION_S:
                        stop_jitter()
                        command_servo(mapped_servo)
                        processed_track_ids.add(locked_track_id)
                        print(f"[CYCLE] Completed target ID {locked_track_id}; moving to next confirmed target.")
                        locked_track_id = None
                        locked_last_seen_time = None
                        smoothed_point = None
                        point_history.clear()
                        last_commanded_actuator = None
                        change_state("NEXT", "Target cycle complete")


            # Stop immediately when the operator pauses. For a brief vision gap on an
            # already locked target, preserve the current absolute actuator target and
            # active jitter until the existing lost-target grace period expires. Calling
            # set_actuator_speed(0) on every missed frame would clear position mode and
            # repeatedly interrupt motion; it also killed jitter during ordinary gaps.
            if not tracking_enabled:
                stop_jitter()
                if actuator_ready:
                    hw.set_actuator_speed(0.0)
                last_commanded_actuator = None
            elif selected is None:
                in_lost_target_grace = (
                    locked_track_id is not None
                    and locked_last_seen_time is not None
                    and now - locked_last_seen_time <= MAX_MISSED_SECONDS
                )
                if not in_lost_target_grace:
                    stop_jitter()
                    if actuator_ready:
                        hw.set_actuator_speed(0.0)
                    last_commanded_actuator = None

            if not actuator_ready:
                status = "ACTUATOR UNREFERENCED - MOTION DISABLED"
                status_color = (0, 0, 255)
            elif not tracking_enabled:
                status = "PAUSED - MOTION STOPPED"
                status_color = (0, 165, 255)
            elif selected is None:
                gap_s = (now - locked_last_seen_time) if locked_last_seen_time is not None else float("inf")
                if locked_track_id is not None and gap_s <= TRACK_RETENTION_SECONDS:
                    status = f"{state}: WAITING FOR TRACK {locked_track_id} TO REAPPEAR; MOTION STOPPED"
                    status_color = (0, 165, 255)
                else:
                    status = f"{state}: NO LOCKED TARGET"
                    status_color = (0, 0, 255)
            else:
                if selected_is_predicted:
                    status = f"{state}: HOLDING TRACK {locked_track_id} THROUGH BRIEF GAP"
                    status_color = (0, 165, 255)
                else:
                    status = f"{state}: {state_note}"
                    status_color = (0, 255, 0) if state in ("SIMULATED_SHOOT", "JITTER", "NEXT") else (0, 165, 255)

            cv2.putText(output, status[:62], (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA)
            cv2.putText(output, f"{fps:.1f} FPS | raw contours {len(tracker.raw_detections)} | {state} | ID {locked_track_id}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
            if target_point is not None:
                est_count = float(hw.get_actuator_position()) if actuator_ready else float("nan")
                servo_text = f"{mapped_servo:.1f}" if mapped_servo is not None else "--"
                actuator_text = f"{mapped_actuator:.0f}" if mapped_actuator is not None else "--"
                est_text = f"{est_count:.0f}" if math.isfinite(est_count) else "--"
                aim_line = (f"AIM PIXEL {target_point} | MAP {map_status} | "
                            f"SERVO {servo_text} | ACT TARGET {actuator_text} | "
                            f"ACT COUNT {est_text} (open-loop)")
            else:
                aim_line = f"AIM PIXEL -- | MAP {map_status}"
            cv2.putText(output, aim_line[:145], (20, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output, f"REC {'ON' if recording else 'OFF'} | R record | C stop | T resume | Q quit", (20, FRAME_HEIGHT - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255) if recording else (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (FRAME_WIDTH - 245, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, output)
            if recording and writer is not None:
                writer.write(output)
                recorded_frames += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                running = False
            elif key == ord('r'):
                recording = not recording
                print(f"[VIDEO] Recording {'ON' if recording else 'OFF'}")
            elif key == ord('c'):
                tracking_enabled = False
                stop_jitter()
                if actuator_ready:
                    hw.set_actuator_speed(0.0)
                locked_track_id = None
                locked_last_seen_time = None
                smoothed_point = None
                point_history.clear()
                last_commanded_actuator = None
                change_state("IDENTIFY", "Motion stopped by operator")
                print("[CONTROL] Stopped. Press T to resume; Q to quit.")
            elif key == ord('t'):
                tracking_enabled = True
                locked_track_id = None
                locked_last_seen_time = None
                smoothed_point = None
                point_history.clear()
                last_commanded_actuator = None
                change_state("IDENTIFY", "Operator resumed tracking")

    finally:
        stop_jitter()
        if motion_started:
            try:
                send_servo_position_direct(SERVO_CENTER_ID)
                print(f"[SHUTDOWN] Servo commanded to center {SERVO_CENTER_ID}.")
            except Exception as exc:
                print(f"[SHUTDOWN] Could not center servo: {exc}")
            if actuator_worker is not None and actuator_worker.is_alive() and actuator_ready:
                try:
                    middle = float(hw.actuator_middle())
                    current = float(hw.get_actuator_position())
                    print(f"[SHUTDOWN] Returning actuator to saved middle {middle:.0f} from estimated position {current:.0f}...")
                    hw.set_actuator_enabled(True)
                    hw.set_actuator_position_target(middle)
                    deadline = time.monotonic() + 60.0
                    reached = False
                    while time.monotonic() < deadline:
                        current = float(hw.get_actuator_position())
                        speed = abs(float(hw.get_actuator_speed()))
                        target = hw.get_actuator_position_target()
                        if target is not None and abs(current - middle) <= 8 and speed <= 10:
                            reached = True
                            break
                        time.sleep(0.05)
                    if reached:
                        print(f"[SHUTDOWN] Actuator reached saved midpoint (estimated {current:.0f}).")
                    else:
                        print(f"[SHUTDOWN] WARNING: midpoint return not confirmed; estimated position={current:.0f}, speed={hw.get_actuator_speed():.1f}")
                        hw.set_actuator_speed(0.0)
                except Exception as exc:
                    print(f"[SHUTDOWN] WARNING: midpoint return failed: {exc}")
                    try:
                        hw.set_actuator_speed(0.0)
                    except Exception:
                        pass
            else:
                try:
                    hw.set_actuator_speed(0.0)
                except Exception:
                    pass
                if actuator_ready:
                    print("[SHUTDOWN] Actuator worker not running; midpoint return unavailable.")

        hw._jitter_running = False
        hw._actuator_running = False
        if actuator_worker is not None:
            actuator_worker.join(timeout=1.0)
        if jitter_worker is not None:
            jitter_worker.join(timeout=1.0)
        try:
            hw.set_actuator_enabled(False)
        except Exception:
            pass
        if writer is not None:
            writer.release()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        try:
            if hw.servo is not None:
                hw.servo.portHandler.closePort()
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass
        if 'video_path' in locals():
            print(f"[VIDEO] Recording path: {os.path.abspath(video_path)}")
            print(f"[VIDEO] Frames written: {recorded_frames}")
            if os.path.isfile(video_path):
                print(f"[VIDEO] File size: {os.path.getsize(video_path)} bytes")
                if os.path.getsize(video_path) == 0 or recorded_frames == 0:
                    print("[VIDEO] WARNING: recording file is empty or no frames were written.")
            else:
                print("[VIDEO] ERROR: recording file was not created.")
        print("[SHUTDOWN] No laser output was configured or controlled.")


if __name__ == "__main__":
    main()
