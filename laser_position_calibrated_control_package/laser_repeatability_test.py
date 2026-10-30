#!/usr/bin/env python3
"""Repeatability test for the existing laser-servo tracker.

Run from the project directory so the existing controller, position_map.py,
position_map.json and servo_calibration.json are available.

Mouse clicks are used to mark target coordinates and the observed laser spot.
This avoids relying on automatic laser detection where reflections are present.
"""

import csv
import datetime as dt
import json
import math
import os
import threading
import time

import cv2
import numpy as np

import laser_servo_tracker_yolo_stable_auto_visit as hw
from position_map import PositionMap, PositionMapError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30  # Fallback recording FPS if the camera does not report its actual rate.
TARGETS_FILE = os.path.join(SCRIPT_DIR, "repeatability_targets.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "repeatability_results")
POSITION_MAP_FILE = os.path.join(SCRIPT_DIR, "position_map.json")
CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "servo_calibration.json")
SETTLE_SECONDS = 1.0
MOVE_TIMEOUT_SECONDS = 20.0
HIT_RADIUS_MM = 2.0  # Set this from the application's required physical targeting tolerance.
HOMOGRAPHY_FILE = os.path.join(SCRIPT_DIR, "pixel_to_mm_homography.json")
DELETE_RADIUS_PX = 32  # Click this close to a target while in delete mode to remove it.
# The homography maps undistorted image pixels to X/Y millimetres on the fixed target plane.
# It accounts for perspective variation across the plane; lens distortion should be corrected first.

state = {
    "mode": "idle",  # idle, add_target, mark_spot
    "target_points": [],
    "spot_click": None,
    "diameter_clicks": [],
    "last_frame": None,
    "active_test_target": None,
    "test_status": "IDLE",
    "record_message": "",
    "recorder": None,
}


def delete_nearest_target(x, y):
    """Delete the closest saved/in-memory target when within the click radius."""
    points = state["target_points"]
    if not points:
        print("[TARGET] No points to delete.")
        return
    distances = [math.hypot(float(p["x"]) - x, float(p["y"]) - y) for p in points]
    index = min(range(len(points)), key=distances.__getitem__)
    if distances[index] > DELETE_RADIUS_PX:
        print(f"[TARGET] No point within {DELETE_RADIUS_PX}px of ({x},{y}); nearest is {distances[index]:.1f}px away.")
        return
    removed = points.pop(index)
    print(f"[TARGET] Deleted point #{index + 1}: ({removed['x']:.1f},{removed['y']:.1f}).")
    # Persist deletions immediately so deleted saved points do not return next launch.
    save_targets(points)


def mouse_callback(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state["mode"] == "add_target":
        state["target_points"].append({"x": int(x), "y": int(y)})
        print(f"[TARGET] Added point {len(state['target_points'])}: ({x}, {y})")
    elif state["mode"] == "delete_target":
        delete_nearest_target(x, y)
    elif state["mode"] == "mark_spot":
        if state["spot_click"] is None:
            state["spot_click"] = (int(x), int(y))
        elif state.get("measure_diameter", False) and len(state["diameter_clicks"]) < 2:
            state["diameter_clicks"].append((int(x), int(y)))


def load_targets():
    if not os.path.exists(TARGETS_FILE):
        return []
    with open(TARGETS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("points", [])
    return [
        {"id": int(p.get("id", i + 1)), "x": float(p["x"]), "y": float(p["y"])}
        for i, p in enumerate(points)
    ]


def save_targets(points):
    data = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "camera_width": FRAME_WIDTH,
        "camera_height": FRAME_HEIGHT,
        "coordinate_system": "camera pixels; origin at top-left",
        "points": [
            {"id": i + 1, "x": int(p["x"]), "y": int(p["y"])}
            for i, p in enumerate(points)
        ],
    }
    with open(TARGETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[TARGET] Saved {len(points)} points to {TARGETS_FILE}")


def _draw_target_overlay(frame, message=""):
    """Annotate every saved point in every view/recorded frame, highlighting the active point."""
    out = frame.copy()
    active = state.get("active_test_target")
    active_xy = None
    if active is not None:
        active_xy = (float(active["x"]), float(active["y"]))

    for i, p in enumerate(list(state.get("target_points", [])), start=1):
        x, y = int(round(float(p["x"]))), int(round(float(p["y"])))
        is_active = active_xy is not None and math.hypot(x - active_xy[0], y - active_xy[1]) <= 1.5
        color = (0, 255, 0) if is_active else (0, 255, 255)  # green=current; yellow=all other targets
        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 22 if is_active else 18, 3 if is_active else 2)
        label = f"P{i}"
        if is_active:
            label += f" ACTIVE T{active.get('trial', '?')}"
        cv2.putText(out, label, (min(max(6, x + 10), max(6, out.shape[1] - 170)),
                                min(max(20, y - 9), out.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58 if is_active else 0.52,
                    color, 2, cv2.LINE_AA)

    # Show the measured spot and optional diameter marks while collecting a measurement.
    spot = state.get("spot_click")
    if spot is not None:
        cv2.drawMarker(out, (int(spot[0]), int(spot[1])), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(out, "MEASURED SPOT", (int(spot[0]) + 10, int(spot[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
    for edge in list(state.get("diameter_clicks", [])):
        cv2.circle(out, (int(edge[0]), int(edge[1])), 5, (255, 0, 255), -1)

    status = str(state.get("test_status", "IDLE"))
    if status:
        cv2.putText(out, status[:115], (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if not message:
        message = state.get("record_message", "")
    if message:
        cv2.putText(out, str(message)[:115], (15, 59), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class CameraRecorder:
    """Continuously read camera frames to avoid gaps while motors are moving.

    Video FPS is set from the camera's reported capture rate (fallback 30 FPS).
    Every captured frame is written, so time spent moving/settling/measuring is
    preserved in real time. Target annotations are burned into the recording.
    """
    def __init__(self, cap, video_path):
        self.cap = cap
        self.video_path = video_path
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.fps = reported_fps if 5.0 <= reported_fps <= 120.0 else float(FRAME_FPS)
        self.width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) or FRAME_WIDTH
        self.height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) or FRAME_HEIGHT
        self.writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height)
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Could not create test video: {video_path}")
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.first_frame_event = threading.Event()
        self.frame_count = 0
        self.read_errors = 0
        self.thread = threading.Thread(target=self._run, name="repeatability-camera-recorder", daemon=True)

    def start(self):
        self.thread.start()
        if not self.first_frame_event.wait(timeout=3.0):
            self.stop()
            raise RuntimeError("Camera recorder did not receive a frame within 3 seconds.")
        print(f"[VIDEO] Recording started: {self.video_path}")
        print(f"[VIDEO] Playback rate={self.fps:.2f} FPS; camera reported={float(self.cap.get(cv2.CAP_PROP_FPS)):.2f} FPS; target points will stay annotated.")

    def _run(self):
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.read_errors += 1
                    time.sleep(0.005)
                    continue
                annotated = _draw_target_overlay(frame)
                if (annotated.shape[1], annotated.shape[0]) != (self.width, self.height):
                    annotated = cv2.resize(annotated, (self.width, self.height))
                self.writer.write(annotated)
                self.frame_count += 1
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                self.first_frame_event.set()
        finally:
            self.writer.release()

    def get_latest_frame(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is not None:
                return frame
            if not self.thread.is_alive() and self.first_frame_event.is_set():
                break
            time.sleep(0.005)
        raise RuntimeError("No camera frame available from the test recorder.")

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
        if self.thread.is_alive():
            print("[VIDEO] WARNING: camera recorder thread did not stop within 3 seconds.")
        else:
            print(f"[VIDEO] Recording stopped: {self.frame_count} frames, approx {self.frame_count / max(self.fps, 1.0):.1f} seconds, {self.read_errors} read errors.")


def get_frame(cap):
    recorder = state.get("recorder")
    if recorder is not None and recorder.thread.is_alive():
        return recorder.get_latest_frame()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera frame read failed")
    return frame


def show_live(cap, message=""):
    frame = get_frame(cap)
    frame = _draw_target_overlay(frame, message)
    cv2.imshow("Laser Repeatability Test", frame)
    state["last_frame"] = frame
    return cv2.waitKey(1) & 0xFF


def wait_for_spot_click(cap, target, point_id, trial):
    """Click blob centre; optionally press D and click opposite blob edges."""
    state["spot_click"] = None
    state["diameter_clicks"] = []
    state["measure_diameter"] = False
    state["mode"] = "mark_spot"
    state["test_status"] = f"MEASURE POINT {point_id} | TRIAL {trial}: click spot centre"
    state["record_message"] = "Click blob centre | D=measure diameter | Enter=accept centre"
    print(
        f"[MEASURE] Point {point_id}, trial {trial}: click the VISUAL CENTRE "
        "of the laser blob. Then press D to measure its diameter, or Enter to skip."
    )
    while True:
        frame = get_frame(cap)
        prompt = "Click blob centre | D=measure diameter | Enter=accept centre"
        if state.get("measure_diameter", False):
            prompt = f"Click opposite blob edges ({len(state['diameter_clicks'])}/2)"
        state["record_message"] = prompt
        frame = _draw_target_overlay(frame, prompt)
        cv2.imshow("Laser Repeatability Test", frame)
        key = cv2.waitKey(10) & 0xFF
        if key in (ord("q"), 27):
            state["mode"] = "idle"
            raise KeyboardInterrupt("Test aborted by user")
        if state["spot_click"] is not None and key == ord("d"):
            state["measure_diameter"] = True
            state["diameter_clicks"] = []
            print("[MEASURE] Click one edge of the blob, then the opposite edge.")
        if state.get("measure_diameter", False) and len(state["diameter_clicks"]) == 2:
            break
        if state["spot_click"] is not None and key in (10, 13, 32):
            break
    state["mode"] = "idle"
    cx, cy = state["spot_click"]
    diameter = None
    if len(state["diameter_clicks"]) == 2:
        (x1, y1), (x2, y2) = state["diameter_clicks"]
        diameter = math.hypot(x2 - x1, y2 - y1)
    state["test_status"] = f"RECORDED POINT {point_id} | TRIAL {trial}"
    state["record_message"] = ""
    return cx, cy, diameter


def move_to_pixel(position_map, x, y, label):
    try:
        servo_pos, actuator_pos, map_status = position_map.pixel_to_position(x, y)
    except PositionMapError as exc:
        raise RuntimeError(f"Target ({x:.1f},{y:.1f}) cannot be mapped: {exc}") from exc
    if map_status == "EXTRAPOLATED":
        raise RuntimeError(
            f"Target ({x:.1f},{y:.1f}) is outside the measured calibration region. "
            f"Choose a point inside the calibrated sample hull. Loaded map: {POSITION_MAP_FILE}."
        )

    servo_pos = int(round(hw.clamp(servo_pos, hw.ST_VISIBLE_MIN_POSITION,
                                   hw.ST_VISIBLE_MAX_POSITION)))
    actuator_pos = float(hw.clamp(actuator_pos, hw.ACTUATOR_MIN_POSITION,
                                  hw.ACTUATOR_MAX_POSITION))

    start = time.monotonic()
    hw.set_servo_position(servo_pos, speed=hw.ST_SPEED_MAX, acc=hw.ST_ACCEL)
    ok = hw.move_actuator_to_position_and_wait(
        actuator_pos, timeout=MOVE_TIMEOUT_SECONDS, label=label
    )
    if not ok:
        raise RuntimeError(
            f"Actuator failed to reach target for {label}; stopping test. "
            "Do not continue until the actuator state is checked."
        )
    time.sleep(SETTLE_SECONDS)
    return servo_pos, actuator_pos, map_status, time.monotonic() - start


def load_pixel_to_mm_homography():
    """Load a planar pixel-to-mm mapping from measured calibration correspondences."""
    if not os.path.exists(HOMOGRAPHY_FILE):
        print(f"[METRIC] {HOMOGRAPHY_FILE} not found; physical mm metrics will be blank.")
        print("[METRIC] Do not convert pixels with one constant scale: perspective varies across the image.")
        return None
    with open(HOMOGRAPHY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    image_points = data.get("image_points_px", [])
    plane_points = data.get("plane_points_mm", [])
    if len(image_points) < 4 or len(image_points) != len(plane_points):
        raise ValueError(f"{HOMOGRAPHY_FILE} must contain at least 4 matching image_points_px and plane_points_mm.")
    src = __import__("numpy").array(image_points, dtype="float32")
    dst = __import__("numpy").array(plane_points, dtype="float32")
    H, mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("Could not calculate pixel-to-mm homography.")
    print(f"[METRIC] Loaded planar pixel-to-mm calibration ({len(image_points)} points).")
    return H


def pixel_to_mm(H, x, y):
    if H is None:
        return None
    point = __import__("numpy").array([[[float(x), float(y)]]], dtype="float32")
    mapped = cv2.perspectiveTransform(point, H)[0, 0]
    return float(mapped[0]), float(mapped[1])


def write_results(rows, path):
    columns = [
        "timestamp", "point_id", "trial", "target_x_px", "target_y_px",
        "spot_x_px", "spot_y_px", "error_x_px", "error_y_px", "radial_error_px",
        "target_x_mm", "target_y_mm", "spot_x_mm", "spot_y_mm",
        "error_x_mm", "error_y_mm", "radial_error_mm", "blob_diameter_px",
        "target_to_blob_edge_miss_px", "servo_command", "actuator_command_steps",
        "map_status", "move_time_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, H):
    if not rows:
        print("[SUMMARY] No completed measurements.")
        return
    errors_px = [float(r["radial_error_px"]) for r in rows]
    errors_sorted = sorted(errors_px)
    p95_px = errors_sorted[max(0, math.ceil(0.95 * len(errors_sorted)) - 1)]
    print("\n========== REPEATABILITY RESULTS ==========")
    print(f"Completed measurements : {len(errors_px)}")
    print(f"Mean radial error       : {sum(errors_px) / len(errors_px):.2f} px (diagnostic only)")
    print(f"Median radial error     : {errors_sorted[len(errors_sorted) // 2]:.2f} px (diagnostic only)")
    print(f"95th-percentile error   : {p95_px:.2f} px (diagnostic only)")
    if H is not None:
        errors_mm = [float(r["radial_error_mm"]) for r in rows]
        ordered_mm = sorted(errors_mm)
        p95_mm = ordered_mm[max(0, math.ceil(0.95 * len(ordered_mm)) - 1)]
        print(f"Mean radial error       : {sum(errors_mm) / len(errors_mm):.3f} mm")
        print(f"Median radial error     : {ordered_mm[len(ordered_mm) // 2]:.3f} mm")
        print(f"RMS radial error        : {math.sqrt(sum(e*e for e in errors_mm) / len(errors_mm)):.3f} mm")
        print(f"95th-percentile error   : {p95_mm:.3f} mm")
        hit_rate = 100.0 * sum(e <= HIT_RADIUS_MM for e in errors_mm) / len(errors_mm)
        print(f"Hit rate within {HIT_RADIUS_MM:.2f} mm : {hit_rate:.1f}%")
    else:
        print("Physical accuracy and hit rate: NOT REPORTED (no pixel-to-mm calibration).")

    print("\nPer-point repeatability (scatter around each point's own mean):")
    for pid in sorted({int(r["point_id"]) for r in rows}):
        group = [r for r in rows if int(r["point_id"]) == pid]
        if H is not None:
            pts = [pixel_to_mm(H, float(r["spot_x_px"]), float(r["spot_y_px"])) for r in group]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            scatter = [math.hypot(x-mx, y-my) for x, y in zip(xs, ys)]
            rms = math.sqrt(sum(d*d for d in scatter) / len(scatter))
            print(f"  Point {pid:>2}: n={len(group):>2}, mean spot=({mx:.2f},{my:.2f}) mm, RMS scatter={rms:.3f} mm, max={max(scatter):.3f} mm")
        else:
            xs = [float(r["spot_x_px"]) for r in group]
            ys = [float(r["spot_y_px"]) for r in group]
            mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
            scatter = [math.hypot(x-mx, y-my) for x, y in zip(xs, ys)]
            rms = math.sqrt(sum(d*d for d in scatter) / len(scatter))
            print(f"  Point {pid:>2}: n={len(group):>2}, mean spot=({mx:.1f},{my:.1f}) px, RMS scatter={rms:.2f} px (diagnostic only)")
    print("==========================================\n")


def return_to_middle_safely(reason="cleanup"):
    """Best-effort return while actuator control and servo communication remain active."""
    try:
        if not hw.actuator_is_referenced():
            print(f"[RETURN] Cannot return to middle during {reason}: actuator is not referenced.")
            return False
        servo_mid = hw.servo_middle_position()
        actuator_mid = hw.actuator_middle()
        print(f"[RETURN] Moving servo and actuator to saved middle ({actuator_mid:.0f} steps): {reason}")
        hw.set_servo_position(servo_mid, speed=hw.ST_SPEED_NEAR, acc=hw.ST_ACCEL)
        ok = hw.move_actuator_to_position_and_wait(
            actuator_mid, timeout=MOVE_TIMEOUT_SECONDS, label=f"return to middle ({reason})"
        )
        if not ok:
            print("[RETURN] FAILED: actuator did not confirm reaching saved middle.")
            return False
        time.sleep(0.3)
        print("[RETURN] Middle command completed. Open-loop steps cannot independently verify physical position.")
        return True
    except Exception as exc:
        print(f"[RETURN] FAILED during {reason}: {exc}")
        return False

def geometric_hull_contains(position_map, x, y):
    """Independently test the convex hull of the saved pixel calibration samples."""
    samples = getattr(position_map, "samples", [])
    if len(samples) < 3:
        return False
    points = np.asarray([[float(s["x"]), float(s["y"])] for s in samples], dtype=np.float32)
    hull = cv2.convexHull(points)
    return cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0


def preflight_target_mappings(position_map, targets):
    """Print both polygon-hull membership and PositionMap's own interpolation status."""
    valid, invalid = [], []
    print("\n[PREFLIGHT] Checking saved target points before any motors move.")
    print(f"[PREFLIGHT] Map type={position_map.map_type}; samples={len(position_map.samples)}; map file={POSITION_MAP_FILE}")
    for target in targets:
        pid = int(target.get("id", len(valid) + len(invalid) + 1))
        x, y = float(target["x"]), float(target["y"])
        geom_inside = geometric_hull_contains(position_map, x, y)
        try:
            servo_pos, actuator_pos, map_status = position_map.pixel_to_position(x, y)
            map_ok = map_status in ("INTERPOLATED", "GRID")
            details = f"status={map_status}, servo={servo_pos:.1f}, actuator={actuator_pos:.1f}"
        except PositionMapError as exc:
            map_ok = False
            map_status = "ERROR"
            details = f"mapping error={exc}"
        geom_text = "IN_HULL" if geom_inside else "OUTSIDE_HULL"
        line = f"[PREFLIGHT] Point {pid}: pixel=({x:.1f},{y:.1f}), geometry={geom_text}, {details}"
        if geom_inside and map_status == "EXTRAPOLATED":
            line += "  <-- MISMATCH: pixel is inside the sample convex hull, but PositionMap reports EXTRAPOLATED"
        print(line)
        if geom_inside and map_ok:
            valid.append(target)
        else:
            invalid.append(target)
    print(f"[PREFLIGHT] {len(valid)} valid, {len(invalid)} invalid.")
    return valid, invalid


def run_test(cap, position_map, targets, repetitions, H):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"repeatability_{stamp}.csv")
    video_path = os.path.join(RESULTS_DIR, f"repeatability_video_{stamp}.mp4")
    rows = []
    recorder = None

    if not hw.actuator_is_referenced():
        raise RuntimeError("Actuator is not referenced. Establish its physical reference first.")
    print(f"[TEST] {len(targets)} points x {repetitions} trials")
    print("[TEST] Moving directly between targets; no midpoint return between targets or repetitions.")
    print("[TEST] Recording will include movement, settling, measurement, and return to middle.")
    print("[TEST] All saved target points will be shown throughout the live view and video.")

    try:
        state["test_status"] = f"TEST STARTING | {len(targets)} points x {repetitions} trials"
        state["record_message"] = "Full-test recording started"
        recorder = CameraRecorder(cap, video_path)
        state["recorder"] = recorder
        recorder.start()

        for trial in range(1, repetitions + 1):
            for target in targets:
                pid = int(target["id"])
                state["active_test_target"] = {
                    "id": pid, "x": float(target["x"]), "y": float(target["y"]), "trial": trial
                }
                state["spot_click"] = None
                state["diameter_clicks"] = []
                state["test_status"] = f"MOVING TO POINT {pid} | TRIAL {trial}/{repetitions}"
                try:
                    servo_cmd, act_cmd, map_status, move_time = move_to_pixel(
                        position_map, target["x"], target["y"],
                        label=f"point {pid}, trial {trial}"
                    )
                except RuntimeError as exc:
                    print(f"[SKIP] Point {pid}, trial {trial}: {exc}")
                    state["test_status"] = f"SKIPPED POINT {pid} | TRIAL {trial}: mapping/motion error"
                    continue
                try:
                    spot_x, spot_y, blob_diameter = wait_for_spot_click(cap, target, pid, trial)
                except KeyboardInterrupt:
                    print("[TEST] Aborted by user. Saving completed measurements.")
                    raise

                dx = float(spot_x) - float(target["x"])
                dy = float(spot_y) - float(target["y"])
                radius_px = math.hypot(dx, dy)
                target_mm = pixel_to_mm(H, target["x"], target["y"])
                spot_mm = pixel_to_mm(H, spot_x, spot_y)
                if target_mm is not None and spot_mm is not None:
                    ex_mm, ey_mm = spot_mm[0] - target_mm[0], spot_mm[1] - target_mm[1]
                    radius_mm = math.hypot(ex_mm, ey_mm)
                else:
                    ex_mm = ey_mm = radius_mm = None
                rows.append({
                    "timestamp": dt.datetime.now().isoformat(timespec="milliseconds"),
                    "point_id": pid, "trial": trial,
                    "target_x_px": f"{float(target['x']):.2f}", "target_y_px": f"{float(target['y']):.2f}",
                    "spot_x_px": f"{spot_x:.2f}", "spot_y_px": f"{spot_y:.2f}",
                    "error_x_px": f"{dx:.2f}", "error_y_px": f"{dy:.2f}",
                    "radial_error_px": f"{radius_px:.2f}",
                    "target_x_mm": f"{target_mm[0]:.4f}" if target_mm else "",
                    "target_y_mm": f"{target_mm[1]:.4f}" if target_mm else "",
                    "spot_x_mm": f"{spot_mm[0]:.4f}" if spot_mm else "",
                    "spot_y_mm": f"{spot_mm[1]:.4f}" if spot_mm else "",
                    "error_x_mm": f"{ex_mm:.4f}" if ex_mm is not None else "",
                    "error_y_mm": f"{ey_mm:.4f}" if ey_mm is not None else "",
                    "radial_error_mm": f"{radius_mm:.4f}" if radius_mm is not None else "",
                    "blob_diameter_px": f"{blob_diameter:.2f}" if blob_diameter is not None else "",
                    "target_to_blob_edge_miss_px": f"{max(0.0, radius_px - blob_diameter / 2.0):.2f}" if blob_diameter is not None else "",
                    "servo_command": servo_cmd, "actuator_command_steps": f"{act_cmd:.1f}",
                    "map_status": map_status, "move_time_s": f"{move_time:.3f}",
                })
                write_results(rows, csv_path)
                if radius_mm is None:
                    print(f"[RESULT] point={pid} trial={trial} radial error={radius_px:.2f} px (mm calibration unavailable)")
                else:
                    print(f"[RESULT] point={pid} trial={trial} radial error={radius_mm:.3f} mm ({radius_px:.2f} px)")
                state["active_test_target"] = None
                state["test_status"] = f"COMPLETED POINT {pid} | TRIAL {trial}/{repetitions}"
    except KeyboardInterrupt:
        print("[TEST] User abort received.")
    finally:
        write_results(rows, csv_path)
        summarize(rows, H)
        print(f"[SAVE] Raw measurements: {csv_path}")
        state["test_status"] = "RETURNING TO SAVED MIDDLE"
        state["active_test_target"] = None
        state["record_message"] = "Returning to saved middle; recording continues"
        return_to_middle_safely("test completion/abort")
        state["test_status"] = "TEST COMPLETE / ABORTED - RECORDING FINISHED"
        state["record_message"] = "Recording ending"
        if recorder is not None:
            recorder.stop()
            state["recorder"] = None
            state["spot_click"] = None
            state["diameter_clicks"] = []
            if os.path.isfile(video_path):
                print(f"[VIDEO] Saved full test video: {video_path}")
                print(f"[VIDEO] File size: {os.path.getsize(video_path)} bytes")
                if recorder.frame_count == 0:
                    print("[VIDEO] WARNING: no frames were written.")



def write_auto_results(rows, path):
    """Write movement-only records from autonomous runs (no manual spot measurements)."""
    columns = [
        "timestamp", "point_id", "trial", "target_x_px", "target_y_px",
        "servo_command", "actuator_command_steps", "estimated_actuator_position_steps",
        "estimated_actuator_speed_steps_s", "map_status", "move_time_s", "result",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_automatic_target_sequence(cap, position_map, targets, repetitions):
    """Visit saved valid points without per-target mouse/keyboard input.

    The only required user interaction is the initial confirmation. This mode
    measures command/move timing, not laser-spot error, because the latter
    currently requires a manual spot-centre click in the existing test.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"automatic_position_run_{stamp}.csv")
    video_path = os.path.join(RESULTS_DIR, f"automatic_position_video_{stamp}.mp4")
    rows = []
    recorder = None
    cancel_event = threading.Event()
    worker_errors = []
    run_counts = {"completed": 0, "attempted": 0}

    if not hw.actuator_is_referenced():
        raise RuntimeError("Actuator is not referenced. Establish its physical reference first.")

    print(f"[AUTO] Starting automatic position run: {len(targets)} points x {repetitions} repetitions.")
    print("[AUTO] No manual spot-centre clicks are requested during this run.")
    print("[AUTO] This logs positioning commands and timing; it does NOT measure physical spot error.")
    print("[AUTO] Press Q in the camera window to request a stop. The script will then try to return to middle.")

    state["active_test_target"] = None
    state["test_status"] = f"AUTO RUN STARTING | {len(targets)} points x {repetitions} trials"
    state["record_message"] = "Automatic mode | no per-target input"
    recorder = CameraRecorder(cap, video_path)
    state["recorder"] = recorder
    recorder.start()

    def motion_worker():
        try:
            for trial in range(1, repetitions + 1):
                for target in targets:
                    if cancel_event.is_set():
                        break
                    pid = int(target.get("id", 0))
                    state["active_test_target"] = {
                        "id": pid, "x": float(target["x"]), "y": float(target["y"]), "trial": trial
                    }
                    state["test_status"] = f"AUTO MOVING TO POINT {pid} | TRIAL {trial}/{repetitions}"
                    state["record_message"] = "Automatic run | movement and settling are recorded"
                    run_counts["attempted"] += 1
                    try:
                        servo_cmd, act_cmd, map_status, move_time = move_to_pixel(
                            position_map, target["x"], target["y"],
                            label=f"AUTO point {pid}, trial {trial}"
                        )
                    except Exception as exc:
                        message = f"Automatic run stopped at point {pid}, trial {trial}: {type(exc).__name__}: {exc}"
                        worker_errors.append(message)
                        state["test_status"] = f"AUTO STOPPED AT POINT {pid}"
                        state["record_message"] = "Movement failed; returning to saved middle"
                        print(f"[AUTO][ERROR] {message}")
                        cancel_event.set()
                        break

                    try:
                        estimated_pos = float(hw.get_actuator_position())
                    except Exception:
                        estimated_pos = ""
                    try:
                        estimated_speed = abs(float(hw.get_actuator_speed()))
                    except Exception:
                        estimated_speed = ""
                    row = {
                        "timestamp": dt.datetime.now().isoformat(timespec="milliseconds"),
                        "point_id": pid,
                        "trial": trial,
                        "target_x_px": f"{float(target['x']):.2f}",
                        "target_y_px": f"{float(target['y']):.2f}",
                        "servo_command": servo_cmd,
                        "actuator_command_steps": f"{act_cmd:.1f}",
                        "estimated_actuator_position_steps": f"{estimated_pos:.1f}" if isinstance(estimated_pos, float) else "",
                        "estimated_actuator_speed_steps_s": f"{estimated_speed:.1f}" if isinstance(estimated_speed, float) else "",
                        "map_status": map_status,
                        "move_time_s": f"{move_time:.3f}",
                        "result": "reached_commanded_position_open_loop_estimate",
                    }
                    rows.append(row)
                    write_auto_results(rows, csv_path)
                    run_counts["completed"] += 1
                    print(f"[AUTO][REACHED] point={pid}, trial={trial}/{repetitions}, "
                          f"pixel=({target['x']:.1f},{target['y']:.1f}), servo={servo_cmd}, "
                          f"actuator_target={act_cmd:.1f}, move={move_time:.2f}s")
                    state["test_status"] = f"AUTO REACHED POINT {pid} | TRIAL {trial}/{repetitions}"
                    state["record_message"] = "Moving automatically to next target"
                if cancel_event.is_set():
                    break
        except Exception as exc:
            worker_errors.append(f"Unexpected automatic-run error: {type(exc).__name__}: {exc}")
            print(f"[AUTO][ERROR] {worker_errors[-1]}")
        finally:
            state["active_test_target"] = None
            state["test_status"] = "AUTO RUN: RETURNING TO SAVED MIDDLE"
            state["record_message"] = "Return-to-middle motion is being recorded"
            return_to_middle_safely("automatic run completion/abort")
            state["test_status"] = "AUTO RUN ABORTED" if cancel_event.is_set() or worker_errors else "AUTO RUN COMPLETE"
            state["record_message"] = f"Completed {run_counts['completed']}/{run_counts['attempted']} attempted moves"

    worker = threading.Thread(target=motion_worker, name="automatic-position-run", daemon=False)
    try:
        worker.start()
        window = "Laser Repeatability Test"
        while worker.is_alive():
            frame = recorder.get_latest_frame()
            preview = _draw_target_overlay(frame)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) and not cancel_event.is_set():
                print("[AUTO] Stop requested from camera window. Motion will stop and the script will attempt to return to middle.")
                cancel_event.set()
                try:
                    # Request an immediate stop; the worker handles the normal return-to-middle sequence.
                    hw.set_actuator_speed(0.0)
                except Exception as exc:
                    print(f"[AUTO][STOP] Stop request warning: {exc}")
            time.sleep(0.002)
        worker.join()
    except KeyboardInterrupt:
        print("[AUTO] Ctrl+C received. Requesting stop and safe return.")
        cancel_event.set()
        try:
            hw.set_actuator_speed(0.0)
        except Exception:
            pass
        if worker.is_alive():
            worker.join(timeout=MOVE_TIMEOUT_SECONDS + 5.0)
    finally:
        if worker.is_alive():
            print("[AUTO][WARNING] Motion worker has not exited; do not disconnect motor power while it is moving.")
        if recorder is not None:
            state["test_status"] = "AUTO RUN FINISHED - VIDEO CLOSING"
            state["record_message"] = "Automatic run recording complete"
            recorder.stop()
            state["recorder"] = None
        state["active_test_target"] = None
        print(f"[AUTO][SUMMARY] Reached {run_counts['completed']} / {run_counts['attempted']} attempted moves.")
        print(f"[AUTO][SAVE] Movement log: {csv_path}")
        if os.path.isfile(video_path):
            print(f"[AUTO][VIDEO] Full-run video: {video_path} ({os.path.getsize(video_path)} bytes)")
        for message in worker_errors:
            print(f"[AUTO][NOTE] {message}")

def main():
    print("=== Laser repeatability test ===")
    print("Use only the calibrated stationary setup. Keep hands clear of the actuator.")
    # Force the hardware controller and this test to use the same files next to this script,
    # regardless of the terminal's current working directory.
    hw.CALIBRATION_FILE = CALIBRATION_FILE
    hw.POSITION_MAP_FILE = POSITION_MAP_FILE
    print(f"[FILES] Script directory: {SCRIPT_DIR}")
    print(f"[FILES] Actuator calibration: {CALIBRATION_FILE}")
    print(f"[FILES] Position map: {POSITION_MAP_FILE}")
    print(f"[FILES] Saved targets: {TARGETS_FILE}")
    if not hw.load_calibration():
        raise RuntimeError(f"Could not load actuator travel from {CALIBRATION_FILE}")
    hw.setup_servo()
    position_map = PositionMap.load(POSITION_MAP_FILE)
    H = load_pixel_to_mm_homography()
    if (position_map.camera_width, position_map.camera_height) != (FRAME_WIDTH, FRAME_HEIGHT):
        print(
            f"[WARNING] Position map resolution is {position_map.camera_width}x"
            f"{position_map.camera_height}; test camera is {FRAME_WIDTH}x{FRAME_HEIGHT}."
        )

    middle = hw.actuator_middle()
    answer = input(
        f"Is the actuator PHYSICALLY at its saved middle ({middle:.0f} steps)? [y/N]: "
    ).strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError(
            "Stopped: establish the actuator reference with your existing safe calibration procedure, "
            "then rerun. Open-loop step counts cannot determine physical position after lost steps."
        )
    hw.accept_saved_middle()

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {CAMERA_DEVICE}")
    # Keep the camera setup consistent with the working project configuration:
    # request resolution only; do not force FPS or FOURCC.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"[CAMERA] Actual capture={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, reported FPS={reported_fps:.2f}")

    cv2.namedWindow("Laser Repeatability Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Laser Repeatability Test", FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback("Laser Repeatability Test", mouse_callback)

    actuator_thread = threading.Thread(target=hw.actuator_thread, daemon=True)
    actuator_thread.start()
    print("\nControls:")
    print("  R : toggle add-target mode; left-click to add points")
    print("  D : toggle delete-target mode; left-click near a point to delete it (saved immediately)")
    print("  X : clear ALL saved and visible target points (requires typing DELETE)")
    print("  S : save target points to repeatability_targets.json")
    print("  L : load saved target points")
    print("  T : manual repeatability test; click each observed laser spot")
    print("  A : automatic position run through saved points; no per-target input")
    print("  Q : quit (attempts a safe return to saved middle)")
    print("During the test, a full MP4 is recorded continuously through the return-to-middle movement.")
    print("All saved target points remain annotated; the current target is highlighted green.")
    print("During a measurement: click blob centre, press D and click opposite edges, or press Enter to skip diameter.")

    targets = load_targets()
    state["target_points"] = [{"x": p["x"], "y": p["y"]} for p in targets]
    try:
        while True:
            key = show_live(cap, f"Mode: {state['mode']} | targets: {len(state['target_points'])} | R add D delete X clear S save L load T manual A auto Q quit")
            if key == ord("r"):
                state["mode"] = "idle" if state["mode"] == "add_target" else "add_target"
                print(f"[UI] Target recording mode: {state['mode']}")
            elif key == ord("d"):
                state["mode"] = "idle" if state["mode"] == "delete_target" else "delete_target"
                print(f"[UI] Target deletion mode: {'ON' if state['mode'] == 'delete_target' else 'OFF'}. Click near a target to delete it.")
            elif key == ord("x"):
                if not state["target_points"]:
                    print("[TARGET] No points to clear.")
                    continue
                confirm_delete = input(f"Type DELETE to clear all {len(state['target_points'])} visible/saved target points: ").strip()
                if confirm_delete == "DELETE":
                    state["target_points"].clear()
                    try:
                        if os.path.exists(TARGETS_FILE):
                            os.remove(TARGETS_FILE)
                    except OSError as exc:
                        print(f"[TARGET] WARNING: could not remove {TARGETS_FILE}: {exc}")
                    print("[TARGET] All target points cleared.")
                else:
                    print("[TARGET] Clear cancelled.")
            elif key == ord("s"):
                if not state["target_points"]:
                    print("[TARGET] No points to save.")
                else:
                    save_targets(state["target_points"])
            elif key == ord("l"):
                loaded = load_targets()
                state["target_points"] = [{"x": p["x"], "y": p["y"]} for p in loaded]
                print(f"[TARGET] Loaded {len(loaded)} points from {TARGETS_FILE}")
            elif key == ord("t"):
                loaded = load_targets()
                if not loaded:
                    print("[TEST] No saved targets. Press R, click points, then S.")
                    continue
                valid_targets, invalid_targets = preflight_target_mappings(position_map, loaded)
                if not valid_targets:
                    print("[TEST] No target points can be mapped inside the measured hull. No motor movement started.")
                    print("[TEST] Use D to delete invalid points, or run position_map_hull_inspector.py to view the hull and mismatch.")
                    continue
                if invalid_targets:
                    ids = [int(p.get("id", i + 1)) for i, p in enumerate(invalid_targets)]
                    answer = input(
                        f"{len(invalid_targets)} point(s) failed preflight (IDs {ids}). "
                        f"Type VALID to test ONLY the {len(valid_targets)} valid point(s), or press Enter to cancel: "
                    ).strip()
                    if answer != "VALID":
                        print("[TEST] Cancelled before any movement. Use D to delete invalid saved points or inspect the hull.")
                        continue
                    loaded = valid_targets
                reps = input("Number of repetitions per point [10]: ").strip()
                repetitions = int(reps) if reps else 10
                if repetitions < 1:
                    print("[TEST] Repetitions must be >= 1.")
                    continue
                confirm = input(
                    f"Run {len(loaded)} points x {repetitions} trials, moving directly between targets? Check clear travel and laser safety. Type RUN: "
                ).strip()
                if confirm != "RUN":
                    print("[TEST] Cancelled.")
                    continue
                run_test(cap, position_map, loaded, repetitions, H)
            elif key == ord("a"):
                loaded = load_targets()
                if not loaded:
                    print("[AUTO] No saved targets. Press R, click points, then S.")
                    continue
                valid_targets, invalid_targets = preflight_target_mappings(position_map, loaded)
                if not valid_targets:
                    print("[AUTO] No saved target is inside the measured hull. No movement started.")
                    continue
                if invalid_targets:
                    ids = [int(p.get("id", i + 1)) for i, p in enumerate(invalid_targets)]
                    answer = input(
                        f"{len(invalid_targets)} point(s) failed preflight (IDs {ids}). "
                        f"Type VALID to automatically visit ONLY the {len(valid_targets)} valid point(s), or press Enter to cancel: "
                    ).strip()
                    if answer != "VALID":
                        print("[AUTO] Cancelled before any movement.")
                        continue
                    loaded = valid_targets
                reps = input("Automatic-run repetitions per point [5]: ").strip()
                repetitions = int(reps) if reps else 5
                if repetitions < 1:
                    print("[AUTO] Repetitions must be >= 1.")
                    continue
                confirm = input(
                    f"Automatically visit {len(loaded)} points x {repetitions} repetitions with no per-target input? "
                    f"Keep the travel path clear. Type RUN: "
                ).strip()
                if confirm != "RUN":
                    print("[AUTO] Cancelled.")
                    continue
                run_automatic_target_sequence(cap, position_map, loaded, repetitions)
            elif key in (ord("q"), 27):
                break
    finally:
        state["mode"] = "idle"
        if state.get("recorder") is not None:
            try:
                state["recorder"].stop()
            except Exception as exc:
                print(f"[VIDEO] Cleanup warning: {exc}")
            state["recorder"] = None
        # A quit from the main UI also attempts to return to the saved middle before disabling drivers.
        try:
            if hw.actuator_is_referenced():
                return_to_middle_safely("program exit")
        except Exception as exc:
            print(f"[RETURN] Skipped on program exit: {exc}")
        try:
            hw.set_actuator_speed(0.0)
            hw.wait_for_actuator_stop(2.0)
        except Exception as exc:
            print(f"[CLEANUP] Stop request: {exc}")
        hw._actuator_running = False
        actuator_thread.join(timeout=2.0)
        try:
            hw.set_actuator_enabled(False)
        except Exception as exc:
            print(f"[CLEANUP] Disable actuator: {exc}")
        try:
            hw.GPIO.cleanup()
        except Exception as exc:
            print(f"[CLEANUP] GPIO cleanup: {exc}")
        try:
            hw.servo.portHandler.closePort()
        except Exception as exc:
            print(f"[CLEANUP] Servo port close: {exc}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
