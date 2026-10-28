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

import laser_servo_tracker_yolo_stable_auto_visit as hw
from position_map import PositionMap, PositionMapError

CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 60
TARGETS_FILE = "repeatability_targets.json"
RESULTS_DIR = "repeatability_results"
SETTLE_SECONDS = 1.0
MOVE_TIMEOUT_SECONDS = 20.0
HIT_RADIUS_MM = 2.0  # Set this from the application's required physical targeting tolerance.
HOMOGRAPHY_FILE = "pixel_to_mm_homography.json"
# The homography maps undistorted image pixels to X/Y millimetres on the fixed target plane.
# It accounts for perspective variation across the plane; lens distortion should be corrected first.

state = {
    "mode": "idle",  # idle, add_target, mark_spot
    "target_points": [],
    "spot_click": None,
    "diameter_clicks": [],
    "last_frame": None,
}


def mouse_callback(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state["mode"] == "add_target":
        state["target_points"].append({"x": int(x), "y": int(y)})
        print(f"[TARGET] Added point {len(state['target_points'])}: ({x}, {y})")
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


def get_frame(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera frame read failed")
    return frame


def show_live(cap, message=""):
    frame = get_frame(cap)
    for i, p in enumerate(state["target_points"], start=1):
        cv2.drawMarker(frame, (int(p["x"]), int(p["y"])), (0, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)
        cv2.putText(frame, str(i), (int(p["x"]) + 8, int(p["y"]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if message:
        cv2.putText(frame, message[:120], (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Laser Repeatability Test", frame)
    state["last_frame"] = frame
    return cv2.waitKey(1) & 0xFF


def wait_for_spot_click(cap, target, point_id, trial):
    """Click blob centre; optionally press D and click opposite blob edges."""
    state["spot_click"] = None
    state["diameter_clicks"] = []
    state["measure_diameter"] = False
    state["mode"] = "mark_spot"
    print(
        f"[MEASURE] Point {point_id}, trial {trial}: click the VISUAL CENTRE "
        "of the laser blob. Then press D to measure its diameter, or Enter to skip."
    )
    while True:
        frame = get_frame(cap)
        cv2.drawMarker(frame, (int(target["x"]), int(target["y"])),
                       (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
        if state["spot_click"] is not None:
            cv2.drawMarker(frame, state["spot_click"], (255, 255, 0),
                           cv2.MARKER_CROSS, 16, 2)
        for edge in state["diameter_clicks"]:
            cv2.circle(frame, edge, 5, (255, 0, 255), -1)
        prompt = "Click blob centre | D=measure diameter | Enter=accept centre"
        if state.get("measure_diameter", False):
            prompt = f"Click opposite blob edges ({len(state['diameter_clicks'])}/2)"
        cv2.putText(frame, prompt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)
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
    return cx, cy, diameter


def move_to_pixel(position_map, x, y, label):
    try:
        servo_pos, actuator_pos, map_status = position_map.pixel_to_position(x, y)
    except PositionMapError as exc:
        raise RuntimeError(f"Target ({x:.1f},{y:.1f}) cannot be mapped: {exc}") from exc
    if map_status == "EXTRAPOLATED":
        raise RuntimeError(
            f"Target ({x:.1f},{y:.1f}) is outside the measured calibration region. "
            "Choose a point inside the calibrated sample hull for a valid repeatability test."
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

def run_test(cap, position_map, targets, repetitions, H):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"repeatability_{stamp}.csv")
    rows = []

    if not hw.actuator_is_referenced():
        raise RuntimeError("Actuator is not referenced. Establish its physical reference first.")
    print(f"[TEST] {len(targets)} points x {repetitions} trials")
    print("[TEST] Moving directly between targets; no midpoint return between targets or repetitions.")
    print("[TEST] Laser must be visible on a matte, non-reflective target plane.")
    try:
        for trial in range(1, repetitions + 1):
            for target in targets:
                pid = int(target["id"])
                try:
                    servo_cmd, act_cmd, map_status, move_time = move_to_pixel(
                        position_map, target["x"], target["y"],
                        label=f"point {pid}, trial {trial}"
                    )
                except RuntimeError as exc:
                    print(f"[SKIP] Point {pid}, trial {trial}: {exc}")
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
    except KeyboardInterrupt:
        print("[TEST] User abort received.")
    finally:
        write_results(rows, csv_path)
        summarize(rows, H)
        print(f"[SAVE] Raw measurements: {csv_path}")
        return_to_middle_safely("test completion/abort")

def main():
    print("=== Laser repeatability test ===")
    print("Use only the calibrated stationary setup. Keep hands clear of the actuator.")
    if not hw.load_calibration():
        raise RuntimeError("Could not load actuator travel from servo_calibration.json")
    hw.setup_servo()
    position_map = PositionMap.load("position_map.json")
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cv2.namedWindow("Laser Repeatability Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Laser Repeatability Test", FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback("Laser Repeatability Test", mouse_callback)

    actuator_thread = threading.Thread(target=hw.actuator_thread, daemon=True)
    actuator_thread.start()
    print("\nControls:")
    print("  R : start/stop adding target points (left-click points in the camera image)")
    print("  S : save clicked target points to repeatability_targets.json")
    print("  L : load saved target points")
    print("  T : run the test on saved points")
    print("  Q : quit (attempts a safe return to saved middle)")
    print("During a measurement: click blob centre, press D and click opposite edges, or press Enter to skip diameter.")

    targets = load_targets()
    state["target_points"] = [{"x": p["x"], "y": p["y"]} for p in targets]
    try:
        while True:
            key = show_live(cap, f"Mode: {state['mode']} | targets: {len(state['target_points'])}")
            if key == ord("r"):
                state["mode"] = "idle" if state["mode"] == "add_target" else "add_target"
                print(f"[UI] Target recording mode: {state['mode']}")
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
            elif key in (ord("q"), 27):
                break
    finally:
        state["mode"] = "idle"
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
