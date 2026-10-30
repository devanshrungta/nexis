#!/usr/bin/env python3
"""Visualize the position-map calibration hull and saved repeatability target points.

This diagnostic does NOT command motors. It loads the same position_map.json beside
this script, draws its calibration samples and convex hull over the live camera view,
and compares geometric hull membership with PositionMap.pixel_to_position().
"""
import datetime as dt
import json
import os
from pathlib import Path

import cv2
import numpy as np
from position_map import PositionMap, PositionMapError

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_FILE = SCRIPT_DIR / "position_map.json"
TARGETS_FILE = SCRIPT_DIR / "repeatability_targets.json"
CAMERA_DEVICE = 0
WINDOW = "Position Map Hull Inspector (NO MOTOR CONTROL)"
SNAPSHOT_DIR = SCRIPT_DIR / "position_map_hull_diagnostics"


def load_targets():
    if not TARGETS_FILE.exists():
        print(f"[TARGETS] No saved target file: {TARGETS_FILE}")
        return []
    data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    pts = data.get("points", [])
    result = [{"id": int(p.get("id", i + 1)), "x": float(p["x"]), "y": float(p["y"])}
              for i, p in enumerate(pts)]
    print(f"[TARGETS] Loaded {len(result)} points from {TARGETS_FILE}")
    return result


def make_hull(position_map):
    pts = np.asarray([[float(s["x"]), float(s["y"])] for s in position_map.samples], dtype=np.float32)
    if len(pts) < 3:
        raise RuntimeError("Position map has fewer than 3 samples; cannot draw a hull.")
    return cv2.convexHull(pts), pts


def classify(position_map, hull, x, y):
    geom_inside = cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0
    try:
        servo, actuator, status = position_map.pixel_to_position(float(x), float(y))
        details = f"map={status}, servo={servo:.1f}, actuator={actuator:.1f}"
    except PositionMapError as exc:
        status = "ERROR"
        details = f"map error={exc}"
    geom = "IN_HULL" if geom_inside else "OUTSIDE_HULL"
    mismatch = geom_inside and status == "EXTRAPOLATED"
    return geom_inside, status, f"{geom}; {details}" + ("; MISMATCH: geometric hull says IN but mapper says EXTRAPOLATED" if mismatch else "")


def mouse_callback(event, x, y, flags, userdata):
    userdata["cursor"] = (int(x), int(y))
    if event == cv2.EVENT_LBUTTONDOWN:
        userdata["last_click"] = (int(x), int(y))
        _, _, details = classify(userdata["map"], userdata["hull"], x, y)
        print(f"[CLICK] pixel=({x},{y}): {details}")


def main():
    print("=== POSITION MAP HULL INSPECTOR — NO MOTOR CONTROL ===")
    print(f"[FILES] Map: {MAP_FILE}")
    print(f"[FILES] Targets: {TARGETS_FILE}")
    position_map = PositionMap.load(MAP_FILE)
    hull, pts = make_hull(position_map)
    targets = load_targets()
    print(f"[MAP] type={position_map.map_type}, samples={len(position_map.samples)}, "
          f"resolution={position_map.camera_width}x{position_map.camera_height}")
    print("[MAP] Hull vertices (camera pixels):")
    for xy in hull.reshape(-1, 2):
        print(f"  ({xy[0]:.1f}, {xy[1]:.1f})")
    for target in targets:
        _, _, details = classify(position_map, hull, target["x"], target["y"])
        print(f"[TARGET {target['id']}] ({target['x']:.1f},{target['y']:.1f}): {details}")

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {CAMERA_DEVICE}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, position_map.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, position_map.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(chr((actual_fourcc_i >> (8 * i)) & 0xFF) for i in range(4))
    print(f"[CAMERA] requested={position_map.camera_width}x{position_map.camera_height} @60 MJPG; "
          f"actual={actual_w}x{actual_h} FPS={actual_fps:.1f} FOURCC={actual_fourcc!r}")
    if (actual_w, actual_h) != (position_map.camera_width, position_map.camera_height):
        print("[WARNING] Camera size differs from calibration-map size. Display will be resized; pixel comparisons may be invalid.")

    state = {"cursor": (0, 0), "last_click": None, "map": position_map, "hull": hull, "targets": targets}
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, position_map.camera_width, position_map.camera_height)
    cv2.setMouseCallback(WINDOW, mouse_callback, state)
    print("[UI] Samples=white labels; hull=yellow; saved targets=green if valid/red if outside; click anywhere to inspect; S=screenshot; L=reload targets; Q=quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[CAMERA] Frame read failed; retrying.")
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            continue
        if frame.shape[1] != position_map.camera_width or frame.shape[0] != position_map.camera_height:
            frame = cv2.resize(frame, (position_map.camera_width, position_map.camera_height))
        overlay = frame.copy()
        # cv2.convexHull() returns float32 points because the source samples are
        # floating-point. OpenCV drawing functions such as fillConvexPoly require
        # integer pixel coordinates (CV_32S), so convert only for drawing. Keep the
        # floating-point hull for pointPolygonTest/classification.
        hull_draw = np.rint(hull).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(overlay, hull_draw, (20, 130, 20))
        frame = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0.0)
        cv2.polylines(frame, [hull_draw], True, (0, 255, 255), 3, cv2.LINE_AA)
        for i, (sx, sy) in enumerate(pts, start=1):
            cv2.drawMarker(frame, (int(round(sx)), int(round(sy))), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(frame, f"S{i}", (int(sx) + 5, int(sy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        for t in targets:
            x, y = int(round(t["x"])), int(round(t["y"]))
            inside, status, _ = classify(position_map, hull, t["x"], t["y"])
            # Magenta denotes a disagreement between geometric hull and mapper status.
            col = (255, 0, 255) if inside and status == "EXTRAPOLATED" else ((0, 255, 0) if inside and status in ("INTERPOLATED", "GRID") else (0, 0, 255))
            cv2.circle(frame, (x, y), 9, col, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (x, y), col, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.putText(frame, f"T{t['id']} {status}", (x + 9, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2, cv2.LINE_AA)
        x, y = state["cursor"]
        inside, status, details = classify(position_map, hull, x, y)
        cursor_color = (0, 255, 0) if inside and status in ("INTERPOLATED", "GRID") else ((255, 0, 255) if inside and status == "EXTRAPOLATED" else (0, 0, 255))
        cv2.drawMarker(frame, (x, y), cursor_color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(frame, f"PIXEL ({x},{y}) | GEOM={'IN' if inside else 'OUT'} | MAP={status}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, cursor_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW outline = convex hull of saved calibration samples; MAGENTA = hull/map mismatch",
                    (12, position_map.camera_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SNAPSHOT_DIR / f"position_map_hull_{stamp}.png"
            if cv2.imwrite(str(path), frame):
                print(f"[SAVE] Annotated hull screenshot: {path}")
            else:
                print(f"[SAVE] Failed to write screenshot: {path}")
        elif key == ord("l"):
            targets = load_targets()
            state["targets"] = targets

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
