#!/usr/bin/env python3
"""Live HSV green-weed detection tuner. Camera/detection only; no motor or laser imports."""
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW = "Green Weed Detection Tuner"
SETTINGS_FILE = Path("green_weed_detection_settings.json")


def nothing(_value):
    pass


def odd_kernel(value):
    if value <= 0:
        return 0
    return value if value % 2 else value + 1


def main():
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {CAMERA_DEVICE}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    # Trackbars directly change segmentation/filter values while the camera runs.
    for name, value, maximum in [
        ("H min", 30, 179), ("H max", 95, 179),
        ("S min", 45, 255), ("S max", 255, 255),
        ("V min", 35, 255), ("V max", 255, 255),
        ("Min area", 350, 30000),
        ("Open kernel (0=off)", 3, 15), ("Close kernel (0=off)", 7, 15),
    ]:
        cv2.createTrackbar(name, WINDOW, value, maximum, nothing)

    print("Controls: adjust sliders live; press S to save settings; Q or ESC to quit.")
    print("Tune while the dummy weed is visible and moving through the full camera view.")
    print("This script does not import or command the servo, actuator, or laser.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed.")
            break

        vals = {name: cv2.getTrackbarPos(name, WINDOW) for name in [
            "H min", "H max", "S min", "S max", "V min", "V max",
            "Min area", "Open kernel (0=off)", "Close kernel (0=off)"]}
        h0, h1 = sorted((vals["H min"], vals["H max"]))
        s0, s1 = sorted((vals["S min"], vals["S max"]))
        v0, v1 = sorted((vals["V min"], vals["V max"]))
        lower = np.array([h0, s0, v0], dtype=np.uint8)
        upper = np.array([h1, s1, v1], dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        k_open = odd_kernel(vals["Open kernel (0=off)"])
        k_close = odd_kernel(vals["Close kernel (0=off)"])
        if k_open:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if k_close:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        annotated = frame.copy()
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < vals["Min area"]:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3:
                continue
            candidates.append((area, x, y, w, h))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(annotated, f"area={area:.0f}", (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

        candidates.sort(key=lambda c: c[0], reverse=True)
        if candidates:
            area, x, y, w, h = candidates[0]
            cx, cy = x + w // 2, y + h // 2
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            status = f"LARGEST: area={area:.0f}, center=({cx},{cy}) | candidates={len(candidates)}"
            color = (0, 220, 0)
        else:
            status = f"NO CANDIDATE | raw green pixels={cv2.countNonZero(mask)}"
            color = (0, 0, 255)

        cv2.putText(annotated, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(annotated, "LEFT: camera + accepted blobs     RIGHT: binary green mask",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Fit both panels to the same size to show them together in one window.
        panel_w = 640
        panel_h = max(1, int(frame.shape[0] * panel_w / frame.shape[1]))
        left = cv2.resize(annotated, (panel_w, panel_h))
        right = cv2.resize(mask_bgr, (panel_w, panel_h))
        display = np.hstack((left, right))
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            settings = {
                "camera_device": CAMERA_DEVICE,
                "green_h_min": h0, "green_h_max": h1,
                "green_s_min": s0, "green_s_max": s1,
                "green_v_min": v0, "green_v_max": v1,
                "min_green_contour_area": vals["Min area"],
                "morph_open_size": k_open,
                "morph_close_size": k_close,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            SETTINGS_FILE.write_text(json.dumps(settings, indent=2) + "\n")
            print(f"Saved current values to {SETTINGS_FILE.resolve()}")
            print(json.dumps(settings, indent=2))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
