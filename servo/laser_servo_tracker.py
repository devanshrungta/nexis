#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime


# ============================================================
# CAMERA
# ============================================================

CAMERA_DEVICE = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 60


# ============================================================
# PWM
# ============================================================

PWM_CHIP = "/sys/class/pwm/pwmchip3"
PWM_PATH = "/sys/class/pwm/pwmchip3/pwm0"

PWM_CHANNEL = 0

PWM_FREQUENCY = 50
PERIOD_NS = 20_000_000


# ============================================================
# SERVO
# ============================================================

MIN_PULSE_US = 500
MAX_PULSE_US = 2500

NEUTRAL_PULSE_US = 1500


# ============================================================
# LASER DETECTION
#
# THESE VALUES ARE COPIED EXACTLY FROM YOUR
# laser_detection.py
# ============================================================

BRIGHTNESS_MIN = 220
BRIGHTNESS_MAX = 255

MORPH_KERNEL_SIZE = 3

MIN_BLOB_AREA = 4


# ============================================================
# CONTROL
# ============================================================

# Start conservative.
#
# This is NOT a detection parameter.
# It controls how aggressively the servo responds
# to pixel error.

PIXEL_TO_US = 0.02

MAX_STEP_US = 10

PIXEL_DEADBAND = 15

SERVO_SETTLE_TIME = 0.3


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = "servo_logs"

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"laser_servo_{timestamp}.csv"
)

# ============================================================
# VIDEO RECORDING (combined: main output + mask side-by-side)
# ============================================================

VIDEO_FILE = os.path.join(
    LOG_DIR,
    f"laser_servo_{timestamp}.mp4"
)

video_writer = None

def setup_video_writer(width, height, fps):

    global video_writer

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Combined frame is output + mask placed side by side
    combined_width = width * 2

    video_writer = cv2.VideoWriter(
        VIDEO_FILE,
        fourcc,
        fps,
        (combined_width, height)
    )

    if not video_writer.isOpened():
        print(f"[VIDEO] Failed to open writer for {VIDEO_FILE}")
        video_writer = None

# ============================================================
# PWM
# ============================================================

def write_sysfs(path, value):

    with open(path, "w") as f:
        f.write(str(value))


def set_pulse_us(pulse_us):

    pulse_us = max(
        MIN_PULSE_US,
        min(MAX_PULSE_US, pulse_us)
    )

    pulse_ns = int(
        round(pulse_us * 1000)
    )

    write_sysfs(
        f"{PWM_PATH}/duty_cycle",
        pulse_ns
    )

    return pulse_us


def setup_pwm():

    if not os.path.exists(PWM_PATH):

        write_sysfs(
            f"{PWM_CHIP}/export",
            PWM_CHANNEL
        )

        time.sleep(0.1)

    try:

        write_sysfs(
            f"{PWM_PATH}/enable",
            0
        )

    except Exception:
        pass

    write_sysfs(
        f"{PWM_PATH}/period",
        PERIOD_NS
    )

    set_pulse_us(
        NEUTRAL_PULSE_US
    )

    write_sysfs(
        f"{PWM_PATH}/enable",
        1
    )


def stop_pwm():

    try:

        write_sysfs(
            f"{PWM_PATH}/enable",
            0
        )

    except Exception:
        pass


# ============================================================
# LASER DETECTION
#
# EXACTLY MATCHES YOUR PROVIDED DETECTOR
# ============================================================

def detect_laser(frame):

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Find very bright pixels
    #
    # EXACT:
    #     245 -> 255
    # --------------------------------------------------------

    mask = cv2.inRange(
        gray,
        BRIGHTNESS_MIN,
        BRIGHTNESS_MAX
    )

    # --------------------------------------------------------
    # Remove tiny noise
    #
    # EXACT:
    #     3x3 kernel
    #     MORPH_OPEN
    # --------------------------------------------------------

    kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # EXACT:
        # Ignore regions below area 2

        if area < MIN_BLOB_AREA:
            continue

        if area > best_area:

            best = contour
            best_area = area

    # --------------------------------------------------------
    # No laser
    # --------------------------------------------------------

    if best is None:

        return None, None, 0, mask, len(contours)

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    M = cv2.moments(best)

    if M["m00"] == 0:

        return None, None, 0, mask, len(contours)

    cx = int(
        M["m10"] / M["m00"]
    )

    cy = int(
        M["m01"] / M["m00"]
    )

    return (
        cx,
        cy,
        best_area,
        mask,
        len(contours)
    )


# ============================================================
# LOG FILE
# ============================================================

log_file = None
csv_writer = None


def setup_logging():

    global log_file
    global csv_writer

    log_file = open(
        LOG_FILE,
        "w",
        newline=""
    )

    csv_writer = csv.writer(
        log_file
    )

    csv_writer.writerow([
        "timestamp",
        "frame",
        "target_x",
        "target_y",
        "laser_x",
        "laser_y",
        "error_x",
        "pulse_us",
        "blob_area",
        "num_contours",
        "tracking"
    ])

    log_file.flush()


def log_data(
    frame_number,
    target_x,
    target_y,
    laser_x,
    laser_y,
    error_x,
    pulse_us,
    blob_area,
    num_contours,
    tracking
):

    if csv_writer is None:
        return

    csv_writer.writerow([
        time.time(),
        frame_number,
        target_x,
        target_y,
        laser_x,
        laser_y,
        error_x,
        pulse_us,
        blob_area,
        num_contours,
        tracking
    ])

    log_file.flush()


# ============================================================
# MOUSE
# ============================================================

target_x = None
target_y = None


def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global target_x
    global target_y

    if event == cv2.EVENT_LBUTTONDOWN:

        target_x = x
        target_y = y

        print()
        print(
            f"[TARGET] "
            f"x={x}, y={y}"
        )


# ============================================================
# DRAW LASER
# ============================================================

def draw_laser(
    output,
    laser_x,
    laser_y
):

    if laser_x is None:
        return

    x = int(laser_x)
    y = int(laser_y)

    # Center
    cv2.circle(
        output,
        (x, y),
        8,
        (0, 0, 255),
        -1
    )

    # Crosshair
    cv2.drawMarker(
        output,
        (x, y),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        30,
        2
    )

    # Coordinates
    cv2.putText(
        output,
        f"Laser: ({x}, {y})",
        (x + 15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


# ============================================================
# DRAW TARGET
# ============================================================

def draw_target(
    output,
    target_x,
    target_y,
    laser_x
):

    if target_x is None:
        return

    x = int(target_x)
    y = int(target_y)

    # Blue target marker

    cv2.drawMarker(
        output,
        (x, y),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        30,
        2
    )

    cv2.putText(
        output,
        f"Target: ({x}, {y})",
        (x + 15, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # Horizontal error line

    if laser_x is not None:

        lx = int(laser_x)

        cv2.line(
            output,
            (lx, y),
            (x, y),
            (255, 255, 0),
            2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global target_x
    global target_y

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"/dev/video{CAMERA_DEVICE}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FRAME_FPS
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    time.sleep(0.5)

    print()
    print("==========================================")
    print(" LASER + SERVO CONTROLLER")
    print("==========================================")
    print()
    print(f"Camera       : /dev/video{CAMERA_DEVICE}")
    print(
        f"Resolution   : "
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
    )
    print(
        f"PWM          : {PWM_PATH}"
    )
    print(
        f"Frequency    : {PWM_FREQUENCY} Hz"
    )
    print(
        f"Pulse range  : "
        f"{MIN_PULSE_US}-{MAX_PULSE_US} us"
    )
    print()
    print("LASER DETECTOR")
    print(
        f"Brightness   : "
        f"{BRIGHTNESS_MIN}-{BRIGHTNESS_MAX}"
    )
    print(
        f"Kernel       : "
        f"{MORPH_KERNEL_SIZE}x"
        f"{MORPH_KERNEL_SIZE}"
    )
    print(
        f"Min area     : "
        f"{MIN_BLOB_AREA}"
    )
    print()
    print("CONTROLS")
    print("  Left click  = select target")
    print("  R           = neutral")
    print("  C           = cancel target")
    print("  Q           = quit")
    print()
    print(
        f"Log file: {LOG_FILE}"
    )
    print()

    # --------------------------------------------------------
    # PWM
    # --------------------------------------------------------

    setup_pwm()

    pulse_us = float(
        NEUTRAL_PULSE_US
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    setup_logging()
    
    
    setup_video_writer(FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS)

    if video_writer is not None:
        print(f"Video file: {VIDEO_FILE}")
    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    WINDOW_NAME = (
        "Camera + Laser + Servo"
    )

    MASK_WINDOW = (
        "Brightness Mask"
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        MASK_WINDOW,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        FRAME_WIDTH,
        FRAME_HEIGHT
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    tracking = False

    frame_number = 0

    # FPS
    fps = 0
    fps_count = 0
    fps_start = time.monotonic()

    last_servo_update = 0
    SERVO_UPDATE_INTERVAL = 0.10

    try:
        filtered_x = None
        ALPHA = 0.4  # lower = smoother, laggier
        while True:

            frame_number += 1

            # ------------------------------------------------
            # Camera
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                print(
                    "[CAMERA] Failed to read frame"
                )

                continue

            # ------------------------------------------------
            # Laser detection
            # ------------------------------------------------

            (
                laser_x,
                laser_y,
                blob_area,
                mask,
                num_contours
            ) = detect_laser(frame)

            # ------------------------------------------------
            # Calculate error
            # ------------------------------------------------

            error_x = None

            if (
                target_x is not None
                and laser_x is not None
                and time.monotonic() - last_servo_update >= SERVO_UPDATE_INTERVAL
            ):
                filtered_x = laser_x if filtered_x is None else ALPHA * laser_x + (1 - ALPHA) * filtered_x
                error_x = (
                    target_x - filtered_x
                )

                # --------------------------------------------
                # Check whether target is reached
                # --------------------------------------------

                if abs(error_x) <= PIXEL_DEADBAND:

                    tracking = False

                else:

                    tracking = True

                    # ----------------------------------------
                    # Proportional correction
                    # ----------------------------------------

                    correction = (
                        -PIXEL_TO_US
                        * error_x
                    )

                    # ----------------------------------------
                    # Limit correction
                    # ----------------------------------------

                    correction = max(
                        -MAX_STEP_US,
                        min(
                            MAX_STEP_US,
                            correction
                        )
                    )

                    pulse_us += correction

                    # ----------------------------------------
                    # Servo limits
                    # ----------------------------------------

                    pulse_us = max(
                        MIN_PULSE_US,
                        min(
                            MAX_PULSE_US,
                            pulse_us
                        )
                    )

                    # ----------------------------------------
                    # Set servo
                    # ----------------------------------------

                    set_pulse_us(
                        pulse_us
                    )
                    
                    last_servo_update = time.monotonic()

                    # ----------------------------------------
                    # Allow servo to move
                    # ----------------------------------------

                    # time.sleep(
                    #     SERVO_SETTLE_TIME
                    # )

            # ------------------------------------------------
            # Output image
            # ------------------------------------------------

            output = frame.copy()

            draw_laser(
                output,
                laser_x,
                laser_y
            )

            draw_target(
                output,
                target_x,
                target_y,
                laser_x
            )

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if laser_x is None:

                status = (
                    "LASER NOT DETECTED"
                )

                status_color = (
                    0, 0, 255
                )

            elif target_x is None:

                status = (
                    "CLICK TO SELECT TARGET"
                )

                status_color = (
                    255, 255, 0
                )

            elif tracking:

                status = "TRACKING"

                status_color = (
                    0, 255, 255
                )

            else:

                status = "TARGET REACHED"

                status_color = (
                    0, 255, 0
                )

            cv2.putText(
                output,
                status,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

            # ------------------------------------------------
            # Debug information
            # ------------------------------------------------

            cv2.putText(
                output,
                f"Pulse: {pulse_us:.1f} us",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                output,
                f"Blob area: {blob_area:.1f}",
                (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                output,
                f"Contours: {num_contours}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if error_x is not None:

                cv2.putText(
                    output,
                    f"X Error: {error_x:+.1f} px",
                    (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            fps_count += 1

            now = time.monotonic()

            if now - fps_start >= 1.0:

                fps = (
                    fps_count
                    / (now - fps_start)
                )

                fps_count = 0
                fps_start = now

            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                output
            )

            cv2.imshow(
                MASK_WINDOW,
                mask
            )

            if video_writer is not None:

                mask_bgr = cv2.cvtColor(
                    mask,
                    cv2.COLOR_GRAY2BGR
                )

                combined = np.hstack(
                    (output, mask_bgr)
                )

                video_writer.write(combined)
            # ------------------------------------------------
            # Log
            # ------------------------------------------------

            log_data(
                frame_number,
                target_x,
                target_y,
                laser_x,
                laser_y,
                error_x,
                pulse_us,
                blob_area,
                num_contours,
                tracking
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            # Quit

            if key == ord("q"):

                break

            # Neutral

            elif key == ord("r"):

                pulse_us = float(
                    NEUTRAL_PULSE_US
                )

                set_pulse_us(
                    pulse_us
                )

                target_x = None
                target_y = None
                tracking = False

                print()
                print(
                    "[SERVO] "
                    "Returned to 1500 us"
                )

            # Cancel target

            elif key == ord("c"):

                target_x = None
                target_y = None
                tracking = False

                print()
                print(
                    "[TARGET] Cancelled"
                )

    finally:

        print()
        print(
            "Returning servo to neutral..."
        )

        try:

            set_pulse_us(
                NEUTRAL_PULSE_US
            )

            time.sleep(0.5)

        except Exception as e:

            print(
                f"Could not return to neutral: {e}"
            )

        stop_pwm()

        cap.release()

        cv2.destroyAllWindows()

        if log_file is not None:
            log_file.close()

        print()
        print(
            f"Log saved to:"
        )
        print(
            f"  {LOG_FILE}"
        )
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to:\n  {VIDEO_FILE}")
        print()
        print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()