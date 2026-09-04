#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import csv
import math
import random
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
# ============================================================

BRIGHTNESS_MIN = 230
BRIGHTNESS_MAX = 255

MORPH_KERNEL_SIZE = 3

MIN_BLOB_AREA = 6

# ============================================================
# CONTROL
# ============================================================

PIXEL_DEADBAND = 15

SERVO_UPDATE_INTERVAL = 0.08
# Minimum time between corrections. Was 0.10 (stepping visibly
# every 100ms on far targets); 0.0 means "update every frame,"
# giving continuous motion — speed and slow-down near the target
# now come entirely from ADAM_LR/EASE_DISTANCE_PX/MAX_STEP_US
# rather than from a coarse fixed interval. Raise this only if
# continuous updates reintroduce oscillation on your hardware.

ALPHA = 0.4
# EMA smoothing factor on the laser x position used for
# error calculation. Lower = smoother but laggier.

# --------------------------------------------------------
# NEW: Adam-style adaptive step
#
# Replaces the flat PIXEL_TO_US proportional gain. See the
# accompanying markdown for what each of these does — in
# short: ADAM_LR is the base gain, BETA1/BETA2 control how
# much recent error history smooths the step (this is what
# resists single-frame noise/jump spikes), and EASE_DISTANCE_PX
# is what actually forces smaller steps near the target.
# --------------------------------------------------------

ADAM_LR = 8
# NOTE: unlike the old PIXEL_TO_US, this is NOT a per-pixel
# gain. Adam normalizes the step by the gradient's own recent
# magnitude, so after the first update the raw step settles to
# roughly +/- ADAM_LR regardless of error size. Set this to the
# step size (us) you want on LARGE errors -- the actual
# "smaller steps near target" behavior comes from EASE_DISTANCE_PX
# below, not from this value shrinking on its own.
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-3

EASE_DISTANCE_PX = 50
# Distance (px) at which step size starts scaling down.
# At |error| >= this value: full step. At |error| == 0:
# step scaled to 0. Linear ramp in between.

MAX_STEP_US = 10
# Hard safety cap on any single correction, regardless of
# what Adam/easing compute.


# --------------------------------------------------------
# NEW: post-target jitter
#
# After the target is reached, deliberately wiggles the
# pulse around the settled position for a short time.
# --------------------------------------------------------

JITTER_ENABLE = False

JITTER_AMPLITUDE_US = 30
# How far (in us) the jitter swings from the settled pulse.
# Larger = more visible/aggressive wiggle.

JITTER_DURATION_S = 3
# How long (seconds) to jitter after reaching target.

JITTER_FREQUENCY_HZ = 5.0
# Speed of the sinusoidal component of the jitter.

JITTER_RANDOMNESS = 0.0
# Blend between smooth sine motion (0.0) and pure random
# noise (1.0). e.g. 0.3 = mostly smooth with some noise.

SETTLE_GRACE_PERIOD_S = 0.4
# After entering SETTLED (jitter finished, or JITTER_ENABLE is
# False), drift-recovery is paused for this many seconds before
# SETTLED starts re-checking the real error again. Without this,
# residual motion from the jitter itself (the servo/laser hasn't
# mechanically caught up to the settled pulse yet) can read as
# "still off target," bounce back into TRACKING, immediately
# re-reach the deadband, and jitter again — forever.
# Too short -> the jitter-loop bug can come back.
# Too long  -> slower to notice genuine drift (e.g. the mount
# gets bumped) once truly settled.


# ============================================================
# LOGGING (unchanged — original log/video files)
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

VIDEO_FILE = os.path.join(
    LOG_DIR,
    f"laser_servo_{timestamp}.mp4"
)

video_writer = None


def setup_video_writer(width, height, fps):

    global video_writer

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

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
# NEW: separate diagnostic log
#
# Does not touch LOG_FILE/log_data above. Captures everything
# needed to debug detection + control behavior from the CSV
# alone, without needing the video or OCR.
# ============================================================

EXTRA_LOG_FILE = os.path.join(
    LOG_DIR,
    f"laser_servo_{timestamp}_extra.csv"
)

extra_log_file = None
extra_csv_writer = None

EXTRA_LOG_FIELDS = [
    "timestamp",
    "frame",
    "status_text",
    "target_state",
    "sequence_running",
    "queue_remaining",
    "laser_x",
    "laser_y",
    "filtered_x",
    "error_x",
    "adam_m",
    "adam_v",
    "adam_step",
    "ease_factor",
    "correction_us",
    "pulse_us",
    "jitter_offset_us",
]


def setup_extra_logging():

    global extra_log_file
    global extra_csv_writer

    extra_log_file = open(
        EXTRA_LOG_FILE,
        "w",
        newline=""
    )

    extra_csv_writer = csv.writer(
        extra_log_file
    )

    extra_csv_writer.writerow(EXTRA_LOG_FIELDS)

    extra_log_file.flush()


def log_extra_data(values):

    if extra_csv_writer is None:
        return

    row = [time.time()] + [
        values.get(field) for field in EXTRA_LOG_FIELDS[1:]
    ]

    extra_csv_writer.writerow(row)

    extra_log_file.flush()


# ============================================================
# NEW: tuned-parameter dump
#
# Writes every tunable constant's active value to its own file
# once at startup, so a given log/video can always be matched
# back to the exact settings that produced it.
# ============================================================

PARAMS_LOG_FILE = os.path.join(
    LOG_DIR,
    f"laser_servo_{timestamp}_params.csv"
)


def get_tunable_params():

    return {
        "BRIGHTNESS_MIN": BRIGHTNESS_MIN,
        "BRIGHTNESS_MAX": BRIGHTNESS_MAX,
        "MORPH_KERNEL_SIZE": MORPH_KERNEL_SIZE,
        "MIN_BLOB_AREA": MIN_BLOB_AREA,
        "PIXEL_DEADBAND": PIXEL_DEADBAND,
        "SERVO_UPDATE_INTERVAL": SERVO_UPDATE_INTERVAL,
        "ALPHA": ALPHA,
        "ADAM_LR": ADAM_LR,
        "ADAM_BETA1": ADAM_BETA1,
        "ADAM_BETA2": ADAM_BETA2,
        "ADAM_EPS": ADAM_EPS,
        "EASE_DISTANCE_PX": EASE_DISTANCE_PX,
        "MAX_STEP_US": MAX_STEP_US,
        "JITTER_ENABLE": JITTER_ENABLE,
        "JITTER_AMPLITUDE_US": JITTER_AMPLITUDE_US,
        "JITTER_DURATION_S": JITTER_DURATION_S,
        "JITTER_FREQUENCY_HZ": JITTER_FREQUENCY_HZ,
        "JITTER_RANDOMNESS": JITTER_RANDOMNESS,
        "SETTLE_GRACE_PERIOD_S": SETTLE_GRACE_PERIOD_S,
        "MIN_PULSE_US": MIN_PULSE_US,
        "MAX_PULSE_US": MAX_PULSE_US,
        "NEUTRAL_PULSE_US": NEUTRAL_PULSE_US,
        "PWM_FREQUENCY": PWM_FREQUENCY,
    }


def save_params_log():

    params = get_tunable_params()

    with open(PARAMS_LOG_FILE, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])

        for key, value in params.items():
            writer.writerow([key, value])

    return params


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
# Restored to your original implementation — largest bright
# blob wins, no continuity gating. The gating introduced
# false rejections/misses, so it's removed rather than tuned.
# ============================================================

def detect_laser(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    mask = cv2.inRange(
        gray,
        BRIGHTNESS_MIN,
        BRIGHTNESS_MAX
    )

    kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_BLOB_AREA:
            continue

        if area > best_area:
            best = contour
            best_area = area

    if best is None:
        return None, None, 0, mask, len(contours)

    M = cv2.moments(best)

    if M["m00"] == 0:
        return None, None, 0, mask, len(contours)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return cx, cy, best_area, mask, len(contours)


# ============================================================
# MOUSE
# ============================================================

target_x = None
target_y = None
target_state = "IDLE"
# One of: IDLE, TRACKING, JITTERING, SETTLED

pending_targets = []
# Queue of (x, y) tuples clicked but not yet active. Filled by
# clicking before pressing G; consumed one at a time once the
# sequence is running.

sequence_running = False
# True from the moment G is pressed until the queue is empty
# and the last target has settled.


def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global pending_targets
    global sequence_running

    if event == cv2.EVENT_LBUTTONDOWN:

        if sequence_running:

            print()
            print(
                "[QUEUE] Sequence already running — "
                "press C to cancel before queuing new targets"
            )

            return

        pending_targets.append((x, y))

        print()
        print(
            f"[QUEUE] Added target #{len(pending_targets)}: "
            f"x={x}, y={y}"
        )


# ============================================================
# DRAW LASER
# ============================================================

def draw_laser(output, laser_x, laser_y):

    if laser_x is None:
        return

    x = int(laser_x)
    y = int(laser_y)

    cv2.circle(output, (x, y), 8, (0, 0, 255), -1)

    cv2.drawMarker(
        output, (x, y), (0, 0, 255),
        cv2.MARKER_CROSS, 30, 2
    )

    cv2.putText(
        output, f"Laser: ({x}, {y})", (x + 15, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )


# ============================================================
# DRAW TARGET
# ============================================================

def draw_target(output, target_x, target_y, laser_x):

    if target_x is None:
        return

    x = int(target_x)
    y = int(target_y)

    cv2.drawMarker(
        output, (x, y), (255, 0, 0),
        cv2.MARKER_CROSS, 30, 2
    )

    cv2.putText(
        output, f"Target: ({x}, {y})", (x + 15, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
    )

    if laser_x is not None:

        lx = int(laser_x)

        cv2.line(output, (lx, y), (x, y), (255, 255, 0), 2)


# ============================================================
# DRAW QUEUE
# ============================================================

def draw_queue(output, pending_targets):

    for i, (x, y) in enumerate(pending_targets):

        cv2.circle(output, (x, y), 6, (0, 165, 255), 2)

        cv2.putText(
            output, str(i + 1), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1
        )


# ============================================================
# DRAW TUNED PARAMETERS
#
# Bakes the active tunable settings into the corner of the
# recorded video (and live window), so a saved clip is always
# self-describing without needing to cross-reference the code
# or the separate params CSV.
# ============================================================

def draw_params_overlay(output, params):

    x0 = output.shape[1] - 280
    y0 = 20
    line_height = 15

    for i, (key, value) in enumerate(params.items()):

        text = f"{key}={value}"

        cv2.putText(
            output, text, (x0, y0 + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global target_x
    global target_y
    global target_state
    global pending_targets
    global sequence_running

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera /dev/video{CAMERA_DEVICE}"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    time.sleep(0.5)

    print()
    print("==========================================")
    print(" LASER + SERVO CONTROLLER")
    print("==========================================")
    print()
    print(f"Camera       : /dev/video{CAMERA_DEVICE}")
    print(f"Resolution   : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"PWM          : {PWM_PATH}")
    print(f"Pulse range  : {MIN_PULSE_US}-{MAX_PULSE_US} us")
    print()
    print("CONTROLS")
    print("  Left click  = queue a target")
    print("  G           = go — start tracking the queued targets in order")
    print("  R           = neutral (also clears queue)")
    print("  C           = cancel current target / sequence / queue")
    print("  Q           = quit")
    print()
    print(f"Log file       : {LOG_FILE}")
    print(f"Extra log file : {EXTRA_LOG_FILE}")
    print(f"Params file    : {PARAMS_LOG_FILE}")
    print()

    setup_pwm()

    pulse_us = float(NEUTRAL_PULSE_US)

    setup_logging()
    setup_extra_logging()
    tunable_params = save_params_log()
    setup_video_writer(FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS)

    if video_writer is not None:
        print(f"Video file: {VIDEO_FILE}")

    WINDOW_NAME = "Camera + Laser + Servo"
    RAW_WINDOW = "Raw Camera"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(RAW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    tracking = False
    frame_number = 0

    fps = 0
    fps_count = 0
    fps_start = time.monotonic()

    last_servo_update = 0

    filtered_x = None

    adam_m = 0.0
    adam_v = 0.0
    adam_t = 0

    jitter_start_time = None
    jitter_base_pulse = None
    settled_since = None

    try:

        while True:

            frame_number += 1

            ret, frame = cap.read()

            if not ret:
                print("[CAMERA] Failed to read frame")
                continue

            # ------------------------------------------------
            # Laser detection (original implementation)
            # ------------------------------------------------

            (
                laser_x, laser_y, blob_area, mask, num_contours
            ) = detect_laser(frame)

            # ------------------------------------------------
            # Control
            # ------------------------------------------------

            error_x = None
            correction = None
            ease = None
            jitter_offset = None

            if target_x is not None:

                if target_state == "IDLE" and laser_x is not None:
                    target_state = "TRACKING"

                # TRACKING always re-evaluates. SETTLED also
                # re-evaluates, but only once SETTLE_GRACE_PERIOD_S
                # has passed since settling — see the constant's
                # comment for why the grace period exists.

                settled_grace_elapsed = (
                    target_state != "SETTLED"
                    or settled_since is None
                    or time.monotonic() - settled_since >= SETTLE_GRACE_PERIOD_S
                )

                if (
                    target_state in ("TRACKING", "SETTLED")
                    and settled_grace_elapsed
                    and laser_x is not None
                    and time.monotonic() - last_servo_update >= SERVO_UPDATE_INTERVAL
                ):

                    filtered_x = (
                        laser_x if filtered_x is None
                        else ALPHA * laser_x + (1 - ALPHA) * filtered_x
                    )

                    error_x = target_x - filtered_x

                    if abs(error_x) <= PIXEL_DEADBAND:

                        tracking = False

                        if target_state == "TRACKING":
                            # Just arrived — enter jitter/settle.
                            target_state = "JITTERING" if JITTER_ENABLE else "SETTLED"
                            jitter_start_time = time.monotonic()
                            jitter_base_pulse = pulse_us
                            settled_since = time.monotonic()
                        # else: already SETTLED and still within
                        # deadband — stay put, nothing to do.

                    else:

                        # Either actively tracking, or drifted back
                        # out of the deadband while SETTLED — in
                        # both cases, (re)apply a correction.

                        tracking = True
                        target_state = "TRACKING"

                        grad = error_x
                        adam_t += 1
                        adam_m = ADAM_BETA1 * adam_m + (1 - ADAM_BETA1) * grad
                        adam_v = ADAM_BETA2 * adam_v + (1 - ADAM_BETA2) * (grad ** 2)

                        m_hat = adam_m / (1 - ADAM_BETA1 ** adam_t)
                        v_hat = adam_v / (1 - ADAM_BETA2 ** adam_t)

                        adam_step = ADAM_LR * m_hat / (math.sqrt(v_hat) + ADAM_EPS)

                        ease = min(1.0, abs(error_x) / EASE_DISTANCE_PX)

                        correction = -adam_step * ease

                        correction = max(
                            -MAX_STEP_US,
                            min(MAX_STEP_US, correction)
                        )

                        pulse_us += correction
                        pulse_us = max(
                            MIN_PULSE_US,
                            min(MAX_PULSE_US, pulse_us)
                        )

                        set_pulse_us(pulse_us)

                    last_servo_update = time.monotonic()

                elif target_state == "JITTERING":

                    # Jitter is a deliberate motion, independent of
                    # whether the laser happens to be detected this
                    # exact frame — it doesn't need laser_x.

                    elapsed = time.monotonic() - jitter_start_time

                    if elapsed >= JITTER_DURATION_S:

                        pulse_us = jitter_base_pulse
                        set_pulse_us(pulse_us)

                        if sequence_running and pending_targets:

                            # Advance to the next queued target.

                            target_x, target_y = pending_targets.pop(0)
                            target_state = "IDLE"
                            filtered_x = None
                            adam_m = 0.0
                            adam_v = 0.0
                            adam_t = 0

                            print()
                            print(
                                f"[SEQUENCE] Advancing to "
                                f"x={target_x}, y={target_y} "
                                f"({len(pending_targets)} remaining)"
                            )

                        else:

                            target_state = "SETTLED"
                            settled_since = time.monotonic()

                            if sequence_running:
                                sequence_running = False
                                print()
                                print("[SEQUENCE] Complete — all targets reached")

                    else:

                        sine_component = JITTER_AMPLITUDE_US * math.sin(
                            2 * math.pi * JITTER_FREQUENCY_HZ * elapsed
                        )
                        noise_component = JITTER_AMPLITUDE_US * random.uniform(-1, 1)

                        jitter_offset = (
                            (1 - JITTER_RANDOMNESS) * sine_component
                            + JITTER_RANDOMNESS * noise_component
                        )

                        jitter_pulse = jitter_base_pulse + jitter_offset
                        jitter_pulse = max(
                            MIN_PULSE_US,
                            min(MAX_PULSE_US, jitter_pulse)
                        )

                        set_pulse_us(jitter_pulse)
                        pulse_us = jitter_pulse

                # SETTLED: nothing to do — servo holds still.

            # ------------------------------------------------
            # Output image
            # ------------------------------------------------

            output = frame.copy()

            draw_laser(output, laser_x, laser_y)
            draw_target(output, target_x, target_y, laser_x)
            draw_queue(output, pending_targets)

            if laser_x is None:
                status = "LASER NOT DETECTED"
                status_color = (0, 0, 255)
            elif target_x is None and pending_targets:
                status = f"{len(pending_targets)} QUEUED — PRESS G TO GO"
                status_color = (0, 165, 255)
            elif target_x is None:
                status = "CLICK TO QUEUE TARGET(S)"
                status_color = (255, 255, 0)
            elif target_state == "TRACKING" and tracking:
                status = "TRACKING"
                status_color = (0, 255, 255)
            elif target_state == "JITTERING":
                status = "JITTERING"
                status_color = (255, 0, 255)
            elif target_state == "SETTLED":
                status = "TARGET REACHED (settled)"
                status_color = (0, 255, 0)
            else:
                status = "TARGET REACHED"
                status_color = (0, 255, 0)

            cv2.putText(
                output, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2
            )

            if sequence_running:
                cv2.putText(
                    output, f"SEQUENCE RUNNING ({len(pending_targets)} remaining)",
                    (30, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                )

            cv2.putText(
                output, f"Pulse: {pulse_us:.1f} us", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            cv2.putText(
                output, f"Blob area: {blob_area:.1f}", (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            cv2.putText(
                output, f"Contours: {num_contours}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            if error_x is not None:
                cv2.putText(
                    output, f"X Error: {error_x:+.1f} px", (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

            fps_count += 1
            now = time.monotonic()

            if now - fps_start >= 1.0:
                fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now

            cv2.putText(
                output, f"FPS: {fps:.1f}", (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            draw_params_overlay(output, tunable_params)

            cv2.imshow(WINDOW_NAME, output)
            cv2.imshow(RAW_WINDOW, frame)

            if video_writer is not None:
                combined = np.hstack((output, frame))
                video_writer.write(combined)

            # ------------------------------------------------
            # Logging (original — untouched)
            # ------------------------------------------------

            log_data(
                frame_number, target_x, target_y, laser_x, laser_y,
                error_x, pulse_us, blob_area, num_contours, tracking
            )

            # ------------------------------------------------
            # Logging (new — extra diagnostic log)
            # ------------------------------------------------

            log_extra_data({
                "frame": frame_number,
                "status_text": status,
                "target_state": target_state,
                "sequence_running": sequence_running,
                "queue_remaining": len(pending_targets),
                "laser_x": laser_x,
                "laser_y": laser_y,
                "filtered_x": filtered_x,
                "error_x": error_x,
                "adam_m": adam_m,
                "adam_v": adam_v,
                "adam_step": (correction / -ease) if (correction is not None and ease) else None,
                "ease_factor": ease,
                "correction_us": correction,
                "pulse_us": pulse_us,
                "jitter_offset_us": jitter_offset,
            })

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("g"):

                if not sequence_running and pending_targets:

                    sequence_running = True
                    target_x, target_y = pending_targets.pop(0)
                    target_state = "IDLE"
                    filtered_x = None
                    adam_m = 0.0
                    adam_v = 0.0
                    adam_t = 0

                    print()
                    print(
                        f"[SEQUENCE] Starting — "
                        f"x={target_x}, y={target_y} "
                        f"({len(pending_targets)} remaining)"
                    )

                elif sequence_running:

                    print()
                    print("[SEQUENCE] Already running")

                else:

                    print()
                    print("[SEQUENCE] No queued targets — click to add some first")

            elif key == ord("r"):

                pulse_us = float(NEUTRAL_PULSE_US)
                set_pulse_us(pulse_us)

                target_x = None
                target_y = None
                target_state = "IDLE"
                tracking = False
                adam_m = 0.0
                adam_v = 0.0
                adam_t = 0
                filtered_x = None
                pending_targets = []
                sequence_running = False
                settled_since = None

                print()
                print("[SERVO] Returned to 1500 us")

            elif key == ord("c"):

                target_x = None
                target_y = None
                target_state = "IDLE"
                tracking = False
                adam_m = 0.0
                adam_v = 0.0
                adam_t = 0
                filtered_x = None
                pending_targets = []
                sequence_running = False
                settled_since = None

                print()
                print("[TARGET] Cancelled")

    finally:

        print()
        print("Returning servo to neutral...")

        try:
            set_pulse_us(NEUTRAL_PULSE_US)
            time.sleep(0.5)
        except Exception as e:
            print(f"Could not return to neutral: {e}")

        stop_pwm()
        cap.release()
        cv2.destroyAllWindows()

        if log_file is not None:
            log_file.close()

        if extra_log_file is not None:
            extra_log_file.close()

        print()
        print(f"Log saved to:\n  {LOG_FILE}")
        print(f"Extra log saved to:\n  {EXTRA_LOG_FILE}")

        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to:\n  {VIDEO_FILE}")

        print()
        print("Done.")


if __name__ == "__main__":
    main()