#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import csv
import math
import random
import Jetson.GPIO as GPIO
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

SERVO_VISIBLE_MIN_PULSE_US = MIN_PULSE_US
SERVO_VISIBLE_MAX_PULSE_US = MAX_PULSE_US
# The pulse range within which the laser is ACTUALLY visible in
# the camera's frame -- not necessarily the servo's full
# mechanical range above, which is very likely wider than what
# the camera can actually see. Defaults to the full range (no
# behavior change) until you calibrate it: press R to go to
# neutral, then manually drive the servo (e.g. quickest way --
# temporarily click a target far off to one side and watch the
# video feed) to find the pulse value where the laser JUST exits
# frame on each side (read it off the "Pulse: ... us" overlay),
# and set these two to just inside those points. Every place the
# control loop below moves the servo clamps to THIS range, not
# the full mechanical range -- so a bad correction (miscalibrated
# gain, an edge case in the recovery sweep, anything) can't drive
# the laser out of view even if it tries to. NEUTRAL_PULSE_US
# above is left alone -- it's the servo's own rest position, used
# only by R and on quit, not part of this visibility clamp.


# ============================================================
# LASER DETECTION
# ============================================================

BRIGHTNESS_MIN = 240
BRIGHTNESS_MAX = 255

MORPH_KERNEL_SIZE = 5

MIN_BLOB_AREA = 10

# ============================================================
# CONTROL
# ============================================================

SERVO_UPDATE_INTERVAL = 0.05
# Minimum time between corrections. 0.0 = update every frame,
# giving continuous motion (was 0.08 before, which visibly
# "stepped" at ~12.5 updates/sec). Speed and slow-down now
# come from GAIN_US_PER_PX / MAX_STEP_US below.

ALPHA = 0.40
# EMA smoothing factor on the laser x position used for
# error calculation. Lower = smoother but laggier.

# --------------------------------------------------------
# Clamped proportional (P) controller
#
# Replaces the previous Adam-based controller. Adam normalizes
# its step by the gradient's own recent magnitude, so the raw
# step settles to roughly +/-ADAM_LR regardless of error size,
# and its momentum term (an EMA of recent error) keeps pushing
# in the old direction for several frames after the error flips
# sign -- both of which produced the overshoot/oscillation you
# were seeing whenever the gap was bigger than the deadband but
# smaller than one step.
#
# A plain clamped P-controller fixes this: the commanded
# correction is directly proportional to the remaining error
# (so it decelerates smoothly and hits ~0 at ~0 error, no
# separate "ease" term needed), and it has no memory of past
# frames, so it can't keep pushing after crossing the target.
# --------------------------------------------------------

GAIN_US_PER_PX = 0.025
# How many microseconds of pulse-width correction to command
# per pixel of remaining error. This should reflect (or be
# slightly BELOW) the real system response -- i.e. how many us
# it actually takes to move the laser 1px. Set too high, it can
# still overshoot despite the clamp; set too low, it just takes
# a few extra frames to converge. Tune by watching the "Error"
# overlay: if it overshoots past zero, lower this; if it's
# sluggish on small errors, raise it.

CONTROL_AXIS = "x"
# Which pixel axis this single servo currently follows: "x"
# (default) or "y". Toggle live with the A key. You don't need
# to re-click or re-queue targets when you switch -- every
# clicked target already stores both its x and y (from the
# original click), so switching axis just changes which
# coordinate of the SAME target(s) the control loop reads.

AXIS_GAIN_US_PER_PX = {
    "x": GAIN_US_PER_PX,
    "y": GAIN_US_PER_PX,
    # ^ Y starts as a copy of your tuned X gain as a starting
    # point only -- the physical response per pixel is not
    # guaranteed to match between axes (different lever arm,
    # different mount geometry, etc). Tune this independently
    # once you've actually tested the Y axis, the same way you
    # tuned GAIN_US_PER_PX for X.
}

AXIS_SIGN = {
    "x": -1,
    "y": 1,
    # ^ x=-1 matches your existing `pulse_us -= correction`
    # convention. Flip y to +1 if increasing pulse_us needs to
    # move the laser in the opposite direction on that axis --
    # you'll see it immediately as the laser moving away from
    # the target instead of toward it when you first test Y.
}

MAX_STEP_US = 15
# Hard cap on any single correction (top speed for large
# errors). Raise this to close large gaps faster; the P-control
# proportionality is what prevents overshoot on the way in, not
# this cap.

SETTLE_DEADBAND_PX = 2
# Error must be within this many px to be considered "reached"
# and enter SETTLED. Kept tight for precision -- with a P-
# controller (not a fixed step), getting this close does not
# cause oscillation the way the old fixed-step approach did.

RETRACK_DEADBAND_PX = 10
# Once SETTLED, the laser must drift out past THIS looser
# threshold before correction resumes. This hysteresis gap
# between SETTLE_DEADBAND_PX and RETRACK_DEADBAND_PX stops
# detection noise right at the settle boundary from repeatedly
# flapping the state back into TRACKING.

SETTLE_DWELL_S = 0.3
# The error must stay continuously within SETTLE_DEADBAND_PX for
# this long before the target is actually marked reached. Without
# this, a fast-moving laser that merely crosses through the
# deadband on its way past the target (e.g. from overshoot, or
# while still converging) gets counted as "reached" on that one
# lucky frame. Any frame the error goes back outside the deadband
# resets the dwell clock -- see settle_dwell_start below. Shared
# by both the servo and the actuator's own settle check.

LASER_LOST_GRACE_S = 0.3
# How long the laser can go undetected before we treat it as
# "off the edge of the frame" and start actively recovering,
# rather than just freezing the servo in place. A short value
# (a frame or two) is normal detection noise and shouldn't
# trigger this -- keep it above a few frame periods.

RECOVERY_STEP_US = 15
# How fast (us per frame) to sweep the pulse back toward
# NEUTRAL_PULSE_US while the laser is missing. Neutral is used
# as the recovery target because, for this rig, it's roughly
# where the camera's field of view is centered -- so sweeping
# back toward it is the best generic way to bring an
# off-frame laser back into view without knowing exactly which
# edge it left from. Recovery stops the instant the laser is
# seen again, wherever that is.


# --------------------------------------------------------
# Post-target jitter
#
# After the target is reached, deliberately wiggles the
# pulse around the settled position for a short time.
# --------------------------------------------------------

JITTER_ENABLE = True

JITTER_AMPLITUDE_US = 30
# How far (in us) the jitter swings from the settled pulse.
# Larger = more visible/aggressive wiggle.

JITTER_DURATION_S = 3
# How long (seconds) to jitter after reaching target.

JITTER_FREQUENCY_HZ = 5.0
# Speed of the sinusoidal component of the jitter. IMPORTANT: this is
# only sampled once per real loop iteration, so it needs the loop
# running well above 2x this frequency (Nyquist) to actually look
# like a smooth wiggle -- ideally 5-10x+ for it to look clean.
# Your last two runs measured ~11.6fps then ~15.3fps (check the extra
# log's timestamp deltas to see your current number -- video encoding
# is the main remaining cost, see VIDEO_WRITE_EVERY_N above). At
# 15.3fps, 5Hz was only ~3x oversampled -- right at the aliasing edge,
# which produces a slow drifting "beat" pattern instead of a clean
# wiggle. 2Hz gives ~7.5x oversampling at that same fps and should
# look clean. If you later get the loop closer to 60fps, feel free to
# raise this back toward 5Hz (keep it under roughly fps/5 as a rule
# of thumb).

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
# re-reach the deadband, and jitter again -- forever.
# Too short -> the jitter-loop bug can come back.
# Too long  -> slower to notice genuine drift (e.g. the mount
# gets bumped) once truly settled.


# ============================================================
# LOGGING (unchanged -- original log/video files)
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

VIDEO_WRITE_EVERY_N = 1
# Write every Nth frame to the output video instead of every frame.
# Encoding a full side-by-side (2x width) frame with mp4v every
# iteration is the other prime suspect (alongside the per-frame log
# flush fixed above) for the ~11.6fps loop your log shows, against a
# nominal 60fps camera. To isolate it: temporarily set this to 3-5,
# re-run, and check the frame dt in the extra log the same way -- if
# the fps jumps back up close to 60, the video encode is the
# bottleneck, and it's worth moving to a hardware encoder (e.g. via
# GStreamer) or a background writer thread rather than skipping
# frames permanently.


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

    # Flushing (fsync-like disk I/O) every single frame was costing
    # real time on every loop iteration -- flush periodically instead.
    # The finally block still flushes+closes on a normal exit, so you
    # only risk losing the last <0.5s of rows on a hard crash.
    if frame_number % 30 == 0:
        log_file.flush()


# ============================================================
# Separate diagnostic log
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
    "control_axis",
    "sequence_running",
    "queue_remaining",
    "laser_x",
    "laser_y",
    "filtered_x",
    "error_x",
    "correction_us",
    "pulse_us",
    "jitter_offset_us",
    "actuator_axis",
    "actuator_state",
    "actuator_error",
    "actuator_position",
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

    # See the matching comment in log_data() -- same fix, same reason.
    if values.get("frame", 0) % 30 == 0:
        extra_log_file.flush()


# ============================================================
# Tuned-parameter dump
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
        "SERVO_UPDATE_INTERVAL": SERVO_UPDATE_INTERVAL,
        "ALPHA": ALPHA,
        "CONTROL_AXIS": CONTROL_AXIS,
        "GAIN_US_PER_PX": GAIN_US_PER_PX,
        "AXIS_GAIN_US_PER_PX_X": AXIS_GAIN_US_PER_PX["x"],
        "AXIS_GAIN_US_PER_PX_Y": AXIS_GAIN_US_PER_PX["y"],
        "AXIS_SIGN_X": AXIS_SIGN["x"],
        "AXIS_SIGN_Y": AXIS_SIGN["y"],
        "MAX_STEP_US": MAX_STEP_US,
        "SETTLE_DEADBAND_PX": SETTLE_DEADBAND_PX,
        "RETRACK_DEADBAND_PX": RETRACK_DEADBAND_PX,
        "SETTLE_DWELL_S": SETTLE_DWELL_S,
        "LASER_LOST_GRACE_S": LASER_LOST_GRACE_S,
        "RECOVERY_STEP_US": RECOVERY_STEP_US,
        "JITTER_ENABLE": JITTER_ENABLE,
        "JITTER_AMPLITUDE_US": JITTER_AMPLITUDE_US,
        "JITTER_DURATION_S": JITTER_DURATION_S,
        "JITTER_FREQUENCY_HZ": JITTER_FREQUENCY_HZ,
        "JITTER_RANDOMNESS": JITTER_RANDOMNESS,
        "SETTLE_GRACE_PERIOD_S": SETTLE_GRACE_PERIOD_S,
        "MIN_PULSE_US": MIN_PULSE_US,
        "MAX_PULSE_US": MAX_PULSE_US,
        "NEUTRAL_PULSE_US": NEUTRAL_PULSE_US,
        "SERVO_VISIBLE_MIN_PULSE_US": SERVO_VISIBLE_MIN_PULSE_US,
        "SERVO_VISIBLE_MAX_PULSE_US": SERVO_VISIBLE_MAX_PULSE_US,
        "PWM_FREQUENCY": PWM_FREQUENCY,
        "VIDEO_WRITE_EVERY_N": VIDEO_WRITE_EVERY_N,
        "ACT_MAX_SPEED_STEPS_S": ACT_MAX_SPEED_STEPS_S,
        "ACT_ACCELERATION_STEPS_S2": ACT_ACCELERATION_STEPS_S2,
        "ACT_RECOVERY_SPEED_STEPS_S": ACT_RECOVERY_SPEED_STEPS_S,
        "ACT_GAIN_STEPS_PER_PX": ACT_GAIN_STEPS_PER_PX,
        "ACT_SIGN": ACT_SIGN,
        "ACT_MAX_STEPS_PER_FRAME": ACT_MAX_STEPS_PER_FRAME,
        "ACT_SETTLE_DEADBAND_PX": ACT_SETTLE_DEADBAND_PX,
        "ACT_RETRACK_DEADBAND_PX": ACT_RETRACK_DEADBAND_PX,
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
# LINEAR ACTUATOR (second DOF)
#
# Only the low-level step/dir/enable control is borrowed from
# your fuyu_actuator.py (TMC2209-driven FUYU FPB30E500RB42) --
# not its calibration/limit-switch homing, keyboard jog loop, or
# acceleration ramps, none of which apply to closed-loop pixel
# tracking the way the servo's own P-controller already works.
#
# This actuator always drives whichever axis the servo is NOT
# currently on -- the complement of CONTROL_AXIS -- so the two
# DOF together cover both X and Y every frame, automatically
# following you if you press A to swap the servo's axis. The
# servo's own code above this point is completely untouched.
#
# Important open-loop caveat, inherited as-is from your script:
# there's no encoder or limit-switch feedback here, same as
# fuyu_actuator.py with CALIBRATION_ENABLED = False. "Position"
# below is just a step count from wherever the actuator happened
# to be when this script started -- not a calibrated absolute
# position. Step 0 is used as this axis's "home"/neutral, the
# same role NEUTRAL_PULSE_US plays for the servo.
# ============================================================

ACT_STEP_PIN = 29
ACT_DIR_PIN = 31
ACT_EN_PIN = 33
# Same physical (BOARD) pin numbers as fuyu_actuator.py.

ACT_MAX_SPEED_STEPS_S = 2000
# Top cruising speed (steps/sec) when there's plenty of distance
# left to cover. Actual speed each frame is acceleration-limited
# below this -- it's never commanded instantly.

ACT_ACCELERATION_STEPS_S2 = 1500
# How fast speed is allowed to ramp up or down (steps/sec^2).
# THIS is the actual fix for the vibration and the overshoot you
# saw: the previous version jumped straight to a target step rate
# every frame with no ramp, which is exactly the failure mode
# that makes a stepper lose sync (miss steps) and vibrate/stall
# instead of turning smoothly -- and once it's missing steps, the
# software's step count stops matching reality, which is how it
# can look like it "kept on moving" past the target. This value
# is borrowed directly from fuyu_actuator.py's own
# MANUAL_ACCELERATION, since that's an already-validated safe
# ramp rate for this exact actuator.
#
# Each frame, actual speed moves toward the frame's target speed
# by at most ACT_ACCELERATION_STEPS_S2 * dt (see actuator_speed
# below) -- a proper trapezoidal motion profile: accelerate while
# far away, then decelerate at exactly the rate needed to arrive
# at zero remaining distance with zero speed (v = sqrt(2 * a * d),
# the standard stopping-distance formula), rather than the
# previous fixed-distance linear ramp, which had no actual
# relationship to how fast the actuator could safely decelerate.

ACT_GAIN_STEPS_PER_PX = 0.5
# Steps of correction per pixel of remaining error on this
# actuator's axis -- i.e. the calibration between pixels and
# steps, used to estimate how many steps are left to close the
# current gap (which the deceleration formula above needs).
# PLACEHOLDER -- unlike GAIN_US_PER_PX for the servo, this has
# never been tuned against real hardware. Tune it the same way:
# watch this axis's error overlay, lower it if it overshoots,
# raise it if it's sluggish.

ACT_SIGN = -1
# +1 or -1. Flip this if the actuator moves away from the target
# instead of toward it.

ACT_MAX_STEPS_PER_FRAME = 40
# Hard safety cap on steps issued in a single frame, independent
# of the speed profile above -- this bounds worst-case per-frame
# blocking time regardless of what speed the ramp reaches. At
# ACT_MAX_SPEED_STEPS_S, 40 steps blocks for about 2*40/2000 =
# 40ms in the worst case -- watch your achieved fps once this is
# live, the same way VIDEO_WRITE_EVERY_N was needed for the video
# encode.

ACT_RECOVERY_SPEED_STEPS_S = 400
# Fixed (non-ramped) speed used only for the laser-lost recovery
# sweep back toward home, not for normal tracking. Borrowed
# directly from fuyu_actuator.py's MANUAL_START_SPEED (500) --
# an already-validated speed this actuator can start at directly
# without ramping, kept slightly under that as a margin. Recovery
# bursts are short (capped by ACT_MAX_STEPS_PER_FRAME same as
# everything else), so the full accel/decel treatment isn't
# needed here.

ACT_SETTLE_DEADBAND_PX = SETTLE_DEADBAND_PX
ACT_RETRACK_DEADBAND_PX = RETRACK_DEADBAND_PX
# Start as copies of the servo's hysteresis thresholds. The
# actuator's real positioning resolution/backlash hasn't been
# characterized -- tune these independently once it has.

GPIO.setmode(GPIO.BOARD)

GPIO.setup(ACT_STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ACT_DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ACT_EN_PIN, GPIO.OUT, initial=GPIO.HIGH)


def act_enable():

    # TMC2209 EN is active LOW -- same convention as
    # fuyu_actuator.py's enable_motor().

    GPIO.output(ACT_EN_PIN, GPIO.LOW)


def act_disable():

    GPIO.output(ACT_EN_PIN, GPIO.HIGH)


def act_set_direction(direction):

    GPIO.output(
        ACT_DIR_PIN,
        GPIO.HIGH if direction > 0 else GPIO.LOW
    )

    time.sleep(0.00001)
    # Matches fuyu_actuator.py's DIR_SETUP_TIME -- lets the DIR
    # signal settle before the first STEP pulse.


def act_step_once(delay):

    GPIO.output(ACT_STEP_PIN, GPIO.HIGH)
    time.sleep(delay)

    GPIO.output(ACT_STEP_PIN, GPIO.LOW)
    time.sleep(delay)


def act_target_speed(remaining_steps):

    # Standard trapezoidal-motion-profile stopping-distance
    # formula: to come to a stop exactly at zero remaining
    # distance under a fixed deceleration `a`, current speed must
    # not exceed sqrt(2 * a * distance). Below this, the speed
    # ramp toward it (actuator_speed, applied per-frame where this
    # is called) naturally decelerates at the right rate instead
    # of slamming into ACT_SETTLE_DEADBAND_PX at full speed.

    stopping_speed = math.sqrt(
        2 * ACT_ACCELERATION_STEPS_S2 * max(0.0, remaining_steps)
    )

    return min(ACT_MAX_SPEED_STEPS_S, stopping_speed)


def act_delay_from_speed(speed_steps_s):

    if speed_steps_s <= 0:
        return None

    return 1.0 / (2.0 * speed_steps_s)


def act_move_steps(n_steps, direction, delay):

    # Blocking -- issues n_steps STEP pulses right away. Callers
    # keep n_steps within ACT_MAX_STEPS_PER_FRAME so this can
    # only block for one frame's worth of time at most.

    if n_steps <= 0:
        return

    act_set_direction(direction)
    act_enable()

    for _ in range(n_steps):
        act_step_once(delay)


# ============================================================
# LASER DETECTION
#
# Largest bright blob wins, no continuity gating.
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
                "[QUEUE] Sequence already running -- "
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
    global CONTROL_AXIS

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
    print(f"Actuator pins: STEP={ACT_STEP_PIN} DIR={ACT_DIR_PIN} EN={ACT_EN_PIN}")
    print()
    print("CONTROLS")
    print("  Left click  = queue a target")
    print("  G           = go -- start tracking the queued targets in order")
    print("  A           = switch servo's control axis (X <-> Y) --")
    print("                the actuator automatically follows the other axis")
    print("  R           = neutral (also clears queue)")
    print("  C           = cancel current target / sequence / queue")
    print("  Q           = quit")
    print()
    print(f"Starting axis  : servo={CONTROL_AXIS.upper()}  "
          f"actuator={('y' if CONTROL_AXIS == 'x' else 'x').upper()}")
    print(f"Log file       : {LOG_FILE}")
    print(f"Extra log file : {EXTRA_LOG_FILE}")
    print(f"Params file    : {PARAMS_LOG_FILE}")
    print()

    setup_pwm()
    act_enable()

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

    laser_last_seen = None
    # Timestamp the laser was last actually detected. Used to notice
    # when it's gone missing long enough to mean "off the edge of
    # the frame" rather than a one-frame detection blip.

    jitter_start_time = None
    jitter_base_pulse = None
    settled_since = None
    target_center_pulse = None

    settle_dwell_start = None
    # When the servo's error first entered SETTLE_DEADBAND_PX,
    # continuously. Reset to None any time the error leaves the
    # deadband. Reaching SETTLE_DWELL_S after this timestamp is
    # required (alongside the actuator also being SETTLED) before
    # the target actually counts as reached.

    actuator_position = 0
    # Step count relative to wherever the actuator happened to be
    # when this script started -- see the caveat in the LINEAR
    # ACTUATOR section above. Used as this axis's own "home".

    actuator_filtered = None
    actuator_state = "IDLE"
    # One of: IDLE, TRACKING, SETTLED. No JITTERING of its own --
    # jitter wasn't asked for on this axis and rapid direction
    # reversals are harder on a leadscrew/stepper than a hobby
    # servo, so it's deliberately left out for now. Laser-lost
    # recovery IS shared with the servo, via the existing
    # target_state == "SEARCHING" check below -- that condition
    # (is the laser visible at all) isn't axis-specific, so this
    # reads it without needing to touch how the servo sets it.

    actuator_settle_dwell_start = None
    # Same dwell-timer idea as settle_dwell_start above, but for
    # the actuator's own axis.

    actuator_speed = 0.0
    # Current commanded speed (steps/sec), persisted across frames
    # and ramped toward each frame's target speed by at most
    # ACT_ACCELERATION_STEPS_S2 * dt -- see act_target_speed().
    # Reset to 0 any time the actuator isn't actively correcting
    # (it isn't physically moving then, so there's no real speed
    # to carry over) and on a direction reversal (must pass
    # through zero speed before turning around).

    actuator_last_direction = 0
    actuator_last_update = None
    # Wall-clock time of the last actuator update, used to get a
    # real dt for the speed ramp above rather than assuming a
    # fixed frame time (your loop's fps isn't constant).

    # --------------------------------------------------------
    # Reaching a target either advances to the next queued
    # target, or settles (with the sequence-complete message)
    # if the queue is empty. Both the "jitter finished" path
    # and the "jitter disabled" path call this -- previously
    # this logic lived ONLY inside the JITTERING branch, so
    # disabling jitter meant the queue never advanced past the
    # first target.
    # --------------------------------------------------------

    def advance_or_settle():

        nonlocal filtered_x
        nonlocal settled_since

        global target_x
        global target_y
        global target_state
        global sequence_running

        if sequence_running and pending_targets:

            target_x, target_y = pending_targets.pop(0)
            target_state = "IDLE"
            filtered_x = None

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
                print("[SEQUENCE] Complete -- all targets reached")

    try:

        while True:

            frame_number += 1

            ret, frame = cap.read()

            if not ret:
                print("[CAMERA] Failed to read frame")
                continue

            # ------------------------------------------------
            # Laser detection
            # ------------------------------------------------

            (
                laser_x, laser_y, blob_area, mask, num_contours
            ) = detect_laser(frame)

            # ------------------------------------------------
            # Control
            # ------------------------------------------------

            error_x = None
            correction = None
            jitter_offset = None

            if target_x is not None:

                if laser_x is not None:
                    laser_last_seen = time.monotonic()

                if (
                    target_state != "SEARCHING"
                    and laser_last_seen is not None
                    and time.monotonic() - laser_last_seen >= LASER_LOST_GRACE_S
                ):
                    target_state = "SEARCHING"
                    print()
                    print("[RECOVERY] Laser lost -- sweeping back toward frame center")

                if target_state == "SEARCHING":

                    if laser_x is not None:
                        # Found it again -- resume normal tracking
                        # from here, wherever "here" turned out to be.
                        target_state = "TRACKING"
                        filtered_x = None
                        settle_dwell_start = None
                    else:
                        recovery_target = (
                            SERVO_VISIBLE_MIN_PULSE_US
                            + SERVO_VISIBLE_MAX_PULSE_US
                        ) / 2
                        # Sweeps toward the middle of the calibrated
                        # VISIBLE range, not NEUTRAL_PULSE_US -- the
                        # servo's mechanical rest position has no
                        # guaranteed relationship to where the camera
                        # is actually pointed, and sweeping toward a
                        # wrong assumed "center" here was the likely
                        # cause of recovery driving the laser further
                        # out of frame instead of back into it.
                        if pulse_us > recovery_target:
                            pulse_us = max(
                                recovery_target, pulse_us - RECOVERY_STEP_US
                            )
                        elif pulse_us < recovery_target:
                            pulse_us = min(
                                recovery_target, pulse_us + RECOVERY_STEP_US
                            )
                        set_pulse_us(pulse_us)

                if target_state == "IDLE" and laser_x is not None:
                    target_state = "TRACKING"

                # TRACKING always re-evaluates. SETTLED also
                # re-evaluates, but only once SETTLE_GRACE_PERIOD_S
                # has passed since settling -- see the constant's
                # comment for why the grace period exists.

                settled_grace_elapsed = (
                    target_state != "SETTLED"
                    or settled_since is None
                    or time.monotonic() - settled_since >= SETTLE_GRACE_PERIOD_S
                )

                can_update = (
                    target_state in ("TRACKING", "SETTLED")
                    and settled_grace_elapsed
                    and laser_x is not None
                    and time.monotonic() - last_servo_update >= SERVO_UPDATE_INTERVAL
                )

                if can_update:

                    # Which coordinate of the laser/target this
                    # single servo is currently following.
                    laser_pos = laser_x if CONTROL_AXIS == "x" else laser_y
                    target_pos = target_x if CONTROL_AXIS == "x" else target_y

                    filtered_x = (
                        laser_pos if filtered_x is None
                        else ALPHA * laser_pos + (1 - ALPHA) * filtered_x
                    )

                    error_x = target_pos - filtered_x

                    if target_state == "SETTLED":

                        if abs(error_x) > RETRACK_DEADBAND_PX:
                            # Drifted far enough to resume tracking.
                            target_state = "TRACKING"
                        # else: still within the looser retrack band --
                        # hold still, no correction below.

                    if target_state == "TRACKING":

                        if abs(error_x) <= SETTLE_DEADBAND_PX:

                            tracking = False

                            # One last small correction to close the
                            # remaining (sub-deadband) error before
                            # freezing/jittering, so the jitter/settle
                            # center is the best estimate of the true
                            # target pulse -- not just wherever the
                            # P-controller happened to stop inside the
                            # deadband.
                            fine_correction = max(
                                -MAX_STEP_US,
                                min(
                                    MAX_STEP_US,
                                    error_x * AXIS_GAIN_US_PER_PX[CONTROL_AXIS]
                                )
                            )
                            pulse_us += AXIS_SIGN[CONTROL_AXIS] * fine_correction
                            pulse_us = max(
                                SERVO_VISIBLE_MIN_PULSE_US,
                                min(SERVO_VISIBLE_MAX_PULSE_US, pulse_us)
                            )
                            set_pulse_us(pulse_us)

                            if settle_dwell_start is None:
                                settle_dwell_start = time.monotonic()

                            dwelled_long_enough = (
                                time.monotonic() - settle_dwell_start
                                >= SETTLE_DWELL_S
                            )

                            # Both conditions required: this axis has
                            # sat inside the deadband continuously for
                            # SETTLE_DWELL_S (not just touched it on
                            # one frame while passing through), AND
                            # the actuator's axis has independently
                            # reached its own SETTLED state. Until
                            # both are true, hold here (already
                            # fine-nudged above) and re-check next
                            # frame -- don't jitter/advance yet.

                            if dwelled_long_enough and actuator_state == "SETTLED":

                                settle_dwell_start = None

                                if JITTER_ENABLE:
                                    target_state = "JITTERING"
                                    jitter_start_time = time.monotonic()
                                    jitter_base_pulse = pulse_us
                                else:
                                    advance_or_settle()

                        else:

                            tracking = True
                            settle_dwell_start = None

                            desired = error_x * AXIS_GAIN_US_PER_PX[CONTROL_AXIS]

                            correction = max(
                                -MAX_STEP_US,
                                min(MAX_STEP_US, desired)
                            )

                            pulse_us += AXIS_SIGN[CONTROL_AXIS] * correction
                            pulse_us = max(
                                SERVO_VISIBLE_MIN_PULSE_US,
                                min(SERVO_VISIBLE_MAX_PULSE_US, pulse_us)
                            )

                            set_pulse_us(pulse_us)

                    last_servo_update = time.monotonic()

                elif target_state == "JITTERING":

                    # Jitter is a deliberate motion, independent of
                    # whether the laser happens to be detected this
                    # exact frame -- it doesn't need laser_x.

                    elapsed = time.monotonic() - jitter_start_time

                    if elapsed >= JITTER_DURATION_S:

                        pulse_us = jitter_base_pulse
                        set_pulse_us(pulse_us)

                        advance_or_settle()

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
                            SERVO_VISIBLE_MIN_PULSE_US,
                            min(SERVO_VISIBLE_MAX_PULSE_US, jitter_pulse)
                        )

                        set_pulse_us(jitter_pulse)
                        pulse_us = jitter_pulse

                # SETTLED (within the retrack band): nothing to do --
                # servo holds still, no re-actuation, no oscillation.

            # ------------------------------------------------
            # Actuator control (second DOF)
            #
            # Fully independent of the servo block above -- own
            # state variable, own error, own deadbands. Drives
            # whichever axis CONTROL_AXIS is NOT currently on.
            # ------------------------------------------------

            actuator_error = None
            actuator_correction = None

            if target_x is not None:

                actuator_axis = "y" if CONTROL_AXIS == "x" else "x"
                actuator_laser_pos = laser_x if actuator_axis == "x" else laser_y
                actuator_target_pos = target_x if actuator_axis == "x" else target_y

                if target_state == "SEARCHING":

                    # Shares the servo's laser-lost trigger (that
                    # condition is "is the laser visible at all",
                    # not axis-specific) -- sweep this axis back
                    # toward its own home (step 0) too, at a fixed
                    # moderate speed (no target to ramp toward, and
                    # bursts here are short anyway).

                    if actuator_position > 0:
                        step_this_frame = min(
                            ACT_MAX_STEPS_PER_FRAME, actuator_position
                        )
                        act_move_steps(
                            step_this_frame, -1,
                            act_delay_from_speed(ACT_RECOVERY_SPEED_STEPS_S)
                        )
                        actuator_position -= step_this_frame

                    elif actuator_position < 0:
                        step_this_frame = min(
                            ACT_MAX_STEPS_PER_FRAME, -actuator_position
                        )
                        act_move_steps(
                            step_this_frame, +1,
                            act_delay_from_speed(ACT_RECOVERY_SPEED_STEPS_S)
                        )
                        actuator_position += step_this_frame

                    actuator_state = "IDLE"
                    actuator_filtered = None
                    actuator_settle_dwell_start = None
                    actuator_speed = 0.0
                    actuator_last_direction = 0
                    actuator_last_update = None

                elif actuator_laser_pos is not None:

                    if actuator_state == "IDLE":
                        actuator_state = "TRACKING"

                    actuator_filtered = (
                        actuator_laser_pos if actuator_filtered is None
                        else ALPHA * actuator_laser_pos
                        + (1 - ALPHA) * actuator_filtered
                    )

                    actuator_error = actuator_target_pos - actuator_filtered

                    if actuator_state == "SETTLED":

                        if abs(actuator_error) > ACT_RETRACK_DEADBAND_PX:
                            actuator_state = "TRACKING"
                            actuator_settle_dwell_start = None

                    if actuator_state == "TRACKING":

                        if abs(actuator_error) <= ACT_SETTLE_DEADBAND_PX:

                            # Same dwell idea as the servo: don't
                            # mark this axis SETTLED off a single
                            # lucky frame -- require it to sit inside
                            # the deadband continuously for
                            # SETTLE_DWELL_S first. Issue no further
                            # steps while waiting; it's already
                            # inside the deadband. Not physically
                            # moving, so there's no speed to carry.

                            actuator_speed = 0.0
                            actuator_last_direction = 0
                            actuator_last_update = None

                            if actuator_settle_dwell_start is None:
                                actuator_settle_dwell_start = time.monotonic()

                            if (
                                time.monotonic() - actuator_settle_dwell_start
                                >= SETTLE_DWELL_S
                            ):
                                actuator_state = "SETTLED"
                                actuator_settle_dwell_start = None

                        else:

                            actuator_settle_dwell_start = None

                            desired_steps = (
                                actuator_error * ACT_GAIN_STEPS_PER_PX
                            )

                            actuator_correction = max(
                                -ACT_MAX_STEPS_PER_FRAME,
                                min(ACT_MAX_STEPS_PER_FRAME, desired_steps)
                            )

                            steps_this_frame_target = int(round(
                                ACT_SIGN * actuator_correction
                            ))

                            if steps_this_frame_target != 0:

                                direction = (
                                    1 if steps_this_frame_target > 0 else -1
                                )

                                # Must pass through zero speed before
                                # reversing -- can't be moving both
                                # ways at once, and an instant
                                # direction flip at speed is exactly
                                # the kind of shock that skips steps.
                                if (
                                    actuator_last_direction != 0
                                    and direction != actuator_last_direction
                                ):
                                    actuator_speed = 0.0

                                now = time.monotonic()
                                dt = (
                                    now - actuator_last_update
                                    if actuator_last_update is not None
                                    else 0.0
                                )
                                actuator_last_update = now
                                actuator_last_direction = direction

                                remaining_steps = (
                                    abs(actuator_error) * ACT_GAIN_STEPS_PER_PX
                                )
                                target_speed = act_target_speed(remaining_steps)

                                max_change = ACT_ACCELERATION_STEPS_S2 * dt
                                speed_diff = target_speed - actuator_speed

                                if abs(speed_diff) <= max_change:
                                    actuator_speed = target_speed
                                else:
                                    actuator_speed += math.copysign(
                                        max_change, speed_diff
                                    )

                                # Never exceed what the ramped speed
                                # allows in this frame's dt, and never
                                # exceed the estimated remaining gap
                                # (steps_this_frame_target) either --
                                # whichever is smaller wins.
                                steps_this_frame = min(
                                    abs(steps_this_frame_target),
                                    max(0, int(round(actuator_speed * dt)))
                                )

                                step_delay = act_delay_from_speed(
                                    actuator_speed
                                )

                                if steps_this_frame > 0 and step_delay is not None:

                                    act_move_steps(
                                        steps_this_frame, direction, step_delay
                                    )

                                    actuator_position += (
                                        direction * steps_this_frame
                                    )

            # ------------------------------------------------
            # Output image
            # ------------------------------------------------

            output = frame.copy()

            draw_laser(output, laser_x, laser_y)
            draw_target(output, target_x, target_y, laser_x)
            draw_queue(output, pending_targets)

            if target_state == "SEARCHING":
                status = "LASER LOST -- RECOVERING TOWARD FRAME CENTER"
                status_color = (0, 0, 255)
            elif laser_x is None:
                status = "LASER NOT DETECTED"
                status_color = (0, 0, 255)
            elif target_x is None and pending_targets:
                status = f"{len(pending_targets)} QUEUED -- PRESS G TO GO"
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
                    output, f"{CONTROL_AXIS.upper()} Error: {error_x:+.1f} px", (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

            cv2.putText(
                output, f"Axis: {CONTROL_AXIS.upper()}  (press A to switch)", (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2
            )

            if target_x is not None:
                actuator_axis_label = "y" if CONTROL_AXIS == "x" else "x"
                act_status = (
                    f"Actuator ({actuator_axis_label.upper()}): {actuator_state}"
                    + (
                        f"  err={actuator_error:+.1f}px"
                        if actuator_error is not None else ""
                    )
                    + f"  pos={actuator_position}"
                )
                cv2.putText(
                    output, act_status, (30, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2
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

            if video_writer is not None and frame_number % VIDEO_WRITE_EVERY_N == 0:
                combined = np.hstack((output, frame))
                video_writer.write(combined)

            # ------------------------------------------------
            # Logging (original -- untouched)
            # ------------------------------------------------

            log_data(
                frame_number, target_x, target_y, laser_x, laser_y,
                error_x, pulse_us, blob_area, num_contours, tracking
            )

            # ------------------------------------------------
            # Logging (extra diagnostic log)
            # ------------------------------------------------

            log_extra_data({
                "frame": frame_number,
                "status_text": status,
                "target_state": target_state,
                "control_axis": CONTROL_AXIS,
                "sequence_running": sequence_running,
                "queue_remaining": len(pending_targets),
                "laser_x": laser_x,
                "laser_y": laser_y,
                "filtered_x": filtered_x,
                "error_x": error_x,
                "correction_us": correction,
                "pulse_us": pulse_us,
                "jitter_offset_us": jitter_offset,
                "actuator_axis": "y" if CONTROL_AXIS == "x" else "x",
                "actuator_state": actuator_state,
                "actuator_error": actuator_error,
                "actuator_position": actuator_position,
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
                    settle_dwell_start = None

                    print()
                    print(
                        f"[SEQUENCE] Starting -- "
                        f"x={target_x}, y={target_y} "
                        f"({len(pending_targets)} remaining)"
                    )

                elif sequence_running:

                    print()
                    print("[SEQUENCE] Already running")

                else:

                    print()
                    print("[SEQUENCE] No queued targets -- click to add some first")

            elif key == ord("a"):

                CONTROL_AXIS = "y" if CONTROL_AXIS == "x" else "x"

                # Reset in-flight control state only -- target_x/
                # target_y, pending_targets, and sequence_running are
                # left alone, since every queued point already has
                # both coordinates and stays valid under either axis.
                filtered_x = None
                error_x = None
                settle_dwell_start = None

                if target_x is not None:
                    target_state = "IDLE"
                    settled_since = None

                print()
                print(f"[AXIS] Switched control axis to {CONTROL_AXIS.upper()}")

            elif key == ord("r"):

                pulse_us = float(NEUTRAL_PULSE_US)
                set_pulse_us(pulse_us)

                target_x = None
                target_y = None
                target_state = "IDLE"
                tracking = False
                filtered_x = None
                settle_dwell_start = None
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
                filtered_x = None
                settle_dwell_start = None
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

        # Deliberately NOT calling stop_pwm() here. Disabling the PWM
        # channel cuts the signal entirely, and most servos only hold
        # a position while they're actively receiving pulses -- once
        # the signal stops, they go limp and can drift away from
        # neutral under their own mechanical load (gravity, cable
        # drag, an off-balance mount, etc). Leaving the channel
        # enabled at NEUTRAL_PULSE_US keeps it actively parked there
        # after the script exits. Call stop_pwm() manually if you
        # ever want to fully power down the channel instead.
        # (A stop_pwm() call had crept back in here -- that's exactly
        # what caused the "not returning to neutral" bug the first
        # time around, so it's been removed again.)

        # Actuator: disabling it here is the OPPOSITE choice from the
        # servo above, on purpose. fuyu_actuator.py itself disables
        # the motor on every exit path, including a clean quit -- and
        # with no limit switches or calibration configured on this
        # axis, holding current indefinitely on an unsupervised,
        # unhomed stepper is a real thermal/mechanical risk that
        # doesn't have a good equivalent on the hobby servo side. The
        # tradeoff: this axis can drift once disabled, same failure
        # mode as the servo bug, just accepted deliberately here. If
        # you'd rather it hold position like the servo does, remove
        # the act_disable() call below -- but note GPIO.cleanup()
        # resets pin state on exit regardless, so holding position
        # reliably after the script exits may need more than that;
        # test it before relying on it.
        act_disable()
        GPIO.cleanup()

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