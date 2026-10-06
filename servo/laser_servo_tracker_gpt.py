#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import csv
import math
import random
import json
import threading
import Jetson.GPIO as GPIO
from datetime import datetime

from st3215 import ST3215
# pip install st3215
# https://github.com/Mickael-Roger/python-st3215 -- verified working
# against this exact ST3215 servo + Waveshare driver board setup.


# ============================================================
# CAMERA
# ============================================================

CAMERA_DEVICE = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 60


# ============================================================
# ST3215 BUS SERVO (via Waveshare ESP32 Servo Driver board)
#
# Setup required ONCE per boot before this script will talk to it:
#   1. On the driver board's own screen, select "Start Serial
#      Forwarding" from its menu. The board is normally in its own
#      WiFi/JSON control mode; this on-screen action (no WiFi/web
#      app needed) switches its USB Type-C port into a transparent
#      bridge to the servo bus. It stays in this mode until you
#      switch it back or power-cycle the board -- unless you've
#      reflashed it with SERIAL_FORWARDING=true in ServoDriverST.ino
#      so it boots directly into this mode every time.
#   2. Plug that Type-C port into the Jetson via USB. It should show
#      up as /dev/ttyUSB0 or /dev/ttyACM0 -- check with `ls /dev/tty*`
#      before and after plugging in.
#
# Positions are 0-4095 over the servo's full 360 degrees (2047/2048 =
# middle). This servo ramps its own speed/acceleration onboard once
# given a target position, speed, and acceleration -- so there is no
# need to hand-compute a smooth approach here. We just keep sending
# it our best current estimate of the right position.
# ============================================================

ST_SERIAL_PORT = "/dev/ttyUSB0"
# Check `ls /dev/ttyUSB* /dev/ttyACM*` on the Jetson right after
# plugging in the driver board's USB cable to confirm this.

ST_BAUDRATE = 115200
# This is the driver board's Serial Forwarding mode baud rate -- NOT
# the servo bus's own native 1,000,000 baud (the board handles that
# internally). The st3215 library's ST3215() constructor doesn't
# expose a baud-rate argument and opens at 1,000,000 by default, so
# setup_st_servo() below reaches into its portHandler.ser (the
# underlying pyserial object) and overrides it to this value -- this
# was confirmed necessary and working on this exact hardware.

ST_SERVO_ID = 1
# Factory default ID. Only matters/needs changing if you have more
# than one ST-series servo on the same bus.

ST_MIN_POSITION = 1981
ST_MAX_POSITION = 2541
# ST_MIDDLE_POSITION = 2047
ST_MIDDLE_POSITION = 2150

ST_VISIBLE_MIN_POSITION = ST_MIN_POSITION
ST_VISIBLE_MAX_POSITION = ST_MAX_POSITION
# Same idea as the old SERVO_VISIBLE_MIN/MAX_PULSE_US -- the position
# range within which the laser is ACTUALLY visible in the camera's
# frame, not the servo's full mechanical range. Defaults to the full
# range until you calibrate it the same way as before: jog toward
# each edge, read the position off the overlay, note where the laser
# exits frame.

ST_CRUISE_SPEED = 1000
# Speed (steps/sec) sent with every position command. Max for this
# servo family is ~3073 steps/sec. This is a moderate starting point
# -- the servo ramps up/down to and from this speed on its own via
# ST_ACCEL below, so raising this mainly affects top cruising speed
# on long moves, not smoothness.

ST_ACCEL = 150
# Servo's onboard start/stop acceleration, 0-254 (larger = snappier
# ramp). Moderate starting point -- raise it if tracking feels
# sluggish to start moving, lower it if motion looks jerky at the
# start/end of a move.

ST_JITTER_AMPLITUDE_POS = 1500
# Jitter amplitude in position units. This is a deliberately modest
# starting amplitude for a fast back-and-forth motion; 1000 units was
# a very large excursion to ask the servo to reproduce at high frequency.


def jitter_thread_loop():
    global _jitter_active

    period = 1.0 / max(1.0, JITTER_COMMAND_HZ)
    next_tick = time.monotonic()

    while _jitter_running:
        now = time.monotonic()

        if not _jitter_active:
            time.sleep(0.005)
            next_tick = time.monotonic()
            continue

        with _servo_command_lock:
            base = _jitter_base_position
            start = _jitter_start_time

        if base is None or start is None:
            time.sleep(0.005)
            continue

        elapsed = now - start
        sine_component = ST_JITTER_AMPLITUDE_POS * math.sin(
            2.0 * math.pi * JITTER_FREQUENCY_HZ * elapsed
        )
        noise_component = (
            ST_JITTER_AMPLITUDE_POS * random.uniform(-1, 1)
        )
        offset = (
            (1.0 - JITTER_RANDOMNESS) * sine_component
            + JITTER_RANDOMNESS * noise_component
        )

        jitter_pos = base + offset
        jitter_pos = max(
            ST_VISIBLE_MIN_POSITION,
            min(ST_VISIBLE_MAX_POSITION, jitter_pos)
        )

        set_servo_position(
            jitter_pos,
            speed=ST_JITTER_SPEED,
            acc=ST_JITTER_ACCEL
        )

        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()


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

CONTROL_AXIS = "y"

AXIS_GAIN_POS_PER_PX = {
    "x": 1,
    "y": 1.0,
}

AXIS_SIGN = {
    "x": -1,
    "y": -1,
}

MAX_POSITION_STEP_PER_UPDATE = 300
# Trust-region clamp on how far the commanded target position can
# move in one control update (out of the 0-4095 range) -- same
# anti-overshoot role MAX_STEP_US played before. The servo's own
# speed limit (ST_CRUISE_SPEED) bounds physical speed regardless;
# this bounds how aggressively our estimate itself jumps.

SERVO_SPEED_FAR = 3000
# Commanded speed (step/s) for large errors -- near this servo
# family's ceiling (~3073 step/s per the library's own notes).
SERVO_SPEED_NEAR = 200
# Commanded speed (step/s) once close to the target -- gentle final
# approach instead of slamming in at full speed and relying only on
# the servo's onboard accel/decel ramp to soften it.
SERVO_SPEED_EASE_PX = 60
# Pixel-error distance at which speed starts scaling down from
# SERVO_SPEED_FAR toward SERVO_SPEED_NEAR. Linear in between.

JOG_STEP_SERVO = 20
# Position units moved per I/K keypress while in jog mode.
JOG_SPEED_SERVO = 200
# Commanded speed (step/s) for jog moves -- deliberately gentle/slow
# since jog mode is manual, closely-watched positioning.
JOG_SPEED_ACTUATOR = 800
# Commanded speed (step/s) for U/O actuator jog pulses.
ACTUATOR_JOG_PULSE_S = 0.15
# Duration of each U/O jog pulse. The actuator is speed-driven, not
# position-driven, and cv2.waitKey only gives discrete keypress
# events (no "key held" state) -- so each tap issues a short timed
# speed pulse rather than a direct position command.
JOG_OVERTRAVEL_ACTUATOR_STEPS = 20000
# How far beyond the CURRENTLY SAVED actuator calibration jog mode is
# allowed to explore, in memory only, while actively re-calibrating.
# See the jog-mode entry/exit handling for why this exists: without
# it, you could only ever narrow a saved range, never widen one,
# since jogging itself was clamped by the same bounds it was trying
# to move past. The servo doesn't need an equivalent constant -- it
# has a real absolute backstop (ST_MIN/MAX_POSITION, 0-4095) that jog
# mode can safely open all the way up to; the actuator has no such
# known hard stop, so this is a deliberately generous but still
# bounded allowance rather than fully unbounded travel. Jog mode is
# manually operated and watched -- you are the safety mechanism
# during calibration, same as before.


def scaled_servo_speed(error_px):

    ease = min(1.0, abs(error_px) / SERVO_SPEED_EASE_PX)
    return int(SERVO_SPEED_NEAR + ease * (SERVO_SPEED_FAR - SERVO_SPEED_NEAR))

SERVO_UPDATE_INTERVAL = 0.01
# Minimum time between servo command updates (~33 Hz). Previously 0.0
# ("every frame") -- at 30-60fps that meant a brand-new MoveTo() command
# was issued before the servo had any chance to finish or settle into
# its previous move, constantly interrupting and re-planning its
# internal speed/accel ramp. That's what produced the visible jerky,
# abrupt motion. Throttling updates gives each move enough time to
# actually execute smoothly before the next correction arrives.

ALPHA = 0.4
# EMA smoothing factor on the laser position used for error calc.

SETTLE_DEADBAND_PX = 5
RETRACK_DEADBAND_PX = 15
# Same hysteresis idea as before: SETTLE to enter "reached", the
# looser RETRACK before resuming active correction.

SETTLE_DWELL_S = 0.3
# Error must sit inside SETTLE_DEADBAND_PX continuously this long
# before actually counting as reached -- guards against a fast pass
# through the deadband being falsely counted. Shared by both axes.

LASER_LOST_GRACE_S = 0.3
RECOVERY_STEP_POS = 20
# Laser-lost recovery: if undetected this long, sweep the servo back
# toward the middle of the VISIBLE range (not necessarily
# ST_MIDDLE_POSITION -- see ST_VISIBLE_MIN/MAX_POSITION above),
# RECOVERY_STEP_POS position-units per update, until it reappears.


# ============================================================
# JITTER (servo axis only)
# ============================================================

JITTER_ENABLE = True

JITTER_DURATION_S = 3.0
JITTER_FREQUENCY_HZ = 20.0
# Mechanical jitter frequency. 50 Hz was too high for the previous
# camera-loop implementation because commands were only sent at camera FPS.
# Jitter is now generated by its own background command loop.

JITTER_COMMAND_HZ = 200.0
# Servo target-command rate, independent of camera FPS.

JITTER_RANDOMNESS = 0.0

ST_JITTER_SPEED = 2600
ST_JITTER_ACCEL = 200
# The jitter motion itself is deliberately small and fast.  Increase
# amplitude/frequency only after confirming the servo is physically
# following the waveform.

SETTLE_GRACE_PERIOD_S = 0.4
# After entering SETTLED, drift-recovery is paused this long before
# re-checking real error -- avoids residual jitter motion reading as
# "still off target" and immediately re-triggering.


# ============================================================
# LINEAR ACTUATOR (second DOF) -- fuyu hardware unchanged
#
# Only the low-level step/dir/enable control is borrowed from
# fuyu_actuator.py, same as before. What's NEW here: step pulses are
# now generated by a dedicated background thread (actuator_thread_
# loop), not bursted from inside the camera loop. That's the actual
# fix for the visible jerking -- the previous version's speed math
# was fine, but bursts of steps synchronized to camera frames (every
# ~65-85ms) meant real dead time between bursts, which looks and
# feels jerky no matter how smooth the commanded speed curve is. The
# thread paces itself in real time, independent of camera fps, so
# the accel/decel ramp is actually continuous.
# ============================================================

ACT_STEP_PIN = 29
ACT_DIR_PIN = 31
ACT_EN_PIN = 33

ACT_MAX_SPEED_STEPS_S = 5000
# Fast cruising speed for the stationary tests. The actual ceiling will
# depend on the Jetson GPIO pulse-generation overhead.

ACT_ACCELERATION_STEPS_S2 = 5000
# High enough to reach cruise speed quickly, while still giving a smooth
# trapezoidal accelerate/decelerate profile.

ACT_GAIN_STEPS_PER_PX = 0.7
# Starting point for incremental position control. Tune from the error
# response after the direction is verified.

ACT_MIN_TRACKING_SPEED_STEPS_S = 0
# IMPORTANT: do not force a minimum speed for tiny residual errors.
# That was causing motion even when the actuator pixel error was only a
# few pixels and was making the axis hunt between its limits.

ACT_SIGN = -1
# Keep the currently reported correct actuator direction.

ACT_RECOVERY_SPEED_STEPS_S = 1000
# Only used for laser-lost recovery. Normal tracking and sweep use the
# trapezoidal profile below.

ACT_JOG_SPEED_STEPS_S = 1400

ACT_SETTLE_DEADBAND_PX = 10
ACT_RETRACK_DEADBAND_PX = 18

ACT_MAX_TARGET_DELTA_PER_UPDATE = 800
# Maximum actuator-position change requested by one vision update.

ACT_ENDPOINT_SLOWDOWN_STEPS = 1200
# The actuator starts reducing speed as it approaches either calibrated
# endpoint, so it slows before the software hard stop instead of arriving
# at full speed.

SWEEP_SERVO_SPEED = 2400
SWEEP_SERVO_ACCEL = 120
SWEEP_SERVO_TOLERANCE = 8
SWEEP_ACTUATOR_TOLERANCE = 5

ACT_UNREFERENCED_JOG_LIMIT = 20000
# Safety envelope while manually referencing/calibrating before a saved
# travel range is available. Jogging remains a manually supervised action.

ACTUATOR_CALIBRATION_EXTENSION_LIMIT = 30000
# Absolute software envelope while explicitly extending calibration.
# This is NOT a physical safe-travel guarantee; the mechanism must be
# watched by the operator during extension.


# ============================================================
# WORLD-FRAME TARGET LOCK (odometry-driven feedforward)
#
# Lets the laser keep pointing at a fixed GROUND point while the vehicle
# moves, instead of chasing a pixel. See the accompanying design notes for
# the full reasoning -- short version: at the moment a pixel-target is
# SETTLED, the servo/actuator's own state already encodes the correct aim
# direction (the laser is sitting on it), so locking ray-casts THAT
# direction, from the current vehicle pose, onto an assumed flat ground
# plane, to get a world-frame (x, y). From then on, every frame, the
# feedforward math re-aims at that same world point using the vehicle's
# CURRENT pose. This does not use the camera at all once locked -- it has
# to work even when the target is out of frame.
#
# No vision trim is blended on top yet (see design notes) -- this is pure
# feedforward. Pixel error is still computed and shown on screen for you
# to judge how much drift there is before adding a trim.
# ============================================================

VEHICLE_POSE_FILE = "vehicle_pose.txt"
# Whatever process knows the vehicle's current pose writes a single line
# "x y yaw_rad unix_timestamp" to this file, overwritten each update. For
# now, with no ROS running, nothing writes it -- get_vehicle_pose() then
# returns None and world-lock mode safely holds its last commanded aim
# rather than guess. The standalone ROS2 node provided separately writes
# to this same file from /odom -- no changes needed here when that's
# wired up, just point ODOM_TOPIC's VEHICLE_POSE_FILE at the same path.

VEHICLE_POSE_MAX_AGE_S = 1.0
# Pose older than this is treated as unavailable. Stale odometry driving
# the aim is worse than no odometry (silently wrong vs. visibly frozen).

MOUNT_HEIGHT_M = 1.2
# PLACEHOLDER -- MEASURE THIS. Height of the servo/actuator mount above
# the ground, straight down. Used to convert an aim direction into a
# ground range via basic trigonometry (this whole model assumes flat,
# locally-planar ground at this fixed height below the mount).

MOUNT_FORWARD_OFFSET_M = 0.0
MOUNT_LATERAL_OFFSET_M = 0.0
# PLACEHOLDER -- measure/estimate: the mount's position relative to
# wherever the vehicle's odometry frame is referenced (base_link),
# forward (+x) and left (+y) in the vehicle's own frame.

MOUNT_YAW_OFFSET_DEG = 0.0
# PLACEHOLDER -- if the mount faces anything other than straight ahead
# relative to the vehicle's forward axis, that offset, in degrees.

AZIMUTH_DEG_PER_SERVO_UNIT = 360.0 / 4096.0
# Exactly right ONLY if the servo's rotation axis is vertical and directly
# drives the pan axis with no linkage/gearing in between. If there's any
# mechanical reduction, measure this directly instead of assuming it.

ELEVATION_HOME_DEG = 45.0
ELEVATION_DEG_PER_ACTUATOR_STEP = 0.01
# PLACEHOLDER, and the shakiest calibration in this whole model. Assumes
# the linear actuator drives elevation through roughly a CONSTANT
# degrees-per-step relationship, which only holds if the mechanical
# linkage (leadscrew into whatever tilts the mount) is close to linear
# over its travel. If it drives a bellcrank/lever arrangement, this is
# almost certainly nonlinear, especially near the ends of travel. Worth
# measuring a handful of (actuator_position, actual elevation angle)
# points by hand before trusting this beyond a rough first pass.

ACTUATOR_MIN_POSITION = -5000
ACTUATOR_MAX_POSITION = 5000
# PLACEHOLDER SAFETY LIMIT -- the actuator has real mechanical travel
# limits that NOTHING in this script has ever enforced before now. Pixel-
# based control stayed safe by construction (bounded by what was visible
# in frame); a computed elevation angle has no such natural bound. These
# numbers are not measured -- get the actuator's real safe step range
# (or wire up limit switches) before trusting world-lock to drive it
# unattended.


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = "servo_logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = os.path.join(LOG_DIR, f"laser_servo_{timestamp}.csv")
VIDEO_FILE = os.path.join(LOG_DIR, f"laser_servo_{timestamp}.mp4")

video_writer = None

VIDEO_WRITE_EVERY_N = 1
# Write every Nth frame to the output video. Set to 3-5 to test
# whether the mp4v encode is costing you fps (it was, previously).


def setup_video_writer(width, height, fps):

    global video_writer

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    combined_width = width * 2

    video_writer = cv2.VideoWriter(
        VIDEO_FILE, fourcc, fps, (combined_width, height)
    )

    if not video_writer.isOpened():
        print(f"[VIDEO] Failed to open writer for {VIDEO_FILE}")
        video_writer = None


log_file = None
csv_writer = None


def setup_logging():

    global log_file, csv_writer

    log_file = open(LOG_FILE, "w", newline="")
    csv_writer = csv.writer(log_file)

    csv_writer.writerow([
        "timestamp", "frame", "target_x", "target_y",
        "laser_x", "laser_y", "error_x", "servo_position",
        "blob_area", "num_contours", "tracking"
    ])
    log_file.flush()


def log_data(
    frame_number, target_x, target_y, laser_x, laser_y,
    error_x, servo_position, blob_area, num_contours, tracking
):
    if csv_writer is None:
        return

    csv_writer.writerow([
        time.time(), frame_number, target_x, target_y,
        laser_x, laser_y, error_x, servo_position,
        blob_area, num_contours, tracking
    ])

    if frame_number % 30 == 0:
        log_file.flush()


EXTRA_LOG_FILE = os.path.join(LOG_DIR, f"laser_servo_{timestamp}_extra.csv")
extra_log_file = None
extra_csv_writer = None

EXTRA_LOG_FIELDS = [
    "timestamp", "frame", "status_text", "target_state", "control_axis",
    "sequence_running", "queue_remaining", "laser_x", "laser_y",
    "filtered_x", "error_x", "servo_position", "actuator_axis",
    "actuator_state", "actuator_error", "actuator_position",
    "actuator_speed", "world_lock_active", "world_lock_x", "world_lock_y",
    "vehicle_pose_x", "vehicle_pose_y", "vehicle_pose_yaw",
]


def setup_extra_logging():

    global extra_log_file, extra_csv_writer

    extra_log_file = open(EXTRA_LOG_FILE, "w", newline="")
    extra_csv_writer = csv.writer(extra_log_file)
    extra_csv_writer.writerow(EXTRA_LOG_FIELDS)
    extra_log_file.flush()


def log_extra_data(values):

    if extra_csv_writer is None:
        return

    row = [time.time()] + [values.get(field) for field in EXTRA_LOG_FIELDS[1:]]
    extra_csv_writer.writerow(row)

    if values.get("frame", 0) % 30 == 0:
        extra_log_file.flush()


PARAMS_LOG_FILE = os.path.join(LOG_DIR, f"laser_servo_{timestamp}_params.csv")


def get_tunable_params():

    return {
        "BRIGHTNESS_MIN": BRIGHTNESS_MIN,
        "BRIGHTNESS_MAX": BRIGHTNESS_MAX,
        "MORPH_KERNEL_SIZE": MORPH_KERNEL_SIZE,
        "MIN_BLOB_AREA": MIN_BLOB_AREA,
        "SERVO_UPDATE_INTERVAL": SERVO_UPDATE_INTERVAL,
        "ALPHA": ALPHA,
        "CONTROL_AXIS": CONTROL_AXIS,
        "AXIS_GAIN_POS_PER_PX_X": AXIS_GAIN_POS_PER_PX["x"],
        "AXIS_GAIN_POS_PER_PX_Y": AXIS_GAIN_POS_PER_PX["y"],
        "AXIS_SIGN_X": AXIS_SIGN["x"],
        "AXIS_SIGN_Y": AXIS_SIGN["y"],
        "MAX_POSITION_STEP_PER_UPDATE": MAX_POSITION_STEP_PER_UPDATE,
        "SERVO_SPEED_FAR": SERVO_SPEED_FAR,
        "SERVO_SPEED_NEAR": SERVO_SPEED_NEAR,
        "SERVO_SPEED_EASE_PX": SERVO_SPEED_EASE_PX,
        "SETTLE_DEADBAND_PX": SETTLE_DEADBAND_PX,
        "RETRACK_DEADBAND_PX": RETRACK_DEADBAND_PX,
        "SETTLE_DWELL_S": SETTLE_DWELL_S,
        "LASER_LOST_GRACE_S": LASER_LOST_GRACE_S,
        "RECOVERY_STEP_POS": RECOVERY_STEP_POS,
        "JITTER_ENABLE": JITTER_ENABLE,
        "JITTER_DURATION_S": JITTER_DURATION_S,
        "JITTER_FREQUENCY_HZ": JITTER_FREQUENCY_HZ,
        "JITTER_COMMAND_HZ": JITTER_COMMAND_HZ,
        "JITTER_RANDOMNESS": JITTER_RANDOMNESS,
        "ST_JITTER_SPEED": ST_JITTER_SPEED,
        "ST_JITTER_ACCEL": ST_JITTER_ACCEL,
        "ST_JITTER_AMPLITUDE_POS": ST_JITTER_AMPLITUDE_POS,
        "SETTLE_GRACE_PERIOD_S": SETTLE_GRACE_PERIOD_S,
        "ST_CRUISE_SPEED": ST_CRUISE_SPEED,
        "ST_ACCEL": ST_ACCEL,
        "ST_VISIBLE_MIN_POSITION": ST_VISIBLE_MIN_POSITION,
        "ST_VISIBLE_MAX_POSITION": ST_VISIBLE_MAX_POSITION,
        "ACT_MAX_SPEED_STEPS_S": ACT_MAX_SPEED_STEPS_S,
        "ACT_ACCELERATION_STEPS_S2": ACT_ACCELERATION_STEPS_S2,
        "ACT_GAIN_STEPS_PER_PX": ACT_GAIN_STEPS_PER_PX,
        "ACT_MIN_TRACKING_SPEED_STEPS_S": ACT_MIN_TRACKING_SPEED_STEPS_S,
        "ACT_MAX_TARGET_DELTA_PER_UPDATE": ACT_MAX_TARGET_DELTA_PER_UPDATE,
        "SWEEP_SERVO_SPEED": SWEEP_SERVO_SPEED,
        "SWEEP_SERVO_ACCEL": SWEEP_SERVO_ACCEL,
        "SWEEP_SERVO_TOLERANCE": SWEEP_SERVO_TOLERANCE,
        "SWEEP_ACTUATOR_TOLERANCE": SWEEP_ACTUATOR_TOLERANCE,
        "ACT_UNREFERENCED_JOG_LIMIT": ACT_UNREFERENCED_JOG_LIMIT,
        "ACT_SETTLE_DEADBAND_PX": ACT_SETTLE_DEADBAND_PX,
        "ACT_RETRACK_DEADBAND_PX": ACT_RETRACK_DEADBAND_PX,
        "ACT_SIGN": ACT_SIGN,
        "ACT_RECOVERY_SPEED_STEPS_S": ACT_RECOVERY_SPEED_STEPS_S,
        "JOG_STEP_SERVO": JOG_STEP_SERVO,
        "JOG_SPEED_SERVO": JOG_SPEED_SERVO,
        "JOG_SPEED_ACTUATOR": JOG_SPEED_ACTUATOR,
        "ACTUATOR_JOG_PULSE_S": ACTUATOR_JOG_PULSE_S,
        "JOG_OVERTRAVEL_ACTUATOR_STEPS": JOG_OVERTRAVEL_ACTUATOR_STEPS,
        "VIDEO_WRITE_EVERY_N": VIDEO_WRITE_EVERY_N,
        "MOUNT_HEIGHT_M": MOUNT_HEIGHT_M,
        "MOUNT_FORWARD_OFFSET_M": MOUNT_FORWARD_OFFSET_M,
        "MOUNT_LATERAL_OFFSET_M": MOUNT_LATERAL_OFFSET_M,
        "MOUNT_YAW_OFFSET_DEG": MOUNT_YAW_OFFSET_DEG,
        "AZIMUTH_DEG_PER_SERVO_UNIT": AZIMUTH_DEG_PER_SERVO_UNIT,
        "ELEVATION_HOME_DEG": ELEVATION_HOME_DEG,
        "ELEVATION_DEG_PER_ACTUATOR_STEP": ELEVATION_DEG_PER_ACTUATOR_STEP,
        "ACTUATOR_MIN_POSITION": ACTUATOR_MIN_POSITION,
        "ACTUATOR_MAX_POSITION": ACTUATOR_MAX_POSITION,
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
# ST3215 SERVO CONTROL
# ============================================================

st_servo = None
_servo_command_lock = threading.Lock()


def setup_st_servo():

    global st_servo

    st_servo = ST3215(ST_SERIAL_PORT)

    # The library opens at 1,000,000 baud internally with no way to pass
    # a different rate to the constructor. Override it directly on the
    # underlying pyserial object -- confirmed to take effect live.
    try:
        st_servo.portHandler.ser.baudrate = ST_BAUDRATE
    except AttributeError:
        print(
            f"[ST3215] WARNING: could not set baud rate on "
            f"portHandler.ser -- inspect with: python3 -c \"from st3215 "
            f"import ST3215; s = ST3215('{ST_SERIAL_PORT}'); "
            f"print(dir(s.portHandler))\""
        )

    if st_servo.PingServo(ST_SERVO_ID):
        print(f"[ST3215] Servo ID {ST_SERVO_ID} responded")
    else:
        print(
            f"[ST3215] WARNING: ping to ID {ST_SERVO_ID} did not succeed. "
            f"Check wiring, ID, and that 'Start Serial Forwarding' is "
            f"active on the driver board."
        )


def set_servo_position(position, speed=ST_CRUISE_SPEED, acc=ST_ACCEL):

    # HARD LIMIT -- enforced here, centrally, regardless of caller.
    # Every code path that moves the servo (tracking, jog, sweep test,
    # jitter, recovery, neutral reset) goes through this function, so
    # putting the clamp here means none of them can ever command the
    # servo past the calibrated safe range, even a future call site
    # that forgets to clamp first. ST_VISIBLE_MIN/MAX_POSITION are
    # loaded from CALIBRATION_FILE at startup (see load_calibration())
    # and updated live when you mark new endpoints.
    position = int(max(
        ST_VISIBLE_MIN_POSITION, min(ST_VISIBLE_MAX_POSITION, position)
    ))
    # Backstop against the servo's own absolute electrical range, in
    # case calibration is somehow missing/corrupt.
    position = int(max(ST_MIN_POSITION, min(ST_MAX_POSITION, position)))

    with _servo_command_lock:
        st_servo.MoveTo(
            ST_SERVO_ID, position, speed=speed, acc=acc, wait=False
        )
    return position


def close_st_servo():

    try:
        st_servo.portHandler.closePort()
    except Exception:
        pass


# ============================================================
# CALIBRATION PERSISTENCE
#
# Saves/loads the manually-jogged safe-range endpoints for both
# axes so they don't need to be re-set every run. Written every
# time an endpoint is marked (keys 1/2/3/4), not just on request.
# ============================================================

CALIBRATION_FILE = "servo_calibration.json"


def load_calibration():

    global ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION
    global ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION

    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE) as f:
                cal = json.load(f)
            ST_VISIBLE_MIN_POSITION = int(cal["servo_min"])
            ST_VISIBLE_MAX_POSITION = int(cal["servo_max"])

            # The actuator coordinate is now defined from the PHYSICAL MIN
            # endpoint, not from wherever the program happened to start.
            # Convert old saved [-range/2, +range/2]-style data into 0..travel.
            old_act_min = int(cal["actuator_min"])
            old_act_max = int(cal["actuator_max"])
            travel = old_act_max - old_act_min
            ACTUATOR_MIN_POSITION = 0
            ACTUATOR_MAX_POSITION = max(0, travel)

            print(
                f"[CALIBRATION] Loaded from {CALIBRATION_FILE}: "
                f"servo=[{ST_VISIBLE_MIN_POSITION}, {ST_VISIBLE_MAX_POSITION}]  "
                f"actuator_travel={ACTUATOR_MAX_POSITION} steps"
            )
            return True
        except Exception as e:
            print(f"[CALIBRATION] WARNING: failed to load {CALIBRATION_FILE}: {e}")

    print(
        "[CALIBRATION] WARNING: no saved calibration found -- currently "
        f"using UNCALIBRATED defaults: servo=[{ST_VISIBLE_MIN_POSITION}, "
        f"{ST_VISIBLE_MAX_POSITION}]  actuator=[{ACTUATOR_MIN_POSITION}, "
        f"{ACTUATOR_MAX_POSITION}]. Press M to jog to your real safe "
        "endpoints and mark them (1/2/3/4) before selecting a target."
    )
    return False


def save_calibration():

    global ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION

    # Normalize order -- if endpoints were marked out of order (e.g.
    # "max" jogged to a smaller position than a previously-marked
    # "min"), max(MIN, min(MAX, x)) with MIN > MAX silently resolves
    # every value to the same bound instead of raising an error. Swap
    # rather than let that happen unnoticed.
    if ST_VISIBLE_MIN_POSITION > ST_VISIBLE_MAX_POSITION:
        ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION = (
            ST_VISIBLE_MAX_POSITION, ST_VISIBLE_MIN_POSITION
        )

    cal = {
        "servo_min": ST_VISIBLE_MIN_POSITION,
        "servo_max": ST_VISIBLE_MAX_POSITION,
        "actuator_min": ACTUATOR_MIN_POSITION,
        "actuator_max": ACTUATOR_MAX_POSITION,
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(cal, f, indent=2)

    print(f"[CALIBRATION] Saved to {CALIBRATION_FILE}: {cal}")


# ============================================================
# LINEAR ACTUATOR -- low-level GPIO control (unchanged pins)
# ============================================================

GPIO.setmode(GPIO.BOARD)
GPIO.setup(ACT_STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ACT_DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ACT_EN_PIN, GPIO.OUT, initial=GPIO.HIGH)


def act_enable():
    GPIO.output(ACT_EN_PIN, GPIO.LOW)


def act_disable():
    GPIO.output(ACT_EN_PIN, GPIO.HIGH)


def act_set_direction(direction):
    GPIO.output(ACT_DIR_PIN, GPIO.HIGH if direction > 0 else GPIO.LOW)
    time.sleep(0.00001)


def act_step_once(delay):
    GPIO.output(ACT_STEP_PIN, GPIO.HIGH)
    time.sleep(delay)
    GPIO.output(ACT_STEP_PIN, GPIO.LOW)
    time.sleep(delay)


def act_target_speed(remaining_steps):
    # Stopping-distance formula: v = sqrt(2 * a * d), same as before.
    stopping_speed = math.sqrt(
        2 * ACT_ACCELERATION_STEPS_S2 * max(0.0, remaining_steps)
    )
    return min(ACT_MAX_SPEED_STEPS_S, stopping_speed)


# --------------------------------------------------------
# Background stepping thread. The vision/control loop only ever
# writes to _actuator_target_speed_signed (a signed steps/sec
# value: sign = direction, magnitude = desired speed, 0 = stop).
# This thread turns that into an actual, continuously-ramped pulse
# train, independent of camera frame timing.
# --------------------------------------------------------

_actuator_lock = threading.Lock()
_actuator_target_speed_signed = 0.0
_actuator_position = 0
_actuator_current_speed_signed = 0.0
_actuator_referenced = False
_actuator_min_reference_done = False
_actuator_calibration_override = False
_actuator_running = True

_jitter_active = False
_jitter_start_time = None
_jitter_base_position = None
_jitter_running = True
_jitter_thread = None


def actuator_thread_loop():
    global _actuator_position, _actuator_current_speed_signed
    global _actuator_target_speed_signed

    current_speed_signed = 0.0
    current_direction = 0
    last_time = time.monotonic()

    while _actuator_running:
        now = time.monotonic()
        dt = min(now - last_time, 0.03)
        last_time = now

        with _actuator_lock:
            target_speed_signed = _actuator_target_speed_signed
            referenced = _actuator_referenced
            calibration_override = _actuator_calibration_override
            position = _actuator_position
            min_pos = ACTUATOR_MIN_POSITION
            max_pos = ACTUATOR_MAX_POSITION

        # Automatic tracking/sweep is gated in the main loop until the
        # actuator is referenced. Manual jog is intentionally still allowed
        # before referencing so the user can physically find MIN.
        target_direction = (
            1 if target_speed_signed > 0
            else -1 if target_speed_signed < 0
            else 0
        )

        # Always pass through zero before reversing.
        if (
            current_direction != 0
            and target_direction != 0
            and target_direction != current_direction
        ):
            target_speed_signed = 0.0

        max_change = ACT_ACCELERATION_STEPS_S2 * dt
        diff = target_speed_signed - current_speed_signed
        if abs(diff) <= max_change:
            current_speed_signed = target_speed_signed
        else:
            current_speed_signed += math.copysign(max_change, diff)

        # Safety: taper and clamp at the ACTUAL calibrated travel endpoints.
        # During explicit calibration-extension mode only, allow manual jog
        # beyond the saved MAX/MIN, but never beyond the larger hard software
        # envelope defined by ACTUATOR_CALIBRATION_EXTENSION_LIMIT.
        if calibration_override:
            if (
                current_speed_signed > 0
                and position >= ACTUATOR_CALIBRATION_EXTENSION_LIMIT
            ) or (
                current_speed_signed < 0
                and position <= -ACTUATOR_CALIBRATION_EXTENSION_LIMIT
            ):
                current_speed_signed = 0.0
                with _actuator_lock:
                    _actuator_target_speed_signed = 0.0
                current_direction = 0
                time.sleep(0.001)
                continue

        elif referenced and max_pos > min_pos:
            remaining_to_limit = (
                max_pos - position
                if current_speed_signed > 0
                else position - min_pos
            )
            if remaining_to_limit <= 0:
                current_speed_signed = 0.0
                with _actuator_lock:
                    _actuator_target_speed_signed = 0.0
                current_direction = 0
                time.sleep(0.001)
                continue

            # Stopping-distance speed limit near either endpoint.
            endpoint_speed = math.sqrt(
                2.0 * ACT_ACCELERATION_STEPS_S2 * max(
                    0.0, remaining_to_limit
                )
            )
            if abs(current_speed_signed) > endpoint_speed:
                current_speed_signed = math.copysign(
                    endpoint_speed, current_speed_signed
                )

        _actuator_current_speed_signed = current_speed_signed

        if abs(current_speed_signed) < 1.0:
            current_direction = 0
            time.sleep(0.001)
            continue

        new_direction = 1 if current_speed_signed > 0 else -1
        if new_direction != current_direction:
            act_set_direction(new_direction)
            current_direction = new_direction
            act_enable()

        # Faster timing than two time.sleep(50 us) calls: the short busy-wait
        # reduces the large sleep jitter that made the actuator feel slow.
        speed = abs(current_speed_signed)
        half_period = 0.5 / speed
        act_step_once_precise(half_period)

        with _actuator_lock:
            _actuator_position += current_direction
            # Final software backstop in case a pulse lands exactly on a
            # calibrated endpoint.
            if _actuator_referenced:
                _actuator_position = max(
                    ACTUATOR_MIN_POSITION,
                    min(ACTUATOR_MAX_POSITION, _actuator_position)
                )


def act_step_once_precise(half_period):
    # Hybrid timing: sleep for most of the interval so the CPU/GIL remain
    # available to the camera and servo threads, then busy-wait only for
    # the last few tens of microseconds.
    def wait_interval(seconds):
        seconds = max(0.000002, seconds)
        guard = 0.00004
        if seconds > guard:
            time.sleep(seconds - guard)
        deadline = time.perf_counter() + min(guard, seconds)
        while time.perf_counter() < deadline:
            pass

    GPIO.output(ACT_STEP_PIN, GPIO.HIGH)
    wait_interval(half_period)
    GPIO.output(ACT_STEP_PIN, GPIO.LOW)
    wait_interval(half_period)


def actuator_is_referenced():
    with _actuator_lock:
        return _actuator_referenced and ACTUATOR_MAX_POSITION > ACTUATOR_MIN_POSITION


def actuator_accept_saved_middle():
    """Declare that the actuator is physically sitting at saved MIDDLE."""
    global _actuator_position, _actuator_referenced
    global _actuator_min_reference_done, _actuator_calibration_override

    middle = actuator_get_middle_position()
    if middle is None:
        return False

    actuator_set_target_speed(0.0)
    time.sleep(0.12)

    with _actuator_lock:
        _actuator_position = middle
        _actuator_referenced = True
        # This is NOT a new MIN reference. Keep this false so a later
        # attempt to redefine the physical MAX still requires the user
        # to visit physical MIN and press 3 first.
        _actuator_min_reference_done = False
        _actuator_calibration_override = False

    return True


def actuator_get_middle_position():
    if ACTUATOR_MAX_POSITION <= ACTUATOR_MIN_POSITION:
        return None
    return 0.5 * (ACTUATOR_MIN_POSITION + ACTUATOR_MAX_POSITION)


def actuator_mark_min():
    global _actuator_position, _actuator_referenced
    global _actuator_min_reference_done, ACTUATOR_MIN_POSITION

    # Stop commanded motion first; allow the background thread to settle.
    actuator_set_target_speed(0.0)
    time.sleep(0.12)

    with _actuator_lock:
        _actuator_position = 0
        ACTUATOR_MIN_POSITION = 0
        _actuator_min_reference_done = True
        _actuator_referenced = ACTUATOR_MAX_POSITION > 0


def actuator_mark_max():
    global ACTUATOR_MAX_POSITION, _actuator_referenced

    with _actuator_lock:
        if not _actuator_min_reference_done:
            return False
        if _actuator_position <= ACTUATOR_MIN_POSITION:
            return False

        ACTUATOR_MAX_POSITION = int(_actuator_position)
        _actuator_referenced = True

    return True


def actuator_target_speed_for_target_position(target_position):
    with _actuator_lock:
        current_position = _actuator_position

    delta = float(target_position) - current_position
    if abs(delta) < 1.0:
        return 0.0

    direction = 1.0 if delta > 0 else -1.0
    stopping_speed = math.sqrt(
        2.0 * ACT_ACCELERATION_STEPS_S2 * abs(delta)
    )
    speed = min(ACT_MAX_SPEED_STEPS_S, stopping_speed)

    return direction * speed


def actuator_set_calibration_override(enabled):
    global _actuator_calibration_override
    with _actuator_lock:
        _actuator_calibration_override = bool(enabled)


def actuator_set_target_speed(signed_speed):
    global _actuator_target_speed_signed
    with _actuator_lock:
        _actuator_target_speed_signed = signed_speed


def actuator_get_position():
    with _actuator_lock:
        return _actuator_position


def actuator_get_speed():
    return _actuator_current_speed_signed


# ============================================================
# LASER DETECTION
# ============================================================

def detect_laser(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, BRIGHTNESS_MIN, BRIGHTNESS_MAX)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
# TARGET / QUEUE
# ============================================================

target_x = None
target_y = None
target_state = "IDLE"
# One of: IDLE, TRACKING, JITTERING, SETTLED, SEARCHING

pending_targets = []
sequence_running = False


def mouse_callback(event, x, y, flags, param):

    global pending_targets, sequence_running

    if event == cv2.EVENT_LBUTTONDOWN:

        if sequence_running:
            print()
            print("[QUEUE] Sequence already running -- press C to cancel first")
            return

        pending_targets.append((x, y))
        print()
        print(f"[QUEUE] Added target #{len(pending_targets)}: x={x}, y={y}")


# ============================================================
# DRAWING
# ============================================================

def draw_laser(output, laser_x, laser_y):
    if laser_x is None:
        return
    x, y = int(laser_x), int(laser_y)
    cv2.circle(output, (x, y), 8, (0, 0, 255), -1)
    cv2.drawMarker(output, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(
        output, f"Laser: ({x}, {y})", (x + 15, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )


def draw_target(output, target_x, target_y, laser_x):
    if target_x is None:
        return
    x, y = int(target_x), int(target_y)
    cv2.drawMarker(output, (x, y), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(
        output, f"Target: ({x}, {y})", (x + 15, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
    )
    if laser_x is not None:
        cv2.line(output, (int(laser_x), y), (x, y), (255, 255, 0), 2)


def draw_queue(output, pending_targets):
    for i, (x, y) in enumerate(pending_targets):
        cv2.circle(output, (x, y), 6, (0, 165, 255), 2)
        cv2.putText(
            output, str(i + 1), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1
        )


def draw_params_overlay(output, params):
    x0 = output.shape[1] - 300
    y0 = 20
    line_height = 15
    for i, (key, value) in enumerate(params.items()):
        cv2.putText(
            output, f"{key}={value}", (x0, y0 + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1
        )


def get_vehicle_pose():

    # Returns (x, y, yaw_rad) in the odom/world frame, or None if
    # unavailable (file missing, malformed, or stale). Callers MUST
    # treat None as "hold current aim, don't apply feedforward this
    # frame" -- never fall back to a guessed pose.

    try:
        with open(VEHICLE_POSE_FILE) as f:
            line = f.read().strip()
        parts = line.split()
        if len(parts) != 4:
            return None
        x, y, yaw, ts = (float(p) for p in parts)
        if time.time() - ts > VEHICLE_POSE_MAX_AGE_S:
            return None
        return (x, y, yaw)
    except (FileNotFoundError, ValueError):
        return None


def azimuth_from_servo(servo_position):
    return math.radians(
        (servo_position - ST_MIDDLE_POSITION) * AZIMUTH_DEG_PER_SERVO_UNIT
        + MOUNT_YAW_OFFSET_DEG
    )


def servo_from_azimuth(azimuth_rad):
    deg = math.degrees(azimuth_rad) - MOUNT_YAW_OFFSET_DEG
    return ST_MIDDLE_POSITION + deg / AZIMUTH_DEG_PER_SERVO_UNIT


def elevation_from_actuator(actuator_position):
    return math.radians(
        ELEVATION_HOME_DEG + actuator_position * ELEVATION_DEG_PER_ACTUATOR_STEP
    )


def actuator_from_elevation(elevation_rad):
    deg = math.degrees(elevation_rad) - ELEVATION_HOME_DEG
    position = deg / ELEVATION_DEG_PER_ACTUATOR_STEP
    return max(ACTUATOR_MIN_POSITION, min(ACTUATOR_MAX_POSITION, position))


def mount_world_position(vehicle_pose):
    vx, vy, vyaw = vehicle_pose
    mount_x = (
        vx + MOUNT_FORWARD_OFFSET_M * math.cos(vyaw)
        - MOUNT_LATERAL_OFFSET_M * math.sin(vyaw)
    )
    mount_y = (
        vy + MOUNT_FORWARD_OFFSET_M * math.sin(vyaw)
        + MOUNT_LATERAL_OFFSET_M * math.cos(vyaw)
    )
    return mount_x, mount_y


def compute_world_lock_point(vehicle_pose, servo_position, actuator_position):

    # Ray-casts the CURRENT aim direction (servo -> azimuth, actuator ->
    # elevation) from the mount onto the assumed flat ground plane, to
    # get a world-frame (x, y). Only meaningful to call when the laser
    # is actually sitting on the real target (i.e. SETTLED) -- this
    # trusts the current aim as ground truth.

    mount_x, mount_y = mount_world_position(vehicle_pose)

    azimuth = vehicle_pose[2] + azimuth_from_servo(servo_position)
    elevation = elevation_from_actuator(actuator_position)

    if elevation <= math.radians(1.0):
        return None  # aimed at/above the horizon -- no ground intersection

    ground_range = MOUNT_HEIGHT_M / math.tan(elevation)

    return (
        mount_x + ground_range * math.cos(azimuth),
        mount_y + ground_range * math.sin(azimuth),
    )


def compute_feedforward_aim(vehicle_pose, world_x, world_y):

    # Given the CURRENT vehicle pose and a locked world-frame ground
    # point, returns the (servo_position, actuator_position) that keeps
    # pointing at it right now.

    mount_x, mount_y = mount_world_position(vehicle_pose)

    dx = world_x - mount_x
    dy = world_y - mount_y
    ground_range = math.hypot(dx, dy)

    world_azimuth = math.atan2(dy, dx)
    azimuth = world_azimuth - vehicle_pose[2]

    elevation = math.atan2(MOUNT_HEIGHT_M, ground_range)

    servo_pos = servo_from_azimuth(azimuth)
    servo_pos = max(ST_VISIBLE_MIN_POSITION, min(ST_VISIBLE_MAX_POSITION, servo_pos))

    return servo_pos, actuator_from_elevation(elevation)


# ============================================================
# MAIN
# ============================================================

def main():

    global target_x, target_y, target_state, pending_targets, sequence_running
    global CONTROL_AXIS, _actuator_running
    global ST_VISIBLE_MIN_POSITION, ST_VISIBLE_MAX_POSITION
    global ACTUATOR_MIN_POSITION, ACTUATOR_MAX_POSITION
    global _actuator_referenced, _actuator_min_reference_done
    global _actuator_calibration_override

    calibration_loaded = load_calibration()

    # With no encoder/limit switch, absolute actuator position cannot be
    # recovered automatically after restart. Since the program normally
    # returns the actuator to MIDDLE at shutdown, offer a deliberate startup
    # declaration instead of forcing a full MIN reference every run.
    if calibration_loaded and ACTUATOR_MAX_POSITION > 0:
        middle = actuator_get_middle_position()
        print()
        print(
            f"[STARTUP] Saved actuator travel = {ACTUATOR_MAX_POSITION} steps; "
            f"saved MIDDLE = {middle:.1f}."
        )
        answer = input(
            "[STARTUP] Is the actuator physically at MIDDLE now? "
            "[Y] accept / [N] reference MIN: "
        ).strip().lower()
        if answer in ("y", "yes"):
            if actuator_accept_saved_middle():
                print(
                    f"[STARTUP] Accepted actuator position as MIDDLE "
                    f"({middle:.1f}). No recalibration required."
                )
        else:
            print(
                "[STARTUP] Actuator NOT referenced. Enter JOG mode, "
                "move to physical MIN, press 3, then use R/T/G."
            )

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera /dev/video{CAMERA_DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    time.sleep(0.5)

    print()
    print("==========================================")
    print(" LASER + ST3215 SERVO + LINEAR ACTUATOR")
    print("==========================================")
    print()
    print(f"Camera         : /dev/video{CAMERA_DEVICE}")
    print(f"Resolution     : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"ST3215 port    : {ST_SERIAL_PORT} @ {ST_BAUDRATE} baud, ID {ST_SERVO_ID}")
    print(f"Actuator pins  : STEP={ACT_STEP_PIN} DIR={ACT_DIR_PIN} EN={ACT_EN_PIN}")
    print()
    print("CONTROLS")
    print("  Left click  = queue a target")
    print("  G           = go -- start tracking the queued targets in order")
    print("  A           = switch servo's control axis (X <-> Y) --")
    print("                the actuator automatically follows the other axis")
    print("  L           = lock current (SETTLED) target to world frame --")
    print("                keeps pointing at it as the vehicle moves")
    print("  R           = neutral (also clears queue and world-lock)")
    print("  C           = cancel current target / sequence / queue / world-lock / sweep")
    print("  M           = toggle jog mode (only when no target is active)")
    print("                  I/K = jog servo -/+        U/O = jog actuator -/+")
    print("                  1/2 = mark servo MIN/MAX")
    print("                  3   = REFERENCE actuator at physical MIN (do each run)")
    print("                  4   = mark actuator MAX / save travel")
    print("  T           = sweep -- choose SERVO / ACTUATOR / BOTH")
    print("                then MIDDLE -> MAX -> MIN -> MIDDLE")
    print("  J           = manually trigger jitter at the current position")
    print("  Q           = quit")
    print()
    print(f"Starting axis  : servo={CONTROL_AXIS.upper()}  "
          f"actuator={('y' if CONTROL_AXIS == 'x' else 'x').upper()}")
    print()
    if ACTUATOR_MAX_POSITION > 0:
        print(
            f"[ACTUATOR] Saved travel = {ACTUATOR_MAX_POSITION} steps. "
            "Reference required each run: jog to PHYSICAL MIN and press 3."
        )
    else:
        print(
            "[ACTUATOR] No saved travel yet. First reference: jog to PHYSICAL "
            "MIN -> press 3, then PHYSICAL MAX -> press 4."
        )
    print(f"Log file       : {LOG_FILE}")
    print(f"Extra log file : {EXTRA_LOG_FILE}")
    print(f"Params file    : {PARAMS_LOG_FILE}")
    print()

    setup_st_servo()

    true_position = st_servo.ReadPosition(ST_SERVO_ID)
    if true_position is not None:
        servo_position = float(true_position)
        print(f"[ST3215] Read true starting position: {true_position}")
    else:
        servo_position = float(ST_MIDDLE_POSITION)
        print(
            "[ST3215] WARNING: could not read true position -- "
            f"assuming middle ({ST_MIDDLE_POSITION}). If the servo was "
            "NOT actually at middle, the first correction will be "
            "computed from a wrong baseline and may move further/faster "
            "than intended."
        )

    act_enable()
    actuator_thread = threading.Thread(target=actuator_thread_loop, daemon=True)
    actuator_thread.start()

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

    jitter_start_time = None
    jitter_base_position = None
    settled_since = None

    global _jitter_active, _jitter_start_time, _jitter_base_position
    global _jitter_running, _jitter_thread
    _jitter_thread = threading.Thread(target=jitter_thread_loop, daemon=True)
    _jitter_thread.start()
    settle_dwell_start = None

    actuator_filtered = None
    actuator_state = "IDLE"
    actuator_settle_dwell_start = None

    world_lock_active = False
    world_lock_x = None
    world_lock_y = None

    # --------------------------------------------------------
    # Jog / calibration / sweep-test state
    # --------------------------------------------------------

    actuator_jog_until = 0.0
    jog_mode = False

    sweep_active = False
    sweep_leg = None
    sweep_mode = None
    sweep_selection_pending = False
    sweep_servo_commanded = False
    return_to_middle = False
    # Sweep sequence:
    #   MIDDLE -> MAX -> MIN -> MIDDLE
    # sweep_mode is "servo", "actuator", or "both".
    # When only one DOF is selected, the other DOF is held stationary.

    def advance_or_settle():

        nonlocal filtered_x, settled_since
        global target_x, target_y, target_state, sequence_running

        if sequence_running and pending_targets:
            target_x, target_y = pending_targets.pop(0)
            target_state = "IDLE"
            filtered_x = None
            print()
            print(
                f"[SEQUENCE] Advancing to x={target_x}, y={target_y} "
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

            laser_x, laser_y, blob_area, mask, num_contours = detect_laser(frame)

            error_x = None
            vehicle_pose = None

            # ------------------------------------------------
            # World-frame lock (odometry-driven feedforward)
            #
            # Bypasses the pixel-based blocks below entirely -- locking
            # (the L key) clears target_x/target_y, so those blocks'
            # own "if target_x is not None" guards naturally go inert.
            # No vision trim blended in yet -- see the design notes.
            # ------------------------------------------------

            if world_lock_active:

                vehicle_pose = get_vehicle_pose()

                if vehicle_pose is not None:

                    ff_servo_pos, ff_actuator_pos = compute_feedforward_aim(
                        vehicle_pose, world_lock_x, world_lock_y
                    )

                    servo_position = set_servo_position(ff_servo_pos)

                    remaining_actuator_steps = (
                        ff_actuator_pos - actuator_get_position()
                    )
                    ff_speed = act_target_speed(abs(remaining_actuator_steps))
                    actuator_set_target_speed(
                        math.copysign(ff_speed, remaining_actuator_steps)
                        if remaining_actuator_steps != 0 else 0.0
                    )

                # else: no fresh pose -- hold last commanded aim, do
                # nothing this frame (see get_vehicle_pose()'s contract).

            # ------------------------------------------------
            # ST3215 servo control
            # ------------------------------------------------

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
                        target_state = "TRACKING"
                        filtered_x = None
                        settle_dwell_start = None
                    else:
                        recovery_target = (
                            ST_VISIBLE_MIN_POSITION + ST_VISIBLE_MAX_POSITION
                        ) / 2
                        if servo_position > recovery_target:
                            servo_position = max(
                                recovery_target, servo_position - RECOVERY_STEP_POS
                            )
                        elif servo_position < recovery_target:
                            servo_position = min(
                                recovery_target, servo_position + RECOVERY_STEP_POS
                            )
                        servo_position = set_servo_position(servo_position)

                if target_state == "IDLE" and laser_x is not None:
                    target_state = "TRACKING"

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

                    laser_pos = laser_x if CONTROL_AXIS == "x" else laser_y
                    target_pos = target_x if CONTROL_AXIS == "x" else target_y

                    filtered_x = (
                        laser_pos if filtered_x is None
                        else ALPHA * laser_pos + (1 - ALPHA) * filtered_x
                    )
                    error_x = target_pos - filtered_x

                    if target_state == "SETTLED":
                        if abs(error_x) > RETRACK_DEADBAND_PX:
                            target_state = "TRACKING"
                            settle_dwell_start = None

                    if target_state == "TRACKING":

                        if abs(error_x) <= SETTLE_DEADBAND_PX:

                            # Inside the deadband, stop sending new servo
                            # position commands. The previous code applied a
                            # final correction even at near-zero error, which
                            # could move a correctly aligned servo slightly.
                            tracking = False

                            if settle_dwell_start is None:
                                settle_dwell_start = time.monotonic()

                            dwelled_long_enough = (
                                time.monotonic() - settle_dwell_start
                                >= SETTLE_DWELL_S
                            )

                            if dwelled_long_enough and actuator_state == "SETTLED":

                                settle_dwell_start = None

                                if JITTER_ENABLE:
                                    target_state = "JITTERING"
                                    jitter_start_time = time.monotonic()
                                    jitter_base_position = servo_position
                                    actuator_set_target_speed(0.0)
                                    time.sleep(0.05)
                                    _jitter_start_time = time.monotonic()
                                    jitter_start_time = _jitter_start_time
                                    _jitter_base_position = jitter_base_position
                                    _jitter_active = True
                                else:
                                    advance_or_settle()

                        else:

                            tracking = True
                            settle_dwell_start = None

                            desired = error_x * AXIS_GAIN_POS_PER_PX[CONTROL_AXIS]
                            correction = max(
                                -MAX_POSITION_STEP_PER_UPDATE,
                                min(MAX_POSITION_STEP_PER_UPDATE, desired)
                            )

                            servo_position += AXIS_SIGN[CONTROL_AXIS] * correction
                            servo_position = max(
                                ST_VISIBLE_MIN_POSITION,
                                min(ST_VISIBLE_MAX_POSITION, servo_position)
                            )
                            servo_position = set_servo_position(
                                servo_position, speed=scaled_servo_speed(error_x)
                            )

                    last_servo_update = time.monotonic()

                elif target_state == "JITTERING":

                    elapsed = time.monotonic() - jitter_start_time

                    # The background jitter thread owns servo commands for
                    # the duration of this state. The main loop only times
                    # the phase and ensures the actuator remains stopped.
                    actuator_set_target_speed(0.0)

                    if elapsed >= JITTER_DURATION_S:
                        _jitter_active = False
                        time.sleep(0.02)
                        servo_position = set_servo_position(
                            jitter_base_position,
                            speed=SERVO_SPEED_NEAR,
                            acc=ST_ACCEL
                        )
                        advance_or_settle()

            # ------------------------------------------------
            # Return-to-middle state (used by R and by end-of-test reset)
            # ------------------------------------------------
            if return_to_middle and target_x is None and not sweep_active:
                servo_position = set_servo_position(
                    ST_MIDDLE_POSITION,
                    speed=SERVO_SPEED_NEAR,
                    acc=ST_ACCEL
                )

                if actuator_is_referenced():
                    middle = actuator_get_middle_position()
                    actuator_speed = actuator_target_speed_for_target_position(
                        middle
                    )
                    actuator_set_target_speed(actuator_speed)

                    if (
                        abs(actuator_get_position() - middle) <= 5
                        and abs(actuator_get_speed()) < 20
                    ):
                        actuator_set_target_speed(0.0)
                        return_to_middle = False
                else:
                    actuator_set_target_speed(0.0)
                    return_to_middle = False

            # ------------------------------------------------
            # Actuator control (second DOF)
            #
            # Stationary-test rule:
            #   - no automatic actuator tracking until it is referenced
            #   - NO actuator motion during JITTERING
            #   - the actuator is commanded by a speed profile bounded by
            #     the calibrated physical MIN/MAX positions
            # ------------------------------------------------

            actuator_error = None

            if target_x is not None and not return_to_middle:

                actuator_axis = "y" if CONTROL_AXIS == "x" else "x"
                actuator_laser_pos = (
                    laser_x if actuator_axis == "x" else laser_y
                )
                actuator_target_pos = (
                    target_x if actuator_axis == "x" else target_y
                )

                if target_state == "JITTERING":
                    # Jitter is SERVO ONLY. Freeze the actuator.
                    actuator_set_target_speed(0.0)
                    actuator_state = "SETTLED"
                    actuator_filtered = None
                    actuator_settle_dwell_start = None

                elif not actuator_is_referenced():
                    actuator_set_target_speed(0.0)
                    actuator_state = "UNREFERENCED"
                    actuator_filtered = None
                    actuator_settle_dwell_start = None

                elif actuator_laser_pos is not None:

                    if actuator_state in ("IDLE", "UNREFERENCED"):
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
                            actuator_set_target_speed(0.0)

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

                            with _actuator_lock:
                                current_pos = _actuator_position
                                min_pos = ACTUATOR_MIN_POSITION
                                max_pos = ACTUATOR_MAX_POSITION

                            # Convert pixel error into a bounded incremental
                            # position request. This is much less prone to
                            # endpoint hunting than forcing a minimum speed.
                            desired_delta = (
                                actuator_error
                                * ACT_GAIN_STEPS_PER_PX
                                * ACT_SIGN
                            )
                            desired_delta = max(
                                -ACT_MAX_TARGET_DELTA_PER_UPDATE,
                                min(
                                    ACT_MAX_TARGET_DELTA_PER_UPDATE,
                                    desired_delta
                                )
                            )

                            desired_position = max(
                                min_pos,
                                min(max_pos, current_pos + desired_delta)
                            )

                            # Slow down in advance of the endpoint.
                            remaining_to_endpoint = (
                                desired_position - current_pos
                                if desired_position > current_pos
                                else current_pos - desired_position
                            )

                            speed = actuator_target_speed_for_target_position(
                                desired_position
                            )

                            if remaining_to_endpoint > 0:
                                signed_dir = (
                                    1 if desired_position > current_pos else -1
                                )
                                distance_to_limit = (
                                    max_pos - current_pos
                                    if signed_dir > 0
                                    else current_pos - min_pos
                                )
                                endpoint_speed = math.sqrt(
                                    2.0 * ACT_ACCELERATION_STEPS_S2
                                    * max(0.0, distance_to_limit)
                                )
                                speed = math.copysign(
                                    min(
                                        abs(speed),
                                        endpoint_speed
                                    ),
                                    signed_dir
                                )

                            actuator_set_target_speed(speed)

                else:
                    actuator_set_target_speed(0.0)

            # ------------------------------------------------
            # Jog mode: actuator jog pulse auto-stop
            #
            # Runs every frame regardless of target state -- U/O each
            # issue a short timed speed pulse (steppers are speed-
            # driven, not position-driven, and cv2.waitKey only gives
            # discrete keypress events, not "key held" state), and
            # this is what cuts that pulse off again once it expires.
            # ------------------------------------------------

            if (
                jog_mode
                and target_x is None
                and not sweep_active
                and actuator_jog_until
                and time.monotonic() >= actuator_jog_until
            ):
                actuator_set_target_speed(0.0)
                actuator_jog_until = 0.0

            # ------------------------------------------------
            # Sweep test
            #
            # The sweep is deliberately a small state machine:
            #   MIDDLE -> MAX -> MIN -> MIDDLE
            #
            # T opens a selection prompt:
            #   1 = SERVO only
            #   2 = ACTUATOR only
            #   3 = BOTH
            #
            # For the servo we command each leg ONCE, then use actual
            # ReadPosition() to determine arrival. Re-sending MoveTo()
            # every video frame was what made the old sweep state machine
            # unreliable.
            # ------------------------------------------------

            if sweep_active:

                # An actuator reference is required only when the selected
                # sweep includes the actuator. A servo-only sweep can be run
                # independently.
                if (
                    sweep_mode in ("actuator", "both")
                    and not actuator_is_referenced()
                ):
                    sweep_active = False
                    sweep_leg = None
                    sweep_mode = None
                    sweep_selection_pending = False
                    sweep_servo_commanded = False
                    actuator_set_target_speed(0.0)
                    print()
                    print(
                        "[SWEEP] Aborted -- actuator sweep requires "
                        "a MIN reference (M -> 3)"
                    )

                else:
                    actuator_middle = actuator_get_middle_position()

                    if sweep_leg == "to_middle_start":
                        servo_target = ST_MIDDLE_POSITION
                        actuator_target = actuator_middle
                    elif sweep_leg == "to_max":
                        servo_target = ST_VISIBLE_MAX_POSITION
                        actuator_target = ACTUATOR_MAX_POSITION
                    elif sweep_leg == "to_min":
                        servo_target = ST_VISIBLE_MIN_POSITION
                        actuator_target = ACTUATOR_MIN_POSITION
                    elif sweep_leg == "to_middle_end":
                        servo_target = ST_MIDDLE_POSITION
                        actuator_target = actuator_middle
                    else:
                        sweep_active = False
                        sweep_leg = None
                        sweep_mode = None
                        sweep_selection_pending = False
                        sweep_servo_commanded = False
                        actuator_set_target_speed(0.0)
                        continue

                    # ---- SERVO LEG ----
                    servo_arrived = True
                    if sweep_mode in ("servo", "both"):
                        if not sweep_servo_commanded:
                            set_servo_position(
                                servo_target,
                                speed=SWEEP_SERVO_SPEED,
                                acc=SWEEP_SERVO_ACCEL
                            )
                            sweep_servo_commanded = True

                        actual_servo = st_servo.ReadPosition(ST_SERVO_ID)
                        if actual_servo is not None:
                            servo_position = float(actual_servo)
                            servo_arrived = (
                                abs(actual_servo - servo_target)
                                <= SWEEP_SERVO_TOLERANCE
                            )
                        else:
                            # Fallback only if an actual read fails.
                            servo_arrived = not st_servo.IsMoving(ST_SERVO_ID)

                    # ---- ACTUATOR LEG ----
                    actuator_arrived = True
                    if sweep_mode in ("actuator", "both"):
                        actuator_speed = actuator_target_speed_for_target_position(
                            actuator_target
                        )
                        actuator_set_target_speed(actuator_speed)

                        actuator_arrived = (
                            abs(
                                actuator_get_position() - actuator_target
                            ) <= SWEEP_ACTUATOR_TOLERANCE
                            and abs(actuator_get_speed()) < 20
                        )
                    else:
                        # Servo-only sweep: actuator must not move.
                        actuator_set_target_speed(0.0)

                    if servo_arrived and actuator_arrived:
                        actuator_set_target_speed(0.0)
                        sweep_servo_commanded = False

                        if sweep_leg == "to_middle_start":
                            sweep_leg = "to_max"
                            print("[SWEEP] MIDDLE reached -> MAX")

                        elif sweep_leg == "to_max":
                            sweep_leg = "to_min"
                            print("[SWEEP] MAX reached -> MIN")

                        elif sweep_leg == "to_min":
                            sweep_leg = "to_middle_end"
                            print("[SWEEP] MIN reached -> MIDDLE")

                        elif sweep_leg == "to_middle_end":
                            completed_mode = sweep_mode.upper() if sweep_mode else "UNKNOWN"

                            sweep_active = False
                            sweep_leg = None
                            sweep_mode = None
                            sweep_selection_pending = False
                            sweep_servo_commanded = False

                            print()
                            print(f"[SWEEP] Complete -- returned {completed_mode} to MIDDLE")

            # ------------------------------------------------
            # Output image
            # ------------------------------------------------

            output = frame.copy()
            draw_laser(output, laser_x, laser_y)
            draw_target(output, target_x, target_y, laser_x)
            draw_queue(output, pending_targets)

            if world_lock_active:
                if vehicle_pose is None:
                    status = "WORLD-LOCK -- NO VEHICLE POSE (holding last aim)"
                    status_color = (0, 165, 255)
                else:
                    status = f"WORLD-LOCK ({world_lock_x:.2f}, {world_lock_y:.2f})"
                    status_color = (255, 0, 255)
            elif target_state == "SEARCHING":
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
                    (30, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                )
            cv2.putText(
                output, f"Servo pos: {servo_position:.0f}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                output, f"Blob area: {blob_area:.1f}", (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                output, f"Contours: {num_contours}", (30, 150),
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

            if world_lock_active:
                pose_str = (
                    f"pose=({vehicle_pose[0]:.2f},{vehicle_pose[1]:.2f},"
                    f"{math.degrees(vehicle_pose[2]):.0f}deg)"
                    if vehicle_pose is not None else "pose=UNAVAILABLE"
                )
                cv2.putText(
                    output,
                    f"World-lock ({world_lock_x:.2f}, {world_lock_y:.2f})  {pose_str}",
                    (30, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2
                )

            if target_x is not None:
                actuator_axis_label = "y" if CONTROL_AXIS == "x" else "x"
                ref_status = "REF" if actuator_is_referenced() else "UNREF"
                act_pos = actuator_get_position()
                act_status = (
                    f"Actuator ({actuator_axis_label.upper()}): {actuator_state} [{ref_status}]"
                    + (f"  err={actuator_error:+.1f}px" if actuator_error is not None else "")
                    + f"  pos={act_pos}"
                    + f"  spd={actuator_get_speed():.0f}"
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

            draw_params_overlay(output, get_tunable_params())
            # Recomputed live every frame rather than reusing the
            # `tunable_params` startup snapshot -- calibration marked
            # mid-run (1/2/3/4) previously never showed up here.

            cv2.imshow(WINDOW_NAME, output)
            cv2.imshow(RAW_WINDOW, frame)

            if video_writer is not None and frame_number % VIDEO_WRITE_EVERY_N == 0:
                combined = np.hstack((output, frame))
                video_writer.write(combined)

            log_data(
                frame_number, target_x, target_y, laser_x, laser_y,
                error_x, servo_position, blob_area, num_contours, tracking
            )

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
                "servo_position": servo_position,
                "actuator_axis": "y" if CONTROL_AXIS == "x" else "x",
                "actuator_state": actuator_state,
                "actuator_error": actuator_error,
                "actuator_position": actuator_get_position(),
                "actuator_speed": actuator_get_speed(),
                "world_lock_active": world_lock_active,
                "world_lock_x": world_lock_x,
                "world_lock_y": world_lock_y,
                "vehicle_pose_x": vehicle_pose[0] if vehicle_pose else None,
                "vehicle_pose_y": vehicle_pose[1] if vehicle_pose else None,
                "vehicle_pose_yaw": vehicle_pose[2] if vehicle_pose else None,
            })

            key = cv2.waitKey(1) & 0xFF

            # Sweep selection takes priority over the normal 1/2/3 keys,
            # because those keys are also calibration keys inside JOG mode.
            if sweep_selection_pending:
                if key == ord("1"):
                    sweep_mode = "servo"
                elif key == ord("2"):
                    sweep_mode = "actuator"
                elif key == ord("3"):
                    sweep_mode = "both"
                elif key in (ord("c"), 27):
                    sweep_selection_pending = False
                    sweep_mode = None
                    print()
                    print("[SWEEP] Selection cancelled")
                    continue
                else:
                    continue

                sweep_selection_pending = False

                if (
                    sweep_mode in ("actuator", "both")
                    and not actuator_is_referenced()
                ):
                    sweep_mode = None
                    print()
                    print(
                        "[SWEEP] Actuator is not referenced. "
                        "Use M -> jog to physical MIN -> 3 first."
                    )
                    continue

                sweep_active = True
                sweep_leg = "to_middle_start"
                sweep_servo_commanded = False
                actuator_set_target_speed(0.0)

                print()
                print(
                    f"[SWEEP] Starting {sweep_mode.upper()} -- "
                    "MIDDLE -> MAX -> MIN -> MIDDLE"
                )
                continue

            if key == ord("q"):
                break

            elif key == ord("g"):
                if not sequence_running and pending_targets:
                    sequence_running = True
                    target_x, target_y = pending_targets.pop(0)
                    target_state = "IDLE"
                    return_to_middle = False
                    actuator_set_target_speed(0.0)
                    filtered_x = None
                    settle_dwell_start = None
                    print()
                    print(
                        f"[SEQUENCE] Starting -- x={target_x}, y={target_y} "
                        f"({len(pending_targets)} remaining)"
                    )
                elif sequence_running:
                    print()
                    print("[SEQUENCE] Already running")
                else:
                    print()
                    print("[SEQUENCE] No queued targets -- click to add some first")

            elif key == ord("l"):
                if world_lock_active:
                    print()
                    print("[WORLD-LOCK] Already locked -- press C to release first")
                elif target_state != "SETTLED":
                    print()
                    print("[WORLD-LOCK] Target must be SETTLED before locking")
                else:
                    pose = get_vehicle_pose()
                    if pose is None:
                        print()
                        print(
                            f"[WORLD-LOCK] No vehicle pose available -- "
                            f"check {VEHICLE_POSE_FILE}"
                        )
                    else:
                        point = compute_world_lock_point(
                            pose, servo_position, actuator_get_position()
                        )
                        if point is None:
                            print()
                            print(
                                "[WORLD-LOCK] Current aim doesn't intersect "
                                "the ground -- can't lock"
                            )
                        else:
                            world_lock_x, world_lock_y = point
                            world_lock_active = True
                            target_x = None
                            target_y = None
                            target_state = "IDLE"
                            print()
                            print(
                                f"[WORLD-LOCK] Locked at world "
                                f"({world_lock_x:.2f}, {world_lock_y:.2f})"
                            )

            elif key == ord("a"):
                CONTROL_AXIS = "y" if CONTROL_AXIS == "x" else "x"
                filtered_x = None
                error_x = None
                settle_dwell_start = None
                if target_x is not None:
                    target_state = "IDLE"
                    settled_since = None
                print()
                print(f"[AXIS] Switched control axis to {CONTROL_AXIS.upper()}")

            elif key == ord("r"):
                _jitter_active = False
                return_to_middle = True
                servo_position = set_servo_position(
                    float(ST_MIDDLE_POSITION),
                    speed=SERVO_SPEED_NEAR,
                    acc=ST_ACCEL
                )
                # Captures the clamped return value -- previously this
                # discarded it and left `servo_position` holding the
                # unclamped ST_MIDDLE_POSITION even when the hardware
                # was actually sent somewhere else (if 2047 falls
                # outside your calibrated range), re-introducing the
                # same software/hardware desync fixed at startup.
                # The main loop will keep recomputing the speed from the
                # remaining distance so the actuator decelerates into middle.
                actuator_set_target_speed(0.0)

                target_x = None
                target_y = None
                target_state = "IDLE"
                tracking = False
                filtered_x = None
                settle_dwell_start = None
                pending_targets = []
                sequence_running = False
                settled_since = None
                world_lock_active = False
                world_lock_x = None
                world_lock_y = None
                jog_mode = False
                actuator_set_calibration_override(False)
                sweep_active = False
                sweep_leg = None
                sweep_mode = None
                sweep_selection_pending = False
                sweep_servo_commanded = False
                actuator_jog_until = 0.0

                print()
                if actuator_is_referenced():
                    print("[SYSTEM] Returned servo + actuator toward middle position")
                else:
                    print(
                        "[SERVO] Returned to middle; actuator held because it "
                        "is not referenced this run"
                    )

            elif key == ord("c"):
                _jitter_active = False
                target_x = None
                target_y = None
                target_state = "IDLE"
                tracking = False
                filtered_x = None
                settle_dwell_start = None
                pending_targets = []
                sequence_running = False
                settled_since = None
                actuator_set_target_speed(0.0)
                world_lock_active = False
                world_lock_x = None
                world_lock_y = None
                sweep_active = False
                sweep_leg = None
                sweep_mode = None
                sweep_selection_pending = False
                sweep_servo_commanded = False
                actuator_jog_until = 0.0
                actuator_set_calibration_override(False)
                return_to_middle = False

                print()
                print("[TARGET] Cancelled")

            elif key == ord("m"):
                if target_x is None and not sweep_active:
                    jog_mode = not jog_mode
                    return_to_middle = False
                    actuator_set_target_speed(0.0)
                    actuator_jog_until = 0.0
                    print()
                    print(f"[JOG] {'Entered' if jog_mode else 'Exited'} jog mode")
                else:
                    print()
                    print("[JOG] Cancel the active target/sweep first (C)")

            elif key == ord("i") and jog_mode:
                servo_position = set_servo_position(
                    servo_position - JOG_STEP_SERVO, speed=JOG_SPEED_SERVO
                )
                print(f"[JOG] Servo -> {servo_position}")

            elif key == ord("k") and jog_mode:
                servo_position = set_servo_position(
                    servo_position + JOG_STEP_SERVO, speed=JOG_SPEED_SERVO
                )
                print(f"[JOG] Servo -> {servo_position}")

            elif key == ord("u") and jog_mode:
                with _actuator_lock:
                    pos = _actuator_position
                    referenced = _actuator_referenced
                    override = _actuator_calibration_override
                if override and pos <= -ACTUATOR_CALIBRATION_EXTENSION_LIMIT:
                    actuator_set_target_speed(0.0)
                    print("[JOG] Calibration envelope reached on MIN side")
                elif referenced and not override and pos <= ACTUATOR_MIN_POSITION:
                    actuator_set_target_speed(0.0)
                    print("[JOG] Actuator already at MIN")
                elif not referenced and pos <= -ACT_UNREFERENCED_JOG_LIMIT:
                    actuator_set_target_speed(0.0)
                    print("[JOG] Unreferenced safety envelope reached")
                else:
                    actuator_set_target_speed(-ACT_JOG_SPEED_STEPS_S)
                    actuator_jog_until = time.monotonic() + ACTUATOR_JOG_PULSE_S
                    print(f"[JOG] Actuator <-  pos={actuator_get_position()}")

            elif key == ord("o") and jog_mode:
                with _actuator_lock:
                    pos = _actuator_position
                    referenced = _actuator_referenced
                    override = _actuator_calibration_override
                if override and pos >= ACTUATOR_CALIBRATION_EXTENSION_LIMIT:
                    actuator_set_target_speed(0.0)
                    print("[JOG] Calibration envelope reached on MAX side")
                elif referenced and not override and pos >= ACTUATOR_MAX_POSITION:
                    actuator_set_target_speed(0.0)
                    print("[JOG] Actuator already at MAX")
                elif not referenced and pos >= ACT_UNREFERENCED_JOG_LIMIT:
                    actuator_set_target_speed(0.0)
                    print("[JOG] Unreferenced safety envelope reached")
                else:
                    actuator_set_target_speed(ACT_JOG_SPEED_STEPS_S)
                    actuator_jog_until = time.monotonic() + ACTUATOR_JOG_PULSE_S
                    print(f"[JOG] Actuator ->  pos={actuator_get_position()}")

            elif key == ord("e") and jog_mode:
                with _actuator_lock:
                    current_override = _actuator_calibration_override
                new_override = not current_override
                actuator_set_target_speed(0.0)
                actuator_set_calibration_override(new_override)
                print()
                if new_override:
                    print(
                        "[CALIBRATION] EXTENSION mode ON -- U/O can now move "
                        "past the saved actuator MIN/MAX. "
                        f"Absolute envelope: +/-{ACTUATOR_CALIBRATION_EXTENSION_LIMIT} steps. "
                        "Manually watch the mechanism; press 4 at the new MAX."
                    )
                else:
                    print("[CALIBRATION] EXTENSION mode OFF -- saved limits enforced")

            elif key == ord("1") and jog_mode:
                ST_VISIBLE_MIN_POSITION = int(servo_position)
                save_calibration()
                print(f"[CALIBRATION] Servo MIN marked at {ST_VISIBLE_MIN_POSITION}")

            elif key == ord("2") and jog_mode:
                ST_VISIBLE_MAX_POSITION = int(servo_position)
                save_calibration()
                print(f"[CALIBRATION] Servo MAX marked at {ST_VISIBLE_MAX_POSITION}")

            elif key == ord("3") and jog_mode:
                # This is the IMPORTANT per-run reference operation.
                # The physical MIN endpoint becomes coordinate 0, so the
                # saved travel length has a stable meaning across restarts.
                actuator_set_target_speed(0.0)
                actuator_mark_min()
                actuator_set_calibration_override(False)
                if ACTUATOR_MAX_POSITION > 0:
                    save_calibration()
                    print(
                        "[CALIBRATION] Actuator MIN referenced at 0. "
                        f"Saved travel={ACTUATOR_MAX_POSITION} steps. "
                        "The actuator is now safe for R/T/G."
                    )
                else:
                    print(
                        "[CALIBRATION] Actuator MIN referenced at 0. "
                        "Now jog to physical MAX and press 4 to establish travel."
                    )

            elif key == ord("4") and jog_mode:
                if actuator_mark_max():
                    actuator_set_calibration_override(False)
                    save_calibration()
                    print(
                        f"[CALIBRATION] Actuator MAX marked at "
                        f"{ACTUATOR_MAX_POSITION}; travel saved."
                    )
                else:
                    print(
                        "[CALIBRATION] Mark MIN first (press 3 at physical MIN), "
                        "then jog toward MAX and press 4."
                    )

            elif key == ord("t"):
                if sweep_active:
                    sweep_active = False
                    sweep_leg = None
                    sweep_mode = None
                    sweep_selection_pending = False
                    sweep_servo_commanded = False
                    actuator_set_target_speed(0.0)
                    print()
                    print("[SWEEP] Aborted")
                elif target_x is not None:
                    print()
                    print("[SWEEP] Cancel the active target first (C)")
                elif jog_mode:
                    print()
                    print("[SWEEP] Exit JOG mode first (M)")
                else:
                    actuator_set_calibration_override(False)
                    return_to_middle = False
                    _jitter_active = False
                    sweep_active = False
                    sweep_leg = None
                    sweep_mode = None
                    sweep_selection_pending = True
                    sweep_servo_commanded = False
                    actuator_set_target_speed(0.0)
                    print()
                    print("[SWEEP] Choose sweep:")
                    print("        1 = SERVO only")
                    print("        2 = ACTUATOR only")
                    print("        3 = BOTH")
                    print("        C = cancel")

            elif key == ord("j"):
                if target_state == "JITTERING":
                    print()
                    print("[JITTER] Already jittering")
                elif laser_x is None:
                    print()
                    print("[JITTER] Laser not currently detected -- can't seed a position")
                else:
                    if target_x is None:
                        target_x = int(laser_x)
                        target_y = int(laser_y) if laser_y is not None else 0
                        target_state = "IDLE"
                    target_state = "JITTERING"
                    jitter_start_time = time.monotonic()
                    jitter_base_position = servo_position
                    actuator_set_target_speed(0.0)
                    time.sleep(0.05)
                    _jitter_start_time = time.monotonic()
                    jitter_start_time = _jitter_start_time
                    _jitter_base_position = jitter_base_position
                    _jitter_active = True
                    print()
                    print(
                        "[JITTER] Manually triggered at current position "
                        "(servo only; actuator frozen)"
                    )

    finally:

        print()
        print("Returning servo + actuator to middle position...")

        try:
            _jitter_active = False
            actuator_set_calibration_override(False)

            set_servo_position(
                ST_MIDDLE_POSITION,
                speed=SERVO_SPEED_NEAR,
                acc=ST_ACCEL
            )

            if actuator_is_referenced():
                middle = actuator_get_middle_position()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    speed = actuator_target_speed_for_target_position(middle)
                    actuator_set_target_speed(speed)
                    if (
                        abs(actuator_get_position() - middle) <= 5
                        and abs(actuator_get_speed()) < 20
                    ):
                        break
                    time.sleep(0.01)
                actuator_set_target_speed(0.0)
            else:
                actuator_set_target_speed(0.0)

            time.sleep(0.2)
        except Exception as e:
            print(f"Could not return to middle: {e}")

        # Servo: left with torque on at its last commanded position --
        # the ST3215 holds position with force by default (that's
        # what the "Torque" state means for this servo), no special
        # action needed to keep it from drifting the way the old PWM
        # servo could.

        # Actuator: the calibrated travel is software-enforced after the
        # physical MIN reference. Return-to-middle is completed above, then
        # the stepper is stopped and disabled.
        actuator_set_target_speed(0.0)
        _jitter_active = False
        _jitter_running = False
        if _jitter_thread is not None:
            _jitter_thread.join(timeout=1.0)

        time.sleep(0.5)
        _actuator_running = False
        actuator_thread.join(timeout=1.0)
        act_disable()
        GPIO.cleanup()

        close_st_servo()

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