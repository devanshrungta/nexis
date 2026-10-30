# Development and Learning Summary

This document records the main engineering decisions, what they revealed, and the current open work. It complements the [system guide](HOW_TO_USE_SYSTEM.md); setup instructions belong there.

## Current system snapshot

- The active controller is `laser_position_calibrated_control_package/green_weed_moving_tracker.py`.
- Perception uses hard-coded HSV green segmentation and contour tracking. The tuner is separate; the active tracker does not load its JSON output automatically.
- Target image coordinates are converted to servo position and actuator steps through `position_map.json`. The supplied map is a scattered map with 32 samples at 1280 x 720.
- The actuator is open-loop. Position is estimated from emitted step pulses, with no encoder or end switches to detect missed steps or mechanical stalls.
- The current script limits servo targets to IDs 1930-2171 around center 2056. It retains a software-only simulated treatment stage; power and duration control for the laser is not implemented in this controller.
- The controller records annotated video and uses an explicit startup midpoint confirmation and `GO` gate before motion.

## How the design evolved

### 1. One-axis PWM servo and laser-spot feedback

- The early setup used a hobby servo driven through Jetson PWM at 50 Hz, with a roughly 1500 microsecond neutral pulse and a broad pulse range.
- Bright-spot detection and proportional corrections were tested. Overshoot, stick-slip, low-resolution movement, reflections, and lack of reliable position feedback made repeatable aiming difficult.
- Sending commands on every camera frame could restart motion ramps and produce overshoot. Smoothing, deadbands, and command pacing helped but did not solve the hardware limitations.

### 2. ST3215 servo and linear actuator

- The system moved to an ST3215 serial servo for one axis and a TMC2209-driven stepper linear actuator for the other.
- The Waveshare ESP32 board requires Serial Forwarding for raw ST3215 control. The USB-facing link is 115200 baud; the `st3215` library constructor initially opened the port at a different baud, so the underlying serial baud had to be set explicitly.
- The actuator has no encoder or limit switches. Its counter records commanded step pulses, not independently measured carriage position. A physical reference must be established at startup, and lost steps remain a possible source of error.
- Software limits and safer stopping behavior were added because a lost reference or an unchecked target could drive the actuator to an end. Software limits are not a substitute for physical switches or independent motion feedback.

### 3. Camera tracking and YOLO experiments

- Camera-only tracking was explored so the system could follow a point while the camera or target moved. Optical flow around a selected point and background motion estimation were developed in an earlier tracker.
- YOLO-based detection and automatic target visiting were added later. The integrated YOLO path remains in the repository for experiments and calibration utilities, but the current moving dummy-weed test uses green thresholding because the tested YOLO models did not reliably detect that target.
- The active green tracker should therefore be treated as the current runtime entry point. Older controller filenames remain for their calibration and diagnostic functions, not as interchangeable replacements.

### 4. A calibrated position map replaced runtime laser-spot search

- The earlier bright-blob detector could select reflections instead of the real laser spot, particularly under difficult lighting and backgrounds.
- The current positioning path maps a target camera pixel to absolute servo and actuator commands. It does not close the loop by detecting the actual laser spot after each movement.
- `position_map.py` supports older regular-grid maps and the newer scattered map. The scattered map uses triangle interpolation inside the convex hull of recorded samples and a quadratic estimate outside the hull.
- The runtime tracker rejects extrapolated points for motion. When an intended target is outside the hull, the safe response is to inspect the coverage and collect more calibration points, not to turn off the rejection.
- A4 sheet placements make it easier to spread the measured points across one physical target plane. Other physical marks can also be used when the image point and corresponding motor position are paired carefully.

### 5. Calibration and repeatability testing

- `calibrate_position_map_multiplacement.py` records multiple image/motor correspondences across placements of the calibration sheet. It now attempts to save a timestamped annotated screenshot when the map is saved.
- `position_map_hull_inspector.py` makes the sample hull visible and helps distinguish an outside-hull point from a point that is inside the map region.
- `click_point_position_test.py` isolates the position map from weed detection by commanding selected image points directly.
- `laser_repeatability_test.py` records target points, motor commands, move times, observed laser-spot centers when manually measured, and annotated video/CSV outputs. Millimetre metrics require `pixel_to_mm_homography.json`; pixel error alone is not physical accuracy.
- A point-to-position test demonstrated that the calibration map, rather than only weed detection, can be the source of pointing error. Errors that change sign across the workspace usually require map/geometry or actuator repeatability analysis, not a single global offset.

### 6. The 2026 poster and the current code describe different stages

The current active tracker has moved to green HSV segmentation for the dummy-weed test and does not contain laser firing control. The images remain useful as historical hardware context, but they should not be read as proof that the current controller supports laser activation or field operation.

<img src="media/Laser-annotated.png" alt="Earlier enclosed laser setup" width="300">

<img src="media/laser_actuation_setup_annotated.png" alt="Current actuation setup" width="300">

## Earlier experiments retained for reference

- **PWM control:** the first axis used Jetson PWM at 50 Hz with about 500-2500 microsecond pulses and a 1500 microsecond neutral point. Stick-slip, coarse movement, queued commands, and overshoot limited repeatability.
- **Bright laser-spot detection:** an older detector used a grayscale threshold of 240-255, morphological opening, and the largest bright blob. Reflections could be selected instead of the true spot, so this is not the active green tracker approach.
- **ST3215 details:** the servo uses a 12-bit position scale of 0-4095 (4096 counts per full rotation). The Waveshare ESP32 board needs Serial Forwarding; the USB-side link is 115200 baud. A write time of about 16.5 ms and a command-rate ceiling near 30 Hz were measured in an earlier setup and should not be assumed for every firmware or runtime.
- **Actuator profiles:** earlier controllers used higher requested speed and acceleration values, including 10,000 steps/s and 5,000 steps/s^2. The current green tracker deliberately uses 2,500 steps/s and 1,250 steps/s^2 while checking accuracy because the actuator has no encoder. The software count can diverge from physical position if steps are missed.
- **Older servo limits:** prior files referenced other ranges, including 1981-2541 and 1740-2440. These are historical values, not the current green tracker's 1930-2171 software window.
- **Automatic calibration:** `laser_position_autocalibration.py` and related experiments tried to collect grid points automatically from the A4 sheet and detected laser spot. Spot detection and serial reads were unreliable, so those results are not trusted as the current calibration path.
- **Retained controllers:** `laser_servo_tracker_actuator_jog_shutdown_fixed.py` is an older non-YOLO controller; `laser_servo_tracker_yolo_stable_auto_visit.py` is the older YOLO controller used by a retained launcher and calibration utilities; `servo/laser_servo_tracker_vision_moving_target.py` contains an earlier optical-flow target tracker. The active weed-tracking entry point remains `green_weed_moving_tracker.py`.
- **Jetson Python environment:** earlier YOLO tests ran slowly when the system Python used CPU inference. The virtual environment at `/home/sunoj-orin/venvs/weed-yolo` was verified to provide the intended Orin/CUDA stack. The documented absolute interpreter path is retained to avoid silently selecting different system packages under `sudo`.
- **Earlier map metrics:** a previous map was reported with 44 samples and training RMSE values of 9.4 servo positions and 122.2 actuator steps. Those values belong to that earlier map and must not be attributed to the 32-sample map in the supplied archive. In either case, training RMSE is not a physical pointing-accuracy measurement.

## Important implementation lessons

- Camera resolution, view geometry, camera placement, and target plane must remain consistent between calibration and runtime. Same pixel dimensions alone do not prove that the camera sees the same geometry.
- Calibration samples must surround the useful workspace. A training RMSE is not the same as physical targeting accuracy; validation points should be independent of the points used to fit the map.
- The actuator is open-loop. Slower speed and acceleration can improve physical repeatability, but physical endpoints must still be measured independently.
- A bounding-box center can shift when visible leaves change shape. Target point, camera-to-motor mapping, and physical actuation need separate measurements so that a change in one layer does not hide an error in another.
- The tuner and tracker have separate settings paths by design in the current source. The tuner JSON is not automatically loaded by the tracker; the chosen HSV values must be copied into its constants.
- Some old scripts use relative file paths. Launch them from `laser_position_calibrated_control_package` so the intended map and calibration files are used.
- The standalone `servo/st3215_servo.py` has a 45-degree default sweep range around its logical midpoint, but manual jog commands in the supplied source can use 0-4095. The active tracker's narrower software limit is not shared automatically with that utility.
- The calibration screenshot save block in the supplied archive refers to `SCRIPT_DIR` without defining it. The map is saved before that code path, but snapshot saving can then raise `NameError`; this should be fixed before relying on the annotated screenshot feature.
- A saved actuator midpoint does not automatically prove the carriage is at the midpoint. It must be physically placed there before the startup confirmation.

## Values and constraints worth retaining

| Item | Current code or saved value |
|---|---|
| Camera image size | 1280 x 720 pixels |
| Tracker timing constant | 30 FPS nominal; tracker reports actual camera FPS at startup |
| ST3215 serial | Servo ID 1, `/dev/ttyUSB0`, 115200 baud through Serial Forwarding |
| Servo software window | 1930-2171, center 2056 |
| Actuator travel | 12954 software steps; saved midpoint 6477 |
| Tracker actuator profile | 2500 steps/s maximum, 1250 steps/s^2 acceleration in this code snapshot |
| Green threshold | H 30-95, S 36-255, V 128-255 |
| Contour/morphology thresholds | Minimum area 2000 px; opening 3; closing 7 |
| Laser mount height | Approximately 12-13 inches (0.32 m) above ground, as recorded for the prototype |
| Position map | `position_map.json`, scattered map, 32 samples in the supplied archive |

## Current limitations and next work

- Validate the position map at independent physical points across the full workspace. Measure signed X/Y errors and radial error, not only training-point residuals.
- Measure actuator repeatability and direction-dependent error to detect backlash or lost steps. Add stall detection or closed-loop position feedback if the hardware supports it.
- Improve calibration coverage and test whether the target remains on the same plane. A change in height or depth requires a different geometric model or additional calibration.
- Finish and validate laser power/time triggering, interlocks, and aim/kill verification before treatment-efficacy tests. The present `SIMULATED_SHOOT` stage is not a laser-fire command.
- For a research paper, record detection precision/recall, target-point error, positioning RMSE and 95th-percentile error, repeatability, target acquisition/move time, and cycle time. Add weed mortality and crop injury only after controlled laser treatment is integrated.
- A ROS 2 pipeline for perception, coordinate transforms, motion commands, and aim verification has been sketched, but is not implemented or validated on the current hardware.

## Selected technical references

- [python-st3215 library](https://github.com/Mickael-Roger/python-st3215)
- [Waveshare ST3215 documentation](https://www.waveshare.com/wiki/ST3215_Servo)
- [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) (for retained YOLO experiments only)
