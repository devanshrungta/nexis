# Laser Weeding Rig — Learning Summary

This file records the major design changes and what they taught us. Status labels: **[HW]** user-confirmed on hardware; **[SIM]** synthetic/local logic test only; **[CANDIDATE]** code created but not yet verified on the rig; **[OPEN]** unresolved.

## Current direction (October 2026)

- **[HW]** Current rig uses an ST3215 servo for image Y and a stepper linear actuator for image X. The camera is an ELP-style USB camera at `/dev/video0`; servo communication goes through the Waveshare ESP32 driver in Serial Forwarding mode at 115200 baud.
- **[HW]** Calibration map is `position_map.json`; latest run reported a `SCATTERED` map with 44 multi-placement samples, training RMSE 9.4 servo positions and 122.2 actuator steps.
- **Reason for the map:** outdoor laser detection was fooled by reflections. The calibrated path converts the current target pixel directly to absolute motor positions and does not search for the laser spot at runtime. This is simpler but open-loop; it cannot correct unexpected missed steps or changing camera/target-plane geometry.
- **[CANDIDATE]** Latest controller `laser_servo_tracker_yolo_stable_auto_visit.py` adds manual-point, YOLO single-weed, and `A` auto-visit modes. It stabilizes detections over time, tracks targets between detector results using optical flow, and visits confirmed weeds one by one with settle → jitter → next target. Syntax and isolated logic tests passed; the integrated moving-camera and motor workflow has not been hardware-verified.
- **[HW]** YOLO GPU environment was verified: Python 3.10; PyTorch 2.8.0 / CUDA 12.6; torchvision 0.23.0; NumPy 1.26.4; OpenCV 4.11.0; Ultralytics 8.4.177; CUDA available and GPU named Orin.
- **Important launcher lesson:** use `sudo /home/sunoj-orin/venvs/weed-yolo/bin/python ...` or `run_yolo_stable_auto_visit.sh`. Plain `sudo python ...` uses system packages (previously PyTorch 2.14.1+cu130, NumPy 2.2.6, OpenCV 5.0), produced a CUDA warning and fell back to CPU inference. Running without sudo could not open `/dev/ttyUSB0`.

## Timeline: tried → learned → changed

### 1. One DOF: PWM hobby servo and laser-dot feedback

- **[HW]** Used Jetson sysfs PWM at 50 Hz, 500–2500 μs, neutral 1500 μs; gray-threshold laser detection and largest bright blob.
- P control overshot, so deadband, step clamps, EMA smoothing, easing and settle grace were tried. EMA state had to live outside the frame loop; otherwise it reset every frame.
- Sending commands every frame caused queued motion while the servo was stuck, then a large overshoot (159 px in a recorded test). Updates had to be paced.
- Jitter could appear random when faster than camera sampling. Reflections near the image edge could become the “largest blob” and cause false aiming or oscillation.
- **Why we moved on:** PWM servo stick-slip, coarse response and no position readback made repeatable precision difficult.

### 2. ST3215 servo + stepper actuator: two DOF

- **[HW]** Changed to ST3215 serial servo plus a linear actuator. Waveshare board was not a plain USB-to-serial adapter; raw servo packets required Serial Forwarding mode and the USB side ran at 115200 baud.
- `st3215` constructor initially appeared to use a different baud; setting `servo.portHandler.ser.baudrate = 115200` after construction fixed communication. Servo ID 1 responded.
- Running under `sudo` changes which Python packages are visible. `st3215` was present for the regular user but initially missing under root; later the YOLO use case required a root-run virtual environment instead.
- Servo travel registers `0/0` mean multi-turn/unlimited, not a literal 0-to-0 position range. Treating them as a range froze commands at zero.
- Repeated `MoveTo` calls restart servo motion ramps. Update throttling matters. Always retain the returned/clamped position and do not confuse commanded servo position with actual readback.
- Actuator has no encoder or limit switches. Its software counter only counts pulses; missed steps or a hard-stop stall are not detected. The startup midpoint prompt and MIN/MAX referencing reduce ambiguity but do not provide true feedback.
- Earlier timed `U/O` jog pulses were inconsistent because the pulse was shorter than the acceleration time. The latest candidate uses fixed-distance jog targets. A prior shutdown/return method repeatedly reversed velocity around the midpoint; the latest candidate uses an absolute position target. These fixes still need physical verification.

### 3. Vision-only target tracking

- **[HW]** A clicked image point can be followed without IMU, GPS or odometry. `VisionTargetTracker` uses local Lucas–Kanade optical flow around the target, background features and a homography, with fusion/smoothing and lost-frame handling.
- LOCAL tracking can follow an independently moving target; GLOBAL background motion helps follow a world-fixed point as the camera/rig moves. The target must remain visible and trackable.
- This tracks a **target image point**, not a weed class and not the actual laser spot. YOLO was added later for weed detection.

### 4. Position-map calibration replaced runtime laser detection

- **Reason:** largest-bright-blob laser detection was not dependable outdoors because reflections could beat the real spot. A detector lock/gating experiment either rejected genuine dots or selected the wrong bright object.
- A4 sheet points were collected at multiple placements. The current map uses scattered-triangle interpolation within the measured sample hull and a global quadratic model outside it. The latest reported map had 44 samples.
- This removed laser detection from normal aiming: pixel → mapped servo position + actuator position. It avoids false laser-spot detection but cannot close the loop on the actual impact point.
- Physical accuracy still needs independent validation across the workspace. A low training RMSE is not proof of low laser miss distance. Extrapolation is especially risky; current auto-visit candidate disables extrapolation by default.

### 5. Automatic laser calibration attempt — on hold

- Goal was to place the A4 sheet at different workspace locations, detect its grid and the laser spot, and automate collection of calibration samples. Reflection-exclusion polygons were added as an idea to mask known trouble regions.
- **[HW] Failure:** laser spot detection was inconsistent and sometimes missed the spot. A separate run also failed in `st3215.MoveTo()` → `ReadPosition()` with `IndexError: list index out of range` during serial response parsing.
- Tightening acceptance to 2.5 px cannot ensure real accuracy if the spot detector or inferred printed-grid locations are wrong. The user chose to set this work aside rather than trust bad samples.

### 6. YOLO weed detection and auto-visit mode

- YOLO mode uses the `.pt` model (currently `/home/sunoj-orin/devansh/camera/best 1.pt`) and supports all classes or an explicit list of class IDs. Box center is the initial aim point; lower-center is an alternative but neither guarantees hitting the crown.
- **[CANDIDATE]** Temporal stabilization requires multiple detections to confirm a weed, smooths box and confidence, tolerates short detection dropouts, conservatively re-associates targets, and uses optical flow between detections. This reduces box jitter but cannot always recover an identity after occlusion or target loss.
- **[CANDIDATE]** `A` mode queues confirmed weeds and aims at each, waits for both axes to settle, runs servo-only jitter, then advances. It favors shorter motor travel, drops stale queue entries, and by default refuses extrapolated map targets and targets near actuator ends. Needs hardware testing before field use.
- **FPS investigation:** below-3-FPS runs logged `device=cpu` because the system Python was used. CUDA worked when the `/home/sunoj-orin/venvs/weed-yolo` interpreter was tested. The controller and its inference FPS still require on-rig benchmarking; a smooth UI does not mean the model itself runs at camera FPS.

## Open items

- **[OPEN]** Reconcile `ST_MAX_POSITION = 2440` with calibration files/logs that report visible max 2500. The controller applies both clamps, so the hard-coded bound currently wins.
- **[OPEN]** Validate pointing error at independent check points across the full work area; do not use training RMSE as the accuracy result.
- **[OPEN]** Confirm actual YOLO inference FPS with the venv, then benchmark TensorRT if needed. Run the stable/auto-visit candidate on the rig before depending on it.
- **[OPEN]** Check whether the target surface is always the same plane and working distance. The camera-to-laser offset makes mapping depth-dependent.
- **[OPEN]** Measure real actuator step rate and check for lost steps under load. The current counter cannot detect them. Physical end protection remains absent.
- **[OPEN]** Separate motion time, settle dwell, and jitter time when reducing time per weed. Do not raise acceleration/speed without tests for vibration and lost steps.
- **[OPEN]** No firing/kill-verification control has been demonstrated in the current Python controller.

## Mistakes to avoid

- Do not run YOLO with the wrong Python environment. With root hardware access, use the venv's absolute interpreter path.

## Numbers and constraints worth remembering

| Item | Value / note |
|---|---|
| Camera | `/dev/video0`, 1280×720, MJPG, 60 fps requested |
| Servo transport | `/dev/ttyUSB0`, 115200 baud, servo ID 1 |
| Servo absolute hardware range in latest candidate | constants 1740–2440 (can calibrate your self during the start of the script according to your need and the safety requirements if any); calibration logs have shown visible range 1740–2500 |
| Actuator driver/control | TMC2209; Jetson BOARD pins STEP 29, DIR 31, EN 33; active-low enable |
| Current candidate actuator profile | max 10,000 steps/s; acceleration 5,000 steps/s² (intentionally made slow, since otherwise the linear actuator makes noise, or else you can have whatever you want. To achieve max speed out of the hardware, you need to learn and implement about micro-stepping) |
| Current candidate fixed jog | 100 steps per accepted `U/O` command; 0.18 s debounce |
| Jitter candidate settings | 20 servo positions amplitude, 3 s duration, 6 Hz command reversals, speed 3400, acceleration 254; physical behavior must be checked on rig |
| Actuator feedback | none; no encoder or physical end switches |
| Laser | 40 W Class 4, nominal 0.3175 m above ground; preserve E-stop/interlock |

---

# Appendix — Earlier or Currently Unused Work

## A. ST3215 / Waveshare notes

- ST3215 uses a 12-bit position range (0–4095); documented 4096 counts per 360° (about 0.0879°/count). Factory ID 1; servo-bus default baud 1,000,000, but this setup forwards over USB at 115200.
- Acceleration register uses one byte, 0–254, in units of 100 steps/s². Documented speed maximum is about 3400 steps/s; confirm library/firmware semantics before relying on the value.
- Useful registers: `0x05` ID, `0x06` baud index, `0x08` reply level, `0x09/0x0B` min/max angle (`0/0` means multi-turn mode), `0x21` mode, `0x28` torque/middle operation, `0x29` acceleration, `0x2A` target position, `0x30` torque limit, `0x38` present position, `0x3A/0x3C` speed/load, `0x3E/0x3F` voltage/temperature, `0x42` moving flag, `0x45` current.
- Packet format: `FF FF | ID | LEN | INSTR | PARAMS | CHK`; checksum is `~(ID + LEN + INSTR + params) & 0xFF`. The `st3215` library exposes `PingServo`, `ReadPosition`, `ReadSpeed`, `ReadLoad`, `ReadVoltage`, `ReadCurrent`, `ReadTemperature`, `IsMoving`, `SetSpeed`, `SetAcceleration`, `MoveTo`, `WritePosition`, `Rotate`, `SetMode`, torque controls and calibration helpers. Many methods return `None` on communication failure.
- Waveshare driver is an ESP32 servo controller, not a generic USB-TTL adapter. Serial Forwarding must be enabled for raw servo packets. The board firmware/menu steps recorded in earlier sessions were not independently verified for every board revision.

## B. Retired scripts and experiments

- `st3215_keyboard_control.py`, `st3215_sweep_control.py`: early servo bring-up.
- `laser_servo_tracker.py`: older two-DOF laser-spot tracker.
- PWM/Adam-style tracker: had target queue, jitter, CSV logs and video; replaced because of stick-slip, no position readback and unreliable blob detection.
- `laser_servo_tracker_stationary_minimal_debugged.py`: earlier stationary controller with jitter diagnostics.
- `laser_position_autocalibration.py` / fixed variants: automatic A4 calibration attempts; currently not trusted because laser detection was inconsistent and serial read errors occurred.

## C. ROS 2 calibration pipeline proposal — not run on hardware

Planned package `laser_weed_calib`: camera → `weed_detection_node` (`Detection2DArray`) → coordinate transform → pan/tilt commands → Arduino servo driver. Additional nodes were proposed for VL53L0X ground range and image-based aim/kill verification.

- Calibration concept: generate an ArUco grid and world-point CSV; calibrate camera intrinsics from 15–25 varied images; calibrate the ground plane with camera pose and laser x/y offset/height; redo ground-plane calibration if the mount moves.
- Geometry concept: undistort image pixel → camera ray → world frame → intersect flat ground plane; reverse-project commanded pan/tilt to image for aim verification.
- Arduino protocol proposal: 115200 baud ASCII lines `P<pan>,T<tilt>\n`; `H` detach, `R` reattach; pins 9 and 10; firmware clamps angles and detaches after a 1 s command watchdog.
- **Not implemented/verified:** fire-control node and `/laser_fire_event` publisher, laser trigger, ROS nodes running on hardware, compiled Arduino firmware, the VL53L0X height sign/offset, and pan-zero direction. This pipeline targets a different pan/tilt architecture than the current ST3215 + linear-actuator pixel-map rig.

## D. Other ideas not currently used

- Robot odometry/IMU/GPS to maintain world coordinates; deferred in favor of camera-based target tracking.
- Other actuator options considered: Dynamixel XM-series, digital PWM servos, brushless gimbal servos, closed-loop steppers and galvo mirrors. None were evaluated on this rig.
- Servo EEPROM min/max registers as an added safety layer; not exposed/validated in current code. `TareServo` and `getBlockPosition` auto-limit ideas were not used.
- Old PWM rig: `/sys/class/pwm/pwmchip3/pwm0`, 20 ms period, 500–2500 μs pulse, 1500 μs neutral, observed roughly 2–3.5 px/μs at the test distance; 10 μs sometimes did not overcome stiction while 30 μs did.
- Earlier serial timing measurements: a servo write took about 16.5 ms; measured jitter ceiling was about 30 Hz in that setup. Do not assume the same timing on different firmware or controller versions.

## E. Historical control values and timing notes

These describe older versions and should **not** be copied into the current controller without checking its actual code and logs.

- Original closed-loop servo axis: `CONTROL_UPDATE_INTERVAL = 0.01 s`, EMA `alpha = 0.4`, `SERVO_SIGN = -1`, gain `0.25 position/px`, maximum 30 servo steps per update, deadband 20 px, speed 100–6000, `ST_ACCEL = 200`.
- Original closed-loop actuator axis: gain `0.7 steps/px`, deadband 7 px, acceleration 5000 steps/s², max speed 15000 steps/s, endpoint tolerance 5 steps. The map-based controller now sends absolute positions instead of continuously applying this pixel-error law.
- Earlier target-tracking parameters: local Lucas–Kanade optical flow with partial-affine RANSAC; background homography; blend local/global 0.75/0.25 when they agree within 25 px; smooth alpha 0.8; reject jumps above 160 px; mark lost after 8 bad frames.
- Old laser detector: grayscale threshold 240–255, morphological opening with kernel size 20, accept blobs area ≥50 px², choose the largest blob centroid. Outdoor reflections made this unreliable.
- Earlier jitter timing test: a serial position write took about 16.5 ms; measured ceiling was around 30 Hz in that setup. One older version used 5 Hz; the current candidate is configured to 6 Hz. Do not assume either value equals physical movement frequency without checking logs.
- Earlier actuation estimate: ideal triangular move time is approximately `T = 2√(d/a)`. For 3000 steps and acceleration 5000 steps/s², this is about 1.55 s before other overhead. A 1 s move over 3000 steps would require about 12000 steps/s² under the idealized model; this was a calculation, not a hardware recommendation.
- Known timing limits: Python bit-banged STEP pulses are not hard real-time. The actual step rate under load should be measured; Linux scheduling and GPIO overhead can cause jitter or missed motion.
- Questions recorded in earlier sessions: working-distance range and camera/laser baseline; whether targets are always on flat ground; measured steps per pixel; motion vs dwell vs jitter time per target; real step rate; and whether high acceleration causes missed steps.

## F. References

- `python-st3215`: https://github.com/Mickael-Roger/python-st3215
- Waveshare ST3215: https://www.waveshare.com/wiki/ST3215_Servo
- Jetson AI Lab package index used for JetPack 6 / CUDA 12.6 wheels: https://pypi.jetson-ai-lab.io/jp6/cu126
- Ultralytics Jetson guide: https://docs.ultralytics.com/guides/nvidia-jetson/
- Waveshare Communication Protocol manual: `Communication_Protocol_User_Manual-EN_191218-0923_.pdf` (local project reference)

## G. Adding notes from another session

Use a short entry and label each claim:

```text
### Session: <who/chat, date>
- Done:
- Worked (how verified):
- Failed / mistakes:
- Change made and reason:
- Open items:
```
