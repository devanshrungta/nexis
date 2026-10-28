# Laser Weeding Rig: README + Learning Summary

Merged from three sessions. **S1** = servo bring-up (ST3215). **S2** = control and tuning (PWM servo, then 2-DOF). **S3** = ROS 2 calibration and pipeline design. New sessions: add a section at the end of Part C (template in C9).

**Tags**
- **(verified)** user pasted output or confirmed on hardware.
- **(docs)** from manuals, papers or library docs.
- **(calc)** arithmetic from verified numbers.
- **(sim)** tested only on synthetic data in Claude's sandbox.
- **(untested)** / **(hypothesis)** reasoning only.
- Claude never had the hardware. "Works" means the user said so.

**Layout:** Part A = README (current system). Part B = learning summary (history). Part C = appendix (not in use now).

---

# PART A. README

## A1. What this is
- Aim a laser at a clicked point on a camera view with a 2-DOF mechanism.
- Use: kill weeds (horseradish research project) from a farm-ng Amiga robot at about 5 mph.
- Camera and mechanism share one frame. If the robot moves, the camera moves too.
- **Compute:** Jetson Orin Nano. Camera `/dev/video0`, 1280×720, MJPG, 60 fps requested.
- **Servo (image Y):** ST3215 serial bus servo on a Waveshare ESP32 driver board (12 V / 3 A, USB-C to Jetson). Driven from our Python, not the Waveshare web app.
- **Actuator (image X):** stepper linear actuator. Step/dir/enable on Jetson pins 29/31/33, `Jetson.GPIO`, bit-banged.
- **Laser:** LaserTree K40 40 W module, Class 4 (docs: product page). Mounted 12.5 in (0.3175 m) above ground.
- **Laser safety:** user says registration, eyewear and enclosure/interlock are done. Keep the e-stop path in any new code.
- **USB bridge:** Silicon Labs CP210x, `/dev/ttyUSB0` (verified).
- **Current script:** `laser_servo_tracker_vision_moving_target.py` (~2000 lines). User: "works like a charm".
- **Hard rule: values and logic were tuned by long testing. Do not change them. Add new things beside them.**

## A2. What the script does
- Click a target, press `G`. Both axes move until the laser dot is on the target.
- "Reached" = both axes inside deadband for 1 s. Then the servo jitters 3 s and returns to base.
- Follows a moving target or moving rig using only the camera (`V` toggles).
- Jog both axes, mark travel ends, sweep test, manual jitter.
- Saves calibration to `servo_calibration.json`, loads it next start.
- Clamps every servo command. Actuator has software travel limits and slows near the ends.
- Prints achieved vs requested jitter Hz and average write time.
- **Not in this script:** per-frame CSV logs, params dump, video recording (older versions had them).
- **Not reviewed by Claude:** code after about line 1480 (jog, sweep, jitter start, drawing, keys). Read it before editing.

## A3. How it works

**Laser detection:** gray threshold 240-255 → morphological open (kernel 20) → blobs with area ≥ 50 → centroid of the largest.

**Servo (image Y), every tick** (`CONTROL_UPDATE_INTERVAL` 0.01 s)
- Error = target_y − filtered laser_y (EMA α 0.4).
- Command = real position (read back) + `SERVO_SIGN` (−1) × clamp(0.25 × error, ±30 steps).
- Speed 100 → 6000 as error grows (full at ≥ 60 px). `ST_ACCEL` 200. Deadband 20 px.

**Actuator (image X), every tick**
- Error = target_x − filtered laser_x. Deadband 7 px.
- Desired = clamp(current + 0.7 steps/px × error).
- Speed = min(15000, √(2 × 5000 × remaining)), signed. Stops within 5 steps.

**Vision target tracking (`VisionTargetTracker`)**
- LOCAL: Lucas-Kanade optical flow on features near the click, partial-affine RANSAC.
- GLOBAL: background features + homography, so a world-fixed point follows camera motion.
- Fusion: if LOCAL and GLOBAL agree within 25 px, blend 0.75 / 0.25. Else LOCAL wins.
- Smoothing α 0.8. Jumps > 160 px in one frame rejected. Features re-detected when few. "Lost" after 8 bad frames.
- While lost: actuator stopped, servo held.
- No IMU, GPS or encoders.

**Calibration and safety**
- Servo absolute range 1981-2541 (560 steps ≈ 49°). `ST_MIDDLE_POSITION` 2150.
- Every servo command: clamp to visible range, then to absolute range.
- Actuator has no encoder or limit switch, so its step count means nothing between runs.
  - At start the script asks: is the actuator at its middle? If yes, counter = travel/2.
  - If no: `M` → jog to physical MIN → `3` → jog to physical MAX → `4`.
- Actuator thread enforces travel limits and a slow-down cap √(2·a·distance_to_end).

**Threads**
- Actuator: speed ramps at 5000 steps/s², `dt` capped at 0.03 s. Direction change goes through zero. One pulse per loop. `precise_wait` = sleep + ~40 µs busy-wait.
- Jitter: alternates two servo endpoints at `JITTER_FREQUENCY_HZ` (5 Hz), amplitude 20 steps, raw `WritePosition`. Speed/accel set once (3400 / 254).

## A4. Keys
| Key | Action |
|---|---|
| click | set target |
| `G` | start tracking |
| `V` | toggle vision tracking |
| `R` | return to middle |
| `C` | cancel |
| `T` | sweep test |
| `J` | jitter by hand |
| `M` | jog mode |
| `Q` | quit |

Jog mode: `I/K` servo −/+, `U/O` actuator −/+, `1/2` mark servo min/max, `3/4` mark actuator min/max, `E` extension mode (not reviewed).

## A5. First run
1. Power the servo board (12 V) first. An unpowered servo once looked like a software bug.
2. Board in **Serial Forwarding** mode. USB side **115200**.
3. `sudo usermod -aG dialout $USER`, log in again. Run **without** sudo.
4. Packages: `opencv-python`, `numpy`, `pyserial`, `st3215`, `Jetson.GPIO`.
5. `ST3215(device)` opens at 1,000,000. Set `servo.portHandler.ser.baudrate = 115200` right after. Close with `servo.portHandler.closePort()`.
6. Start, answer the actuator-middle question (or calibrate with `M`).
7. Give a target. On-screen error must **shrink**. If it grows, flip that axis's sign constant.

## A6. Troubleshooting
| Symptom | Likely cause | Fix |
|---|---|---|
| `Permission denied` on port | not in `dialout` | `usermod -aG dialout`, re-login |
| Port name unknown | enumerates differently | `ls /dev/ttyUSB* /dev/ttyACM*` |
| Reads `None`, writes "succeed" | wrong baud, no forwarding, wrong ID | forwarding on, 115200, ID 1 |
| Library ignores baud | constructor takes only `device` | set `portHandler.ser.baudrate` |
| `ModuleNotFoundError` under sudo | root has own site-packages | run without sudo |
| Servo does nothing, no error | not powered | check 12 V first |
| Servo stuck at 0 or one end | limits 0/0 read as range; wrong sign | 0/0 = unlimited; check sign |
| Swings to an end after `G` | wrong sign, error grows | flip sign (clamp holds it) |
| Jerky motion | new `MoveTo` every frame | throttle updates |
| Moves with no target | not software (log showed constant command) | check power, load, signal noise |
| False laser spot outdoors | reflections, noise | Goal A |

## A7. Open goals (proposals, nothing built on the rig)

### Goal A. Remove laser-spot detection
- **Why:** detection fails outdoors. Indoors, a reflection near the frame edge beat the real dot as "largest blob" (S2 logs).
- **Idea:** one-time calibration of pixel (u, v) → (servo_pos, actuator_pos). Aim open loop. No laser detection at run time.
- **Calibration sketch**
  1. Print a grid of markers (S3 made an ArUco sheet, see C6).
  2. Put the laser on each marker (jog by hand, or closed loop indoors).
  3. Record marker pixel + servo and actuator readback.
  4. Fit a mapping (polynomial or homography). Hold a few points out to check error.
- **Depth caveat (hypothesis):** laser and camera are at different points, so the mapping depends on distance. Laser is 0.3175 m above ground: on flat ground, distance follows from the pixel. Repeat at several places or print larger.
- **Risks:** no feedback, so the stepper must not lose steps; backlash; camera and laser must stay rigid.
- `VisionTargetTracker` still gives the moving target's pixel, so tracking carries over.

### Goal B. Faster actuation (5-10 s → < 1 s)
- **Measure first.** `SETTLE_DWELL_S` 1 s + `AUTO_JITTER` 3 s = 4 s fixed waiting (calc). Motion may be only 1-6 s.
- **Hardware is not the limit (calc).** Move time T = 2·√(d/a).
  - Servo, 560 steps at `ST_ACCEL` 200 (20000 steps/s²): ≈ 0.34 s.
  - Actuator: a = 4d/T². For d = 3000 steps, T = 1 s: a ≥ 12000 steps/s². Now 5000 → 1.55 s.
- **The loop only hops a short way ahead.** Servo ≤ 30 steps per update. Actuator target = current + 0.7 steps/px × error. Speed follows gain and loop rate.
- **√(2·a·remaining)** stays low if `remaining` is under-estimated (steps/px too small).
- **Serial cost:** one write ≈ 16.5 ms (verified). If a read is similar (hypothesis), read + move ≈ 33 ms, so a 0.01 s tick is not reachable.
- **Servo ramp:** a new `MoveTo` restarts the ramp. At accel 254, 3000 steps/s needs 0.12 s. Re-commanding faster never cruises (docs + calc).
- **Filter lag:** EMA 0.4 plus detection delay.
- **Real step rate** is bit-banged in Python, probably far below 15000 steps/s (hypothesis; measure).
- **`ST_SPEED_MAX` 6000** but register max is 3400 (docs). Effect untested.
- **Open-loop idea:** with Goal A, compute absolute targets once, send one command per axis with the √ profile. Test for lost steps by moving out and back to a physical mark.

### Questions to answer first
1. Working distance range and laser-to-camera baseline?
2. Targets on flat ground, or any depth?
3. Actuator travel in steps, and steps per pixel?
4. Time split of a typical move: motion / dwell / jitter?
5. Real actuator step rate?
6. Has the actuator ever lost steps at high acceleration?

## A8. Next: ROS 2 pipeline (S3, not run on any hardware)
- Package `laser_weed_calib` (details in C6): detect weed → bbox to real-world point → pan/tilt → servos → check aim and kill with the same camera.
- Checked **(sim)**: pixel → world → pan/tilt → pixel math under a tilted camera; ArUco sheet detection; serial line format.
- **Not done:**
  - No fire-control node. Nothing publishes `/laser_fire_event`. K40 trigger is not coded.
  - Arduino sketch never compiled. No node ever ran under ROS.
  - Weed model format assumed to be Ultralytics YOLO.
  - VL53L0X height sign/offset in `on_height` not checked.
  - Pan zero direction assumed to be world +X. Check on the rig.
- **Hardware conflict to settle:** the package outputs pan/tilt angles for two Arduino-driven servos. The tuned rig is ST3215 + stepper in pixel space. Options: (a) use S3 calibration to feed Goal A on the current rig; (b) build the pan/tilt head. Not decided.

## A9. Rules for anyone continuing
- Do not change tuned values or logic. Add beside them.
- Send every servo command through one clamped function (visible range, then absolute). Store the value it returns.
- Read the real servo position at start. Never auto-snap to "middle".
- Actuator steps are valid only after the middle question or a re-home.
- Do not trust old code comments. Verify with real output.
- Mark unverified claims.
- Measure: phase timestamps, step rate, jitter Hz. Log per-frame state and a params dump to CSV (S2 did; every diagnosis came from these).
- Ask the user to run introspection commands (C7). Do not guess APIs.
- Do not re-apply old numbers from earlier sessions (e.g. `SERVO_UPDATE_INTERVAL` 0.03 / 0.12, `ST_ACCEL` 254).

---

# PART B. LEARNING SUMMARY

## B1. Timeline
| # | Setup | Result | Why we moved on |
|---|---|---|---|
| 1 | PWM hobby servo, 1 DOF (X), laser-dot detection, click target | Tracks, but imprecise, stuck-prone | Stick-slip, no readback, noise. Wanted precise, strong, with feedback. Added a second axis |
| 2 | ST3215 serial servo + stepper actuator, 2 DOF | Works after bring-up and fixes | Rig will move (Amiga). Odometry idea deferred → vision tracking |
| 3 | Vision-only moving target (current) | "Works like a charm" | Open: laser detection outdoors, speed |
| 4 | ROS 2 calibration + pan/tilt pipeline (S3, design only) | Math checked in sim only | Not yet tied to the real rig (A8) |

## B2. Phase 1: PWM servo, 1 DOF (S2)
Setup: Jetson sysfs PWM `pwmchip3/pwm0`, 50 Hz, pulse 500-2500 µs, neutral 1500. Dot found by gray threshold 245-255, 3×3 open, largest blob.

**Tried → learned**
- P control only → overshoot. Added deadband, step clamp, EMA, Adam-style step with easing, settle grace.
- Adam step ≈ ±LR whatever the error. LR is a step size in µs, not a per-pixel gain. LR 0.05 → no motion; ~8 works. The easing term shrinks steps near target, not Adam.
- EMA variable created inside the loop reset every frame. Keep filter state outside.
- Update every frame → commands piled up while the servo was stuck, then released: 159 px overshoot (verified). Pace updates.
- SETTLED had no way back → stuck on "reached". Fix: re-check error after a 0.4 s grace. Without grace, jitter re-triggered tracking → endless jitter.
- Jitter 40 µs swept 150-285 px (≈ 2-3.5 px/µs). 10 µs gave no motion from rest (stiction). 30 µs moved (verified). Keep amplitude near the deadband.
- Jitter at 50 Hz sampled by a ~20 fps camera looked random (aliasing). Keep under half the fps, or use its own thread.
- Convergence was accurate: 0-11 px at "reached", ~6 px final (verified). Imprecision came from jitter size.
- Threshold 245 too tight → dropouts (lowered to 220-230, area 4-6). A reflection near the right edge (x ≈ 1240-1278) made "largest blob" jump → oscillation (CSV, verified).
- Nearest-blob gating rejected valid hits or locked the wrong blob → removed (user's call).
- "Moves with no target": log showed commanded pulse constant 1500 µs (verified), so not software. First case: unpowered servo. Later case unresolved (load, supply, noise?).

**Built:** target queue + `G`, jitter after reach, per-run logs (main CSV, extra CSV, params CSV), annotated + raw video.

**Why moved on:** stick-slip, coarse resolution, no readback. User chose ST3215 and added a linear actuator for the second axis.

## B3. Phase 2: ST3215 + actuator, 2 DOF

**Bring-up (S1)**
- Raw pyserial packets at 1 Mbaud → `Permission denied` → `dialout`.
- Writes ok, reads empty, servo still → board is a Waveshare ESP32 needing Serial Forwarding, USB at 115200. (Mistake: assumed a plain USB-TTL bridge.)
- Servo moved only 600-1300. Blamed limit registers. Wrong: they were 0/0 (disabled). Real cause unknown.
- Treated limits 0/0 as a literal range → froze servo at 0. 0/0 = unlimited.
- Counter tracked commanded position, not real → drift. Use `ReadPosition`.
- Moved to `st3215` library. `ModuleNotFoundError` under sudo. Constructor takes only `device`. Library swallows errors, returns `None`. Fix: set `portHandler.ser.baudrate`. **Working (verified).**
- Swapped three tracker functions: `setup_st_servo`, `set_servo_position`, `close_st_servo`.
- "Abrupt motion after Go": new `MoveTo` every frame restarted the ramp → interval 0.03 s. Actuator `dt` spiked (GIL) → capped.

**Audit and features (S2)**
- Startup snapped the servo to a fixed middle → read real position, no snap. Same bug in `R`: clamped return value thrown away.
- Actuator thread had no travel clamp → clamp in the step loop + slow-down near ends.
- One clamped `set_servo_position` for all commands. Calibration JSON. Jog / mark / sweep / jitter keys designed here.
- Jog was clamped by the old saved range, so the range could only shrink → open range while jogging, reload file on exit.
- No encoder or limit switch → step count meaningless across runs → re-home each run. (Now: the middle question.)
- "Search" recovery swept the actuator to 0 = MIN end → go to calibrated middle. Not during jitter. Actuator frozen during jitter.
- `IsMoving()` can return `None` → treat as "not arrived".
- 180° swing at first target: placeholder sign constants. The clamp bounded it.
- Accel unit = 100 steps/s². 40 → 0.75 s to 3000 steps/s. 254 → 0.12 s.
- Pixel-distance actuator speed schedule tried. Current script uses the √ profile.

**Jitter speed**
- Jitter thread with raw `WritePosition` (user's work).
- Verified: write ≈ 16.5 ms for amplitude 20-80 and 5-20 Hz. Achieved = requested at 5, 10, 20 Hz. Ceiling ≈ 30 Hz.
- FTDI `latency_timer` guess was wrong: adapter is CP210x (`lsusb` 10c4:ea60).
- Higher baud not available in forwarding mode; wire time only ≈ 1.3 ms (calc).
- Untested: `set_low_latency_mode(True)`; servo register 0x08 (response level 0). Hypothesis: the 16.5 ms comes from the board / forwarding.
- **Open:** jitter above ~30 Hz not achieved.

## B4. Phase 3: vision-only moving target (user's work)
- Reason: the rig will move. S2 discussed Amiga odometry + ground-plane math; user deferred it. Vision needs no extra sensors.
- Worked: "like a charm".
- New problems: laser detection outdoors, 5-10 s per target (A7).

## B5. Phase 4: ROS 2 calibration + pipeline (S3, design only)

**Tried → learned**
- Started from the literature. Closest refs (docs): Mwitta et al. 2024 (diode laser on pan/tilt over a Cartesian arm, YOLOv4-tiny, ROS state machine; pan/tilt only 47% single-pass kill, 63% with a 10° downward tilt; 8-9.5 s per weed) and LiteWeed 2026 (low-cost edge-AI, stop-and-align, >96% removal in field trials).
- First idea: 2D homography. Only valid for one flat plane. Switched to intrinsics + `solvePnP` + ray-cast to the ground plane, so a tilted camera is handled.
- Checkerboard → ArUco sheet: each marker has an ID, so partial occlusion is fine. One CSV holds ground-truth mm, so no manual point entry.
- Aim at bbox bottom-center, not centroid (closer to the stem/crown).
- PCA9685 → plain Arduino over USB serial. Only 2 channels. Added hardware detach on e-stop and a 1 s firmware watchdog. Angles clamped in both ROS and firmware.
- VL53L0X (30-1000 mm) added behind an `enabled` flag. Off or unwired → flat-plane fallback, no crash.
- Same camera used for two checks: aim error (reproject commanded pan/tilt, find blue blob nearby) and kill check (Excess Green drop before/after). Thresholds are placeholders.
- First synthetic test failed. Cause: the test camera pointed up (wrong rotation matrix). The library was fine. Fixed the test, then the round trip matched to 1e-13 mm **(sim)**.

## B6. Mistakes to avoid
- Assuming the driver board type. Check the hardware.
- Stating unverified facts (forwarding switch "on the screen", "~200 mA no-load", "ammeter too coarse"). Tag them.
- Guessing library signatures. Inspect them.
- Treating 0/0 limits as a range.
- Tracking commanded position instead of real position.
- Discarding a clamp's return value.
- Leaving placeholder gains and signs in.
- Changing numbers without reading logs first.
- Adding detection hacks without checking their cost.
- Gain meaning changes with the control law (Adam LR).
- Half-finished edits; unclear which file is latest.
- Shipping untested code. Run `py_compile`; missing `global` and forward references were caught this way.
- Sandbox has no internet and no hardware. The user must run introspection and hardware tests.
- (S3) Designing a new pipeline without checking the user's actual hardware first (A8 conflict).
- (S3) Writing a verification node that listens to a topic nothing publishes.

## B7. Numbers worth keeping
| Item | Value |
|---|---|
| ST3215 | 4096 steps = 360° (0.0879°/step) (docs) |
| Servo range used | 1981-2541 (560 steps), middle 2150 |
| Serial write | ≈ 16.5 ms; jitter ceiling ≈ 30 Hz (verified) |
| USB side baud | 115200 (forwarding mode) |
| PWM rig | ≈ 2-3.5 px/µs; stiction break-away 10-30 µs |
| Laser height | 0.3175 m |
| ArUco sheet (S3) | 15 markers, 35 mm, 12 mm gap, 223×129 mm; print at 100% |
| Arduino link (S3) | 115200 baud, 1 s watchdog |

---

# PART C. APPENDIX (not used now)

## C1. ST3215 reference (docs)
- 12-bit magnetic encoder. Range 0-4095. Middle 2048 after `DefineMiddle`. Multi-byte registers little-endian.
- Factory ID 1. Servo-bus baud 1,000,000. Baud index (0x06): 0-7 = 1M, 500k, 250k, 128k, 115200, 76.8k, 57.6k, 38.4k.
- Speed max 3400 steps/s (50 steps/s ≈ 0.732 RPM). Acceleration 0-254, unit 100 steps/s².

| Addr | Meaning | Notes |
|---|---|---|
| 0x05 | ID | 0-253, 254 = broadcast |
| 0x06 | Baud index | |
| 0x08 | Response level | 0 = no reply to writes; 1 = reply to all |
| 0x09 / 0x0B | Min / Max angle (2 B) | defaults 0 / 4095. **0 and 0 = multi-turn mode** |
| 0x1F | Position correction | ±2047 |
| 0x21 | Mode | 0 position, 1 speed, 2 PWM, 3 step |
| 0x28 | Torque switch | 0 off, 1 on, **128 = current position becomes 2048** |
| 0x29 | Acceleration | 1 B |
| 0x2A | Target position (2 B) | then time (0x2C), speed (0x2E) |
| 0x30 | Torque limit | |
| 0x37 | EEPROM lock | 0 = writes persist |
| 0x38 | Present position (2 B) | |
| 0x3A / 0x3C | Present speed / load | |
| 0x3E / 0x3F | Voltage (0.1 V) / temperature (°C) | |
| 0x42 | Moving flag | |
| 0x45 | Current (6.5 mA units) | |

**Protocol:** `FF FF | ID | LEN | INSTR | PARAMS | CHK`. `LEN` = params + 2. `CHK` = `~(ID + LEN + INSTR + params) & 0xFF`. Instructions: PING 01, READ 02, WRITE 03, REG WRITE 04, ACTION 05, RESET 06, SYNC READ 82, SYNC WRITE 83. Reply: `FF FF | ID | LEN | ERROR | PARAMS | CHK`. Broadcast 0xFE gets no reply (except PING). S1's hand-written packets matched the manual's examples.

**`st3215` library:** `PingServo`, `ListServos`, `ReadPosition`, `ReadSpeed`, `ReadLoad`, `ReadVoltage`, `ReadCurrent`, `ReadTemperature`, `IsMoving`, `SetSpeed`, `SetAcceleration`, `MoveTo(id, pos, speed=2400, acc=50, wait=False)`, `WritePosition`, `Rotate`, `SetMode`, `StartServo`/`StopServo` (torque), `CorrectPosition`, `DefineMiddle`, `TareServo`, `getBlockPosition`, `ChangeId`, `ChangeBaudrate`, `LockEprom`/`UnLockEprom`. Most return `None` on error.

## C2. Board notes
- Waveshare driver is **not** a plain USB-TTL adapter. Own ESP32 firmware (WiFi / web app by default).
- Raw bytes reach the servo bus only in "Serial Forwarding" mode. USB side 115200.
- Wiki: sketch `ServoDriverST.ino` has a `SERIAL_FORWARDING` flag; if true the board boots into forwarding (docs, never done as far as recorded).
- Claude said to pick "Start Serial Forwarding" from the board menu. From wiki wording only. Unverified for this board revision.

## C3. Older scripts
- `st3215_keyboard_control.py`: hand-written protocol, pyserial. Keys `a/d`, `[ ]`, `m`, `SPACE`, `t`, `i`, `q`. Superseded.
- `st3215_sweep_control.py`: uses `st3215`. `c` = `DefineMiddle`, `m`, `- =` sweep range (±30° default, max ±180), `s` sweep, `t`, `i`, `q`. `DefineMiddle` and sweep not individually confirmed.
- `laser_servo_tracker.py`: older 2-DOF tracker with ST3215 layer (three functions + two fixes). Superseded.
- S2 PWM tracker: Adam + easing, jitter, target queue, logs, video. Superseded.
- "minimal debugged" version: jitter thread + achieved-Hz print. Folded into the current script.

## C4. Old PWM rig facts
- sysfs `/sys/class/pwm/pwmchip3/pwm0`, period 20 ms, pulse 500-2500 µs, neutral 1500 µs.
- ≈ 2-3.5 px per µs at the camera distance used.

## C5. Ideas not used
- **Odometry / world lock:** Amiga ROS bridge gives wheel, IMU, GPS. At click, store pose; each frame, reproject the ground point by distance travelled using mount height 0.3175 m. Deferred by user.
- **Other hardware considered:** Dynamixel XM-series, digital metal-gear PWM servos (e.g. Hitec HS-7955TG), brushless gimbal servos, closed-loop steppers. Galvo mirrors are the usual choice for fast laser aiming (idea). None evaluated.
- **Servo EEPROM min/max angle (0x09 / 0x0B)** as a hardware safety net. Not exposed by the library. Untested.
- `TareServo` / `getBlockPosition` to auto-find limits. Not used.
- Second repo the user mentioned: `github.com/alessiodam` (not examined).
- Other laser-weeding refs from S3 (not read in depth): Open-Weeding-Delta (ROS 2, GitHub), Nindamani (GitHub), FarmBot laser prototype video, Frontiers 2022 blue-laser corn robot (5-DOF arm), `easy_handeye2` for hand-eye calibration, micro-ROS.

## C6. S3 ROS 2 package `laser_weed_calib` (untested on hardware)

**Pipeline:** camera → `weed_detection_node` → `/weed_detections` (`Detection2DArray`) → `coordinate_transform_node` → `/laser_pan_tilt_cmd` (`JointState`, radians) → `servo_driver_node` → Arduino → servos. `verification_node` watches the camera. `vl53l0x_range_node` feeds `/laser_head/ground_range`.

**Calibration (run in order)**
1. `generate_grid.py` → `aruco_calibration_grid.png` + `aruco_world_points.csv`. Print at 100%, check one marker = 35 mm, keep it flat.
2. `camera_intrinsics_calib.py`: 15-25 photos at varied angles. RMS error < 0.5 px good, < 1 px ok.
3. `ground_plane_calib.py`: one photo of the sheet on the ground with the camera mounted. Also give laser x/y offset and height (tape measure). Error < 1 px good.
4. Redo step 3 whenever the mount moves.

**Math (`geometry.py`):** undistort pixel → ray in camera frame → rotate to world → intersect ground plane `z = ground_z` → pan/tilt about the laser pivot. Reverse functions (`pan_tilt_to_world`, `world_to_pixel`) are used by aim check.

**Nodes**
- `weed_detection_node`: Ultralytics YOLO, filters to `weed_class_names` so crop is never a target. Debug image topic.
- `vl53l0x_range_node`: `enabled` flag. Drops readings outside 30-1000 mm. Init failure → logs and stops, no crash.
- `coordinate_transform_node`: aims at bbox bottom-center (or centroid). Skips targets beyond `max_valid_horiz_dist_mm`. Uses range sensor only if `ground_height_topic` is set.
- `servo_driver_node`: clamp, slew limit, `/estop` sends `H` (detach). `simulate: true` = no serial writes.
- `verification_node`: aim error px on `/verification/aim_error_px`; kill check = Excess Green drop after `kill_check_delay_sec`; result on `/verification/kill_confirmed`. Needs a fire-event publisher (missing).

**Arduino link:** 115200 baud, ASCII lines. `P<pan>,T<tilt>\n` → `OK ...`. `H` detach, `R` reattach. Pins 9 (pan), 10 (tilt). Firmware clamps angles and auto-detaches after 1 s without a command. Servo power from a separate supply, common ground.

**Install:** `pyserial`, `ultralytics`, `adafruit-circuitpython-vl53l0x`, `adafruit-blinka`. Launch: `ros2 launch laser_weed_calib full_pipeline.launch.py`.

**Terrain:** flat plane at calibration height by default. Range sensor corrects height per detection (sign convention unchecked).

## C7. Introspection commands (no sudo)
```bash
python3 -c "from st3215 import ST3215; import inspect; print(inspect.signature(ST3215.__init__))"
python3 -c "from st3215 import ST3215; s=ST3215('/dev/ttyUSB0'); print([a for a in dir(s) if not a.startswith('__')])"
python3 -c "from st3215 import ST3215; s=ST3215('/dev/ttyUSB0'); print([a for a in dir(s.portHandler) if not a.startswith('__')]); print(s.portHandler.getBaudRate())"
python3 -c "from st3215 import ST3215; help(ST3215)"
```

## C8. Files and references
| File | What it is |
|---|---|
| `laser_servo_tracker_vision_moving_target.py` | current tuned tracker (user's; not modified by Claude) |
| `laser_servo_tracker_stationary_minimal_debugged.py` | earlier: jitter thread + Hz print |
| `laser_servo_tracker.py` | older 2-DOF tracker (superseded) |
| `st3215_keyboard_control.py`, `st3215_sweep_control.py` | S1 bring-up scripts |
| `laser_weed_calib_ros2_package.zip` | S3 ROS 2 package (C6) |
| `Communication_Protocol_User_Manual-EN_191218-0923_.pdf` | Waveshare protocol manual |

- python-st3215 (used): https://github.com/Mickael-Roger/python-st3215 (pip: `st3215`)
- Waveshare ST3215 wiki: https://www.waveshare.com/wiki/ST3215_Servo
- Mwitta et al. 2024: https://www.frontiersin.org/journals/agronomy/articles/10.3389/fagro.2024.1388452/full
- LiteWeed 2026: https://www.sciencedirect.com/science/article/pii/S2772375526005654
- Open-Weeding-Delta: https://github.com/Agroecology-Lab/Open-Weeding-Delta

## C9. Template for other sessions
```
### Session: <who / which chat, date>
- What was done:
- What worked (verified how?):
- What failed / mistakes:
- Iterations (tried → learned → moved on):
- Open items:
```

### Session: (add yours here)
