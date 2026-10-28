# Laser Weeding Rig — README

## Purpose

Aim a laser at a selected image point or a YOLO-detected weed using a 2-DOF mechanism. The target pixel is converted to servo and actuator positions by `position_map.json`. **The current calibrated-map path does not detect the laser spot at runtime**; pointing accuracy therefore depends on calibration, repeatable mechanics, and the actuator not missing steps.

## Current setup

- **Compute:** Jetson Orin Nano, L4T R36.4.7; camera `/dev/video0`, requested 1280×720 MJPG at 60 fps.
- **Servo (image Y):** ST3215 bus servo through a Waveshare ESP32 Servo Driver in Serial Forwarding mode; USB `/dev/ttyUSB0` at 115200 baud.
- **Actuator (image X):** stepper linear actuator driven by TMC2209 STEP/DIR/EN from Jetson.GPIO BOARD pins 29/31/33. Pulses are generated in Python.
- **Laser:** LaserTree K40 40 W, Class 4; documented mount height 12.5 in (0.3175 m). Preserve the hardware E-stop/interlock. This tracker does not implement verified firing or weed-kill confirmation.
- **Calibration:** `servo_calibration.json` stores safe servo range and actuator travel. `position_map.json` stores pixel → servo/actuator mapping. Latest reported map: 44 scattered samples; training RMSE was 9.4 servo positions and 122.2 actuator steps. RMSE is not a guarantee of real-world pointing accuracy. The map assumes a stable camera/mechanism and the calibrated target plane; different depth can increase error.

## Current controller candidate

Latest delivered version: `laser_servo_tracker_yolo_stable_auto_visit.py` with `run_yolo_stable_auto_visit.sh`. It passed Python syntax and isolated logic tests, but the full workflow has **not** been verified on the physical rig yet.

### Modes and keys

| Key | Action |
|---|---|
| `P` | Manual-point mode; click a target, then `G` |
| `Y` | YOLO preview / single-weed mode; click a stabilized box, then `G` |
| `A` | Auto-visit confirmed weeds one by one: aim → settle → servo jitter → next weed |
| `G` | Start aiming at selected target; in YOLO mode, uses the highest-confidence stable weed if none is selected |
| `V` | Toggle vision tracking in manual-point mode |
| `C` | Cancel tracking, auto-visit, sweep, or jitter |
| `R` | Return servo and actuator to middle |
| `M` | Enter jog mode: `I/K` servo −/+, `U/O` actuator −/+ |
| `1/2` | Mark servo MIN/MAX in jog mode |
| `3/4` | Mark actuator MIN/reference and MAX/save travel |
| `E` | Calibration-extension mode; bypasses normal travel range, so use only with direct supervision and ample clearance |
| `T` | Select sweep: servo / actuator / both |
| `J` | Manual servo-only jitter |
| `Q` | Quit and run shutdown handling |

### YOLO configuration

At the top of the controller:

- `YOLO_MODEL_PATH` — current default `/home/sunoj-orin/devansh/camera/best 1.pt`.
- `YOLO_ALLOWED_CLASS_IDS = None` means all classes; use a list such as `[0, 2]` to select class IDs.
- Current candidate settings use image size 640, confidence threshold 0.25, confirmation over multiple inference results, box/confidence smoothing, limited missed-result tolerance, and optical flow between detector updates.
- Target point defaults to bounding-box `center`; `lower_center` is also supported. Neither guarantees that the spot is on the weed crown.
- Auto-visit defaults to the nearest estimated motor target and refuses map extrapolation. It keeps a software margin from actuator endpoints. These are safeguards, not physical limit switches.

## Install and run

Keep the controller, launcher, `position_map.py`, `position_map.json`, and `servo_calibration.json` in:

`~/devansh/laser_position_calibrated_control_package/`

The tested YOLO environment is `/home/sunoj-orin/venvs/weed-yolo/`. It has Python 3.10, PyTorch 2.8.0 (CUDA 12.6 build), torchvision 0.23.0, NumPy 1.26.4, OpenCV 4.11.0 and Ultralytics 8.4.177; `torch.cuda.is_available()` was confirmed `True`.

Use the launcher, which invokes `sudo` with the virtual-environment interpreter so both GPU packages and hardware permissions are available:

```bash
cd ~/devansh/laser_position_calibrated_control_package
chmod +x run_yolo_stable_auto_visit.sh
./run_yolo_stable_auto_visit.sh
```

Equivalent direct command:

```bash
sudo /home/sunoj-orin/venvs/weed-yolo/bin/python \
  laser_servo_tracker_yolo_stable_auto_visit.py
```

Do not use plain `sudo python ...` for YOLO: it selects the system Python, where the earlier installation had PyTorch 2.14.1+cu130, NumPy 2.2.6 and OpenCV 5.0. That combination reported CUDA unavailable and ran inference on CPU.

At startup, power the servo driver first and confirm it is in Serial Forwarding mode. Confirm the actuator is physically at the saved midpoint when prompted. If it is not, establish MIN and MAX using jog mode before trusting the step counter.

## Safety and known checks

- The actuator has **no encoder or physical limit switches**. Position is an open-loop STEP count and can drift if steps are missed. Keep away from hard stops; software limits cannot guarantee protection when unreferenced.
- Test manual mode, single-weed YOLO mode, then auto-visit mode in that order. First trials should be performed with the laser disabled or in a safe controlled test setup.
- Calibration is known to be imperfect; laser-spot detection was unreliable outdoors because reflections caused false detections. Avoid claiming <3 px physical accuracy from map-fit RMSE alone.
- **Range consistency check:** the latest controller candidate defines `ST_MAX_POSITION = 2440`, while calibration logs have reported a visible maximum of 2500. `set_servo_position()` applies both clamps, so the effective upper command bound is 2440 until this is checked and reconciled safely.
- If YOLO FPS is low, first verify the launch command uses `/home/sunoj-orin/venvs/weed-yolo/bin/python` and the startup log confirms CUDA. UI FPS and inference FPS are different measurements.
- `/dev/ttyUSB0` permission errors occur when run as an ordinary user; the provided launcher uses `sudo`. A `dialout` group setup is an alternative only after permissions are deliberately configured.

See [`LEARNING_SUMMARY_laser_servo_project.md`](LEARNING_SUMMARY_laser_servo_project.md) for the experiment history, failures, retired designs, ROS 2 proposal, and technical appendix.
