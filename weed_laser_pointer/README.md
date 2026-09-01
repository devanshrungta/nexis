# weed_laser_pointer

ROS2 package: YOLO weed detection + closed-loop laser pointing on a Jetson
Orin Nano. **Currently 1 DOF: rotation only.** Translation axis is
deliberately deferred - the controller only corrects horizontal (x-pixel)
error for now; vertical error is ignored until translation is added.

## Data flow

```
camera_node --/camera/image_raw--> yolo_detector_node --/weed_target (x,y px)--> \
                                                                                    servo_controller_node --/servo_cmd (deg)--> servo_driver_node
                                    laser_detector_node --/laser_position (x,y px)--> /
```

`servo_controller_node` compares the weed's x-pixel position against the
laser dot's actual x-pixel position (as seen by the camera) and nudges the
rotation angle until they match - this is the feedback loop. It only uses
horizontal error since there's only one rotational axis to correct with.

## Before running: calibrate your limits

Use the standalone (non-ROS) calibration script to jog the servo to each
physical end and read off the pulse widths - same approach you've been
using:

```python
import Jetson.GPIO as GPIO
import time

SERVO_PIN = 32
FREQ = 50

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, FREQ)
pwm.start(0)


def set_pulse_us(pulse_us, hold=0.4):
    duty = (pulse_us / 20000.0) * 100.0
    pwm.ChangeDutyCycle(duty)
    time.sleep(hold)
    pwm.ChangeDutyCycle(0)


def calibrate_limit(label, start_us=1500.0):
    current = start_us
    set_pulse_us(current)
    print(f"\n--- Finding {label} limit ---")
    while True:
        cmd = input(f"[{label}] pulse us (currently {current:.0f}): ").strip().lower()
        if cmd == 'set':
            print(f"{label} locked at {current:.0f}us")
            return current
        try:
            current = float(cmd)
            set_pulse_us(current)
        except ValueError:
            print("Enter a number, or 'set' to lock this position")


try:
    min_us = calibrate_limit("MIN")
    max_us = calibrate_limit("MAX")
    print(f"\nUse these in servo_driver_node params: min_us={min_us:.0f}, max_us={max_us:.0f}")
finally:
    pwm.stop()
    GPIO.cleanup()
```

Take the `min_us` / `max_us` it prints and set them as `servo_driver_node`
params (in the launch file or via `ros2 param set`). Also set `max_angle`
if your servo's real swing isn't a full 180 (e.g. it's the 270 deg
variant).

These same limits should also be reflected in `servo_controller_node`'s
`angle_min_deg` / `angle_max_deg` (defaults 0/180) so the controller
never tries to command an angle the driver would have to clamp away.

## Other tuning before the full loop

1. **Laser color.** `laser_detector_node`'s HSV thresholds default to a
   red laser. Run it standalone and view `/laser_detector/image_debug` in
   `rqt_image_view`, adjusting `hsv_lower`/`hsv_upper` params until only
   the dot shows up white.
2. **Sign convention.** If the laser moves *away* from the weed instead
   of toward it when you first run the full loop, set `invert: true` on
   `servo_controller_node` - don't try to fix this by rewiring.
3. **Laser safety.** Enclose the work area, add a hardware interlock so
   the laser only fires on a confirmed lock, and check your local laser
   class regulations before running at real power.

## Build

```bash
cd ~/ros2_ws/src
# copy this weed_laser_pointer/ folder in here
cd ~/ros2_ws
colcon build --packages-select weed_laser_pointer
source install/setup.bash
```

## Run

```bash
ros2 launch weed_laser_pointer weed_laser.launch.py
```

Or run nodes individually while tuning:
```bash
ros2 run weed_laser_pointer camera_node
ros2 run weed_laser_pointer yolo_detector_node
ros2 run weed_laser_pointer laser_detector_node
ros2 run weed_laser_pointer servo_controller_node
ros2 run weed_laser_pointer servo_driver_node --ros-args -p min_us:=500.0 -p max_us:=2500.0 -p max_angle:=180.0
```

## Tuning the controller

- `kp`: degrees of correction per pixel of x-error. Too high -> oscillation.
  Too low -> slow convergence. Start at 0.02 and increase gradually.
- `max_step_deg`: caps movement per control cycle - keeps motion smooth.
- `pixel_tolerance`: how close (in x-pixels) counts as "on target."

## Adding translation later

When you're ready to add the translation axis, `laser_detector_node` and
`yolo_detector_node` already publish full (x, y) pixel positions - only
`servo_controller_node` (add a y-error branch) and a new driver for the
translation actuator need to change.

## Swapping in your fine-tuned model later

Only `model_path` (and possibly `target_class_ids`) in
`yolo_detector_node` needs to change - nothing else in the pipeline
depends on which model is loaded.
