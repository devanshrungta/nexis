#!/usr/bin/env python3
"""Click a camera pixel and move the calibrated servo/actuator to that point.

This is a positioning test only. It does not switch a laser on or off.
Run with sudo from the project's Python virtual environment.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import subprocess
import threading
from pathlib import Path

import cv2

# Match the current system's required Jetson pinmux setup. Run this BEFORE
# importing the hardware controller, because that module configures Jetson.GPIO.
def configure_jetson_pins() -> None:
    busybox = shutil.which("busybox")
    if not busybox:
        for candidate in ("/bin/busybox", "/usr/bin/busybox"):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                busybox = candidate
                break
    if not busybox:
        raise RuntimeError("busybox was not found; cannot configure actuator pins safely")

    for address in ("0x2430068", "0x2430070"):
        cmd = [busybox, "devmem", address, "w", "0x8"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Pin setup failed: {' '.join(cmd)}\n"
                f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
            )
        print(f"[PINMUX] Configured {address}")


# ------------------------------ SETTINGS ------------------------------------
CAMERA_DEVICE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "Click Point - Calibrated Position Test"

# Conservative actuator profile for accuracy testing (the controller uses
# open-loop step counting, not a physical position encoder).
ACTUATOR_TEST_MAX_SPEED_STEPS_S = 2500
ACTUATOR_TEST_ACCEL_STEPS_S2 = 1250
ACTUATOR_END_MARGIN_STEPS = 100

SERVO_CENTER_ID = 2056
ALLOWED_MAP_STATUSES = {"INTERPOLATED", "GRID"}


def main() -> int:
    if os.geteuid() != 0:
        print("ERROR: run this script with sudo so it can access Jetson GPIO and /dev/ttyUSB0.")
        print("Example: sudo /home/sunoj-orin/venvs/weed-yolo/bin/python click_point_position_test.py")
        return 2

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "position_map.json"
    servo_cal_path = script_dir / "servo_calibration.json"

    try:
        configure_jetson_pins()
    except Exception as exc:
        print(f"[PINMUX] ERROR: {exc}")
        return 2

    # Import only after pinmux setup because importing this module configures GPIO pins.
    sys.path.insert(0, str(script_dir))
    try:
        try:
            import laser_servo_tracker_yolo_stable_auto_visit as hw
        except ImportError:
            import laser_servo_tracker_point_yolo_modes_async_display as hw
        from position_map import PositionMap, PositionMapError
    except Exception as exc:
        print(f"[INIT] Could not import project hardware/map modules: {exc}")
        return 2

    camera = None
    actuator_worker = None
    actuator_ready = False
    motion_started = False
    servo_ready = False
    map_obj = None
    state = {
        "pending_click": None,
        "target_pixel": None,
        "servo_target": None,
        "actuator_target": None,
        "map_status": "WAITING FOR CLICK",
        "message": "Left-click a point after GO",
        "last_command_time": None,
    }

    try:
        # Avoid loading a different calibration due to the terminal's current directory.
        hw.CALIBRATION_FILE = str(servo_cal_path)
        hw.POSITION_MAP_FILE = str(map_path)
        if hasattr(hw, "ACT_MAX_SPEED_STEPS_S"):
            hw.ACT_MAX_SPEED_STEPS_S = ACTUATOR_TEST_MAX_SPEED_STEPS_S
        if hasattr(hw, "ACT_ACCELERATION_STEPS_S2"):
            hw.ACT_ACCELERATION_STEPS_S2 = ACTUATOR_TEST_ACCEL_STEPS_S2

        print(f"[FILES] Position map: {map_path}")
        print(f"[FILES] Servo/actuator travel calibration: {servo_cal_path}")
        print(f"[ACTUATOR] Accuracy-test profile: max={hw.ACT_MAX_SPEED_STEPS_S} steps/s, "
              f"acceleration={hw.ACT_ACCELERATION_STEPS_S2} steps/s^2")

        if not map_path.is_file():
            raise RuntimeError(f"Position map not found: {map_path}")
        if not servo_cal_path.is_file():
            raise RuntimeError(f"Travel calibration not found: {servo_cal_path}")

        if not hw.load_calibration() or int(hw.ACTUATOR_MAX_POSITION) <= 0:
            raise RuntimeError("Could not load valid actuator travel calibration.")

        map_obj = PositionMap.load(str(map_path))
        print(f"[MAP] Type={map_obj.map_type}; resolution={map_obj.camera_width}x{map_obj.camera_height}; "
              f"samples={len(map_obj.samples)}")
        xs = [float(sample["x"]) for sample in map_obj.samples]
        ys = [float(sample["y"]) for sample in map_obj.samples]
        servos = [float(sample["servo"]) for sample in map_obj.samples]
        actuators = [float(sample["actuator"]) for sample in map_obj.samples]
        print(f"[MAP COVERAGE] samples X={min(xs):.0f}..{max(xs):.0f}, Y={min(ys):.0f}..{max(ys):.0f}; "
              f"servo={min(servos):.0f}..{max(servos):.0f}; actuator={min(actuators):.0f}..{max(actuators):.0f}")
        print(f"[LIMITS] servo={hw.ST_VISIBLE_MIN_POSITION}..{hw.ST_VISIBLE_MAX_POSITION}; "
              f"actuator={hw.ACTUATOR_MIN_POSITION}..{hw.ACTUATOR_MAX_POSITION} steps; "
              f"end margin={ACTUATOR_END_MARGIN_STEPS} steps")

        if (map_obj.camera_width, map_obj.camera_height) != (FRAME_WIDTH, FRAME_HEIGHT):
            raise RuntimeError(
                f"Map is {map_obj.camera_width}x{map_obj.camera_height}, but this tester is configured "
                f"for {FRAME_WIDTH}x{FRAME_HEIGHT}. Use the calibrated resolution."
            )

        hw.setup_servo()
        if hw.servo is None:
            raise RuntimeError("ST3215 servo was not initialized")
        servo_ready = True
        try:
            # A modest speed is easier to observe during a point-to-point test.
            hw.servo.SetAcceleration(hw.ST_SERVO_ID, 100)
            hw.servo.SetSpeed(hw.ST_SERVO_ID, 1000)
        except Exception as exc:
            print(f"[SERVO] WARNING: could not set speed/acceleration: {exc}")

        camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(CAMERA_DEVICE)
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera /dev/video{CAMERA_DEVICE}")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        ok, first_frame = camera.read()
        if not ok:
            raise RuntimeError("Camera opened but did not return a frame")
        actual_h, actual_w = first_frame.shape[:2]
        print(f"[CAMERA] Actual frame={actual_w}x{actual_h}")
        if (actual_w, actual_h) != (map_obj.camera_width, map_obj.camera_height):
            raise RuntimeError(
                f"Camera delivered {actual_w}x{actual_h}, but map expects "
                f"{map_obj.camera_width}x{map_obj.camera_height}. No motion was enabled."
            )

        middle = hw.actuator_middle()
        answer = input(f"Is the actuator PHYSICALLY at its saved middle ({middle:.0f} steps)? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("[ACTUATOR] Not confirmed at midpoint. Motion will remain disabled.")
            return 2
        if not hw.accept_saved_middle():
            raise RuntimeError("Could not accept saved actuator midpoint")
        actuator_ready = True
        print("[ACTUATOR] Midpoint accepted; position values are step-count estimates, not encoder readings.")

        print("\nSAFETY: no movement starts until you type GO.")
        if input("Type GO to enable point-to-move testing: ").strip() != "GO":
            print("[START] GO not entered. No movement started.")
            return 0

        motion_started = True
        hw._actuator_running = True
        actuator_worker = threading.Thread(target=hw.actuator_thread, daemon=True, name="actuator-position-worker")
        actuator_worker.start()

        # Known starting angle; the actuator remains where the operator referenced it.
        try:
            hw.servo.WritePosition(hw.ST_SERVO_ID, SERVO_CENTER_ID)
            print(f"[SERVO] Sent safe center command {SERVO_CENTER_ID}.")
        except Exception as exc:
            raise RuntimeError(f"Could not command servo center: {exc}") from exc

        def on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["pending_click"] = (int(x), int(y))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        print("\n[CONTROLS] Left-click a pixel to move to it. Press Q or Esc to quit.")
        print("[RULE] Targets outside the measured calibration map are rejected; no extrapolation is allowed.")
        print("[LASER] This script only positions the mount; it does not switch a laser on.")

        while True:
            ok, frame = camera.read()
            if not ok:
                state["message"] = "CAMERA READ FAILED"
                cv2.waitKey(1)
                continue

            click = state["pending_click"]
            state["pending_click"] = None
            if click is not None:
                px, py = click
                print(f"\n[CLICK] Pixel=({px},{py})")
                try:
                    servo_value, actuator_value, status = map_obj.pixel_to_position(px, py)
                except PositionMapError as exc:
                    message = f"REJECTED: map error: {exc}"
                    state["message"] = message
                    print(f"[REJECT] {message}")
                else:
                    print(f"[MAP] pixel=({px},{py}) -> servo={servo_value:.2f}, "
                          f"actuator={actuator_value:.1f}, status={status}")
                    if status not in ALLOWED_MAP_STATUSES:
                        message = f"REJECTED: {status}; click inside calibrated map"
                        state["message"] = message
                        print(f"[REJECT] {message}")
                    elif not (hw.ST_VISIBLE_MIN_POSITION <= servo_value <= hw.ST_VISIBLE_MAX_POSITION):
                        message = (f"REJECTED: servo target {servo_value:.1f} outside safe limits "
                                   f"{hw.ST_VISIBLE_MIN_POSITION}..{hw.ST_VISIBLE_MAX_POSITION}")
                        state["message"] = message
                        print(f"[REJECT] {message}")
                    elif not (hw.ACTUATOR_MIN_POSITION + ACTUATOR_END_MARGIN_STEPS <= actuator_value <=
                              hw.ACTUATOR_MAX_POSITION - ACTUATOR_END_MARGIN_STEPS):
                        message = (f"REJECTED: actuator target {actuator_value:.1f} is outside the "
                                   f"safe travel window with {ACTUATOR_END_MARGIN_STEPS}-step end margin")
                        state["message"] = message
                        print(f"[REJECT] {message}")
                    elif not hw.actuator_is_referenced():
                        message = "REJECTED: actuator is not referenced"
                        state["message"] = message
                        print(f"[REJECT] {message}")
                    else:
                        servo_target = int(round(servo_value))
                        try:
                            result = hw.servo.WritePosition(hw.ST_SERVO_ID, servo_target)
                            if result is False:
                                raise RuntimeError("python-st3215 WritePosition reported failure")
                        except Exception as exc:
                            message = f"SERVO COMMAND FAILED: {type(exc).__name__}: {exc}"
                            state["message"] = message
                            print(f"[SERVO] {message}")
                            # Do not command the actuator if the servo target could not be sent.
                            continue

                        actuator_target = hw.set_actuator_position_target(actuator_value)
                        state["target_pixel"] = (px, py)
                        state["servo_target"] = servo_target
                        state["actuator_target"] = actuator_target
                        state["map_status"] = status
                        state["last_command_time"] = time.monotonic()
                        state["message"] = f"MOVING TO ({px},{py})"
                        print(f"[COMMAND] servo={servo_target}; actuator={actuator_target:.1f} steps")
                        print("[INFO] Re-clicking the same point is useful for repeatability testing.")

            # Use an unresized, exact-resolution view so mouse pixel coordinates match map pixels.
            if state["target_pixel"] is not None:
                px, py = state["target_pixel"]
                cv2.drawMarker(frame, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                label = f"TARGET PIXEL ({px},{py}) | SERVO {state['servo_target']} | ACT {state['actuator_target']:.0f}"
                cv2.putText(frame, label, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

            estimated_actuator = hw.get_actuator_position()
            actuator_speed = hw.get_actuator_speed()
            target_act = state["actuator_target"]
            if target_act is None:
                move_status = "WAITING FOR CLICK"
            else:
                remaining = float(target_act) - float(estimated_actuator)
                if abs(remaining) <= 10 and abs(actuator_speed) <= 10:
                    move_status = "ACTUATOR AT TARGET (ESTIMATED)"
                else:
                    move_status = f"ACT EST {estimated_actuator:.0f} / TARGET {float(target_act):.0f} / SPEED {actuator_speed:.0f}"
            cv2.putText(frame, f"{state['message'][:92]}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if "MOVING" in state["message"] else (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, move_status[:100], (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "LEFT CLICK: MOVE  |  Q/ESC: QUIT", (20, FRAME_HEIGHT - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        return 0

    except KeyboardInterrupt:
        print("\n[EXIT] Keyboard interrupt")
        return 130
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    finally:
        # On exit, return to the known middle/center only if GO was accepted and
        # the actuator worker actually started. This mirrors the tracker shutdown behavior.
        if motion_started:
            try:
                if servo_ready and hw.servo is not None:
                    try:
                        hw.servo.WritePosition(hw.ST_SERVO_ID, SERVO_CENTER_ID)
                        print(f"[SHUTDOWN] Servo commanded to center {SERVO_CENTER_ID}.")
                    except Exception as exc:
                        print(f"[SHUTDOWN] WARNING: could not center servo: {exc}")

                if actuator_worker is not None and actuator_worker.is_alive() and actuator_ready:
                    middle_target = float(hw.actuator_middle())
                    print(f"[SHUTDOWN] Returning actuator to saved middle {middle_target:.0f} steps...")
                    ok = hw.move_actuator_to_position_and_wait(middle_target, timeout=45.0, label="shutdown return to middle")
                    if not ok:
                        hw.set_actuator_speed(0.0)

            except Exception as exc:
                print(f"[SHUTDOWN] WARNING: return-to-middle failed: {exc}")

        try:
            hw._actuator_running = False
            if actuator_worker is not None:
                actuator_worker.join(timeout=2.0)
        except Exception:
            pass
        try:
            hw.set_actuator_speed(0.0)
        except Exception:
            pass
        try:
            hw.set_actuator_enabled(False)
        except Exception:
            pass
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        if servo_ready:
            try:
                hw.servo.portHandler.closePort()
            except Exception:
                pass
        try:
            import Jetson.GPIO as GPIO
            GPIO.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
