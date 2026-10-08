POSITION-MAP TARGETING VERSION

Files
-----
1. position_map.py
   Shared piecewise-linear calibration map.

2. calibrate_position_map.py
   One-time manual calibration utility.

3. laser_servo_tracker_calibrated_position_control.py
   Runtime controller. Uses the existing vision target tracker, but does NOT
   detect or track the laser spot at runtime. The clicked/current visual target
   is converted directly to absolute servo and actuator positions from
   position_map.json.

4. position_map_calibration_A4.pdf
   5 x 6 numbered A4 landscape calibration sheet. Print at 100% / actual size.

One-time calibration
--------------------
1. Put the printed A4 sheet in the actual target plane used by the system.
2. Make sure the camera is at 1280x720, same as the runtime controller.
3. Put the actuator at the saved physical middle and run:
      sudo python3 calibrate_position_map.py
4. For points 1 through 30, in order:
      - click the center of the printed cross
      - use I/K for servo motion
      - use U/O for actuator motion
      - visually center the laser on the printed cross
      - press SPACE to record
5. After point 30, press S.
6. This creates position_map.json.
7. Copy position_map.py, position_map.json, and the runtime controller into
   the same directory.

Runtime
-------
Run:
      sudo python3 laser_servo_tracker_calibrated_position_control.py

The existing vision tracker remains responsible for updating the target's
image coordinate when the target or camera/setup moves. The new position map
converts that current image coordinate directly to absolute actuator positions.

Important
---------
- Runtime laser detection has been removed from the control loop.
- The laser is used only as a human-visible reference during the one-time
  calibration procedure.
- Existing tested controller constants are preserved. The new absolute
  position mode removes the old gain*pixel-error incremental actuator command.
- The current actuator speed and acceleration values are still the motion
  limits. If a large move is still slower than 1 s, a faster motion profile or
  hardware-timed TMC2209 step generation will be required; software cannot
  beat the acceleration/velocity limits mathematically.
- Recalibrate if the camera-to-actuator mechanical relationship changes,
  or if the working target plane changes enough to invalidate the empirical
  pixel-to-position relationship.
