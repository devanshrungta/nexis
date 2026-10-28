SPARSE MULTI-PLACEMENT POSITION-MAP CALIBRATION

WHY THIS VERSION
----------------
The previous calibration map represented one regular 5x6 grid on one A4 sheet.
Outside that grid, it used a quadratic extrapolation based on the local grid.
That extrapolation can be inaccurate far away from the sheet.

This version records sparse, arbitrary pixel/motor samples from several A4
placements. Inside the convex hull of the recorded sample pixels, it uses
Delaunay-triangle interpolation. Only pixels outside that hull use the global
quadratic model. Place the samples around the whole intended operating area so
normal targets are INSIDE the calibrated hull.

FILES
-----
laser_servo_tracker_calibrated_position_control.py
    Runtime controller. Existing motion-control constants are retained; startup
    map reporting now supports scattered multi-placement maps.

position_map.py
    Backwards-compatible map loader. It can read old regular-grid maps and new
    SCATTERED maps.

calibrate_position_map_multiplacement.py
    New calibration program; supports four A4 outer crosses per placement,
    checkpoint/resume, arbitrary corner selection, fine/fast actuator jog, and
    restoring the servo/actuator to their starting positions.

position_map_calibration_A4.pdf
    Existing A4-style 30-point sheet. For each placement record only crosses
    #1 (top-left), #6 (top-right), #30 (bottom-right), and #25 (bottom-left).

BEFORE YOU START
----------------
1. Keep your existing servo_calibration.json. Do not overwrite it with a guessed
   travel value. Verify actuator MIN/MAX and travel calibration first.
2. Put the A4 sheet flat in the same physical target plane where you will aim.
   Moving it around within that plane is fine; do not tilt it or raise/lower it
   between placements.
3. The camera resolution and camera mounting must remain the same as runtime.
4. The calibration program asks you to confirm the actuator is physically at
   its saved middle. This is required because the stepper has no position sensor.

RECOMMENDED CALIBRATION
-----------------------
Use 4 sheet placements (16 samples) minimum; 5 placements (20 samples) is
better. Place the sheet around the extremes of the useful view, and add a center
placement if practical. Keep every sheet corner inside the camera image.
The sample points should surround the entire region you plan to target.

At each placement:
- Default corner order is #1, #6, #30, #25.
- Click the cross in the camera view.
- Jog servo with I/K and actuator with U/O until the laser is visually centered
  on the printed cross.
- Press SPACE to save the sample.
- After the four corners are recorded, move the sheet to a new region and press N.

Keys:
I/K: servo jog
U/O: actuator jog
F: toggle actuator jog mode between existing FINE settings and FAST settings
P: choose paper cross #1, #6, #25, or #30 (any order)
SPACE: record selected cross
N: advance to the next A4 placement
Z: undo the latest sample
S: save position_map.json (requires at least 12 non-degenerate samples)
Q: checkpoint and exit

Every SPACE press writes position_map_sparse_checkpoint.json. On restart, answer
Y to resume. Saving a final map removes the sparse checkpoint. On either S or Q,
the program commands both axes back to their positions at calibration startup.

RUN
---
Run the new calibration program in the same directory as servo_calibration.json:

    sudo python3 calibrate_position_map_multiplacement.py

When the map is saved, run the runtime controller:

    sudo python3 laser_servo_tracker_calibrated_position_control.py

RUNTIME STATUS
--------------
INTERPOLATED means the target pixel lies inside the measured sample hull and
is mapped using local triangle interpolation.
EXTRAPOLATED means it lies outside the sample hull and uses a global quadratic
estimate. Do not rely on this for accuracy-critical operation; reposition the
A4 during calibration so the planned working region is inside the sample hull.

IMPORTANT LIMITATIONS
---------------------
- All calibration placements and targets must lie in the same physical plane.
  Height/depth changes introduce parallax and need separate calibration/3D geometry.
- Calibration samples must surround the workspace. A few points in one corner
  do not calibrate the rest of the viewport.
- Keep early tests slow/supervised and within known safe actuator/servo limits.
- A lower global-model training RMSE alone does not guarantee physical accuracy;
  validate several points not used to fit the calibration, especially near edges.
