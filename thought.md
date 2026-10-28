
there are a few things
the calibiration is one time thing for each use, can you think of a continuous validation, becuase it may drift while in use and we will keep on accumulating error without even knowing
also if we have a threshold based detection of laser coordinate, then I have the fear of its performance in real world scenarios. Because in the actual field, it would be bright and the laser detection would not just detect or may have a lot of false positives. Also there can be irregular reflections, ie, laser bouncing off curved/irregular surfaces and thus the laser detection would simply not work

1. how do we account for vibrations when the vehicle is in motion
2. how to use >1 cameras? how to stich their output together
3. will a "time of flight sensor" or a depth camera help?


not returning when pressing q
whenever the laser gets off the camera's frmae, try to bring it back within the camera's view

decresase the pixel error



- PHASES
  - 1-DOF laser tracking
  - 2-DOF laser tracking
  - Moving vehicle integration
  - ROS


# Important
Always run the below command on startup for linear actuator to work
``` 
sudo busybox devmem 0x2430068 w 0x8
sudo busybox devmem 0x2430070 w 0x8
```


TODO:
[] real laser code integration
  - another dof
  - time and intensity needs to be controlled by the code
[] Switch from target coordinates to target bounding box
  - Figure out what best strategy to implement for this
  - Have configurable options like - select all bounding box or a list of class for the bounding box
[] Make code lightweight
  - Optimise to run on edge
  - Remove unwanted code
[] Have the actuator return to middle position from the last position in case of power failure. Or else we will need to calibrate the actuator again.




- Documentation
- Github
- Angle adjustable camera mount
- ask claude and gpt to create readme and a detailed report of what I have done till now along with the learnings


These are things that I want to also mention in the readme/summary (you can decide where it should go)
- Problem with the servo driver
  - Even if the power is pluuged & turned on and still the driver is not pulling any Amps, then there DC jack on board might be the problem. Try moving the dc jack a bit on the board, since once the dc jack had an accident and i tried to put solder on the pin that got disconnected.
- If the linear actuator is not working 
  - Always write the `sudo busybox` command to make the linear actuator work (once plugged in). Won't work otherwise.
- I have the servo driver wiring schematic in the readme/summare too.
- Write that the esp32 must be changeg to "serial port forwading" mode 
- how to calibrate
- explain what and how to tune the parameters/variables





the current language is hard to read, rewrite the whole in simple english
for example use "servo's hard-coded limit (for safety concerns)" and write more about there is a script just for this purpose in ./servo/st3215_servo.py and use it to limit the maximum what the servo can travel to minimize the risk of laser escaping and due to the safety concerns instead of"Servo absolute hardware range in latest candidate"
I am sharing my rough thoughts with you. Its on you to stich my agendas to a coherent story in simple and good english


Regarding the jitter:
- \>6 Hz doesn't materialise good. If we instruct higher frequency (that is expected from our system) the servo doesn't work good. I tried to debug this, and got to know that there is 16-20ms communication time required to send the position id and wait and receive if the servo has reached there or not. I tried skipping to wait, the time got down to aroung 1 ms, but the servo missed a lot of position, resulting in not good jitter. Still in that case, the 50 Hz can be achieved, but since I played with different frequencies, I found that after 6 Hz, the servo doesn't perform as good. I also tried the different values of amplitudes, and I settled to the current value in the code after trying a lot of values. Increasing the value, doesn't work as intended. So this is a problem that needs to be solved since faster jitter is expected, but not in top priority.

- talk about the steps to get started
  - like the calibration (althought the printed a4 sheet is made available, but i tried myself that we can calibrate without it. Like, we can aim the laser at multiple positions and mark the cross where the laser is and do this for multiple points, and thus we can calibrate)
- this was the calibration between the camera frame of reference and the real world coordinate
- regarding the servo and linear actuation, we can use the script itself, since it provided to calibrate at the start
- the height of the laser above the ground is ~12-13 inches
- currently the code to integrate the laser is still not there (like the code to trigger the laser at what power percent and what time duration)
- To connect to servo driver board. Connect to the wifi named ESP32_DEV and the password is 12345678. Then go to http://192.168.4.1/ and click the button saying start "Start Serial Forwading" 




+- 2.5 inch = 1920, 2160
middle point = 2048


middle = 2056
=- 2.5" =  1930, 2171

GREEN_H_MIN = 24
GREEN_H_MAX = 179
GREEN_S_MIN = 33
GREEN_S_MAX = 154
GREEN_V_MIN = 104
GREEN_V_MAX = 167
MIN_GREEN_CONTOUR_AREA = 2000       # pixels; reduce if the weed is small
MORPH_OPEN_SIZE = 4
MORPH_CLOSE_SIZE = 7



Potential Improvement:
- Add stall detection in linear actuator
- Auto calibration



If the weed is not being detected: tune the H,S,V min,max and Morph open and close variables using this script laser_position_calibrated_control_package/green_weed_detection_tuner.py
- Current workflow
  - Connect the power to both the servo and the linear actuator
  - Write the sudo box command for the linear actuator
  - Calibrate the 2D surface (if not done already). Move the laser to the differnet points on the surface and mark the according point on the screen with the mouse. Better the calibration, better the results. (Use the script laser_position_calibrated_control_package/calibrate_position_map_multiplacement.py)
  - then run the laser_position_calibrated_control_package/green_weed_moving_tracker.py

- to calibrate the servo/actuator maximum endpoints, use laser_position_calibrated_control_package/run_yolo_stable_auto_visit.sh

Chronology of the files
- laser_position_calibrated_control_package/laser_servo_tracker_actuator_jog_shutdown_fixed.py - w/o yolo
- laser_position_calibrated_control_package/laser_servo_tracker_yolo_stable_auto_visit.py - with yolo (use this to run it laser_position_calibrated_control_package/run_yolo_stable_auto_visit.sh)
- laser_position_calibrated_control_package/green_weed_moving_tracker.py - switched from yolo to green thresholding for weed detection
This the order in which the files were developed. Progressively, simple things might not work, like the calibration of the servo and actuator will work in the initial script but not in other
Not deleted them, since might need them sometime. All do the same thing, with progressively added features. The latest one should be used, but if something doesnt work use the previous script for that specific part

There are 2 types of calibration: 1st the servo and actuator maximum travel and 2nd is the camera frame of reference and the real world coordinates transformation.

use the venv created to run
like `sudo /home/sunoj-orin/venvs/weed-yolo/bin/python green_weed_moving_tracker.py`