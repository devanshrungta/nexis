- >12V variable power supply
  (We must have this, since we have already used the linear actuator and that needs 24V)
- servo driver board




say we ignore the open/closed loop motor for now
and concentrate on how to validate if the laser is pointing to a target coordinate given by a simple rgb camera's output, how would we approach this

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