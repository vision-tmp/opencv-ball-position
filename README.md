# opencv-ball-position
## Summary
This is an example of computer vision program based on OpenCV library.
It is developed for the experimental system called "ball and beam". 

This program can be seen as a module of a more complex program because it just estimates the position of the ball. Other modules should be developed yet, and they are needed to control the ball position by moving the beam, to implement user interface etc.
It is supposed that chessboard markers are attached to each end of the beam as well as to the servo that controls the beam. The pattern for the markers is in the comment in [the header file](src/vision.h).
You can install this program and test it under various conditions.

The source code is written according to the convention that can be called "minimize comments by making code more self-explanatory". For example, make names of variables and functions more descriptive. And even add unnecessary variables and functions with self-explanatory names if this can make the code easier for understanding.

## Prerequisites and Installation
The program is tested for `xubuntu 18.05.4` with `OpenCV 3.2.0` and `ubuntu 22.04.1` with `OpenCV 4.5.x`.
Be sure that `universe` is included in `/etc/apt/sources.list`

```
sudo apt install g++ libopencv-dev cmake
cd <root dir of the project>
cmake src
```

## Build and Run

```
make
./opencv-ball-position
```

By default, ball and beam are simulated on top of the frame taken from camera. Try to move different trackbars to see how dynamic changes affect the estimation of ball position (which is on the chart and indicated on the main picture taken from camera).
Also, you can test the performance by moving the camera and thus changing the background.
To switch off the simulation, e.g., to test on real ball and beam in [CMakeLists.txt](src/CMakeLists.txt) change `-DDEBUG_IF_WithEmulatedBallAndBeam=1` to `-DDEBUG_IF_WithEmulatedBallAndBeam=0` and build again.

Also, you can toggle other `-DDEBUG_*`, for example, to control what to display. The starting positions of the windows are in the files located in directory `debug`.
