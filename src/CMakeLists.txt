cmake_minimum_required(VERSION 3.10)

project(opencv-ball-position LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_drawChessboardMarkers=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_drawChessboardMarkerTypes=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_searchImgOfChessboardMarkers=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_rotateBallAndBeamFrame=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_ChessboardMarkersFinder__findAllUsing__=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_ChessboardMarkersFinder_findMarker=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_BallAndBeamVision__findBeamRoi=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_BallAndBeamVision__findBallPos=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_BallAndBeamVision__drawBallPos=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_IF_WithEmulatedBallAndBeam=1")

add_executable(opencv-ball-position vision_debug.cpp vision.cpp opencv-ball-position.cpp)
target_include_directories(opencv-ball-position PRIVATE /usr/include/opencv4)
link_directories(/usr/local/lib)
target_link_libraries(opencv-ball-position opencv_core opencv_highgui opencv_imgproc opencv_calib3d opencv_videoio opencv_imgcodecs)
