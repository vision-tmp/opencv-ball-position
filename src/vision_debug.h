#ifndef VISION_DEBUG_H_
#define VISION_DEBUG_H_

#include <opencv2/opencv.hpp>

#ifdef DEBUG

#define DEBUG_BLOCK(x) do {x} while (0)
#define DEBUG_DECL(x) x
#define DEBUG_COUT(x)                                           \
    do { std::cout << ' ' << #x << ':' << x; } while (0)
#define DEBUG_COUT_ENDL(x)                                              \
    do { std::cout << ' ' << #x << ':' << x << std::endl; } while (0)

#ifndef DEBUG_IF_drawChessboardMarkers
#define DEBUG_IF_drawChessboardMarkers 0
#endif // DEBUG_IF_drawChessboardMarkers
#define DEBUG_drawChessboardMarkers(x)                          \
    do {if(DEBUG_IF_drawChessboardMarkers) {x}} while (0)

#ifndef DEBUG_IF_drawChessboardMarkerTypes
#define DEBUG_IF_drawChessboardMarkerTypes 0
#endif // DEBUG_IF_drawChessboardMarkerTypes
#define DEBUG_drawChessboardMarkerTypes(x)                \
    do {if(DEBUG_IF_drawChessboardMarkerTypes) {x}} while (0)

#ifndef DEBUG_IF_searchImgOfChessboardMarkers
#define DEBUG_IF_searchImgOfChessboardMarkers 0
#endif // DEBUG_IF_searchImgOfChessboardMarkers
#define DEBUG_searchImgOfChessboardMarkers(x)                      \
    do {if(DEBUG_IF_searchImgOfChessboardMarkers) {x}} while (0)

#ifndef DEBUG_IF_rotateBallAndBeamFrame
#define DEBUG_IF_rotateBallAndBeamFrame 0
#endif // DEBUG_IF_rotateBallAndBeamFrame
#define DEBUG_rotateBallAndBeamFrame(x)                          \
    do {if(DEBUG_IF_rotateBallAndBeamFrame) {x}} while (0)

#ifndef DEBUG_IF_ChessboardMarkersFinder__findAllUsing__
#define DEBUG_IF_ChessboardMarkersFinder__findAllUsing__ 0
#endif // DEBUG_IF_ChessboardMarkersFinder__findAllUsing__
#define DEBUG_ChessboardMarkersFinder__findAllUsing__(x)                \
    do {if(DEBUG_IF_ChessboardMarkersFinder__findAllUsing__) {x}} while (0)

#ifndef DEBUG_IF_ChessboardMarkersFinder_findMarker
#define DEBUG_IF_ChessboardMarkersFinder_findMarker 0
#endif // DEBUG_IF_ChessboardMarkersFinder_findMarker
#define DEBUG_ChessboardMarkersFinder_findMarker(x)                     \
    do {if(DEBUG_IF_ChessboardMarkersFinder_findMarker) {x}} while (0)

#ifndef DEBUG_IF_BallAndBeamVision__findBeamRoi
#define DEBUG_IF_BallAndBeamVision__findBeamRoi 0
#endif // DEBUG_IF_BallAndBeamVision__findBeamRoi
#define DEBUG_BallAndBeamVision__findBeamRoi(x)                         \
    do {if(DEBUG_IF_BallAndBeamVision__findBeamRoi) {x}} while (0)

#ifndef DEBUG_IF_BallAndBeamVision__findBallPos
#define DEBUG_IF_BallAndBeamVision__findBallPos 0
#endif // DEBUG_IF_BallAndBeamVision__findBallPos
#define DEBUG_BallAndBeamVision__findBallPos(x)                         \
    do {if(DEBUG_IF_BallAndBeamVision__findBallPos) {x}} while (0)

#ifndef DEBUG_IF_BallAndBeamVision__drawBallPos
#define DEBUG_IF_BallAndBeamVision__drawBallPos 0
#endif // DEBUG_IF_BallAndBeamVision__drawBallPos
#define DEBUG_BallAndBeamVision__drawBallPos(x)                         \
    do {if(DEBUG_IF_BallAndBeamVision__drawBallPos) {x}} while (0)

#ifndef DEBUG_IF_BallAndBeamVision__drawBallPos
#define DEBUG_IF_BallAndBeamVision__drawBallPos 0
#endif // DEBUG_IF_BallAndBeamVision__drawBallPos
#define DEBUG_BallAndBeamVision__drawBallPos(x)                         \
    do {if(DEBUG_IF_BallAndBeamVision__drawBallPos) {x}} while (0)

#ifndef DEBUG_IF_WithEmulatedBallAndBeam
#define DEBUG_IF_WithEmulatedBallAndBeam 0
#endif // DEBUG_IF_WithEmulatedBallAndBeam
#define DEBUG_WithEmulatedBallAndBeam(x)                         \
    do {if(DEBUG_IF_WithEmulatedBallAndBeam) {x}} while (0)

#else // DEBUG
#define DEBUG_BLOCK(x)
#define DEBUG_DECL(x)
#define DEBUG_COUT(x)
#define DEBUG_COUT_ENDL(x)
#define DEBUG_drawChessboardMarkers(x)
#define DEBUG_drawChessboardMarkerTypes(x)
#define DEBUG_searchImgOfChessboardMarkers(x)
#define DEBUG_rotateBallAndBeamFrame(x)
#define DEBUG_ChessboardMarkersFinder__findAllUsing__(x)
#define DEBUG_ChessboardMarkersFinder_findMarker(x)
#define DEBUG_BallAndBeamVision__findBeamRoi(x)
#define DEBUG_BallAndBeamVision__findBallPos(x)
#define DEBUG_BallAndBeamVision__drawBallPos(x)
#define DEBUG_WithEmulatedBallAndBeam(x)
#endif // DEBUG

void createTrackbars(cv::Mat const& img);
void createCVWindow(cv::String const& name);
float getEmulatedBallPos();
void addEmulatedBallAndBeam(cv::Mat& img, int& frameRotationAngle, bool autoMoving);
void addBallPosToDataForChart(cv::Mat const& img, float ballPos);
void addRealBallPosToDataForChart(cv::Mat const& img, float realBallPos);
void updateChart(cv::Mat& img);


#endif // VISION_DEBUG_H_
