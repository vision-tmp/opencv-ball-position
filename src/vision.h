#ifndef VISION_H_
#define VISION_H_

#include <opencv2/opencv.hpp>
#include "vision_debug.h"

namespace ballAndBeam
{

namespace chessboardMarkers
{

enum class SearchMethod
{
    NONE, IMAGE, ROI, ONE_RECENT, TWO_RECENT, THREE_RECENT
};

// The pattern of ChessboardMarker is the following:
//
// WWWWWWWWWW
// WBBWWBBWWW
// WBBWWBBWWW
// WWWBBWWBBW
// WWWBBWWBBW
// WBBWWBBWWW
// WBBWWBBWWW
// WWWBBWWBBW
// WWWBBWWBBW
// WWWWWWWWWW
//
// where W is a quarter of a white filled square,
// and B is a quarter of a black filled square.
// There must be no gaps between the W and B quarters.
// Notice the white frame around the chessboard formed by W quarters.

class Marker
{
public:
    explicit Marker(SearchMethod searchMethod = SearchMethod::NONE)
        : found_(false),
          searchMethod_(searchMethod) {}
    Marker(bool found, SearchMethod searchMethod,
           std::vector<cv::Point2f> const& squareCenters)
        : found_(found),
          searchMethod_(searchMethod),
          squareCenters_(squareCenters),
          center_(getCenter()),
          minAreaRect_(getMinAreaRect()) {}
    bool found() const { return found_; }
    SearchMethod searchMethod() const { return searchMethod_; }
    std::vector<cv::Point2f> const& squareCenters() const
        { assert(found_); return squareCenters_; }
    cv::Point2f center() const
        { assert(found_); return center_; }
    cv::RotatedRect minAreaRect() const
        { assert(found_); return minAreaRect_; }
private:
    bool found_;
    SearchMethod searchMethod_;
    std::vector<cv::Point2f> squareCenters_;
    cv::Point2f center_;
    cv::RotatedRect minAreaRect_;
    cv::Point2f getCenter();
    cv::RotatedRect getMinAreaRect();
};

class SearchResult
{
public:
    explicit SearchResult(std::vector<Marker> const& markers);
    SearchResult(Marker const& central,
                 Marker const& left,
                 Marker const& right);
    bool allIdentified() const { return allIdentified_; }
    bool allFound() const;
    bool allFoundAndIdentified() const
        { return (allFound() && allIdentified()); }
    bool allFoundUsingImageOrRoi() const;
    bool atLeastOneFoundUsingImageOrRoi() const;
    std::vector<Marker> const& markers() const { return markers_; }
    Marker const& central() const;
    Marker const& left() const;
    Marker const& right() const;
private:
    bool allIdentified_{false};
    unsigned centralIndex_;
    unsigned leftIndex_;
    unsigned rightIndex_;
    std::vector<Marker> markers_;
    bool allFound(std::vector<Marker> const& markers) const;
    bool identifyCentralLeftRight(std::vector<Marker> const& markers);
};

class History
{
public:
    explicit History(unsigned depth = 3) : depth_(depth) {}
    void add(SearchResult const& newItem);
    std::deque<SearchResult> const& deque() const { return deque_; }
private:
    unsigned depth_;
    std::deque<SearchResult> deque_;
};

class Finder
{
public:
    Finder(bool usingRoi = true,
           unsigned toleratedDepthOfNotFoundResults = 5,
           float ratioForInitialRoi = 5.0,
           float scaleImageFactor = 1.0,
           float scaleRoiFactor = 2.0)
        : usingRoi_(usingRoi),
          toleratedDepthOfNotFoundResults_(toleratedDepthOfNotFoundResults),
          ratioForInitialRoi_(ratioForInitialRoi),
          scaleImageFactor_(scaleImageFactor),
          scaleRoiFactor_(scaleRoiFactor),
          history_(toleratedDepthOfNotFoundResults + 1) {}
    SearchResult const& find(cv::Mat const& img);
private:
    bool usingRoi_;
    unsigned toleratedDepthOfNotFoundResults_;
    float ratioForInitialRoi_;
    float scaleImageFactor_;
    float scaleRoiFactor_;
    History history_;
    bool findAllUsingImage(cv::Mat const& img,
                           std::vector<Marker>& markers);
    bool findAllUsingRoi(cv::Mat const& img, std::vector<Marker>& markers);
    bool findAllUsingEstimation(std::vector<Marker>& markers);
    bool findAllUsingFoundInHistory();
    bool findAllUsingOneFoundInHistory();
    bool findAllUsingTwoFoundInHistory(std::vector<Marker>& markers);
    SearchResult const& recentResult() const;
};

} // namespace chessboardMarkers

class Beam
{
public:
    Beam() : found_(false) {}
    Beam(cv::Mat const& rotatedImgRoi,
         cv::Point2f centerOnRotatedImgRoi,
         cv::RotatedRect rectOnOriginalImg,
         cv::Point2f centerOnOriginalImg)
        : found_(true),
          rotatedImgRoi_(rotatedImgRoi),
          centerOnRotatedImgRoi_(centerOnRotatedImgRoi),
          rectOnOriginalImg_(rectOnOriginalImg),
          centerOnOriginalImg_(centerOnOriginalImg) {}
    cv::Mat const& rotatedImgRoi() const {
        assert(found_); return rotatedImgRoi_; }
    cv::Point2f centerOnRotatedImgRoi() const {
        assert(found_); return centerOnRotatedImgRoi_; }
    cv::RotatedRect rectOnOriginalImg() const {
        assert(found_); return rectOnOriginalImg_; }
    cv::Point2f centerOnOriginalImg() const {
        assert(found_); return centerOnOriginalImg_; }
private:
    bool found_;
    cv::Mat rotatedImgRoi_;
    cv::Point2f centerOnRotatedImgRoi_;
    cv::RotatedRect rectOnOriginalImg_;
    cv::Point2f centerOnOriginalImg_;
};

class Vision
{
public:
    explicit Vision(cv::VideoCapture& cap);
    bool processNewFrame();
    void updateRealBallPos();
private:
    Beam beam_;
    float ballPos_;
    float realBallPos_;
    bool realBallPosIsKnown_;
    chessboardMarkers::Finder markersFinder_;
    cv::VideoCapture& cap_;
};

} // namespace ballAndBeam

#endif // VISION_H_
