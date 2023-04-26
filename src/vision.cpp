#include <array>
#include <iostream>
#include "vision.h"
#include "vision_debug.h"

namespace
{
DEBUG_DECL(
    cv::Mat debugImg;
    cv::Mat imgForChart;
    int frameRotationAngle;
    )
}

using namespace ballAndBeam;
using namespace chessboardMarkers;

cv::Point2f Marker::getCenter()
{
    cv::Point2f c(0, 0);
    if (squareCenters_.size() == 0)
        return c;
    for (auto const& e : squareCenters_)
        c += e;
    c *= 1.0 / squareCenters_.size();
    return c;
}

cv::RotatedRect Marker::getMinAreaRect()
{
    if (squareCenters_.size() == 0)
        return cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(0, 0), 0);
    return cv::minAreaRect(squareCenters_);
}

SearchResult::SearchResult(std::vector<Marker> const& markers)
    : markers_(markers)
{
    while (markers_.size() < 3)
        markers_.push_back(Marker(SearchMethod::NONE));
    while (markers_.size() > 3)
        markers_.pop_back();
    if (allFound(markers_)) {
        allIdentified_ = identifyCentralLeftRight(markers_);
    }
}

SearchResult::SearchResult(Marker const& central,
                           Marker const& left,
                           Marker const& right)
{
    markers_.push_back(central);
    centralIndex_ = markers_.size() - 1;
    markers_.push_back(left);
    leftIndex_ = markers_.size() - 1;
    markers_.push_back(right);
    rightIndex_ = markers_.size() - 1;
    allIdentified_ = true;
}

inline bool SearchResult::allFound() const
{
    return allFound(markers_);
}

bool SearchResult::allFoundUsingImageOrRoi() const
{
    for (auto const& e : markers()) {
        if (e.searchMethod() != SearchMethod::IMAGE &&
            e.searchMethod() != SearchMethod::ROI) {
            return false;
        }
    }
    return true;
}

bool SearchResult::atLeastOneFoundUsingImageOrRoi() const
{
    for (auto const& e : markers()) {
        if (e.searchMethod() == SearchMethod::IMAGE ||
            e.searchMethod() == SearchMethod::ROI)
            return true;
    }
    return false;
}

bool SearchResult::allFound(std::vector<Marker> const& markers) const
{
    if (markers.size() < 3)
        return false;
    for (auto const& e : markers) {
        if (!e.found())
            return false;
    }
    return true;
}

Marker const& SearchResult::central() const
{
    assert(allFound());
    assert(allIdentified());
    assert(centralIndex_ < markers_.size());
    return markers_[centralIndex_];
}

Marker const& SearchResult::left() const
{
    assert(allFound());
    assert(allIdentified());
    assert(leftIndex_ < markers_.size());
    return markers_[leftIndex_];
}

Marker const& SearchResult::right() const
{
    assert(allFound());
    assert(allIdentified());
    assert(rightIndex_ < markers_.size());
    return markers_[rightIndex_];
}

bool SearchResult::identifyCentralLeftRight(std::vector<Marker> const& markers)
{
    assert(markers.size() == 3);
    assert(allFound(markers));

    std::array<float, 3> dists = {{
            (float)cv::norm(markers[1].center()-markers[2].center()),
            (float)cv::norm(markers[0].center()-markers[2].center()),
            (float)cv::norm(markers[0].center()-markers[1].center())
        }};
    auto centralIt = markers.begin() +
        (std::max_element(dists.begin(), dists.end()) - dists.begin());

    std::vector<std::vector<Marker>::const_iterator> sideIt;
    for (auto it = markers.begin(); it != markers.end(); ++it) {
        if (it != centralIt)
            sideIt.push_back(it);
    }

    std::vector<float> angle;
    for (auto const& e : sideIt) {
        float dy = centralIt->center().y - e->center().y;
        float dx = e->center().x - centralIt->center().x;
        if (dx == 0.0f && dy == 0.0f)
            return false;
        angle.push_back(std::atan2(dy, dx));
    }

    if (fabs(angle[0] - angle[1]) > CV_PI) {
        if (angle[0] < 0)
            angle[0] += 2 * CV_PI;
        else
            angle[1] += 2 * CV_PI;
    }
    if (angle[0] > angle[1])
        std::swap(sideIt[0], sideIt[1]);

    centralIndex_ = centralIt - markers_.begin();
    leftIndex_ = sideIt[0] - markers_.begin();
    rightIndex_ = sideIt[1] - markers_.begin();

    return true;
}

void History::add(SearchResult const& newItem)
{
    while (deque_.size() >= depth_)
        deque_.pop_front();
    deque_.push_back(newItem);
}

static void getPointsOfRotatedRect(cv::RotatedRect const& rect,
                                   std::vector<cv::Point2f>& points)
{
    cv::Point2f vertices[4];
    rect.points(vertices);
    points.clear();
    for(auto e : vertices)
        points.push_back(e);
}

DEBUG_DECL(
static void drawRotatedRect(cv::Mat& img, cv::RotatedRect r, cv::Scalar color)
{
    cv::Point2f vertices[4];
    r.points(vertices);
    for (unsigned i = 0; i < 4; ++i)
        cv::line(img, vertices[i], vertices[(i + 1) % 4], color);
}
)

static int getNumberOfMarkersFound(std::vector<Marker>& markers)
{
    int n = 0;
    for (auto const& e : markers) {
        if (e.found())
            ++n;
    }
    return n;
}

SearchResult const& Finder::find(cv::Mat const& img)
{
    bool allFound = false;
    std::vector<Marker> markers;

    if (history_.deque().size() == 0) {
        allFound = findAllUsingImage(img, markers);
    } else {
        if (usingRoi_ &&
            recentResult().allFoundAndIdentified() &&
            recentResult().allFoundUsingImageOrRoi()) {
            allFound = findAllUsingRoi(img, markers);
        } else {
            allFound = findAllUsingImage(img, markers);
        }
    }

    if (!allFound)
        allFound = findAllUsingEstimation(markers);

    if (!allFound) {
        DEBUG_ChessboardMarkersFinder__findAllUsing__(
            std::cout << "!allFound: add only " << getNumberOfMarkersFound(markers)
                      << " marker(s)" << std::endl;
            );
        SearchResult result(markers);
        history_.add(result);
    }

    DEBUG_BLOCK(
        for (auto const& e_markers : recentResult().markers()) {
            if (e_markers.found()) {
                DEBUG_drawChessboardMarkers(
                    for (auto const& e : e_markers.squareCenters()) {
                        cv::circle(debugImg, e, 1,
                                   cv::Scalar(255, 0, 255), -1);
                    }
                    drawRotatedRect(debugImg, e_markers.minAreaRect(),
                                    cv::Scalar(255, 0, 255));
                    cv::circle(debugImg, e_markers.center(), 3,
                               cv::Scalar(255, 0, 255), -1);
                    );
            }
        }
        DEBUG_drawChessboardMarkerTypes(
            if (recentResult().allFoundAndIdentified()) {
                cv::putText(debugImg, "C", recentResult().central().center(),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 255), 2);
                cv::putText(debugImg, "L", recentResult().left().center(),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 255), 2);
                cv::putText(debugImg, "R", recentResult().right().center(),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 255), 2);
            }
            );
        );

    assert(history_.deque().size() > 0);
    return recentResult();
}

static void findBaseRectNearestNeighborsAndDistsToThem(
    std::vector<cv::RotatedRect> const& rects,
    cv::RotatedRect const& baseRect,
    unsigned maxNumberOfFound,
    std::vector<cv::RotatedRect>& found,
    std::vector<float>& dists)
{
    assert(rects.size() > 0);

    auto baseRectIt = rects.begin();
    for (auto const& e : rects) {
        if (e.center == baseRect.center)
            break;
        ++baseRectIt;
    }
    assert(baseRectIt < rects.end());

    found.clear();
    dists.clear();

    if (rects.size() == 1)
        return;

    struct RectItAndDistPair
    {
        std::vector<cv::RotatedRect>::const_iterator it;
        float dist;
    };
    std::vector<RectItAndDistPair> pairs;
    for (auto it = rects.begin(); it != rects.end(); ++it){
        if (it != baseRectIt)
            pairs.push_back({it, (float)cv::norm(it->center - baseRectIt->center)});
    }

    unsigned maxNumberOfFoundBounded = maxNumberOfFound;
    if (maxNumberOfFoundBounded > pairs.size())
        maxNumberOfFoundBounded = pairs.size();
    std::partial_sort(pairs.begin(), pairs.begin() + maxNumberOfFoundBounded,
                      pairs.end(),
                      [](RectItAndDistPair const& a, RectItAndDistPair const& b)
                      {
                          return a.dist < b.dist;
                      });
    pairs.resize(maxNumberOfFoundBounded);

    for(auto e : pairs) {
        found.push_back(*e.it);
        dists.push_back(e.dist);
    }
}

static float getRectAspectRatio(cv::RotatedRect r)
{
    return (std::min(r.size.width, r.size.height) /
            std::max(std::max(r.size.width, r.size.height), 1.0f));
}

static void getRectsFromContoursUsingSimpleFilter(
    std::vector<std::vector<cv::Point> > const& contours,
    std::vector<cv::RotatedRect>& rects)
{
    const float MIN_ROTATED_RECT_ASPECT_RATIO = 0.2f;
    const float MIN_ROTATED_RECT_SIZE = 1.0f;
    const float MAX_ROTATED_RECT_SIZE = 20.0f;

    for (auto const& e : contours) {
        cv::RotatedRect r = cv::minAreaRect(e);
        float size = std::max(r.size.width, r.size.height);
        if (size < MIN_ROTATED_RECT_SIZE || size > MAX_ROTATED_RECT_SIZE)
            continue;
        if(getRectAspectRatio(r) < MIN_ROTATED_RECT_ASPECT_RATIO)
            continue;
        rects.push_back(r);
    }
}

static bool rectIsNotCentralRectOfChessboard(
    cv::RotatedRect const& rect,
    std::vector<cv::RotatedRect> const& nearests,
    std::vector<float> const& dists)
{
    const float TOLERATED_DIST_RATIO = 0.2f;
    const float MAX_DIST_RATIO = 7.0f;
    const float TOLERATED_SIZE_RATIO = 0.7f;
    const float TOLERATED_ASPECT_RATIO_RATIO = 0.7f;
    const float TOLERATED_ALMOST_RECT_CENTER_RATIO = 0.1f;
    const float MIN_NEARESTS_AREA_RECT_ASPECT_RATIO = 0.8f;
    const float TOLERATED_IN_RECT_CORNERS_RATIO = 0.2f;

    if (nearests.size() < 4)
        return true;

    float minDist = *min_element(dists.begin(), dists.end());
    float maxDist = *max_element(dists.begin(), dists.end());

    bool distancesAreAlmostEqual = (maxDist - minDist <
                                    TOLERATED_DIST_RATIO * (minDist + maxDist));
    if (!distancesAreAlmostEqual)
        return true;

    bool distancesAreBounded = (maxDist <
                                std::max(rect.size.width, rect.size.height) *
                                MAX_DIST_RATIO);
    if(!distancesAreBounded)
        return true;

    bool sizesAreAlmostEqual = true;
    float sizeBase = std::max(rect.size.width, rect.size.height);
    for (auto const& e : nearests) {
        float sizeNearest = std::max(e.size.width, e.size.height);
        if (sizeNearest < sizeBase * (1 - TOLERATED_SIZE_RATIO) ||
            sizeNearest > sizeBase * (1 + TOLERATED_SIZE_RATIO)) {
            sizesAreAlmostEqual = false;
            break;
        }
    }
    if (!sizesAreAlmostEqual)
        return true;

    bool aspectRatiosAreAlmostEqual = true;
    float aspectRatioBase = getRectAspectRatio(rect);
    for (auto const& e_nearests : nearests) {
        float aspectRatioNearest = getRectAspectRatio(e_nearests);
        if (aspectRatioNearest < aspectRatioBase *
            (1 - TOLERATED_ASPECT_RATIO_RATIO) ||
            aspectRatioNearest > aspectRatioBase *
            (1 + TOLERATED_ASPECT_RATIO_RATIO)) {
            aspectRatiosAreAlmostEqual = false;
            break;
        }
    }
    if (!aspectRatiosAreAlmostEqual)
        return true;

    std::vector<cv::Point2f> centers;
    for (auto const& e : nearests) {
        centers.push_back(e.center);
    }
    cv::RotatedRect nearestsAreaRect = cv::minAreaRect(centers);

    bool nearestsAreaRectCenterIsAlmostRectCenter =
        cv::norm(rect.center - nearestsAreaRect.center) <
        TOLERATED_ALMOST_RECT_CENTER_RATIO * (minDist + maxDist) / 2;
    if (!nearestsAreaRectCenterIsAlmostRectCenter)
        return true;

    bool nearestsAreaRectHasValidAspectRatio =
        getRectAspectRatio(nearestsAreaRect) >
        MIN_NEARESTS_AREA_RECT_ASPECT_RATIO;
    if (!nearestsAreaRectHasValidAspectRatio)
        return true;

    cv::Point2f vertices[4];
    nearestsAreaRect.points(vertices);
    bool nearestsAreInRectCorners = true;
    for (auto const& e_vertices : vertices) {
        std::vector<float> dists;
        for (auto const& e_nearests : nearests) {
            dists.push_back(cv::norm(e_nearests.center - e_vertices));
        }
        if (*min_element(dists.begin(), dists.end()) >
            TOLERATED_IN_RECT_CORNERS_RATIO * (minDist + maxDist) / 2) {
            nearestsAreInRectCorners = false;
            break;
        }
    }
    if (!nearestsAreInRectCorners)
        return true;

    return false;
}

static void findChessboardRectsForThreshold(
    cv::Mat const& grayImg,
    int threshold,
    std::vector<cv::RotatedRect>& foundRects)
{
    foundRects.clear();

    cv::Mat thr;
    cv::threshold(grayImg, thr, threshold, 255, cv::THRESH_BINARY_INV);
    cv::erode(thr, thr, getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thr, contours, hierarchy, cv::RETR_CCOMP,
                     cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::RotatedRect> rects;
    getRectsFromContoursUsingSimpleFilter(contours, rects);

    struct FilteredItem
    {
        cv::RotatedRect rect;
        std::vector<cv::RotatedRect> nearests;
    };
    std::vector<FilteredItem> filtered;

    DEBUG_DECL(
        std::vector<cv::RotatedRect> nearestsAreaRects;
        );

    for (auto const& e_rects : rects) {
        std::vector<cv::RotatedRect> nearests;
        std::vector<float> dists;
        findBaseRectNearestNeighborsAndDistsToThem(rects, e_rects, 4,
                                                   nearests, dists);

        if (rectIsNotCentralRectOfChessboard(e_rects, nearests, dists))
            continue;

        DEBUG_ChessboardMarkersFinder_findMarker(
            std::vector<cv::Point2f> centers;
            for (auto const& e : nearests)
                centers.push_back(e.center);
            nearestsAreaRects.push_back(cv::minAreaRect(centers));
            );

        bool chessboardRectsAreFound = false;
        for (auto const& e_filtered : filtered) {
            auto matchedNearestIt = std::find_if(
                nearests.begin(), nearests.end(),
                [&e_filtered](cv::RotatedRect& e_nearests)
                {
                    return (e_nearests.center == e_filtered.rect.center);
                });
            if (matchedNearestIt < nearests.end()) {
                foundRects.push_back(e_rects);
                for (auto const& e : nearests)
                    foundRects.push_back(e);
                for (auto const& e : e_filtered.nearests) {
                    if (e.center != e_rects.center)
                        foundRects.push_back(e);
                }
                assert(foundRects.size() == 8);
                chessboardRectsAreFound = true;
                break;
            }
        }
        if (chessboardRectsAreFound)
            break;

        filtered.push_back({e_rects, nearests});
    }
    DEBUG_ChessboardMarkersFinder_findMarker(
        if (filtered.size() > 0) {
            cv::imshow("thr", thr);
            cv::Mat cntr = cv::Mat::zeros(thr.rows, thr.cols, CV_8UC3);
            for (auto const& e : rects) {
                drawRotatedRect(cntr, e, cv::Scalar(0, 0, 255));
            }
            for (auto const& e : filtered) {
                drawRotatedRect(cntr, e.rect, cv::Scalar(0, 255, 0));
            }
            for (auto const& e : nearestsAreaRects) {
                drawRotatedRect(cntr, e, cv::Scalar(255, 255, 0));
            }
            for (auto const& e : foundRects) {
                cv::circle(cntr, e.center, 4, cv::Scalar::all(255), 1);
            }
            cv::imshow("cntr", cntr);
        }
        );
}

static void findChessboardSquareCenters(cv::Mat const& grayImg,
                                        std::vector<cv::Point2f>& squareCenters)
{
    const int THRESHOLD_MIN = 70;
    const int THRESHOLD_MAX = 220;
    const int THRESHOLD_DELTA = 50;

    squareCenters.clear();
    std::vector<cv::RotatedRect> rects;
    for (int th = THRESHOLD_MIN; th <= THRESHOLD_MAX; th += THRESHOLD_DELTA) {
        findChessboardRectsForThreshold(grayImg, th, rects);
        if (rects.size() > 0)
            break;
    }
    for (auto const& e : rects)
        squareCenters.push_back(e.center);
}

static void hideMarkerOnImage(std::vector<cv::Point2f> squareCenters,
                              cv::Mat& img)
{
    cv::Point2f c;
    float r;
    cv::minEnclosingCircle(squareCenters, c, r);
    r *= 3;
    cv::circle(img, c, r, cv::Scalar::all(128), -1);
}

static void scaleSquareCenters(cv::Point2f scale,
                               std::vector<cv::Point2f>& squareCenters)
{
    for (auto& e : squareCenters) {
        e.x = e.x * scale.x;
        e.y = e.y * scale.y;
    }
}

bool Finder::findAllUsingImage(cv::Mat const& img, std::vector<Marker>& markers)
{
    DEBUG_ChessboardMarkersFinder__findAllUsing__(
        std::cout << "findAllUsingImage" << std::endl;
        );

    cv::Mat searchImg;
    cv::cvtColor(img, searchImg, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(searchImg, searchImg);

    float scaleFactor = scaleImageFactor_;
    if (scaleFactor != 1.0) {
        cv::resize(searchImg, searchImg, cv::Size(0, 0),
                   scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }

    markers.clear();
    bool allFound = true;

    for (int i = 0; i < 3; ++i) {
        std::vector<cv::Point2f> squareCenters;
        findChessboardSquareCenters(searchImg, squareCenters);

        if (squareCenters.size() > 0) {
            hideMarkerOnImage(squareCenters, searchImg);
            scaleSquareCenters(cv::Point2f(1.0 / scaleFactor,
                                           1.0 / scaleFactor),
                               squareCenters);
            markers.push_back(Marker(true, SearchMethod::IMAGE, squareCenters));
        } else {
            allFound = false;
            break;
        }
    }

    DEBUG_searchImgOfChessboardMarkers(
        cv::imshow("searchImg", searchImg);
        cv::moveWindow("searchImg", img.cols + 64, img.rows + 128);
        cv::imwrite("searchImg.jpg", searchImg);
        );

    if (allFound) {
        SearchResult result(markers);
        history_.add(result);
    }

    return allFound;
}

static cv::Rect getMarkerRoi(cv::Mat const& img, Marker const& m, float ratio)
{
    cv::Point2f c;
    float r;
    cv::minEnclosingCircle(m.squareCenters(), c, r);
    r *= ratio;
    return (cv::Rect(c.x - r + 1, c.y - r + 1, r * 2, r * 2) &
            cv::Rect(0, 0, img.cols, img.rows));
}

static void shiftSquareCenters(std::vector<cv::Point2f>& squareCenters,
                               cv::Point2f shift)
{
    for (auto& e : squareCenters)
        e += shift;
}

bool Finder::findAllUsingRoi(cv::Mat const& img, std::vector<Marker>& markers)
{
    DEBUG_ChessboardMarkersFinder__findAllUsing__(
        std::cout << "findAllUsingRoi" << std::endl;
        );

    assert(history_.deque().size() > 0);
    assert(recentResult().allFoundAndIdentified());
    assert(recentResult().atLeastOneFoundUsingImageOrRoi());

    std::vector<cv::Rect> roi;
    roi.push_back(getMarkerRoi(img, recentResult().central(),
                               ratioForInitialRoi_));
    roi.push_back(getMarkerRoi(img, recentResult().left(),
                               ratioForInitialRoi_));
    roi.push_back(getMarkerRoi(img, recentResult().right(),
                               ratioForInitialRoi_));

    markers.clear();
    bool allFound = true;

    for (int i = 0; i < 3; ++i) {
        cv::Mat roiImg(img, roi[i]);
        cv::cvtColor(roiImg, roiImg, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(roiImg, roiImg);

        float scaleFactor = scaleRoiFactor_;
        if (scaleFactor != 1.0) {
            cv::resize(roiImg, roiImg, cv::Size(0, 0),
                       scaleFactor, scaleFactor, cv::INTER_LINEAR);
        }

        std::vector<cv::Point2f> squareCenters;
        findChessboardSquareCenters(roiImg, squareCenters);

        if (squareCenters.size() > 0) {
            scaleSquareCenters(cv::Point2f(1.0 / scaleFactor,
                                           1.0 / scaleFactor),
                               squareCenters);
            shiftSquareCenters(squareCenters, cv::Point2f(roi[i].x, roi[i].y));
            markers.push_back(Marker(true, SearchMethod::ROI, squareCenters));
            DEBUG_searchImgOfChessboardMarkers(
                std::stringstream out;
                out << "roiImg_" << i;
                cv::imshow(out.str(), roiImg);
                cv::moveWindow(out.str(), i * 384, img.rows + 160);
                out << ".jpg";
                cv::imwrite(out.str(), roiImg);
                );
        } else {
            allFound = false;
        }
    }

    if (allFound) {
        SearchResult result(markers);
        history_.add(result);
    }
    return allFound;
}

bool Finder::findAllUsingEstimation(std::vector<Marker>& markers)
{
    DEBUG_ChessboardMarkersFinder__findAllUsing__(
        std::cout << "findAllUsingEstimation" << std::endl;
        );

    assert(markers.size() < 3);
    bool allFound = false;

    switch (getNumberOfMarkersFound(markers)) {
    case 0:
        allFound = findAllUsingFoundInHistory();
        break;
    case 1:
        allFound = findAllUsingOneFoundInHistory();
        if (!allFound)
            allFound = findAllUsingFoundInHistory();
        break;
    case 2:
        allFound = findAllUsingTwoFoundInHistory(markers);
        if (!allFound)
            allFound = findAllUsingFoundInHistory();
        break;
    }
    return allFound;
}

bool Finder::findAllUsingFoundInHistory()
{
    DEBUG_ChessboardMarkersFinder__findAllUsing__(
        std::cout << "findAllUsingFoundInHistory" << std::endl;
        );

    if (history_.deque().empty())
        return false;

    unsigned depthOfNotFoundResults = 0;
    auto foundItemIt = history_.deque().rbegin();
    for (; foundItemIt != history_.deque().rend(); ++foundItemIt) {
        if (foundItemIt->allFoundAndIdentified() && foundItemIt->atLeastOneFoundUsingImageOrRoi()) {
            break;
        }
       if (++depthOfNotFoundResults > toleratedDepthOfNotFoundResults_) {
           break;
       }
    }

    bool allFound = false;
    if (depthOfNotFoundResults <= toleratedDepthOfNotFoundResults_ &&
        foundItemIt != history_.deque().rend()) {
        assert(foundItemIt->allIdentified());
        SearchResult result(
            Marker(foundItemIt->central().found(), SearchMethod::THREE_RECENT,
                   foundItemIt->central().squareCenters()),
            Marker(foundItemIt->left().found(), SearchMethod::THREE_RECENT,
                   foundItemIt->left().squareCenters()),
            Marker(foundItemIt->right().found(), SearchMethod::THREE_RECENT,
                   foundItemIt->right().squareCenters()));
        history_.add(result);
        allFound = true;
    } else {
        allFound = false;
    }
    return allFound;
}

bool Finder::findAllUsingOneFoundInHistory()
{
    DEBUG_ChessboardMarkersFinder__findAllUsing__(
        std::cout << "findAllUsingOneFoundInHistory" << std::endl;
        );
    return false;
}

static bool getMarkerItOfSimilarMarker(
    std::vector<cv::Point2f> const& squareCenters,
    std::vector<Marker> const& markers,
    std::vector<Marker>::const_iterator& markerIt)
{
    assert(squareCenters.size() != 0);
    assert(markers.size() != 0);
    float r;
    cv::Point2f c;
    cv::minEnclosingCircle(squareCenters, c, r);
    markerIt = markers.end();
    for (auto it = markers.begin(); it != markers.end(); ++it) {
        float mr;
        cv::Point2f mc;
        cv::minEnclosingCircle(it->squareCenters(), mc, mr);
        if (cv::norm(c - mc) <= (r + mr)) {
            markerIt = it;
            return true;
        }
    }
    return false;
}

bool Finder::findAllUsingTwoFoundInHistory(std::vector<Marker>& markers)
{
    DEBUG_ChessboardMarkersFinder__findAllUsing__(
        std::cout << "findAllUsingTwoFoundInHistory" << std::endl;
        );
    if (history_.deque().size() == 0)
        return false;
    if (!recentResult().allFoundAndIdentified())
        return false;

    assert(markers.size() == 2);

    bool allFound = false;

    std::vector<Marker>::const_iterator centralIt = markers.end();
    bool centralFound =
        getMarkerItOfSimilarMarker(recentResult().central().squareCenters(),
                                   markers, centralIt);
    std::vector<Marker>::const_iterator leftIt = markers.end();
    bool leftFound =
        getMarkerItOfSimilarMarker(recentResult().left().squareCenters(),
                                   markers, leftIt);
    std::vector<Marker>::const_iterator rightIt = markers.end();
    bool rightFound =
        getMarkerItOfSimilarMarker(recentResult().right().squareCenters(),
                                   markers, rightIt);

    if (centralFound) {
        if (leftFound && leftIt != centralIt && !rightFound){
            SearchResult result(*centralIt, *leftIt,
                                Marker(recentResult().right().found(),
                                       SearchMethod::ONE_RECENT,
                                       recentResult().right().squareCenters()));
            history_.add(result);
            allFound = true;
        }
        if (rightFound && rightIt != centralIt && !leftFound){
            SearchResult result(*centralIt,
                                Marker(recentResult().left().found(),
                                       SearchMethod::ONE_RECENT,
                                       recentResult().left().squareCenters()),
                                *rightIt);
            history_.add(result);
            allFound = true;
        }
    }
    return allFound;
}

inline SearchResult const& Finder::recentResult() const
{
    assert(history_.deque().size() > 0);
    return history_.deque().back();
}

Vision::Vision(cv::VideoCapture& cap)
    : ballPos_(0.0),
      realBallPos_(0.0),
      realBallPosIsKnown_(false),
      markersFinder_(chessboardMarkers::Finder(true)),
      cap_(cap)
{
    DEBUG_BLOCK(
        createCVWindow("debug");
        createCVWindow("sliders");
        cv::Mat imgForSliders = cv::Mat::zeros(1, 300, CV_8UC3);
        cv::imshow("sliders", imgForSliders);
        createCVWindow("ballPosChart");
        imgForChart = cv::Mat::zeros(200, 420, CV_8UC3);
        cv::imshow("ballPosChart", imgForChart);
        );
    DEBUG_ChessboardMarkersFinder_findMarker(
        createCVWindow("thr");
        createCVWindow("cntr");
        );
    DEBUG_BallAndBeamVision__findBeamRoi(
        createCVWindow("beamRotatedImgRoi");
        createCVWindow("rotatedImgRoi");
        );
    DEBUG_BallAndBeamVision__findBallPos(
        createCVWindow("beamImg");
        createCVWindow("beamImgBlur");
        createCVWindow("beamImgBlurThresh");
        );
    DEBUG_rotateBallAndBeamFrame(
        frameRotationAngle = 0;
        cv::createTrackbar("frameRotationAngle", "sliders",
                           &frameRotationAngle, 360, NULL);
        );
}

static void findNearestTwoRectPointsToBasePoint(cv::RotatedRect rect,
                                                cv::Point2f basePoint,
                                                cv::Point2f& point1,
                                                cv::Point2f& point2)
{
    std::vector<cv::Point2f> points;
    getPointsOfRotatedRect(rect, points);
    std::partial_sort(points.begin(), points.begin() + 2, points.end(),
                      [basePoint](cv::Point2f const& a, cv::Point2f const& b)
                      {
                          return (cv::norm(a - basePoint) <
                                  cv::norm(b - basePoint));
                      });
    point1 = points[0];
    point2 = points[1];
}

static bool findBeam(cv::Mat const& img, SearchResult const& result,
                     Beam& beam)
{
    if (!result.allFoundAndIdentified()) {
        beam = Beam();
        return false;
    }

    Marker const& central = result.central();
    Marker const& left = result.left();
    Marker const& right = result.right();

    cv::RotatedRect rect;

    rect = left.minAreaRect();
    rect.size.width *= 2;
    rect.size.height *= 2;
    cv::Point2f pLeft1;
    cv::Point2f pLeft2;
    findNearestTwoRectPointsToBasePoint(rect, right.center(), pLeft1, pLeft2);

    rect = right.minAreaRect();
    rect.size.width *= 2;
    rect.size.height *= 2;
    cv::Point2f pRight1;
    cv::Point2f pRight2;
    findNearestTwoRectPointsToBasePoint(rect, left.center(), pRight1, pRight2);

    cv::RotatedRect beamRect;
    cv::Point2f pLeftCenter = (pLeft1 + pLeft2) * 0.5;
    cv::Point2f pRightCenter = (pRight1 + pRight2) * 0.5;
    beamRect.center = (pLeftCenter + pRightCenter) * 0.5;
    beamRect.size.width = cv::norm(pLeftCenter - pRightCenter);
    beamRect.size.height = (cv::norm(pLeft1 - pLeft2) +
                            cv::norm(pRight1 - pRight2)) / 4;
    float beamAngle = std::atan2(pLeftCenter.y - pRightCenter.y,
                                 pLeftCenter.x - pRightCenter.x);
    beamRect.angle = beamAngle * 180.0 / CV_PI;

    float angleToBeamCenter = beamAngle - CV_PI / 2;
    if (angleToBeamCenter < -CV_PI)
        angleToBeamCenter += 2 * CV_PI;
    float angleOfCentralMarker = central.minAreaRect().angle * CV_PI / 180.0;
    std::vector<float> angleDelta(7);
    int i = -angleDelta.size() / 2 + 1;
    for (auto& e : angleDelta)
        e = CV_PI / 2 * i++;
    std::vector<float> angleAbsDiff;
    for (auto e : angleDelta) {
        angleAbsDiff.push_back(fabs((angleOfCentralMarker + e) -
                                    angleToBeamCenter));
    }
    float angleToBeamFromCentral = (angleOfCentralMarker +
                                    angleDelta[min_element(angleAbsDiff.begin(),
                                                           angleAbsDiff.end()) -
                                               angleAbsDiff.begin()]);

    float beamCenterXDelta = std::tan(angleToBeamFromCentral -
                                      angleToBeamCenter) *
        cv::norm(central.center() - beamRect.center);
    float fromCentralToLeft = cv::norm(central.center() - pLeftCenter);
    float fromCentralToRight = cv::norm(central.center() - pRightCenter);
    cv::Point2f beamCenterOnRotatedImgRoi;
    beamCenterOnRotatedImgRoi.x = (beamRect.size.width +
                                   (fromCentralToLeft * fromCentralToLeft -
                                    fromCentralToRight * fromCentralToRight) /
                                   beamRect.size.width) / 2 - beamCenterXDelta;
    beamCenterOnRotatedImgRoi.y = beamRect.size.height / 2;
    DEBUG_DECL(
        float fromCentralToBeamCenter = cv::norm(central.center() -
                                                 beamRect.center);
        );

    std::vector<cv::Point2f> points;
    getPointsOfRotatedRect(beamRect, points);
    cv::Rect roiRect = cv::boundingRect(points);
    int addToWidth = std::max(beamRect.size.width - roiRect.width, 0.0f);
    cv::Mat rotatedImgRoi = cv::Mat::zeros(roiRect.height, roiRect.width + addToWidth + 2, CV_8UC3);

    int whereToCopyX1 = rotatedImgRoi.cols / 2 - roiRect.width / 2;
    int whereToCopyX2 = whereToCopyX1 + roiRect.width;
    cv::Rect whereToCopy(cv::Point(whereToCopyX1, 0), cv::Point(whereToCopyX2, roiRect.height));
    img(roiRect).copyTo(rotatedImgRoi(whereToCopy));

    cv::Point2f center(rotatedImgRoi.cols / 2.0, rotatedImgRoi.rows / 2.0);
    cv::Mat rotation;
    rotation = getRotationMatrix2D(center,
                                   beamAngle * 180.0 / CV_PI - 180, 1.0);
    warpAffine(rotatedImgRoi, rotatedImgRoi, rotation, rotatedImgRoi.size());

    cv::Rect roiRect2(center.x - beamRect.size.width / 2.0,
                      center.y - beamRect.size.height / 2.0,
                      beamRect.size.width, beamRect.size.height);
    cv::Mat beamRotatedImgRoi = rotatedImgRoi(roiRect2).clone();

    beam = Beam(beamRotatedImgRoi, beamCenterOnRotatedImgRoi,
                beamRect, pLeftCenter -
                cv::Point2f(beamCenterOnRotatedImgRoi.x *
                            std::cos(beamAngle),
                            beamCenterOnRotatedImgRoi.x *
                            std::sin(beamAngle)));

    DEBUG_BallAndBeamVision__findBeamRoi(
        drawRotatedRect(debugImg, beamRect, cv::Scalar::all(255));
        cv::line(debugImg, central.center(), central.center() +
                 cv::Point2f(1.5 * fromCentralToBeamCenter *
                             std::cos(angleToBeamCenter),
                             1.5 * fromCentralToBeamCenter *
                             std::sin(angleToBeamCenter)),
                 cv::Scalar(0, 255, 255));
        cv::imshow("beamRotatedImgRoi", beamRotatedImgRoi);
        cv::imshow("rotatedImgRoi", rotatedImgRoi);
        );

    return true;
}

static bool findBallPos(Beam const& beam, float& ballPos)
{
    const int THRESHOLD_MIN = 70;
    const int THRESHOLD_MAX = 220;
    const int THRESHOLD_DELTA = 50;
    const float BALL_SIZE_TO_BEAM_MIN_SIZE_RATIO = 1.5;
    const float MIN_BALL_SIZE_ASPECT_RATIO = 0.7;

    cv::Mat beamImg = beam.rotatedImgRoi();
    cv::Mat beamImgGray;
    cv::cvtColor(beamImg, beamImgGray, cv::COLOR_BGR2GRAY);

    cv::Mat beamImgBlur;
    medianBlur(beamImgGray, beamImgBlur, 5);

    cv::Mat beamImgBlurThresh;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Point ballCenterOnBeamImg(-1, -1);

    static int th = THRESHOLD_MIN;
    bool isFirstLoop = true;
    int thInitial = th;
    do {
        if (isFirstLoop || (thInitial != th)) {
            cv::threshold(beamImgBlur, beamImgBlurThresh, th, 255,
                          cv::THRESH_BINARY);
            cv::findContours(beamImgBlurThresh, contours, hierarchy,
                             cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
            if (contours.size() == 1) {
                cv::RotatedRect r = cv::minAreaRect(contours[0]);
                float ballSize = std::max(r.size.width, r.size.height);
                float minSizeOfBeam = std::min(beam.rectOnOriginalImg().size.width,
                                               beam.rectOnOriginalImg().size.height);
                if (ballSize <= minSizeOfBeam * BALL_SIZE_TO_BEAM_MIN_SIZE_RATIO) {
                    if(getRectAspectRatio(r) >= MIN_BALL_SIZE_ASPECT_RATIO) {
                        ballCenterOnBeamImg = r.center;
                        int estimatedBallRadius = (r.size.width + r.size.height) / 4;
                        float ballPosOnBallArea = r.center.x - estimatedBallRadius;
                        int estimatedBallArea = (beam.rectOnOriginalImg().size.width - 2 * estimatedBallRadius);
                        ballPos = ballPosOnBallArea / estimatedBallArea;
                        if (ballPos < 0.0) {
                            ballPos = 0.0;
                        }
                        if (ballPos > 1.0) {
                            ballPos = 1.0;
                        }
                        break;
                    }
                }
            }
        }
        if (isFirstLoop) {
            th = THRESHOLD_MIN;
            isFirstLoop = false;
        } else {
            th += THRESHOLD_DELTA;
        }
    } while (th <= THRESHOLD_MAX);

    DEBUG_BallAndBeamVision__drawBallPos(
        if ((ballCenterOnBeamImg.x >= 0) && (ballCenterOnBeamImg.y >= 0)) {
            cv::circle(beamImg, ballCenterOnBeamImg, 3, cv::Scalar(0, 0, 255));
            cv::line(beamImg, cv::Point(ballCenterOnBeamImg.x, 0), cv::Point(ballCenterOnBeamImg.x, beamImg.rows - 1),
                     cv::Scalar(0, 0, 255), 3);
        }
        );
    DEBUG_BallAndBeamVision__findBallPos(
        cv::drawContours(beamImg, contours, -1, cv::Scalar(0, 255, 0));
        cv::imshow("beamImg", beamImg);
        cv::imshow("beamImgBlur", beamImgBlur);
        cv::imshow("beamImgBlurThresh", beamImgBlurThresh);
        );
    return true;
}


void Vision::updateRealBallPos()
{
    DEBUG_WithEmulatedBallAndBeam(
        realBallPos_ = getEmulatedBallPos();
        realBallPosIsKnown_ = true;
        );

    DEBUG_BLOCK(
        addBallPosToDataForChart(imgForChart, ballPos_);
        if (realBallPosIsKnown_) {
            addRealBallPosToDataForChart(imgForChart, realBallPos_);
        }
        updateChart(imgForChart);
        imshow("ballPosChart", imgForChart);
        );
}

bool Vision::processNewFrame()
{
    cv::Mat frame;
    cap_ >> frame;

    DEBUG_BLOCK(
        debugImg = frame;
        );
    DEBUG_WithEmulatedBallAndBeam(
        createTrackbars(frame);
        addEmulatedBallAndBeam(frame, frameRotationAngle, true);
        );

    updateRealBallPos();

    DEBUG_rotateBallAndBeamFrame(
        cv::Point2f center(debugImg.cols / 2.0, debugImg.rows / 2.0);
        cv::Mat rotation = getRotationMatrix2D(center, frameRotationAngle, 1.0);
        warpAffine(debugImg, debugImg, rotation, debugImg.size());
        );
    bool retValue = false;
    SearchResult result = markersFinder_.find(frame);

    if (!result.allFoundAndIdentified()) {
        std::cout << "Chessboard Markers: not found" << std::endl;
    } else {
        if (!findBeam(frame, result, beam_)) {
            std::cout << "Beam: not found" << std::endl;
        } else {
            if(!findBallPos(beam_, ballPos_)) {
                std::cout << "Ball: not found" << std::endl;
            } else {
                DEBUG_COUT_ENDL(ballPos_);
                retValue = true;
            }
        }
    }
    DEBUG_BallAndBeamVision__drawBallPos(
        std::ostringstream strStream;
        strStream << "estimated ballPos: " << std::setprecision(2) << ballPos_;
        cv::putText(debugImg, strStream.str(), cv::Point2f(50,50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 255), 1);
        if (realBallPosIsKnown_) {
            std::ostringstream strStream;
            strStream << "      real ballPos: " << std::setprecision(2) << realBallPos_;
            cv::putText(debugImg, strStream.str(), cv::Point2f(50,70),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 1);
        }
        );
    DEBUG_BLOCK(
        cv::imshow("debug", debugImg);
        );

    return retValue;
}

