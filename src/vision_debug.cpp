#include <cmath>
#include <fstream>
#include "vision.h"
#include "vision_debug.h"

float g_emulatedBallPos = 0.0;

const int INTERVAL_IN_PIXELS_FOR_CHART = 5;
std::deque<float> g_realBallPosData;
std::deque<float> g_ballPosData;

int g_ballVelocityForScaling = 100;
int g_scale = 35;
int g_shiftX = 100;
int g_shiftY = 100;
int g_beamScale = 300;
int g_rotationSpeed = 0;
bool g_trackBarsCreated = false;

void createTrackbars(cv::Mat const& img)
{
    if (g_trackBarsCreated)
        return;
    g_trackBarsCreated = true;
    g_shiftX = img.cols / 5;
    g_shiftY = img.rows / 3;
    cv::createTrackbar("markers_ballVelocity", "sliders", &g_ballVelocityForScaling, 1000, NULL);
    cv::createTrackbar("markers_scale", "sliders", &g_scale, 100, NULL);
    cv::createTrackbar("markers_shiftX", "sliders", &g_shiftX, img.cols, NULL);
    cv::createTrackbar("markers_shiftY", "sliders", &g_shiftY, img.rows, NULL);
    cv::createTrackbar("markers_beamScale", "sliders", &g_beamScale, std::min(img.rows, img.cols), NULL);
    cv::createTrackbar("markers_rotationSpeed", "sliders", &g_rotationSpeed, 50, NULL);
}

void createCVWindow(cv::String const& name)
{
    cv::namedWindow(name);
    std::ifstream layoutFile;
    std::ostringstream strStream;
    strStream << "debug/" << name;
    layoutFile.open(strStream.str());
    if (layoutFile.is_open()) {
        int x, y;
        layoutFile >> x >> y;
        std::cout << "window \"" << name << "\" x:" << x << " y:" << y << std::endl;
        cv::moveWindow(name, x, y);
        layoutFile.close();
    }
}

void drawEmulatedChessboardMarker(cv::Mat& img, int x, int y, int w, int h)
{
    cv::rectangle(img, cv::Point(x, y), cv::Point(x + 5 * w, y + 5 * h),
                  cv::Scalar(255, 255, 255), cv::FILLED);
    for (int dyCounter = 0; dyCounter < 4; dyCounter++) {
        int dy = dyCounter * h;
        unsigned char bgrComponent = (dyCounter % 2) ? 255 : 0;
        for (int dxCounter = 0; dxCounter < 4; dxCounter++) {
            int dx = dxCounter * w;
            int xi = x + dx + w / 2;
            int yi = y + dy + h / 2;
            bgrComponent = (bgrComponent == 0) ? 255 : 0;
            cv::rectangle(img, cv::Point(xi, yi), cv::Point(xi + w, yi + h),
                          cv::Scalar(bgrComponent, bgrComponent, bgrComponent), cv::FILLED);
        }
    }
}

float getEmulatedBallPos()
{
    return g_emulatedBallPos;
}

void drawEmulatedBall(cv::Mat& img, int x, int y, int w, int h)
{
    static float time = 0;
    time += g_ballVelocityForScaling / 1000.0;
    float position = (sin(time) + 1) / 2;
    g_emulatedBallPos = position;

    int r = h;
    int usedLength = w - r * 2;
    float positionForDrawing = usedLength * position;

    cv::circle(img, cv::Point(x + r + positionForDrawing, y + h / 2), r, cv::Scalar(255, 255, 255), cv::FILLED);
}

void drawEmulatedBallAndBeam(cv::Mat& img, int x, int y, int w, int h)
{
    cv::rectangle(img, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(128, 128, 128), cv::FILLED);
    cv::line(img, cv::Point(x, y), cv::Point(x, y + h), cv::Scalar(0, 255, 0), 3);
    cv::line(img, cv::Point(x + w, y), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 3);

    drawEmulatedBall(img, x, y, w, h);
}

void addEmulatedBallAndBeam(cv::Mat& img, int& frameRotationAngle, bool autoMoving)
{
    if (autoMoving) {
        frameRotationAngle = (frameRotationAngle + g_rotationSpeed) % 360;
    }

    int w = 20 * g_scale / 100.0;
    int h = 20 * g_scale / 100.0;
    int x = g_shiftX;
    int y = g_shiftY;
    int beamLength = 100 * g_beamScale / 100.0;

    int beamLengthGap = w;
    int beamWidthGap = 2 * h;
    drawEmulatedBallAndBeam(img, x + 5 * w + beamLengthGap, y + beamWidthGap,
                     beamLength - 5 * w - 2 * beamLengthGap, 5 * h - 2 * beamWidthGap);

    drawEmulatedChessboardMarker(img, x, y, w, h);
    drawEmulatedChessboardMarker(img, x + beamLength, y, w, h);
    drawEmulatedChessboardMarker(img, x + beamLength / 2, y - beamLength / 5, w, h);
}

void addValueToDataForChart(cv::Mat const& img, float value, std::deque<float>& data)
{
    assert((value >= 0.0) && (value <= 1.0));

    unsigned sizeOfData = ceil((float)img.cols / INTERVAL_IN_PIXELS_FOR_CHART);

    data.push_back(value);
    while(data.size() > sizeOfData) {
        data.pop_front();
    }
}

void addBallPosToDataForChart(cv::Mat const& img, float ballPos)
{
    addValueToDataForChart(img, ballPos, g_ballPosData);
}

void addRealBallPosToDataForChart(cv::Mat const& img, float realBallPos)
{
    addValueToDataForChart(img, realBallPos, g_realBallPosData);
}

void drawChartForData(cv::Mat& img, std::deque<float> const& data, cv::Scalar const& color)
{
    int x = 0;
    float previousValue = 0.0;
    for (auto value : data) {
        cv::circle(img, cv::Point(x, img.rows * (1.0 - value)), 1, color);
        if (x > 0) {
            cv::line(img, cv::Point(x - INTERVAL_IN_PIXELS_FOR_CHART, img.rows * (1.0 - previousValue)),
                     cv::Point(x, img.rows * (1.0 - value)), color, 1);
        }
        previousValue = value;
        x += INTERVAL_IN_PIXELS_FOR_CHART;
    }
}

void updateChart(cv::Mat& img)
{
    img = cv::Mat::zeros(img.size(), img.type());
    drawChartForData(img, g_ballPosData, cv::Scalar(0, 255, 255));
    drawChartForData(img, g_realBallPosData, cv::Scalar(0, 255, 0));
}
