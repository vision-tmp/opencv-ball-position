#include "vision.h"
#include <opencv2/opencv.hpp>

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    ballAndBeam::Vision vision(cap);

    int key;
    const int kbEsc = 27;

    while ((key = cv::waitKey(30)) != kbEsc) {
        vision.processNewFrame();
    }

    return 0;
}
