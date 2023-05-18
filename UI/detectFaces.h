
#ifndef UNTITLED_DETECTFACES_THRESHOLDING_H
#define UNTITLED_DETECTFACES_THRESHOLDING_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <QString>
using namespace std;
using namespace cv;


#include "logistic.h"
#include "recognition.h"


class detect_faces {
public:
    detect_faces();
    static void apply(Mat inputImg, Mat& output);
    static QString recognize(Mat inputImg);

};


#endif //UNTITLED_OTSU_THRESHOLDING_H
