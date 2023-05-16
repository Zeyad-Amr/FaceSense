//
// Created by Zeyad on 5/15/2023.
//

#ifndef FACE_RECOGNITION_RECOGNITION_H
#define FACE_RECOGNITION_RECOGNITION_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <math.h>
#include <algorithm>
#include <random>
#include <fstream>
#include <ctime>
#include <iostream>

using namespace cv::ml;
using namespace std;
using namespace cv;

class recognition
{

public:
    static double calculateAccuracy(const Mat &y_pred, const Mat &Y_test);
    static double getClassFromName(string name); // extracts the class of the image from its naming.
    static void train_test_split(vector<vector<double>> x, vector<double> y, float train_ratio, int seed, vector<vector<double>> &xTrain, vector<vector<double>> &xTest, vector<double> &yTrain, vector<double> &yTest);
    static vector<double> flatten(cv::Mat image); // converts the image to one row, it assumes that the image is greyscale

    // normalize the data through subtracting mean and dividing by standarad deviation.
    static pair<vector<double>, vector<double>> preprocess_data(vector<vector<double>> faces_train, vector<vector<double>> faces_test, vector<vector<double>> &preprocessedXTrain, vector<vector<double>> &preprocessedXTest);
};

#endif // FACE_RECOGNITION_RECOGNITION_H
