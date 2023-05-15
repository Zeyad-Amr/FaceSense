#ifndef RECOGNITION_H
#define RECOGNITION_H


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
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


double calculateAccuracy(const Mat& y_pred, const Mat& Y_test);
double getClassFromName(string name);//extracts the class of the image from its naming.
void train_test_split(vector<vector<double>> x, vector<double> y, float train_ratio, int seed,vector<vector<double>> &xTrain,vector<vector<double>> &xTest,vector<double> &yTrain,vector<double> &yTest);
vector<double> flatten(cv::Mat image);//converts the image to one row, it assumes that the image is greyscale

//normalize the data through subtracting mean and dividing by standarad deviation.
pair<vector<double>,vector<double>>  preprocess_data(vector<vector<double>> faces_train, vector<vector<double>> faces_test,vector<vector<double>> &preprocessedXTrain, vector<vector<double>> &preprocessedXTest);


#endif // RECOGNITION_H
