#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include "recognition.h"

double sigmoid(double z);
std::pair<cv::Mat, double> compute_gradient_logistic(const cv::Mat& X, const cv::Mat& y, const cv::Mat& w, double b, double lambda_);
std::pair<cv::Mat, double> gradient_descent(const cv::Mat& X, const cv::Mat& y, const cv::Mat& w_in, double b_in, double alpha, int num_iters, double lambda_);
cv::Mat one_vs_all(const cv::Mat& y, int class_label);
std::unordered_map<int, std::pair<cv::Mat, double>> train_one_vs_all(const cv::Mat& X, const cv::Mat& y, const cv::Mat& w_in, double b_in, double alpha, int num_iters, double lambda_);
cv::Mat predict_multi_class(const cv::Mat& X, const std::unordered_map<int, std::pair<cv::Mat, double>>& models);
float calculate_accuracy(const cv::Mat& y_true, const cv::Mat& y_pred);
