#include "logistic.h"
#include <opencv2/core/mat.hpp>
#include <cmath>
#include <algorithm>
#include <unordered_map>


cv::Mat sigmoid(const cv::Mat& z) {
    // Apply numerical stability techniques
    const double clip_value = 500.0;
    cv::Mat g(z.size(), z.type());

    for (int row = 0; row < z.rows; row++) {
        for (int col = 0; col < z.cols; col++) {
            double value = z.at<double>(row, col);
            double clipped_value = std::min(std::max(value, -clip_value), clip_value);
            double sigmoid_value = 1 / (1 + std::exp(-clipped_value));
            g.at<double>(row, col) = sigmoid_value;
        }
    }

    return g;
}


std::pair<cv::Mat, double> compute_gradient_logistic(const cv::Mat& X, const cv::Mat& y, const cv::Mat& w, double b, double lambda_) {
    int m = X.rows;
    cv::Mat z = X * w + b;
    cv::Mat f_wb = sigmoid(z);
    cv::Mat err = f_wb - y;
    cv::Mat dj_dw = (X.t() * err) / m + (lambda_ / m) * w;
    double dj_db = cv::mean(err)[0];
    return std::make_pair(dj_dw, dj_db);
}






std::pair<cv::Mat, double> gradient_descent(const cv::Mat& X, const cv::Mat& y, const cv::Mat& w_in, double b_in, double alpha, int num_iters, double lambda_) {
    cv::Mat w = w_in.clone();
    double b = b_in;

    for (int i = 0; i < num_iters; i++) {
        cv::Mat dj_dw;
        double dj_db;
        std::tie(dj_dw, dj_db) = compute_gradient_logistic(X, y, w, b, lambda_);

        w -= alpha * dj_dw;
        b -= alpha * dj_db;
    }

    return std::make_pair(w, b);
}


cv::Mat one_vs_all(const cv::Mat& y, int class_label) {
    cv::Mat y_binary = cv::Mat::zeros(y.size(), CV_32S);

    for (int i = 0; i < y.rows; i++) {
        if (y.at<int>(i) == class_label) {
            y_binary.at<int>(i) = 1;
        }
    }

    return y_binary;
}


std::unordered_map<int, std::pair<cv::Mat, double>> train_one_vs_all(const cv::Mat& X, const cv::Mat& y, const cv::Mat& w_in, double b_in, double alpha, int num_iters, double lambda_) {
    cv::Mat classes;
    cv::reduce(y, classes, 0, cv::REDUCE_MAX);  // Find the maximum value along each column

    std::unordered_map<int, std::pair<cv::Mat, double>> models;

    for (int i = 0; i < classes.rows; i++) {
        int class_label = classes.at<int>(i);
        cv::Mat y_binary = one_vs_all(y, class_label);
        std::pair<cv::Mat, double> model = gradient_descent(X, y_binary, w_in, b_in, alpha, num_iters, lambda_);
        models[class_label] = model;
    }

    return models;
}


cv::Mat predict_multi_class(const cv::Mat& X, const std::unordered_map<int, std::pair<cv::Mat, double>>& models) {
    cv::Mat y_pred(X.rows, 1, CV_32S);

    for (int i = 0; i < X.rows; i++) {
        std::unordered_map<int, double> class_scores;
        for (const auto& model : models) {
            int class_label = model.first;
            cv::Mat w = model.second.first;
            double b = model.second.second;

//            double z = X.row(i) * w + b;
//            double z = (X.row(i) * w + cv::Scalar(b)).at<double>(0, 0);
//            double z = (X.row(i) * w + cv::Scalar(b)).toScalar();
//            double z = (X.row(i) * w + cv::Scalar(b)).at<double>(0);
            double z = cv::sum(X.row(i).mul(w))[0] + b;

            double score = 1.0 / (1.0 + std::exp(-z));

            class_scores[class_label] = score;
        }

        int predicted_class = std::max_element(class_scores.begin(), class_scores.end(),
            [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                return a.second < b.second;
            })->first;

        y_pred.at<int>(i) = predicted_class;
    }

    return y_pred;
}

float calculate_accuracy(const cv::Mat& y_true, const cv::Mat& y_pred) {
    CV_Assert(y_true.rows == y_pred.rows && y_true.cols == 1 && y_pred.cols == 1);

    int correct = 0;
    for (int i = 0; i < y_true.rows; i++) {
        if (y_true.at<int>(i) == y_pred.at<int>(i)) {
            correct++;
        }
    }

    float accuracy = static_cast<float>(correct) / y_true.rows * 100.0f;
    return accuracy;
}
