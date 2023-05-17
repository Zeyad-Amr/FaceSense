#include "logistic.h"
#include <opencv2/core/mat.hpp>
#include <cmath>
#include <algorithm>
#include <unordered_map>


std::vector<double> matToVector(const cv::Mat& mat) {
    CV_Assert(mat.cols == 1); // Ensure the input matrix has a single column

    std::vector<double> vec;
    vec.reserve(mat.rows);

    for (int i = 0; i < mat.rows; ++i) {
        vec.push_back(mat.at<int>(i, 0));
    }

    return vec;
}

cv::Mat vectorToMat(const std::vector<double>& vec) {
    std::vector<double> uniqueValues(vec);
    std::sort(uniqueValues.begin(), uniqueValues.end());
    auto last = std::unique(uniqueValues.begin(), uniqueValues.end());
    uniqueValues.erase(last, uniqueValues.end());

    cv::Mat mat(uniqueValues.size(), 1, CV_32S);
    for (int i = 0; i < (int)uniqueValues.size(); ++i) {
        mat.at<int>(i, 0) = static_cast<int>(uniqueValues[i]);

    }

    return mat;
}

double sigmoid(double z) {

    double g = 1 / (1 + exp(-z));
    return g;
}


std::pair<cv::Mat, double> compute_gradient_logistic(const cv::Mat& X, const cv::Mat& y, const cv::Mat& w, double b, double lambda_) {

    int m = X.rows;
    int n = X.cols;
    cv::Mat dj_dw = cv::Mat::zeros(n, 1, CV_32F);
    double dj_db = 0.0;
    for (int i = 0; i < m; ++i) {

        double sum = 0;
        for(int j=0;j<w.rows;j++){
            sum += (X.at<float>(i,j)*w.at<float>(j,0));
        }

        double f_wb_i = sigmoid(sum + b);

        double err_i = f_wb_i - y.at<int>(i, 0);
        cv::Mat X_row = X.row(i);
        dj_dw += err_i * X_row.t();
        dj_db += err_i;
    }
    dj_dw /= m;
    dj_db /= m;

    dj_dw += (lambda_ / m) * w;

    return std::make_pair(dj_dw, dj_db);
}



std::pair<cv::Mat, double> gradient_descent(const cv::Mat& X, const cv::Mat& y, const cv::Mat& w_in, double b_in, double alpha, int num_iters, double lambda_) {
    cv::Mat w = w_in.clone();
    double b = b_in;

    for (int i = 0; i < num_iters; ++i) {

        // Calculate the gradient and update the parameters
        std::pair<cv::Mat, double> gradients = compute_gradient_logistic(X, y, w, b, lambda_);

        cv::Mat dj_dw = gradients.first;
        double dj_db = gradients.second;

        // Update parameters using w, b, alpha, and gradients
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
    cv::Mat w = w_in.clone();
    double b = b_in;
    std::unordered_map<int, std::pair<cv::Mat, double>> models;

    vector<double> vec = matToVector(y);
    cv::Mat unique_classes;

    unique_classes = vectorToMat(vec);


    for (int i = 0; i < unique_classes.rows; ++i) {
//        cout<<unique_classes.at<int>(i,0)<<' ';
        int class_label = unique_classes.at<int>(i, 0);
        cv::Mat y_binary = one_vs_all(y, class_label);

        std::pair<cv::Mat, double> model = gradient_descent(X.clone(), y_binary.clone(), w, b, alpha, num_iters, lambda_);

        w = w_in.clone();
        b = b_in;
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

            //dot product
            double sum = 0;
            for(int j=0;j<w.rows;j++){
                sum += (X.at<float>(i,j)*w.at<float>(j,0));
            }
            double z = sum+b;

            double score = sigmoid(z);
            class_scores[class_label] = score;
            cout<<score<<' ';
        }
        cout<<'\n';

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
    cout<<"accuracy function:\n";
    for (int i = 0; i < y_true.rows; i++) {
        if (y_true.at<int>(i) == y_pred.at<int>(i)) {
            correct++;
        }
        cout<<y_true.at<int>(i)<<' '<<y_pred.at<int>(i)<<'\n';
    }

    float accuracy = static_cast<float>(correct) / y_true.rows * 100.0f;
    return accuracy;
}
