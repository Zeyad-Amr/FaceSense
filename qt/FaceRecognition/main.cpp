#include "mainwindow.h"
#include <QApplication>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <algorithm>
#include <random>
#include <fstream>
#include <ctime>




using namespace cv::ml;

//#include <svm.h>


using namespace std;
using namespace cv;

#include <iostream>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;




double calculateAccuracy(const Mat& y_pred, const Mat& Y_test) {
    CV_Assert(y_pred.size() == Y_test.size());

    int correctPredictions = 0;
    for (int i = 0; i < y_pred.rows; i++) {
//        cout<<y_pred.at<float>(i)<<' '<<Y_test.at<int>(i)<<'\n';
        if (y_pred.at<float>(i) == Y_test.at<int>(i)) {
            correctPredictions++;
        }
    }
//    cout<<"Correct Predictions: "<<correctPredictions<<'\n';
    return static_cast<double>(correctPredictions) / y_pred.rows;
}

double getClassFromName(string name){
    double tenth = name[name.size()-6] - '0';
    double ones = name[name.size()-5] - '0';
    if(!(tenth > 0 && tenth < 10 ) ) return ones;
    return ones+tenth*10;
}

void train_test_split(vector<vector<double>> x, vector<double> y, float train_ratio, int seed,vector<vector<double>> &xTrain,vector<vector<double>> &xTest,vector<double> &yTrain,vector<double> &yTest) {
    // Set the seed for the random number generator
    mt19937 g(seed);

    // Shuffle the vectors randomly
    vector<pair<vector<double>, double>> data(x.size());
    for (int i = 0; i <(int) x.size(); i++) {
        data[i] = make_pair(x[i], y[i]);
    }
    shuffle(data.begin(), data.end(), g);
    vector<vector<double>> shuffled_x(x.size());
    vector<double> shuffled_y(x.size());
    for (int i = 0; i <( int)x.size(); i++) {
        shuffled_x[i] = data[i].first;
        shuffled_y[i] = data[i].second;
    }

    // Compute sizes of training and testing sets
    int n_samples = x.size();
    int n_train = round(n_samples * train_ratio);
    int n_test = n_samples - n_train;

    // Split data into training and testing sets
    vector<vector<double>> x_train(shuffled_x.begin(), shuffled_x.begin() + n_train);
    vector<vector<double>> x_test(shuffled_x.begin() + n_train, shuffled_x.end());
    vector<double> y_train(shuffled_y.begin(), shuffled_y.begin() + n_train);
    vector<double> y_test(shuffled_y.begin() + n_train, shuffled_y.end());

    xTrain = x_train;
    xTest  = x_test ;
    yTrain = y_train;
    yTest  = y_test ;
}


//converts the image to one row, it assumes that the image is greyscale
vector<double> flatten(cv::Mat image){
    vector<double> imageVector;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int pixelIntensity = image.at<uchar>(i, j);
            imageVector.push_back(pixelIntensity);
        }
    }
    return imageVector;
}
void preprocess_data(vector<vector<double>> faces_train, vector<vector<double>> faces_test,vector<vector<double>> &preprocessedXTrain, vector<vector<double>> &preprocessedXTest) {
    int n_samples_train = faces_train.size();
    int n_features = faces_train[0].size();

    // Compute mean and standard deviation of training set
    vector<double> mu(n_features, 0.0);
    vector<double> std_dev(n_features, 0.0);
    for (int i = 0; i < n_samples_train; i++) {
        for (int j = 0; j < n_features; j++) {
            mu[j] += faces_train[i][j];
        }
    }
    for (int j = 0; j < n_features; j++) {
        mu[j] /= n_samples_train;
    }
    for (int i = 0; i < n_samples_train; i++) {
        for (int j = 0; j < n_features; j++) {
            std_dev[j] += pow(faces_train[i][j] - mu[j], 2.0);
        }
    }
    for (int j = 0; j < n_features; j++) {
        std_dev[j] = sqrt(std_dev[j] / (n_samples_train-1));
    }

    // Normalize training set
    vector<vector<double>> X_train(n_samples_train, vector<double>(n_features, 0.0));
    vector<double> std_dev_mod(n_features, 0.0);
    for (int j = 0; j < n_features; j++) {
        std_dev_mod[j] = (std_dev[j] == 0) ? 1 : std_dev[j];
    }
    for (int i = 0; i < n_samples_train; i++) {
        for (int j = 0; j < n_features; j++) {
            X_train[i][j] = (faces_train[i][j] - mu[j]) / std_dev_mod[j];
//            X_train[i][j] = (faces_train[i][j] - mu[j]) / 1;
        }
    }

    // Normalize test set with same mean and standard deviation values as training set
    int n_samples_test = faces_test.size();
    vector<vector<double>> X_test(n_samples_test, vector<double>(n_features, 0.0));
    for (int i = 0; i < n_samples_test; i++) {
        for (int j = 0; j < n_features; j++) {
            X_test[i][j] = (faces_test[i][j] - mu[j]) / std_dev_mod[j];
//            X_test[i][j] = (faces_test[i][j] - mu[j]) / 1;
        }
    }
    preprocessedXTrain = X_train;
    preprocessedXTest = X_test;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    vector<vector<double>> X;
    vector<double> y;

    string path = "D:/SBME/3rd year/2nd term/CV/Ass 5/FaceSense/orl faces/archive/*"; // path to directory containing images
    vector<string> filenames;
    glob(path, filenames);

    for (size_t i = 0; i < filenames.size(); i++) {
        Mat img = imread(filenames[i], 0); // read image

        if (img.empty()) {
            cout << "Could not read image " << filenames[i] << endl;
            continue;
        }

        vector<double> flattenedImg = flatten(img);
        X.push_back(flattenedImg);
        y.push_back(getClassFromName(filenames[i]));
    }
    cout<<X.size()<<' '<<X[0].size()<<' '<<y.size();
    cout<<"Finished getting input\n";

    // Step 1: Load input data from file or some other source.
    vector<vector<double>> x_train;
    vector<vector<double>> x_test;
    vector<double> y_train;
    vector<double> y_test;

    train_test_split(X,y,0.8,42,x_train,x_test,y_train,y_test);

    cout<<"Finished train test split\n";

    preprocess_data(x_train,x_test,x_train,x_test);

    cout<<"Finished preprocessing\n";

    // Convert the input data to OpenCV format
    int num_train_samples = x_train.size();
    int num_test_samples = x_test.size();
    int num_features = x_train[0].size();

    cv::Mat train_data(num_train_samples, num_features, CV_32F);
    cv::Mat test_data(num_test_samples, num_features, CV_32F);
    cv::Mat train_labels(num_train_samples, 1, CV_32S);
    cv::Mat test_labels(num_test_samples, 1, CV_32S);
    //filling trainig matrices
    for (int i = 0; i < num_train_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            train_data.at<float>(i, j) = static_cast<float>(x_train[i][j]);
        }
        train_labels.at<int>(i, 0) = static_cast<int>(y_train[i]);
    }
    //filling testing matrices
    for (int i = 0; i < num_test_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            test_data.at<float>(i, j) = static_cast<float>(x_test[i][j]);
        }
        test_labels.at<int>(i, 0) = static_cast<int>(y_test[i]);
    }

    // Reduce the dimensionality of the data using PCA
    PCA pca(train_data, cv::Mat(), PCA::DATA_AS_ROW, 150);
    cv::Mat reduced_train_data = pca.project(train_data);
    cv::Mat reduced_test_data = pca.project(test_data);


    // Train an SVM classifier on the reduced data
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setGamma(1e-4);
    svm->setC(100);
    svm->train(reduced_train_data, ROW_SAMPLE, train_labels);

    // Predict labels for the test data using the trained SVM classifier
    cv::Mat predictions;
    svm->predict(reduced_test_data, predictions);

    //printing accuracy
    double accuracy = calculateAccuracy(predictions, test_labels);
    cout <<"Accuracy: " << accuracy << endl;


    return 0;
    return a.exec();
}
