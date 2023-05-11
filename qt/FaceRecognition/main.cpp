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

double calculateAccuracy(const Mat& y_pred, const Mat& Y_test) {
    CV_Assert(y_pred.size() == Y_test.size());

    int correctPredictions = 0;
    for (int i = 0; i < y_pred.rows; i++) {
        if (y_pred.at<double>(i) == Y_test.at<double>(i)) {
            correctPredictions++;
        }
    }

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
    for (int i = 0; i < x.size(); i++) {
        data[i] = make_pair(x[i], y[i]);
    }
    shuffle(data.begin(), data.end(), g);
    vector<vector<double>> shuffled_x(x.size());
    vector<double> shuffled_y(x.size());
    for (int i = 0; i < x.size(); i++) {
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
vector<double> convertMatToVector(cv::Mat image){
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
        std_dev[j] = sqrt(std_dev[j] / n_samples_train);
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
        }
    }

    // Normalize test set with same mean and standard deviation values as training set
    int n_samples_test = faces_test.size();
    vector<vector<double>> X_test(n_samples_test, vector<double>(n_features, 0.0));
    for (int i = 0; i < n_samples_test; i++) {
        for (int j = 0; j < n_features; j++) {
            X_test[i][j] = (faces_test[i][j] - mu[j]) / std_dev_mod[j];
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
            vector<double> flattenedImg = convertMatToVector(img);
            X.push_back(flattenedImg);
            y.push_back(getClassFromName(filenames[i]));
        }
    cout<<"Finished getting input\n";
    // Step 1: Load input data from file or some other source.
    vector<vector<double>> x_train; // assume this is loaded with training data
    vector<vector<double>> x_test; // assume this is loaded with testing data
    vector<double> y_train; // assume this is loaded with labels for training data
    vector<double> y_test; // assume this is loaded with labels for testing data

    train_test_split(X,y,0.8,42,x_train,x_test,y_train,y_test);
    cout<<"Finished train test split\n";
    preprocess_data(x_train,x_test,x_train,x_test);
    cout<<"Finished preprocessing\n";

    // Step 2: Split dataset into training and testing
    // and convert them to matrices so that pca and svm can deal with them.
    Mat X_train(x_train.size(), x_train[0].size(), CV_64F);
    Mat X_test(x_test.size(), x_test[0].size(), CV_64F);
    Mat Y_train(y_train.size(), 1, CV_64F);
    Mat Y_test(y_test.size(), 1, CV_64F);
    for (int i = 0; i < x_train.size(); i++) {
        for (int j = 0; j < x_train[i].size(); j++) {
            X_train.at<double>(i, j) = x_train[i][j];
        }
        Y_train.at<double>(i, 0) = y_train[i];
    }
    for (int i = 0; i < x_test.size(); i++) {
        for (int j = 0; j < x_test[i].size(); j++) {
            X_test.at<double>(i, j) = x_test[i][j];
        }
        Y_test.at<double>(i, 0) = y_test[i];
    }
    cout<<"Finished converting to Mats\n";

    // Step 3: Perform PCA
    PCA pca(X_train, Mat(), PCA::DATA_AS_ROW, 150);
    Mat eigenvalues = pca.eigenvalues;
    Mat eigenvectors = pca.eigenvectors;
//    cout << "Explained variance ratio: " << sum(eigenvalues.col(0)).at<double>(0, 0) / sum(eigenvalues).at<double>(0, 0) << endl;

    cout<<"Finished PCs\n";

    // Step 4: Project Training data to PCA
    cout << "Projecting the input data on the eigenfaces orthonormal basis" << endl;
    Mat Xtrain_pca = pca.project(X_train);

    Xtrain_pca.convertTo(Xtrain_pca, CV_32F);
    Y_train.convertTo(Y_train, CV_32S);

    // Step 5: Initialize Classifier and fit training data
//    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
//    svm->setType(cv::ml::SVM::C_SVC);
//    svm->setKernel(cv::ml::SVM::RBF);
//    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

//    svm->trainAuto(Xtrain_pca, cv::ml::ROW_SAMPLE, Y_train);
        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::RBF);
        svm->setGamma(1e-3);
        svm->setC(1000);
        svm->train(Xtrain_pca, ROW_SAMPLE, Y_train);

    cout<<"Finished SVM\n";

    // Step 6: Perform testing and get classification report
    cout << "Predicting people's names on the test set" << endl;
    clock_t t0 = clock();
    Mat Xtest_pca = pca.project(X_test);
    Xtest_pca.convertTo(Xtest_pca, CV_32F);
    Mat y_pred;
    svm->predict(Xtest_pca, y_pred);
    cout << "done in " << (double)(clock() - t0) / CLOCKS_PER_SEC << "s" << endl;
    std::cout << "Y_test size: " << Y_test.rows << "x" << Y_test.cols << std::endl;
    std::cout << "y_pred size: " << y_pred.rows << "x" << y_pred.cols << std::endl;
//    auto booleanArray = (Y_test == y_pred);

//    double accuracy = cv::countNonZero(booleanArray) / Y_test.rows;
    double accuracy = calculateAccuracy(y_pred, Y_test);

    //No member named ;at; in 'cv::Scalar_<double>'
    cout << "Accuracy: " << accuracy << endl;
    return 0;
    return a.exec();
}

//#include <opencv2/opencv.hpp>

//using namespace cv;
//using namespace cv::ml;

//int main(int argc, char *argv[])
//{
//    QApplication a(argc, argv);

//    // Create a training dataset with 100 samples of 2D points
//    Mat trainData(100, 2, CV_32FC1);
//    Mat labels(100, 1, CV_32SC1);
//    for (int i = 0; i < 50; i++) {
//        trainData.at<float>(i, 0) = float(rand()) / RAND_MAX * 2 - 1;
//        trainData.at<float>(i, 1) = float(rand()) / RAND_MAX * 2 - 1;
//        labels.at<int>(i, 0) = -1;
//    }
//    for (int i = 50; i < 100; i++) {
//        trainData.at<float>(i, 0) = float(rand()) / RAND_MAX * 2 + 1;
//        trainData.at<float>(i, 1) = float(rand()) / RAND_MAX * 2 + 1;
//        labels.at<int>(i, 0) = 1;
//    }

//    // Train an SVM model with RBF kernel
//    Ptr<SVM> svm = SVM::create();
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::RBF);
//    svm->setGamma(0.1);
//    svm->setC(10);
//    svm->train(trainData, ROW_SAMPLE, labels);

//    // Test the model with a new sample
//    Mat sample(1, 2, CV_32FC1);
//    sample.at<float>(0, 0) = 0.5f;
//    sample.at<float>(0, 1) = 0.5f;
//    float prediction = svm->predict(sample);

//    std::cout << "Prediction: " << prediction << std::endl;

//    return 0;
//    return a.exec();

//}

