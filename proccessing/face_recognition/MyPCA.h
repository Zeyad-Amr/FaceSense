#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;
using namespace cv;

class MyPCA
{
public:
    MyPCA(const Mat data, int maxComponents);
    Mat reduceData(const Mat data);

private:
    int numSamples;
    int numFeatures;
    VectorXd mean;
    MatrixXd normalizedData;
    MatrixXd covariance;
    SelfAdjointEigenSolver<MatrixXd> eigenSolver;
    MatrixXd selectedEigenVectors;

    VectorXd calculateMean(const MatrixXd &data);
    MatrixXd normalizeData(const MatrixXd &data, const VectorXd &mean);
    MatrixXd calculateCovariance(const MatrixXd &normalizedData);
    SelfAdjointEigenSolver<MatrixXd> calculateEigenSolver(const MatrixXd &covariance);
    MatrixXd selectTopEigenVectors(const MatrixXd &eigenVectors, int maxComponents);
    Eigen::MatrixXd cvMatToEigen(const cv::Mat &cvMat);
    Mat eigenToCvMat(const MatrixXd &eigenMat);
    void storeSelectedEigenVectors(const MatrixXd &selectedEigenVectors, const string &filename);
    void loadSelectedEigenVectors(const string &filename, MatrixXd &selectedEigenVectors);
};
