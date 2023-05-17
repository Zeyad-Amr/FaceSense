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
    MyPCA(const Mat& data, int maxComponents);
    Mat reduceData(const Mat& data);

private:
    long long numSamples;
    long long numFeatures;
    VectorXd mean;
    MatrixXd normalizedData;
    MatrixXd covariance;
    SelfAdjointEigenSolver<MatrixXd> eigenSolver;
    MatrixXd selectedEigenVectors;

    static VectorXd calculateMean(const MatrixXd &data);
    static MatrixXd normalizeData(const MatrixXd &data, const VectorXd &mean);
    MatrixXd calculateCovariance(const MatrixXd &data) const;
    static SelfAdjointEigenSolver<MatrixXd> calculateEigenSolver(const MatrixXd &covar);
    static MatrixXd selectTopEigenVectors(const MatrixXd &eigenVectors, int maxComponents);
    static Eigen::MatrixXd cvMatToEigen(const cv::Mat &cvMat);
    static Mat eigenToCvMat(const MatrixXd &eigenMat);
    static void storeSelectedEigenVectors(const MatrixXd &eigenVectors, const string &filename);
    static void loadSelectedEigenVectors(const string &filename, MatrixXd &eigenVectors);
};
