#include "MyPCA.h"

MyPCA::MyPCA(const Mat data, int maxComponents)
{
    MatrixXd eigen_data = cvMatToEigen(data);
    numSamples = eigen_data.rows();
    numFeatures = eigen_data.cols();
    mean = calculateMean(eigen_data);
    normalizedData = normalizeData(eigen_data, mean);
    covariance = calculateCovariance(normalizedData);
    eigenSolver = calculateEigenSolver(covariance);
    selectedEigenVectors = selectTopEigenVectors(eigenSolver.eigenvectors(), maxComponents);
}

Mat MyPCA::reduceData(const Mat data)
{

    MatrixXd eigen_data = cvMatToEigen(data);

    numSamples = eigen_data.rows();
    numFeatures = eigen_data.cols();
    mean = calculateMean(eigen_data);
    normalizedData = normalizeData(eigen_data, mean);

    cout << "Data Dimension: " << numSamples << " x " << numFeatures << endl;

    Mat reducedMat = eigenToCvMat(normalizedData * selectedEigenVectors);

    cout << "Reduced Data Dimension: " << reducedMat.rows << " x " << reducedMat.cols << endl;

    return reducedMat;
}

VectorXd MyPCA::calculateMean(const MatrixXd &data)
{
    return data.colwise().mean();
}

MatrixXd MyPCA::normalizeData(const MatrixXd &data, const VectorXd &mean)
{
    return data.rowwise() - mean.transpose();
}

MatrixXd MyPCA::calculateCovariance(const MatrixXd &normalizedData)
{
    return (normalizedData.transpose() * normalizedData) / (numSamples - 1);
}

SelfAdjointEigenSolver<MatrixXd> MyPCA::calculateEigenSolver(const MatrixXd &covariance)
{
    return SelfAdjointEigenSolver<MatrixXd>(covariance);
}

MatrixXd MyPCA::selectTopEigenVectors(const MatrixXd &eigenVectors, int maxComponents)
{
    return eigenVectors.rightCols(maxComponents);
}

MatrixXd MyPCA::cvMatToEigen(const cv::Mat &cvMat)
{
    cout << "start convert to eigen" << endl;
    cv::Mat cvMatDouble;
    cvMat.convertTo(cvMatDouble, CV_64FC1);

    Eigen::MatrixXd eigenMat(cvMatDouble.rows, cvMatDouble.cols);
    Eigen::Map<Eigen::MatrixXd>(eigenMat.data(), eigenMat.rows(), eigenMat.cols()) = Eigen::Map<const Eigen::MatrixXd>(cvMatDouble.ptr<double>(), cvMatDouble.rows, cvMatDouble.cols);

    cout << "end convert to eigen" << endl;
    return eigenMat;
}

Mat MyPCA::eigenToCvMat(const Eigen::MatrixXd &eigenMat)
{
    cout << "start convert to mat" << endl;

    cv::Mat cvMat(eigenMat.rows(), eigenMat.cols(), CV_64FC1);
    Eigen::Map<Eigen::MatrixXd>(cvMat.ptr<double>(), cvMat.rows, cvMat.cols) = eigenMat;

    // Convert training and test data to CV_32F
    cv::Mat cvMat32F;
    cvMat.convertTo(cvMat32F, CV_32F);

    cout << "end convert to mat" << endl;
    return cvMat32F;
}
