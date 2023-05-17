//
// Created by Zeyad on 5/15/2023.
//

#include "MyPCA.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


// Function to calculate the mean of each column (feature) in the input matrix
VectorXd calculateMean(const MatrixXd& input) {
    VectorXd mean(input.cols());
    for (int i = 0; i < input.cols(); i++) {
        mean(i) = input.col(i).mean();
    }
    return mean;
}

// Function to normalize the input matrix by subtracting the mean
MatrixXd normalize(MatrixXd input, const VectorXd& mean) {
    MatrixXd normalized_data=input;
    cout<<"Data Size: "<<normalized_data.rows()<<" x "<<normalized_data.cols()<<endl;

    for (int i = 0; i < normalized_data.rows(); i++) {
        normalized_data.row(i) -= mean.transpose();
    }
    return  normalized_data;
}

MatrixXd performPCA(const MatrixXd data, int numComponents) {
    // Calculate the mean of each column (feature) in the input data
    VectorXd mean = data.colwise().mean();

    // Normalize the data by subtracting the mean
    MatrixXd normalizedData = data.rowwise() - mean.transpose();

    // Compute the covariance matrix
    MatrixXd covariance = (normalizedData.transpose() * normalizedData) / (normalizedData.rows() - 1);

    // Perform eigenvalue decomposition on the covariance matrix
    SelfAdjointEigenSolver<MatrixXd> eigenSolver(covariance);

    // Sort eigenvalues and eigenvectors in descending order
    MatrixXd eigenVectors = eigenSolver.eigenvectors().rowwise().reverse();
    VectorXd eigenValues = eigenSolver.eigenvalues().reverse();

    // Select the top 'numComponents' eigenvectors
    MatrixXd selectedEigenVectors = eigenVectors.leftCols(numComponents);

    cout<<"Data=> "<<normalizedData.rows()<<" x "<<normalizedData.cols()<<endl;
    cout<<"selectedEigenVectors=> "<<selectedEigenVectors.rows()<<" x "<<selectedEigenVectors.cols()<<endl;

    // Project the normalized data onto the selected eigenvectors to obtain the reduced data
    MatrixXd reducedData = normalizedData * selectedEigenVectors;

    cout<<"reducedData=> "<<reducedData.rows()<<" x "<<reducedData.cols()<<endl;

    return reducedData;
}

MatrixXd performPCA2(const MatrixXd data, int numComponents) {
    // Calculate the mean and normalize the data
    VectorXd mean = calculateMean(data);
    MatrixXd normalized_data = data;
    normalize(normalized_data, mean);

    // Perform Singular Value Decomposition (SVD) on the normalized data
    JacobiSVD<MatrixXd> svd(normalized_data, ComputeThinU | ComputeThinV);

    // Retrieve the singular vectors (eigenvectors) from the SVD
    MatrixXd singularVectors = svd.matrixU();

    // Select the top 'numComponents' singular vectors
    MatrixXd selectedSingularVectors = singularVectors.leftCols(numComponents);

    cout<<"Data=> "<<normalized_data.rows()<<" x "<<normalized_data.cols()<<endl;
    cout<<"selectedSingularVectors=> "<<selectedSingularVectors.rows()<<" x "<<selectedSingularVectors.cols()<<endl;

    // Project the data onto the selected singular vectors
    MatrixXd reducedData = normalized_data * selectedSingularVectors;

    cout<<"reducedData=> "<<reducedData.rows()<<" x "<<reducedData.cols()<<endl;

    return reducedData;
}


Eigen::MatrixXd cvMatToEigen(const cv::Mat& cvMat) {
    cout<<"start convert to eigen"<<endl;
    cv::Mat cvMatDouble;
    cvMat.convertTo(cvMatDouble, CV_64FC1);

    Eigen::MatrixXd eigenMat(cvMatDouble.rows, cvMatDouble.cols);
    Eigen::Map<Eigen::MatrixXd>(eigenMat.data(), eigenMat.rows(), eigenMat.cols()) = Eigen::Map<const Eigen::MatrixXd>(cvMatDouble.ptr<double>(), cvMatDouble.rows, cvMatDouble.cols);

    cout<<"end convert to eigen"<<endl;
    return eigenMat;
}

cv::Mat eigenToCvMat(const Eigen::MatrixXd& eigenMat) {
    cout<<"start convert to mat"<<endl;

    cv::Mat cvMat(eigenMat.rows(), eigenMat.cols(), CV_64FC1);
    Eigen::Map<Eigen::MatrixXd>(cvMat.ptr<double>(), cvMat.rows, cvMat.cols) = eigenMat;

    // Convert training and test data to CV_32F
    cv::Mat cvMat32F;
    cvMat.convertTo(cvMat32F, CV_32F);

    cout<<"end convert to mat"<<endl;
    return cvMat32F;
}


MyPCA::MyPCA(){


}

Mat MyPCA::apply(Mat data){
    cout<<"Train to eigen"<<endl;
    MatrixXd train_data_eigen = cvMatToEigen(data);

    cout<<"Start Perform"<<endl;
    MatrixXd reduced_data_eigen ;

    cout<<"Data Dimension: "<<data.rows<<" x "<<data.cols<<endl;

    reduced_data_eigen= performPCA2(train_data_eigen,150);
    Mat reducedMat =eigenToCvMat(reduced_data_eigen);


    cout<<"Reduced Data Dimension: "<<reducedMat.rows<<" x "<<reducedMat.cols<<endl;

    return reducedMat;
}
