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
void normalize(MatrixXd& input, const VectorXd& mean) {
    for (int i = 0; i < input.rows(); i++) {
        input.row(i) -= mean.transpose();
    }
}


// Function to perform PCA on the input matrix
void performPCA(const MatrixXd& train_data, const MatrixXd& test_data, int numComponents,
                MatrixXd& reduced_train_data, MatrixXd& reduced_test_data) {
    // Combine the train_data and test_data into a single matrix
    MatrixXd combinedData(train_data.rows() + test_data.rows(), train_data.cols());
    combinedData << train_data, test_data;

    // Calculate the mean and normalize the combined data
    VectorXd mean = calculateMean(combinedData);
    normalize(combinedData, mean);

    // Perform eigendecomposition of the covariance matrix of the combined data
    MatrixXd cov = (combinedData.transpose() * combinedData) / (combinedData.rows() - 1);
    SelfAdjointEigenSolver<MatrixXd> eigenSolver(cov);
    MatrixXd eigenVectors = eigenSolver.eigenvectors();

    // Sort the eigenvectors in descending order of eigenvalues
    VectorXd eigenValues = eigenSolver.eigenvalues();
    std::vector<std::pair<double, int>> eigenPairs;
    for (int i = 0; i < eigenValues.size(); i++) {
        eigenPairs.push_back(make_pair(eigenValues(i), i));
    }
    sort(eigenPairs.rbegin(), eigenPairs.rend());

    // Select the top 'numComponents' eigenvectors
    MatrixXd selectedEigenVectors(combinedData.cols(), numComponents);
    for (int i = 0; i < numComponents; i++) {
        selectedEigenVectors.col(i) = eigenVectors.col(eigenPairs[i].second);
    }

    // Project the train_data and test_data onto the selected eigenvectors
    reduced_train_data = train_data * selectedEigenVectors;
    reduced_test_data = test_data * selectedEigenVectors;
}
MyPCA::MyPCA(cv::Mat train_data, cv::Mat test_data){
    // Reduce the dimensionality of the data using PCA
    PCA pca(train_data, cv::Mat(), PCA::DATA_AS_ROW, 150);
    reduced_train_data = pca.project(train_data);
    reduced_test_data = pca.project(test_data);

}

void MyPCA::apply_pca(cv::Mat train_data, cv::Mat test_data){

}
