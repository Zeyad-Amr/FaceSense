#include "recognition.h"

cv::Mat pca(cv::Mat X) {
    // Center the data by subtracting the mean of each feature
    cv::Mat mean;
    cv::reduce(X, mean, 0, cv::REDUCE_AVG);
    cv::Mat centered_data = X - cv::repeat(mean, X.rows, 1);

    // Compute the covariance matrix of the centered data
    cv::Mat X_double;
    centered_data.convertTo(X_double, CV_64FC1);
    cv::Mat cov;
    cv::calcCovarMatrix(X_double, cov, cv::Mat(), cv::COVAR_NORMAL | cv::COVAR_ROWS);

    // Compute the eigenvalues and eigenvectors of the covariance matrix
    cv::Mat eigenvalues, eigenvectors;
    eigen(cov, eigenvalues, eigenvectors);

    // Sort the eigenvectors in descending order based on their corresponding eigenvalues
    cv::Mat sorted_eigenvectors;
    cv::sortIdx(eigenvalues, eigenvalues, cv::SORT_DESCENDING);
    eigenvectors.convertTo(sorted_eigenvectors, CV_32FC1);
    for (int i = 0; i < sorted_eigenvectors.cols; i++) {
        cv::Mat eigenvector = sorted_eigenvectors.col(i);
        eigenvectors.col(eigenvalues.at<int>(i)).copyTo(eigenvector);
    }

    // Return the sorted eigenvectors
    return sorted_eigenvectors;
}

cv::Mat transformData(const cv::Mat& data, const cv::Mat& eigenvectors, int num_components) {

    // Map the centered data onto the principal axes
    cv::Mat transformed_data = data * eigenvectors(cv::Range::all(), cv::Range(0, num_components));

    // Return the transformed data
    return transformed_data;
}
//its code in main:
//    cv::Mat eigenvectors;
//    int num_components = 150;
//    eigenvectors = pca(train_data);

//    // Project train_data and test_data onto the principal axes
//    cv::Mat reduced_train_data = transformData(train_data, eigenvectors, num_components);
//    cv::Mat reduced_test_data = transformData(test_data, eigenvectors, num_components);

