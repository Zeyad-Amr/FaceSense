//
// Created by Zeyad on 5/15/2023.
//

#include "MyPCA.h"

MyPCA::MyPCA(cv::Mat train_data, cv::Mat test_data){
// Reduce the dimensionality of the data using PCA
    PCA pca(train_data, cv::Mat(), PCA::DATA_AS_ROW, 150);
    MyPCA::reduced_train_data = pca.project(train_data);
    MyPCA::reduced_test_data = pca.project(test_data);
}

