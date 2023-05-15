//
// Created by Zeyad on 5/15/2023.
//

#ifndef FACE_RECOGNITION_MYPCA_H
#define FACE_RECOGNITION_MYPCA_H
#include "recognition.h"


class MyPCA {
public:
    cv::Mat reduced_train_data;
    cv::Mat reduced_test_data;
    MyPCA(cv::Mat train_data, cv::Mat test_data);
private:
    void apply_pca(cv::Mat train_data, cv::Mat test_data);

};


#endif //FACE_RECOGNITION_MYPCA_H
