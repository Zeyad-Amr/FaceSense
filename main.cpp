#include <set>
#include "k-means.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>


int main(int argc, char *argv[])
{

    cv::CascadeClassifier faceCascade;
    faceCascade.load("haarcascade_frontalface_alt.xml");

    cv::Mat image = cv::imread("02.jfif");

//     if you want to convert to grayscale
//    cv::Mat grayImage;
//    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0, cv::Size(30, 30));

    for (const cv::Rect& faceRect : faces)
    {
        cv::rectangle(image, faceRect, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Face Detection", image);
//    cv::imshow("Face Detection", grayImage);
    cv::waitKey(0);



    return 0;

}
