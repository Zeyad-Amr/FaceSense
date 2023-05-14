
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/opencv.hpp>

int main()
{
    // Load the cascade classifier XML file for face detection
    cv::CascadeClassifier faceCascade;
    faceCascade.load("C:/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml");

    // Load the image file
    cv::Mat image = cv::imread("D:/SBME/ComputerVision/Final-Project/myWorkSpace/01.jpg");
    if (image.empty())
    {
        std::cout << "Failed to open the image file." << std::endl;
        return -1;
    }

    // Convert the image to grayscale for face detection
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Perform face detection
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    // Draw rectangles around the detected faces
    for (const auto &face : faces)
    {
        cv::rectangle(image, face, cv::Scalar(0, 255, 0), 2);
    }

    // Display the image with detected faces
    cv::imshow("Face Detection", image);
    cv::waitKey(0);

    return 0;
}
