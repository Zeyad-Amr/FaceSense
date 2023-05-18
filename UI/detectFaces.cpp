
#include "detectFaces.h"

detect_faces::detect_faces() {}

void detect_faces::apply(Mat inputImg, Mat& output)
{


    // Load the cascade classifier XML file for face detection
    cv::CascadeClassifier faceCascade;
    faceCascade.load("C:/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml");

// Convert the image to grayscale for face detection
            cv::Mat grayImage;
            cv::cvtColor(inputImg, grayImage, cv::COLOR_BGR2GRAY);

            // Perform face detection
            std::vector<cv::Rect> faces;
            faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

            output = inputImg;
            // Draw rectangles around the detected faces
            for (const auto &face : faces)
            {
                cv::rectangle(output, face, cv::Scalar(0, 255, 0), 2);
            }

}

QString detect_faces::recognize(Mat inputImg)
{
    // recognition part:
    vector<vector<double>> X; // the whole data set
    vector<double> y;         // the whole labels

    // Load the cascade classifier XML file for face detection
    CascadeClassifier faceCascade;
    faceCascade.load("C:/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml");

    vector<Rect> faces;
    string path = "D:/SBME/ComputerVision/Final-Project/FaceSense/orl faces/archive/*"; // path to directory containing images
    vector<string> filenames;
    glob(path, filenames);

    for (size_t i = 0; i < filenames.size(); i++)
    {

        Mat grayImage = imread(filenames[i], 0); // read image
        // Resize the image
        Size targetSize(50, 50);
        resize(grayImage, grayImage, targetSize);

        if (grayImage.empty())
        {
            cout << "Could not read image " << filenames[i] << endl;
            continue;
        }

        // Perform face detection
        faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // Display the detected faces as separate images
        //        int faceCount = 0;
        //        for (const auto &faceRect : faces)
        //        {
        //            Mat faceImage = grayImage(faceRect); // Extract the region of interest (face) from the image

        vector<double> flattenedImg = recognition().flatten(grayImage);
        X.push_back(flattenedImg);
        y.push_back(recognition().getClassFromName(filenames[i]));
        //        }
    }

    cout << "Finished getting input\n";

    // train test split.
    vector<vector<double>> x_train;
    vector<vector<double>> x_test;
    vector<double> y_train;
    vector<double> y_test;

    recognition().train_test_split(X, y, 0.8, 45, x_train, x_test, y_train, y_test);

    cout << "Finished train test split\n";

    pair<vector<double>, vector<double>> mu_std = recognition().preprocess_data(x_train, x_test, x_train, x_test);

    cout << "Finished preprocessing\n";

    // Convert the input data to OpenCV format
    int num_train_samples = x_train.size();
    int num_test_samples = x_test.size();
    int num_features = x_train[0].size();

    Mat train_data(num_train_samples, num_features, CV_32F);
    Mat test_data(num_test_samples, num_features, CV_32F);
    Mat train_labels(num_train_samples, 1, CV_32S);
    Mat test_labels(num_test_samples, 1, CV_32S);

    // filling trainig matrices
    for (int i = 0; i < num_train_samples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            train_data.at<float>(i, j) = static_cast<float>(x_train[i][j]);
        }
        train_labels.at<int>(i, 0) = static_cast<int>(y_train[i]);
    }
    // filling testing matrices
    for (int i = 0; i < num_test_samples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            test_data.at<float>(i, j) = static_cast<float>(x_test[i][j]);
        }
        test_labels.at<int>(i, 0) = static_cast<int>(y_test[i]);
    }

    // Reduce the dimensionality of the data using PCA
    PCA pca(train_data, Mat(), PCA::DATA_AS_ROW, 150);

    Mat reduced_train_data = pca.project(train_data);
    Mat reduced_test_data = pca.project(test_data);

    //    MyPCA pca(train_data, 150);

    //    Mat reduced_train_data = pca.reduceData(train_data);
    //    Mat reduced_test_data = pca.reduceData(test_data);

    // LOGISTIC
    cv::Mat w_tmp(reduced_train_data.cols, 1, CV_32F, cv::Scalar(1.0));
    double b_tmp = 0.0;
    double alph = 1;
    int iters = 200;
    double lambda_tmp = 0.7;

    // get k models for k classes
    std::unordered_map<int, std::pair<cv::Mat, double>> models = train_one_vs_all(reduced_train_data, train_labels, w_tmp, b_tmp, alph, iters, lambda_tmp);

    cv::Mat y_pred = predict_multi_class(reduced_test_data, models);

    float accuracy = calculate_accuracy(test_labels, y_pred);
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    // LOGISTIC

    // Predicting for incoming image
    Mat grayImage; // read image
    cvtColor(inputImg, grayImage, COLOR_BGR2GRAY);

    if (grayImage.empty())
    {
        cout << "Could not read image" << endl;
    }

    // Resize the image
    Size targetSize(50, 50);
    resize(grayImage, grayImage, targetSize);

    // Perform face detection
    faces.clear();
    vector<vector<double>> incoming_data_vec; // the whole incoming data
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    //    bool thereIsFace = false;
    //    for (const auto &faceRect : faces)
    //    {
    //        thereIsFace = true;
    //        Mat faceImage = grayImage(faceRect); // Extract the region of interest (face) from the image

    vector<double> flattenedImg = recognition().flatten(grayImage);
    incoming_data_vec.push_back(flattenedImg);
    //    }

    //    if (!thereIsFace)
    //    {
    //        cout << "No faces in the incoming Photo\n";
    //        return 0;
    //    }

    // Convert the input data to OpenCV format
    int numOfFaces = incoming_data_vec.size();
    num_features = incoming_data_vec[0].size();

    Mat incomingData(numOfFaces, num_features, CV_32F);

    // filling trainig matrices
    for (int i = 0; i < numOfFaces; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            incoming_data_vec[i][j] = (incoming_data_vec[i][j] - (mu_std.first)[j]) / ((mu_std.second)[j]);
            incomingData.at<float>(i, j) = static_cast<float>(incoming_data_vec[i][j]);
        }
    }

    // Reduce the dimensionality of the data using PCA
    //    Mat reduced_incoming_data = pca.project(incomingData);
    Mat reduced_incoming_data = pca.project(incomingData);

    // Predict labels for the test data using the trained SVM classifier
    //    svm->predict(reduced_incoming_data, predictions_for_incoming);
    Mat predictions_for_incoming = predict_multi_class(reduced_incoming_data, models);

    int result;
    for (int i = 0; i < predictions_for_incoming.rows; ++i)
    {
        result = predictions_for_incoming.at<int>(i);
        cout << "prediction for incoming: " << predictions_for_incoming.at<int>(i) << endl;
    }

    switch (result)
    {
    case 42:
        return "Micheal";
    case 43:
        return "Mazen";
    case 44:
        return "Zeyad";
    case 45:
        return "Ahmed";
    case 46:
        return "Mo'men";

    default:
        return "Couldn't recognize";
    }
}


