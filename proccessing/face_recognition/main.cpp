#include "recognition.h"
#include "MyPCA.h"

int main(int argc, char *argv[])
{
    // recognition part:
    vector<vector<double>> X; // the whole data set
    vector<double> y;         // the whole labels

    // Load the cascade classifier XML file for face detection
    CascadeClassifier faceCascade;
    faceCascade.load("C:/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml");

    vector<Rect> faces;
    string path = "../../../orl faces/cropped/*"; // path to directory containing images
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

    recognition().train_test_split(X, y, 0.8, 40, x_train, x_test, y_train, y_test);

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

    // Reduce the dimensionality of the data using PCA
//    PCA pca(train_data, Mat(), PCA::DATA_AS_ROW, 150);

//    Mat  reduced_train_data = pca.project(train_data);
//    Mat  reduced_test_data = pca.project(test_data);

    MyPCA pca(train_data, 150);

    Mat reduced_train_data = pca.reduceData(train_data);
    Mat reduced_test_data = pca.reduceData(test_data);

    // Train an SVM classifier on the reduced data
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setGamma(1e-4);
    svm->setC(100);
    svm->train(reduced_train_data, ROW_SAMPLE, train_labels);

    // Predict labels for the test data using the trained SVM classifier
    Mat predictions;
    svm->predict(reduced_test_data, predictions);

    // printing accuracy
    double accuracy = recognition().calculateAccuracy(predictions, test_labels);
    cout << "Accuracy: " << accuracy << endl;

    // Predicting for incoming image
    Mat grayImage = imread("../../../orl faces/test_42.jpg", 0); // read image

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

    vector<double> flattenedImg = recognition().flatten(grayImage);
    incoming_data_vec.push_back(flattenedImg);

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
    Mat reduced_incoming_data = pca.reduceData(incomingData);
    // Predict labels for the test data using the trained SVM classifier

    Mat predictions_for_incoming;
    svm->predict(reduced_incoming_data, predictions_for_incoming);
    for (int i = 0; i < predictions_for_incoming.rows; ++i)
    {
        cout << "prediction for incoming: " << predictions_for_incoming.at<float>(i) << endl;
    }
    return 0;
}