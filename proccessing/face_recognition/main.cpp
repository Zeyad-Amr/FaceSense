#include "recognition.h"
#include "MyPCA.h"

int main(int argc, char *argv[])
{
    vector<vector<double>> X;
    vector<double> y;

    string path = "../../../orl faces/archive/*"; // path to directory containing images
    vector<string> filenames;
    glob(path, filenames);

    for (size_t i = 0; i < filenames.size(); i++) {
        Mat img = imread(filenames[i], 0); // read image

        if (img.empty()) {
            cout << "Could not read image " << filenames[i] << endl;
            continue;
        }

        vector<double> flattenedImg = recognition().flatten(img);
        X.push_back(flattenedImg);
        y.push_back(recognition().getClassFromName(filenames[i]));
    }

    cout<<"Finished getting input\n";

    //train test split.
    vector<vector<double>> x_train;
    vector<vector<double>> x_test;
    vector<double> y_train;
    vector<double> y_test;

    recognition().train_test_split(X,y,0.8,42,x_train,x_test,y_train,y_test);

    cout<<"Finished train test split\n";

    recognition().preprocess_data(x_train,x_test,x_train,x_test);

    cout<<"Finished preprocessing\n";

    // Convert the input data to OpenCV format
    int num_train_samples = x_train.size();
    int num_test_samples = x_test.size();
    int num_features = x_train[0].size();

    cv::Mat train_data(num_train_samples, num_features, CV_32F);
    cv::Mat test_data(num_test_samples, num_features, CV_32F);
    cv::Mat train_labels(num_train_samples, 1, CV_32S);
    cv::Mat test_labels(num_test_samples, 1, CV_32S);

    //filling trainig matrices
    for (int i = 0; i < num_train_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            train_data.at<float>(i, j) = static_cast<float>(x_train[i][j]);
        }
        train_labels.at<int>(i, 0) = static_cast<int>(y_train[i]);
    }

    //filling testing matrices
    for (int i = 0; i < num_test_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            test_data.at<float>(i, j) = static_cast<float>(x_test[i][j]);
        }
        test_labels.at<int>(i, 0) = static_cast<int>(y_test[i]);
    }

    // Reduce the dimensionality of the data using PCA
//    PCA pca(train_data, cv::Mat(), PCA::DATA_AS_ROW, 150);
//    cv::Mat reduced_train_data = pca.project(train_data);
//    cv::Mat reduced_test_data = pca.project(test_data);
//
    MyPCA pca= MyPCA(train_data, test_data);

    // Train an SVM classifier on the reduced data
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setGamma(1e-4);
    svm->setC(100);
    svm->train(pca.reduced_train_data, ROW_SAMPLE, train_labels);

    // Predict labels for the test data using the trained SVM classifier
    cv::Mat predictions;
    svm->predict(pca.reduced_test_data, predictions);

    //printing accuracy
    double accuracy = recognition().calculateAccuracy(predictions, test_labels);
    cout <<"Accuracy: " << accuracy << endl;


    return 0;

}
