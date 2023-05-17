
//    // Step 2: Split dataset into training and testing
//    // and convert them to matrices so that pca and svm can deal with them.
//    Mat X_train(x_train.size(), x_train[0].size(), CV_64F);
//    Mat X_test(x_test.size(), x_test[0].size(), CV_64F);
//    Mat Y_train(y_train.size(), 1, CV_32S);
//    Mat Y_test(y_test.size(), 1, CV_32S);
//    for (int i = 0; i <( int)x_train.size(); i++) {
//        for (int j = 0; j <( int)x_train[i].size(); j++) {
//            X_train.at<double>(i, j) = x_train[i][j];
//        }
//        Y_train.at<double>(i, 0) = static_cast<int>(y_train[i]);
//    }
//    for (int i = 0; i <( int)x_test.size(); i++) {
//        for (int j = 0; j <( int)x_test[i].size(); j++) {
//            X_test.at<double>(i, j) = x_test[i][j];
//        }
//        Y_test.at<double>(i, 0) = static_cast<int>(y_test[i]);
//    }
//    cout<<"Finished converting to Mats\n";

//    // Step 3: Perform PCA
//    PCA pca(X_train, Mat(), PCA::DATA_AS_ROW, 150);
//    Mat eigenvalues = pca.eigenvalues;
//    Mat eigenvectors = pca.eigenvectors;
////    cout << "Explained variance ratio: " << sum(eigenvalues.col(0)).at<double>(0, 0) / sum(eigenvalues).at<double>(0, 0) << endl;

//    cout<<"Finished PCs\n";

//    // Step 4: Project Training data to PCA
//    cout << "Projecting the input data on the eigenfaces orthonormal basis" << endl;
//    Mat Xtrain_pca = pca.project(X_train);
//    Xtrain_pca.convertTo(Xtrain_pca, CV_32F);
//    Y_train.convertTo(Y_train, CV_32S);

//    // Step 5: Initialize Classifier and fit training data
//    Ptr<SVM> svm = SVM::create();
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::RBF);
//    svm->setGamma(1e-3);
//    svm->setC(1000);
//    svm->train(Xtrain_pca, ROW_SAMPLE, Y_train);

//    cout<<"Finished SVM\n";

//    // Step 6: Perform testing and get classification report
//    cout << "Predicting people's names on the test set" << endl;
//    clock_t t0 = clock();

//    Mat Xtest_pca = pca.project(X_test);

//    Xtest_pca.convertTo(Xtest_pca, CV_32F);

//    Mat y_pred;
//    svm->predict(Xtest_pca, y_pred);
//    cout << "done in " << (double)(clock() - t0) / CLOCKS_PER_SEC << "s" << endl;
//    std::cout << "Y_test size: " << Y_test.rows << "x" << Y_test.cols << std::endl;
//    std::cout << "y_pred size: " << y_pred.rows << "x" << y_pred.cols << std::endl;

//    double accuracy = calculateAccuracy(y_pred, Y_test);

//    cout << "Accuracy: " << accuracy << endl;


//#include <opencv2/opencv.hpp>

//using namespace cv;
//using namespace cv::ml;

//int main(int argc, char *argv[])
//{
//    QApplication a(argc, argv);

//    // Create a training dataset with 100 samples of 2D points
//    Mat trainData(100, 2, CV_32FC1);
//    Mat labels(100, 1, CV_32SC1);
//    for (int i = 0; i < 50; i++) {
//        trainData.at<float>(i, 0) = float(rand()) / RAND_MAX * 2 - 1;
//        trainData.at<float>(i, 1) = float(rand()) / RAND_MAX * 2 - 1;
//        labels.at<int>(i, 0) = -1;
//    }
//    for (int i = 50; i < 100; i++) {
//        trainData.at<float>(i, 0) = float(rand()) / RAND_MAX * 2 + 1;
//        trainData.at<float>(i, 1) = float(rand()) / RAND_MAX * 2 + 1;
//        labels.at<int>(i, 0) = 1;
//    }

//    // Train an SVM model with RBF kernel
//    Ptr<SVM> svm = SVM::create();
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::RBF);
//    svm->setGamma(0.1);
//    svm->setC(10);
//    svm->train(trainData, ROW_SAMPLE, labels);

//    // Test the model with a new sample
//    Mat sample(1, 2, CV_32FC1);
//    sample.at<float>(0, 0) = 0.5f;
//    sample.at<float>(0, 1) = 0.5f;
//    float prediction = svm->predict(sample);

//    std::cout << "Prediction: " << prediction << std::endl;

//    return 0;
//    return a.exec();

//}
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/ml.hpp>
//#include <iostream>

//using namespace cv;
//using namespace cv::ml;
//using namespace std;

//int main(int argc,char *argv[]) {
//    QApplication a(argc,argv);
//    // Create the input training and test data
//    vector<vector<double>> x_train = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} };
//    vector<vector<double>> x_test = { {2.0, 3.0, 4.0}, {5.0, 6.0, 7.0} };
//    vector<double> y_train = { 0.0, 1.0, 0.0 };
//    vector<double> y_test = { 1.0, 0.0 };

//    // Convert the input data to OpenCV format
//    int num_train_samples = x_train.size();
//    int num_test_samples = x_test.size();
//    int num_features = x_train[0].size();

//    cv::Mat train_data(num_train_samples, num_features, CV_32F);
//    cv::Mat test_data(num_test_samples, num_features, CV_32F);
//    cv::Mat train_labels(num_train_samples, 1, CV_32S);
//    cv::Mat test_labels(num_test_samples, 1, CV_32S);

//    for (int i = 0; i < num_train_samples; i++) {
//        for (int j = 0; j < num_features; j++) {
//            train_data.at<float>(i, j) = static_cast<float>(x_train[i][j]);
//        }
//        train_labels.at<int>(i, 0) = static_cast<int>(y_train[i]);
//    }

//    for (int i = 0; i < num_test_samples; i++) {
//        for (int j = 0; j < num_features; j++) {
//            test_data.at<float>(i, j) = static_cast<float>(x_test[i][j]);
//        }
//        test_labels.at<int>(i, 0) = static_cast<int>(y_test[i]);
//    }

//    // Reduce the dimensionality of the data using PCA
//    PCA pca(train_data, cv::Mat(), PCA::DATA_AS_ROW, 2);
//    cv::Mat reduced_train_data = pca.project(train_data);
//    cv::Mat reduced_test_data = pca.project(test_data);

//    // Train an SVM classifier on the reduced data
//    Ptr<SVM> svm = SVM::create();
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::LINEAR);
//    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//    svm->train(reduced_train_data, ROW_SAMPLE, train_labels);

//    // Predict labels for the test data using the trained SVM classifier
//    cv::Mat predictions;
//    svm->predict(reduced_test_data, predictions);

//    // Print the predicted labels
//    for (int i = 0; i < num_test_samples; i++) {
//        cout << "Test sample " << i << " has predicted label " << predictions.at<float>(i, 0) << endl;
//    }

//    a.exec();
//    return 0;
//}


//    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
//    svm->setType(cv::ml::SVM::C_SVC);
//    svm->setKernel(cv::ml::SVM::RBF);
//    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
//    svm->trainAuto(Xtrain_pca, cv::ml::ROW_SAMPLE, Y_train);


//void get_explained_variance_ratios(const cv::PCA& pca) {
//    // Get eigenvalues (variances) of the principal components
//    cv::Mat eigenvalues = pca.eigenvalues;

//    // Calculate explained variance ratios
//    cv::Mat explained_variance_ratios(eigenvalues.size(), eigenvalues.type());
//    cv::Mat cum_sum(eigenvalues.size(), eigenvalues.type());
//    float sum_eigenvalues = cv::sum(eigenvalues)[0];
//    for (int i = 0; i < eigenvalues.rows; i++) {
//        explained_variance_ratios.at<float>(i, 0) = eigenvalues.at<float>(i, 0) / sum_eigenvalues;
//        if (i > 0) {
//            cum_sum.at<float>(i, 0) = cum_sum.at<float>(i-1, 0) + explained_variance_ratios.at<float>(i, 0);
//        } else {
//            cum_sum.at<float>(i, 0) = explained_variance_ratios.at<float>(i, 0);
//        }
//    }
//    // Print explained variance ratios and cumulative sum
////    cout << "Explained variance ratios:" << endl;
////    for (int i = 0; i < explained_variance_ratios.rows; i++) {
////        cout << "PC " << i+1 << ": " << explained_variance_ratios.at<float>(i, 0) << endl;
////    }

////    cout << "Cumulative sum of explained variance ratios:" << endl;
////    for (int i = 0; i < cum_sum.rows; i++) {
////        cout << "PC " << i+1 << ": " << cum_sum.at<float>(i, 0) << endl;
////    }

//    // Return explained variance ratios and cumulative sum
////    cv::Mat results(2, explained_variance_ratios.rows, explained_variance_ratios.type());
////    explained_variance_ratios.copyTo(results.row(0));
////    cum_sum.copyTo(results.row(1));
////    return results;
//}





//PCA implementation
//#include "recognition.h"
//#include "cmath"

//void computeEigen_2(std::vector<std::vector<double>>& A,
//                  std::vector<double>& eigenvalues,
//                  std::vector<std::vector<double>>& eigenvectors)
//{
//    int n = A.size();

//    // Initialize eigenvectors to the identity matrix
//    eigenvectors.resize(n, std::vector<double>(n, 0));
//    for (int i = 0; i < n; i++) {
//        eigenvectors[i][i] = 1;
//    }

//    // Compute the eigenvalues and eigenvectors using Jacobi algorithm
//    double tolerance = 1e-12;
//    double delta = 1;
//    int max_iterations = n * n;
//    int iteration = 0;

//    while (delta > tolerance && iteration < max_iterations) {
//        delta = 0;
//        for (int i = 0; i < n; i++) {
//            for (int j = i + 1; j < n; j++) {
//                double a_ii = A[i][i];
//                double a_jj = A[j][j];
//                double a_ij = A[i][j];

//                double c, s;
//                if (std::abs(a_ij) > tolerance) {
//                    double tau = (a_jj - a_ii) / (2 * a_ij);
//                    double t = (tau > 0) ? 1.0 / (tau + std::sqrt(1 + tau*tau))
//                                         : -1.0 / (-tau + std::sqrt(1 + tau*tau));
//                    c = 1 / std::sqrt(1 + t*t);
//                    s = t * c;
//                }
//                else {
//                    c = 1;
//                    s = 0;
//                }

//                // Apply Givens rotation to A and eigenvectors
//                for (int k = 0; k < n; k++) {
//                    double a_ik = A[i][k];
//                    double a_jk = A[j][k];
//                    A[i][k] = c * a_ik - s * a_jk;
//                    A[j][k] = s * a_ik + c * a_jk;

//                    double v_ik = eigenvectors[i][k];
//                    double v_jk = eigenvectors[j][k];
//                    eigenvectors[i][k] = c * v_ik - s * v_jk;
//                    eigenvectors[j][k] = s * v_ik + c * v_jk;
//                }

//                delta += std::abs(a_ij);
//            }
//        }
//        iteration++;
//    }

//    // Extract the eigenvalues from the diagonal of A
//    eigenvalues.resize(n, 0);
//    for (int i = 0; i < n; i++) {
//        eigenvalues[i] = A[i][i];
//    }
//}


//void computeEigen_1(const cv::Mat& A, cv::Mat& eigenvalues, cv::Mat& eigenvectors)
//{
//    CV_Assert(A.rows == A.cols && A.type() == CV_64FC1);

//    int n = A.rows;

//    // Initialize eigenvectors to the identity matrix
//    eigenvectors = cv::Mat::eye(n, n, CV_64FC1);

//    // Convert input matrix to a std::vector of std::vector
//    std::vector<std::vector<double>> A_data(n, std::vector<double>(n, 0));
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            A_data[i][j] = A.at<double>(i, j);
//        }
//    }

//    // Compute the eigenvalues and eigenvectors using Jacobi algorithm
//    std::vector<double> eigenvalues_data;
//    std::vector<std::vector<double>> eigenvectors_data;
//    computeEigen_2(A_data, eigenvalues_data, eigenvectors_data);

//    // Convert output eigenvalues to cv::Mat
//    eigenvalues = cv::Mat(eigenvalues_data).clone();

//    // Convert output eigenvectors to cv::Mat
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            eigenvectors.at<double>(i, j) = eigenvectors_data[i][j];
//        }
//    }
//}


//cv::Mat pca(cv::Mat X, int num_components) {
//    // Center the data by subtracting the mean of each feature
////    cv::Mat mean;
////    cv::reduce(X, mean, 0, cv::REDUCE_AVG);
////    cv::Mat centered_data = X - cv::repeat(mean, X.rows, 1);

//    // Compute the covariance matrix of the centered data
//    cv::Mat X_double;
//    X.convertTo(X_double, CV_64FC1);
//    cv::Mat cov;
//    cv::calcCovarMatrix(X_double, cov, cv::Mat(), cv::COVAR_NORMAL | cv::COVAR_ROWS);//I think we may need to transpose X for this function.

//    // Compute the eigenvalues and eigenvectors of the covariance matrix
//    cv::Mat eigenvalues, eigenvectors;
//    computeEigen_1(cov, eigenvalues, eigenvectors);

//    // Sort the eigenvectors in descending order based on their corresponding eigenvalues
//    cv::sortIdx(eigenvalues, eigenvalues, cv::SORT_DESCENDING);
//    eigenvectors = eigenvectors(cv::Range::all(), cv::Range(0, num_components));
//    eigenvalues = eigenvalues(cv::Range(0, num_components), cv::Range::all());

//    // Map each point to its new axes
//    cv::Mat transformed_data = X * eigenvectors.t();

//    // Return the sorted eigenvectors, eigenvalues, and transformed data
//    return eigenvectors;
//}


//cv::Mat transformData(const cv::Mat& data, const cv::Mat& eigenvectors, int num_components) {

//    // Map the centered data onto the principal axes
//    cv::Mat transformed_data = data * eigenvectors(cv::Range::all(), cv::Range(0, num_components));

//    // Return the transformed data
//    return transformed_data;
//}

//// int main(int argc,char *argv[])
////{
////     QApplication a(argc,argv);

////    // Load the cascade classifier XML file for face detection
////    cv::CascadeClassifier faceCascade;
////    faceCascade.load("C:/opencv/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml");

////    // Load the image file
////    cv::Mat image = cv::imread("D:/SBME/3rd year/2nd term/CV/Ass 5/FaceSense/orl faces/archive/zeyad9_44.jpg");
////    if (image.empty())
////    {
////        std::cout << "Failed to open the image file." << std::endl;
////        return -1;
////    }

////    // Convert the image to grayscale for face detection
////    cv::Mat grayImage;
////    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

////    // Perform face detection
////    std::vector<cv::Rect> faces;
////    faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

////    // Display the detected faces as separate images
////    int faceCount = 0;
////    for (const auto& faceRect : faces)
////    {
////        cv::Mat faceImage = grayImage(faceRect); // Extract the region of interest (face) from the image

////        std::string windowName = "Detected Face " + std::to_string(faceCount);
////        cv::imshow(windowName, faceImage);
////        cv::waitKey(0);
////        cv::destroyWindow(windowName);

////        faceCount++;
////    }
////    a.exec();
////    return 0;
////}


//SVM
    // Train an SVM classifier on the reduced data
//    Ptr<SVM> svm = SVM::create();
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::RBF);
//    svm->setGamma(1e-4);
//    svm->setC(100);
//    svm->train(reduced_train_data, ROW_SAMPLE, train_labels);

//    // Predict labels for the test data using the trained SVM classifier
//    Mat predictions;
//    svm->predict(reduced_test_data, predictions);

    // printing accuracy
//    double accuracy = recognition().calculateAccuracy(predictions, test_labels);
//    cout << "Accuracy: " << accuracy << endl;
//SVM
