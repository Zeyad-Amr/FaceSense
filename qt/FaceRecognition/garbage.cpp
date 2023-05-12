
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

