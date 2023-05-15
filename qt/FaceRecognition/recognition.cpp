#include "recognition.h"


double calculateAccuracy(const Mat& y_pred, const Mat& Y_test) {
    CV_Assert(y_pred.size() == Y_test.size());

    int correctPredictions = 0;
    for (int i = 0; i < y_pred.rows; i++) {
        cout<<y_pred.at<float>(i)<<' '<<Y_test.at<int>(i)<<'\n';
        if (y_pred.at<float>(i) == Y_test.at<int>(i)) {
            correctPredictions++;
        }
    }
//    cout<<"Correct Predictions: "<<correctPredictions<<'\n';
    return static_cast<double>(correctPredictions) / y_pred.rows;
}
double getClassFromName(string name){
    double tenth = name[name.size()-6] - '0';
    double ones = name[name.size()-5] - '0';
    if(!(tenth > 0 && tenth < 10 ) ) return ones;
    return ones+tenth*10;
}

void train_test_split(vector<vector<double>> x, vector<double> y, float train_ratio, int seed,vector<vector<double>> &xTrain,vector<vector<double>> &xTest,vector<double> &yTrain,vector<double> &yTest) {
    // Set the seed for the random number generator
    mt19937 g(seed);

    // Shuffle the vectors randomly
    vector<pair<vector<double>, double>> data(x.size());
    for (int i = 0; i <(int) x.size(); i++) {
        data[i] = make_pair(x[i], y[i]);
    }
    shuffle(data.begin(), data.end(), g);
    vector<vector<double>> shuffled_x(x.size());
    vector<double> shuffled_y(x.size());
    for (int i = 0; i <( int)x.size(); i++) {
        shuffled_x[i] = data[i].first;
        shuffled_y[i] = data[i].second;
    }

    // Compute sizes of training and testing sets
    int n_samples = x.size();
    int n_train = round(n_samples * train_ratio);
    int n_test = n_samples - n_train;

    // Split data into training and testing sets
    vector<vector<double>> x_train(shuffled_x.begin(), shuffled_x.begin() + n_train);
    vector<vector<double>> x_test(shuffled_x.begin() + n_train, shuffled_x.end());
    vector<double> y_train(shuffled_y.begin(), shuffled_y.begin() + n_train);
    vector<double> y_test(shuffled_y.begin() + n_train, shuffled_y.end());

    xTrain = x_train;
    xTest  = x_test ;
    yTrain = y_train;
    yTest  = y_test ;
}
vector<double> flatten(cv::Mat image){
    vector<double> imageVector;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int pixelIntensity = image.at<uchar>(i, j);
            imageVector.push_back(pixelIntensity);
        }
    }
    return imageVector;
}


void preprocess_data(vector<vector<double>> faces_train, vector<vector<double>> faces_test,vector<vector<double>> &preprocessedXTrain, vector<vector<double>> &preprocessedXTest) {
    int n_samples_train = faces_train.size();
    int n_features = faces_train[0].size();

    // Compute mean and standard deviation of training set
    vector<double> mu(n_features, 0.0);
    vector<double> std_dev(n_features, 0.0);
    for (int i = 0; i < n_samples_train; i++) {
        for (int j = 0; j < n_features; j++) {
            mu[j] += faces_train[i][j];
        }
    }
    for (int j = 0; j < n_features; j++) {
        mu[j] /= n_samples_train;
    }
    for (int i = 0; i < n_samples_train; i++) {
        for (int j = 0; j < n_features; j++) {
            std_dev[j] += pow(faces_train[i][j] - mu[j], 2.0);
        }
    }
    for (int j = 0; j < n_features; j++) {
        std_dev[j] = sqrt(std_dev[j] / (n_samples_train-1));
    }

    // Normalize training set
    vector<vector<double>> X_train(n_samples_train, vector<double>(n_features, 0.0));
    vector<double> std_dev_mod(n_features, 0.0);
    for (int j = 0; j < n_features; j++) {
        std_dev_mod[j] = (std_dev[j] == 0) ? 1 : std_dev[j];
    }
    for (int i = 0; i < n_samples_train; i++) {
        for (int j = 0; j < n_features; j++) {
            X_train[i][j] = (faces_train[i][j] - mu[j]) / std_dev_mod[j];
//            X_train[i][j] = (faces_train[i][j] - mu[j]) / 1;
        }
    }

    // Normalize test set with same mean and standard deviation values as training set
    int n_samples_test = faces_test.size();
    vector<vector<double>> X_test(n_samples_test, vector<double>(n_features, 0.0));
    for (int i = 0; i < n_samples_test; i++) {
        for (int j = 0; j < n_features; j++) {
            X_test[i][j] = (faces_test[i][j] - mu[j]) / std_dev_mod[j];
//            X_test[i][j] = (faces_test[i][j] - mu[j]) / 1;
        }
    }
    preprocessedXTrain = X_train;
    preprocessedXTest = X_test;
}
