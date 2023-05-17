#include "MyPCA.h"

MyPCA::MyPCA(const Mat& data, int maxComponents)
{
    MatrixXd eigen_data = cvMatToEigen(data);
    numSamples = eigen_data.rows();
    numFeatures = eigen_data.cols();
    mean = calculateMean(eigen_data);
    normalizedData = normalizeData(eigen_data, mean);

    string resultsFile = "pca_results.txt";

    ifstream file(resultsFile);
    if (file.peek() == ifstream::traits_type::eof())
    {
        cout << "File is empty, storing ..." << endl;
        // File is empty, calculate and store the selectedEigenVectors
        covariance = calculateCovariance(normalizedData);
        eigenSolver = calculateEigenSolver(covariance);
        selectedEigenVectors = selectTopEigenVectors(eigenSolver.eigenvectors(), maxComponents);

        storeSelectedEigenVectors(selectedEigenVectors, resultsFile);
    }
    else
    {
        cout << "File is not empty, loading ..." << endl;
        // File is not empty, load the selectedEigenVectors from the file

        loadSelectedEigenVectors(resultsFile, selectedEigenVectors);
    }
}

Mat MyPCA::reduceData(const Mat& data)
{

    MatrixXd eigen_data = cvMatToEigen(data);

    numSamples = eigen_data.rows();
    numFeatures = eigen_data.cols();
    mean = calculateMean(eigen_data);
    normalizedData = normalizeData(eigen_data, mean);

    cout << "Data Dimension: " << numSamples << " x " << numFeatures << endl;

    Mat reducedMat = eigenToCvMat(normalizedData * selectedEigenVectors);

    cout << "Reduced Data Dimension: " << reducedMat.rows << " x " << reducedMat.cols << endl;

    return reducedMat;
}

VectorXd MyPCA::calculateMean(const MatrixXd &data)
{
    return data.colwise().mean();
}

MatrixXd MyPCA::normalizeData(const MatrixXd &data, const VectorXd &mean)
{
    return data.rowwise() - mean.transpose();
}

MatrixXd MyPCA::calculateCovariance(const MatrixXd &data) const
{
    return (data.transpose() * data) / (numSamples - 1);
}

SelfAdjointEigenSolver<MatrixXd> MyPCA::calculateEigenSolver(const MatrixXd &covar)
{
    return SelfAdjointEigenSolver<MatrixXd>(covar);
}

MatrixXd MyPCA::selectTopEigenVectors(const MatrixXd &eigenVectors, int maxComponents)
{
    return eigenVectors.rightCols(maxComponents);
}

MatrixXd MyPCA::cvMatToEigen(const cv::Mat &cvMat)
{
    cout << "start convert to eigen" << endl;
    cv::Mat cvMatDouble;
    cvMat.convertTo(cvMatDouble, CV_64FC1);

    Eigen::MatrixXd eigenMat(cvMatDouble.rows, cvMatDouble.cols);
    Eigen::Map<Eigen::MatrixXd>(eigenMat.data(), eigenMat.rows(), eigenMat.cols()) = Eigen::Map<const Eigen::MatrixXd>(cvMatDouble.ptr<double>(), cvMatDouble.rows, cvMatDouble.cols);

    cout << "end convert to eigen" << endl;
    return eigenMat;
}

Mat MyPCA::eigenToCvMat(const Eigen::MatrixXd &eigenMat)
{
    cout << "start convert to mat" << endl;

    cv::Mat cvMat(eigenMat.rows(), eigenMat.cols(), CV_64FC1);
    Eigen::Map<Eigen::MatrixXd>(cvMat.ptr<double>(), cvMat.rows, cvMat.cols) = eigenMat;

    // Convert training and test data to CV_32F
    cv::Mat cvMat32F;
    cvMat.convertTo(cvMat32F, CV_32F);

    cout << "end convert to mat" << endl;
    return cvMat32F;
}

void MyPCA::storeSelectedEigenVectors(const MatrixXd &eigenVectors, const string &filename)
{
    ofstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < eigenVectors.rows(); i++)
        {
            for (int j = 0; j < eigenVectors.cols(); j++)
            {
                file << eigenVectors(i, j) << " ";
            }
            file << "\n";
        }
        file.close();
        cout << "Selected EigenVectors stored in " << filename << endl;
    }
    else
    {
        cout << "Unable to open file " << filename << " for storing the selected EigenVectors." << endl;
    }
}

void MyPCA::loadSelectedEigenVectors(const string &filename, MatrixXd &eigenVectors)
{
    ifstream file(filename);
    if (file.is_open())
    {
        vector<vector<double>> data;
        string line;
        while (getline(file, line))
        {
            vector<double> row;
            istringstream iss(line);
            double value;
            while (iss >> value)
            {
                row.push_back(value);
            }
            data.push_back(row);
        }
        file.close();

        int rows = data.size();
        int cols = (rows > 0) ? data[0].size() : 0;
        eigenVectors.resize(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                eigenVectors(i, j) = data[i][j];
            }
        }

        cout << "Selected EigenVectors loaded from " << filename << endl;
    }
    else
    {
        cout << "Unable to open file " << filename << endl;
    }
}
