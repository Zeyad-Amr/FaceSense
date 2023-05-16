#include "MyPCA.h"

void test_PCA()
{

    // Example usage
    int numSamples = 30;
    int numFeatures = 900;
    int maxComponents = 150;

    //    MatrixXd data(numSamples, numFeatures);  // Input data matrix of size 25x10000
    Mat image1(numSamples, numFeatures, CV_8UC3);
    Mat image2(numSamples / 2, numFeatures, CV_8UC3);
    Mat image3(numSamples / 3, numFeatures, CV_8UC3);

    cout << "Start" << endl;

    // Create PCA object and compute PCA
    MyPCA pca = MyPCA(image1, maxComponents);
    std::cout << "End" << std::endl;

    Mat reducedData = pca.reduceData(image1);

    // Print the dimensions of the reduced data
    std::cout << "Reduced data dimensions: " << reducedData.rows << " x " << reducedData.cols << std::endl;

    reducedData = pca.reduceData(image2);

    // Print the dimensions of the reduced data
    std::cout << "Reduced data dimensions: " << reducedData.rows << " x " << reducedData.cols << std::endl;

    reducedData = pca.reduceData(image3);

    // Print the dimensions of the reduced data
    std::cout << "Reduced data dimensions: " << reducedData.rows << " x " << reducedData.cols << std::endl;
}
