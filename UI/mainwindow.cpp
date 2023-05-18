#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setFixedSize(750, 500);

    //initializing labels
    inputImage = cv::imread("D:/SBME/ComputerVision/Task_4/otsu/boat.jpg");
    outputImage = cv::imread("D:/SBME/ComputerVision/Task_4/otsu/boat.jpg");

    cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
    cv::cvtColor(outputImage, outputImage, cv::COLOR_BGR2RGB);


    QImage qimageInput((uchar*)inputImage.data, inputImage.cols, inputImage.rows, inputImage.step, QImage::Format_RGB888);
    QImage qimageOutput((uchar*)outputImage.data, outputImage.cols, outputImage.rows, outputImage.step, QImage::Format_RGB888);

    unsigned int wi = ui->inputImg->width(),hi = ui->inputImg->height();
    unsigned int wo = ui->outputImg->width(),ho = ui->outputImg->height();

    ui->inputImg->setPixmap(QPixmap::fromImage(qimageInput).scaled(wi,hi));
    ui->outputImg->setPixmap(QPixmap::fromImage(qimageOutput).scaled(wo,ho));


}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_input_btn_clicked()
{
    //showing file selection dialogue and getting fileName
    QString fileName = getFile();
    if (fileName == " ") return;

    //reading image as cv::Mat and making sure the selected file is an image
    cv::Mat imageMat = getMat(fileName);
    if(imageMat.empty()) return;
    inputImage = imageMat;

    //converting cv::Mat to qimage
    QImage qimage = convertMatToQimage(imageMat);

    setLabelImg(ui->inputImg,qimage);
    setLabelImg(ui->outputImg,qimage);
}



// ****************************************** Helper functions **********************************

QString MainWindow::getFile()
{
     QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image Files (*.png *.jpg *.bmp)"));

     if (fileName.isEmpty())
         return " ";
     return fileName;
}
cv::Mat MainWindow::getMat(QString fileName)
{

     cv::Mat image;
     image = cv::imread(fileName.toStdString());

     if (image.empty())
         return image;

     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
     return image;

}
QImage MainWindow::convertMatToQimage(cv::Mat imgMat,int flag )
{
     QImage::Format format = QImage::Format_RGB888;
     if(flag){
        format = QImage::Format_Grayscale8;
     }
     QImage qimage((uchar*)imgMat.data, imgMat.cols, imgMat.rows, imgMat.step, format);
     return qimage;
}

void MainWindow::setLabelImg(QLabel *label, QImage qimage,unsigned int w,unsigned int h)
{
     if(!w) w = label->width();
     if(!h) h = label->height();
     label->setPixmap(QPixmap::fromImage(qimage).scaled(w,h,Qt::KeepAspectRatio));
}


void MainWindow::on_detect_clicked()
{
//     processing image
    cv::Mat processedImage;
    detect_faces().apply(inputImage,processedImage);

    //showing processed image
    QImage qimageProcessed = convertMatToQimage(processedImage);
    setLabelImg(ui->outputImg,qimageProcessed);
}


void MainWindow::on_recoginition_clicked()
{
    cv::Mat processedImage;
    QString result = detect_faces().recognize(inputImage);

    ui->outputName->setText(result);
}

