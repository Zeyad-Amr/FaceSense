#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <QLabel>
#include <QFileDialog>
#include <QButtonGroup>
#include <opencv2/core.hpp>
#include <vector>

#include "detectFaces.h"
//#include "recognizeFace.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    cv::Mat inputImage;
    cv::Mat outputImage;

private slots:
    void on_input_btn_clicked();

    QString getFile();
    cv::Mat getMat(QString fileName);
    QImage convertMatToQimage(cv::Mat imgMat,int flag=0);
    void setLabelImg(QLabel *label, QImage qimage, unsigned int w = 0, unsigned int h = 0);

    void on_detect_clicked();
    void on_recoginition_clicked();
};
#endif // MAINWINDOW_H
