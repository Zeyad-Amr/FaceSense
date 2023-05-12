QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    garbage.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -LC:/opencv/opencv/build/x64/vc16/lib/ -lopencv_world470
else:win32:CONFIG(debug, debug|release): LIBS += -LC:/opencv/opencv/build/x64/vc16/lib/ -lopencv_world470d
else:unix: LIBS += -LC:/opencv/opencv/build/x64/vc16/lib/ -lopencv_world470

INCLUDEPATH += C:/opencv/opencv/build/include
DEPENDPATH += C:/opencv/opencv/build/include
LIBS += C:\opencv\opencv\release\bin\libopencv_core470.dll
LIBS += C:\opencv\opencv\release\bin\libopencv_highgui470.dll
LIBS += C:\opencv\opencv\release\bin\libopencv_imgcodecs470.dll
LIBS += C:\opencv\opencv\release\bin\libopencv_imgproc470.dll
LIBS += C:\opencv\opencv\release\bin\libopencv_calib3d470.dll
LIBS += C:\opencv\opencv\release\bin\libopencv_ml470.dll

