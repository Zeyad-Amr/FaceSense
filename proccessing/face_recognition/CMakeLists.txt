cmake_minimum_required(VERSION 3.25)
project(face_recognition)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/opencv/minwg-build/install/x64/mingw/lib")
set(EIGEN_DIR "C:/eigen")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${EIGEN_DIR})

add_executable(face_recognition main.cpp recognition.cpp recognition.h MyPCA.cpp MyPCA.h)

target_link_libraries(face_recognition ${OpenCV_LIBS} )

