cmake_minimum_required(VERSION 3.19)
project(colorization)
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/colorization/include)

add_executable(${PROJECT_NAME} colorization/main.cpp colorization/include/colorization.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
