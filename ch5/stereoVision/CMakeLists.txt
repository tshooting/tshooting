cmake_minimum_required(VERSION 2.8)

project(stereoVision)

set(CMAKE_BUILD_TYPE "Release")
# 添加c++ 11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

include_directories("/usr/include/eigen3")
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
 

# build/imageBasics ubuntu.png
# 顺序不能改变 得先添加库再添加这个可执行文件
# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

 
 
add_executable(stereoVision stereoVision.cpp)
# 链接OpenCV库
target_link_libraries(stereoVision ${OpenCV_LIBS})
target_link_libraries(stereoVision ${Pangolin_LIBRARIES})

 