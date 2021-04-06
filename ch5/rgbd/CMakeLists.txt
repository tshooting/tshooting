cmake_minimum_required(VERSION 2.8)

project(rgbd)

set(CMAKE_BUILD_TYPE "Release")
# 添加c++ 11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

include_directories("/usr/include/eigen3")
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
 
 

 
 
add_executable(joinmap joinmap.cpp)
# 链接OpenCV库
target_link_libraries(joinmap ${OpenCV_LIBS} ${Pangolin_LIBRARIES} )
target_link_libraries(joinmap Sophus::Sophus)

 