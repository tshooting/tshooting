cmake_minimum_required(VERSION 2.8)
project(chapter4)

# 为使用sophus
find_package(Sophus REQUIRED)
# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus)