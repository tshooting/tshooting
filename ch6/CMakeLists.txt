# 针对于 gaussNewton.cpp 使用了 opencv 和eigen
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11")
project(ch6)

# set(G2O_ROOT /usr/local/include/g2o) 

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


include_directories("/usr/include/eigen3")
# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
#g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# 可执行文件 链接库
add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})