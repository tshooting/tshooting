find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# 为使用sophus
find_package(Sophus REQUIRED)
# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )
 
 

add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError ${Pangolin_LIBRARIES})
target_link_libraries(trajectoryError Sophus::Sophus)