#声明要求的cmake最低的版本
cmake_minimum_required(VERSION 2.8)
#声明一个cmake工程
project(HelloSLAM)
#添加一个可执行程序
#语法：add_executable(程序名 源代码文件)
add_executable(helloSLAM helloSLAM.cpp)
add_executable(useHello useHello.cpp)

# 静态库以.a 作为后缀名，共享库以.so
# 静态库每次被调用都会生成一个副本，而共享库则只有一个副本，
add_library(hello_shared SHARED libHelloSLAM.cpp)
target_link_libraries(useHello hello_shared)