/**
 * g++ helloWorld.cpp 
 * ./a.out
 * 
 * 
 * CMake
 * mkdir build 
 * cmake ..  //通过 cmake .. 命令，对上一层文件夹，也就是代码所在的文件夹进行编译
 * make      //当我们发布源代码时，只要把 build 文件夹删掉即可。
 * ./helloSLAM
 * */
#include<iostream>
using namespace std;
int main(int argc,char ** argv)
{
    cout<<"Hello World!"<<endl;
    return 0;
}