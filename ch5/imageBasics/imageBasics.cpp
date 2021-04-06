#include <iostream>
#include <chrono>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
int main(int argc, char **argv)
{
    // 读入argv[1]指定的图像
    cv::Mat image;
    // 通过cv::imread 函数读取指定路径下的图像
    image = cv::imread("../ubuntu.png");
    // 判断图像是否被正确读入
    if (image.data == nullptr)
    {
        cerr << "文件" << argv[1] << "不存在" << endl;
        return 0;
    }
    cout << "图像宽为" << image.cols << ", 高为" << image.rows << endl;
    cv::imshow("image", image);
    // 暂停程序 等待一个按键的输入
    cv::waitKey(0);
    // 判断image的类型
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3)
    {
        // 图像类型不符合要求
        cout << "请输入一张彩色图或灰度图." << endl;
        return 0;
    }
    // 遍历图像
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (size_t y = 0; y < 10; y++)
    {
        // cv::Mat::ptr 获取图像的行指针 row_ptr 是第y行的头指针
        // 每一个通道都是8位
        unsigned char *row_ptr = image.ptr<unsigned char>(y);
        for (size_t x = 0; x < 10; x++)
        {
            // 访问位于x，y处的像素 // data_ptr 指向待访问的像素数据
            unsigned char *data_ptr = &row_ptr[x * image.channels()];
            for (int c = 0; c < image.channels(); c++)
            {
                // data为I(x,y)第c个通道的值
                unsigned char data = data_ptr[c];
                cout<< (int)data <<", ";
            }
            cout<<";";
        }
        cout<<endl;
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "遍历图像用时：" << time_used.count() << " 秒。" << endl;

    // 关于cv::Mat 的拷贝
    // 直接赋值并不会拷贝数据
    cv::Mat image_another = image;
    // 修改image_another 会导致image 发生变化
    // 将左上角100*100的块置零
     
    image_another (cv::Rect(0,0,100,100)).setTo(0);
    // 这两句在一起 要不然这个image就会闪现
    cv::imshow("image",image);
    cv::waitKey(0);
     

    // 使用clone函数来拷贝数据
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("image",image);
    cv::waitKey(0);
    cv::imshow("image_clone",image_clone);
    cv::waitKey(0);
    // 对于图像还有很多基本的操作,如剪切,旋转,缩放等,限于篇幅就不一一介绍了,请参看OpenCV官方文档查询每个函数的调用方法.
    cv::destroyAllWindows();
    return 0;
}