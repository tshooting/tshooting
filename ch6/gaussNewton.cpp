#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
int main(int argc, char **argv)
{
    // 真实的参数值
    double ar = 1.0, br = 2.0, cr = 1.0;
    // 估计的参数值
    double ae = 2.0, be = -1.0, ce = 5.0;
    // 数据点
    int N = 100;
    // 噪声sigma 的 值
    double w_sigma = 1.0;
    // 问题： 这么做的意义在哪里
    double inv_sigma = 1.0 / w_sigma;
    // OpenCV随机数产生器
    cv::RNG rng;
    // 数据
    vector<double> x_data, y_data;
    // 现在这个带噪声的数据已经弄好了
    for (int i = 0; i < N; i++)
    {
        // [0,1)
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    //开始Gauss-Newton 迭代
    // 设置迭代次数
    int iterations = 100;
    // 本次迭代的cost 和上次迭代的cost
    double cost = 0, lastCost = 0;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    
    for (int iter = 0; iter < iterations; iter++)
    {
        // H = J^T * J 但是参考代码是 Hessian = J^T W^{-1} J in Gauss-Newton
        Matrix3d H = Matrix3d::Zero();
        // bias
        Vector3d b = Vector3d::Zero();
        cost = 0;
        for (int i = 0; i < N; i++)
        {
            // 第i个数据点
            double xi = x_data[i], yi = y_data[i];
            // 真实的值减去 估计的值
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            // 雅克比矩阵 J error 对 ae ,be,ce 的导数
            Vector3d J;
            J[0] = (-1) * xi * xi * exp(ae * xi * xi + be * xi + ce);
            J[1] = (-1) * xi * exp(ae * xi * xi + be * xi + ce);
            J[2] = (-1) * exp(ae * xi * xi + be * xi + ce);
            //  1/(sigma^2) 问题 ：为什么要左右两边都成上 inv_sigma * inv_sigma
            // 是为了方便计算将值变小吗
            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;
            cost += error * error;
        }
        // 求解方程 Hx = b
        Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0]))
        {
            cout << " result id nan !" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost)
        {
            // iter>0 是防止第0次的cost与lastcost相比较
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
        }
        // 更新估计的值
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        lastCost = cost;
        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }
    // 记录用时多少
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;

    return 0;
}