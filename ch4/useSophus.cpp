#include <iostream>
#include <cmath>
#include <Eigen/Core>
// 几何  主要是旋转矩阵 旋转向量 欧拉角 四元数在用
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

// 主要演示sophus的基本用法
int main(int argv, char **argc)
{
    // 沿Z轴旋转90度的旋转矩阵
    Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
    cout << " R = \n"
         << R << endl;
    // R << 0,0,1,0,1,0,1,0,0;

    // cout << " R2 = \n"
    //      << R << endl;
    //或者四元数 旋转矩阵 变成四元数
    Quaterniond q(R);
    // Sophus::SO3d可以直接从旋转矩阵构造 其实这个就是旋转矩阵 特殊正交群
    Sophus::SO3d SO3_R(R);
    // 通过四元数进行构造
    Sophus::SO3d SO3_q(q);
    // 这两者是等价的 问题：为什么和R有偏差 得到的不是完全一样的矩阵
    cout << "SO(3) from matrix:\n"
         << SO3_R.matrix() << endl;
    cout << "SO(3) from quaternion:\n"
         << SO3_q.matrix() << endl;
    cout << "they are equal" << endl;
    // 使用对数映射获得它的李代数
    // 一个李群（旋转矩阵）对应的李代数就是旋转矩阵所转成的旋转向量
    Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;
    // hat 为向量到反对称矩阵
    cout << "so3 hat = \n"
         << Sophus::SO3d::hat(so3) << endl;
    // 相对的 vee是反对称到向量
    cout << "so3 hat vee = " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

    // 使用李代数解决求导问题的思路有两种
    // 1 ： 使用李代数表示姿态，然后根据李代数的加法来对李代数求导
    // 2 ： 对李群左乘或者右乘微小扰动，然后对扰动求导

    // 扰动模型
    // 增量扰动模型的更新 假设增量为这么多
    Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    cout << "SO3 updated = \n"
         << SO3_updated.matrix() << endl;
    cout << "*******************************" << endl;
    // 对SE(3)操作
    // 平移操作，沿着X轴平移1
    Vector3d t(1, 0, 0);
    // 使用R t来进行构造
    Sophus::SE3d SE3_Rt(R, t);
    // 使用 q，t来进行构造
    Sophus::SE3d SE3_qt(q, t);
    cout << "SE3 from R,t= \n"
         << SE3_Rt.matrix() << endl;
    cout << "SE3 from q,t= \n"
         << SE3_qt.matrix() << endl;
    // 李代数se(3)是一个六维向量
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    // 通过这个变化矩阵T 然后做对数映射 得到对应的李代数
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl;
    // 平移在前 旋转在后
    // 会发现 se3 的平移部分 并不是变换矩阵中的平移部分 是有区别的
    cout << "se3 hat = \n"
         << Sophus::SE3d::hat(se3) << endl;
    cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

    // 演示更新
    Vector6d update_se3;
    update_se3.setZero();
     
    // 6 * 1 的矩阵
    update_se3(0, 0) = 1e-4;
    cout << "update_se3 = \n"
         << update_se3.transpose() << endl;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3 updated = " << endl
         << SE3_updated.matrix() << endl;

    return 0;
}