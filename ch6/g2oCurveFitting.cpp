#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;
 
// 曲线模型的顶点，模板参数：优化变量的维度 和数据类型 有3维度 数据类型是3*1 的向量
// 在这里面重写了 重置和更新函数
// 注意 class 是有分号结尾的
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 重置
    virtual void setToOriginImpl() override
    {
        _estimate << 0, 0, 0;
    }
    // 更新
    virtual void oplusImpl(const double *update) override
    {
        _estimate += Eigen::Vector3d(update);
    }
    // 存盘和读盘 ：留空
    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};
// 误差模型 模板参数： 观测值维度(我们能拿到的值)，类型，链接的顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    // 这个是x的值，y的值是 _measurement
    double _x;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数
    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
    // 计算曲线模型的误差 bug 写成了computerError 是computeError
    virtual void computeError() override
    {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        // 拿到 顶点中的待优化变量 因为我们的优化变量的类型是vector3d的
        const Eigen::Vector3d abc = v->estimate();
        // _measurement 就是观测值 y
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }
    // 计算雅克比矩阵
    virtual void linearizeOplus() override
    {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }
    // 读盘和存盘
    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};
int main(int argc, char **argv)
{
    // 设置真实参数值 估计参数值 数据点个数 噪声sigma的值 opencv随机数产生器
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;
    // 现在就制造带有噪声的数据
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        // bug 这里写的是 减号 原版是加号所以结果不一样
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }
    // 构建图优化，先设定g2o
    // 每个误差项优化变量维度为3，误差值维度为1
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolveType;
    // 线性求解器模型
    typedef g2o::LinearSolverDense<BlockSolveType::PoseMatrixType> LinearSolverType;
    // 梯度下降的方式 ，可以从GN,LM,DogLeg 中选择
    auto solve = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolveType>(g2o::make_unique<LinearSolverType>()));
    //图模型 设置求解器 打开调试输出
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solve);
    optimizer.setVerbose(true);
    // 往图中添加顶点
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);
    //往 图中添加边
    for (int i = 0; i < N; i++)
    {

        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        // 设置链接的顶点 void setVertex(size_t i, Vertex* v)
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        // 问题： 这个信息矩阵在公式中没有被用到 为什么需要信息矩阵
        // 信息矩阵：协方差矩阵之逆
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
        
    }
    // 执行优化
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    
    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout<<"estimated model :" << abc_estimate.transpose() <<endl;
    return 0;
}
// #include <iostream>
// #include <g2o/core/g2o_core_api.h>
// #include <g2o/core/base_vertex.h>
// #include <g2o/core/base_unary_edge.h>
// #include <g2o/core/block_solver.h>
// #include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/core/optimization_algorithm_gauss_newton.h>
// #include <g2o/core/optimization_algorithm_dogleg.h>
// #include <g2o/solvers/dense/linear_solver_dense.h>
// #include <Eigen/Core>
// #include <opencv2/core/core.hpp>
// #include <cmath>
// #include <chrono>

// using namespace std;

// // 曲线模型的顶点，模板参数：优化变量维度和数据类型
// class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//     // 重置
//     virtual void setToOriginImpl() override
//     {
//         _estimate << 0, 0, 0;
//     }

//     // 更新
//     virtual void oplusImpl(const double *update) override
//     {
//         _estimate += Eigen::Vector3d(update);
//     }

//     // 存盘和读盘：留空
//     virtual bool read(istream &in) {}

//     virtual bool write(ostream &out) const {}
// };

// // 误差模型 模板参数：观测值维度，类型，连接顶点类型
// class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//     CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

//     // 计算曲线模型误差
//     virtual void computeError() override
//     {
//         const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
//         const Eigen::Vector3d abc = v->estimate();
//         _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
//     }

//     // 计算雅可比矩阵
//     virtual void linearizeOplus() override
//     {
//         const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
//         const Eigen::Vector3d abc = v->estimate();
//         double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
//         _jacobianOplusXi[0] = -_x * _x * y;
//         _jacobianOplusXi[1] = -_x * y;
//         _jacobianOplusXi[2] = -y;
//     }

//     virtual bool read(istream &in) {}

//     virtual bool write(ostream &out) const {}

// public:
//     double _x; // x 值， y 值为 _measurement
// };

// int main(int argc, char **argv)
// {
//     double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
//     double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
//     int N = 100;                          // 数据点
//     double w_sigma = 1.0;                 // 噪声Sigma值
//     double inv_sigma = 1.0 / w_sigma;
//     cv::RNG rng; // OpenCV随机数产生器

//     vector<double> x_data, y_data; // 数据
//     for (int i = 0; i < N; i++)
//     {
//         double x = i / 100.0;
//         x_data.push_back(x);
//         y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
//     }

//     // 构建图优化，先设定g2o
//     typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;           // 每个误差项优化变量维度为3，误差值维度为1
//     typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

//     // 梯度下降方法，可以从GN, LM, DogLeg 中选
//     auto solver = new g2o::OptimizationAlgorithmGaussNewton(
//         g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
//     g2o::SparseOptimizer optimizer; // 图模型
//     optimizer.setAlgorithm(solver); // 设置求解器
//     optimizer.setVerbose(true);     // 打开调试输出

//     // 往图中增加顶点
//     CurveFittingVertex *v = new CurveFittingVertex();
//     v->setEstimate(Eigen::Vector3d(ae, be, ce));
//     v->setId(0);
//     optimizer.addVertex(v);

//     // 往图中增加边
//     for (int i = 0; i < N; i++)
//     {
//         CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
//         edge->setId(i);
//         edge->setVertex(0, v);                                                                   // 设置连接的顶点
//         edge->setMeasurement(y_data[i]);                                                         // 观测数值
//         edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // 信息矩阵：协方差矩阵之逆
//         optimizer.addEdge(edge);
//     }

//     // 执行优化
//     cout << "start optimization" << endl;
//     chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//     optimizer.initializeOptimization();
//     optimizer.optimize(10);
//     chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//     chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//     cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

//     // 输出优化值
//     Eigen::Vector3d abc_estimate = v->estimate();
//     cout << "estimated model: " << abc_estimate.transpose() << endl;

//     return 0;
// }

// #include <iostream>
// #include <g2o/core/g2o_core_api.h>
// #include <g2o/core/base_vertex.h>
// #include <g2o/core/block_solver.h>
// #include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/core/optimization_algorithm_gauss_newton.h>
// #include <g2o/core/optimization_algorithm_dogleg.h>
// #include <g2o/solvers/dense/linear_solver_dense.h>
// #include <Eigen/Core>
// #include <opencv2/core/core.hpp>
// #include <cmath>
// #include <chrono>

// using namespace std;
// using namespace Eigen;
// // 曲线模型的顶点，模板参数：优化变量维度和数据类型
// // 顶点初始化和顶点的更新
// class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//     // 重置
//     virtual void setToOriginImpl() override
//     {
//         _estimate << 0, 0, 0;
//     }
//     // 更新
//     virtual void oplusImpl(const double *update) override
//     {
//         // 这个updata 应该就是一个数组
//         _estimate += Eigen::Vector3d(update);
//     }
//     // 存盘和读盘：留空
//     virtual bool read(istream &in)
//     {
//     }
//     virtual bool write(ostream &out)
//     {
//     }
// }
// // 误差模板 模板参数：观测值的维度，类型，链接顶点的类型
// class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
// {
// public:
//     // x 值， y 值为 _measurement
//     double _x;

// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//     // 构造函数
//     CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x)
//     {
//     }
//     // 计算曲线模型误差 去查一下虚函数的用法
//     virtual void computeError() override
//     {
//         // 得去查一下 static_cast的用法

//         const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
//         // 应该是得到_estimate
//         const Eigen::Vector3d abc = v->estimate();
//         _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
//     }
//     // 计算雅克比矩阵
//     virtual void linearizeOplus() override
//     {
//         const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
//         const Eigen::Vector3d abc = v->estimate();
//         double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
//         _jacobianOplusXi[0] = -_x * _x * y;
//         _jacobianOplusXi[1] = -_x * y;
//         _jacobianOplusXi[2] = -y;
//     }
//     virtual bool read(istream &in) {}
//     virtual bool write(ostream &out) const {}
// }

// int main(int argc, char **argv)
// {
//     double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
//     double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
//     int N = 100;                          // 数据点
//     double w_sigma = 1.0;                 // 噪声Sigma值
//     double inv_sigma = 1.0 / w_sigma;
//     cv::RNG rng; // OpenCV随机数产生器

//     vector<double> x_data, y_data; // 数据
//     for (int i = 0; i < N; i++)
//     {
//         double x = i / 100.0;
//         x_data.push_back(x);
//         y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
//     }

//     // 构建图优化 ，先设定g2o
//     // 每个误差项优化变量维度为3 ，误差值维度为1
//     typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
//     // 线性求解器类型
//     typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
//     // 梯度下降的方法可以从 GN LM DogLeg 中选择
//     auto solver = new g2o::OptimizationAlgorithmGaussNewton(
//         g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
//     // 图模型 设置求解器 打开调试输出
//     g2o::SparseOptimizer optimizer;
//     optimizer.setAlgorithm(solver);
//     // 往图中增加顶点
//     CurveFittingVertex *v = new CurveFittingVertex();
//     v->setEstimate(Eigen::Vector3d(ae, be, ce));
//     v->setId(0);
//     optimizer.addVertex(v);
//     // 加入边
//     for (int i = 0; i < N; i++)
//     {
//         CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
//         edge->setId(i);
//         // 设置链接的顶点 问题：这个0 应该是没有设置顶点吧
//         edge->setVertex(0, v);
//         // 观测数值
//         edge->setMeasurement(y_data[i]);
//         // 信息矩阵 ：协方差矩阵之逆
//         edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
//         optimizer.addEdge(edge);
//     }
//     // 执行优化
//     cout << "start optimization" << endl;
//     chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//     // 初始化
//     optimizer.initializeOptimization();
//     optimizer.optimize(10);
//     chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//     chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//     cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

//     // 输出优化值
//     Eigen::Vector3d abc_estimate = v->estimate();
//     cout << "estimated model: " << abc_estimate.transpose() << endl;

//     return 0;
// }