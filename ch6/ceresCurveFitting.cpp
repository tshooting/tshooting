 /**
 * ceres 曲线拟合
 * 参数块：abc 我们要优化的参数
 * 代价函数： f(x)
 * 残差块：y真实-f(x估计)
 * 核函数： 在f^2外面有加了一个函数
 * 我们需要做的事情是:
 * 1:定义参数块、残差块的计算方式（有时候需要定义雅克比的计算方式）
 * 2:将残差块和参数块加入到ceres定义的problem对象中，调用solve函数求解。
 * 求解之前，可以传入一些配置信息，例如迭代次数、终止条件等，也可以使用默认的配置
 * */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>
using namespace std;
// 代价函数的计算模型
struct CURVE_FITTING_COST
{
    // x y 数据 代价函数需要的内容
    const double _x, _y;
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}
    // 残差的计算 问题：感觉没必要有两个const
    template <typename T>
    bool operator()(const T *const abc, T *residual) const
    {
        // y - exp (ax^2 + bx + c)
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        // 这个地方一开始写成了return 0
        return true;
    }
};

int main(int argc, char **argv)
{
    //真实参数值 估计参数值 数据点 噪声sigma的值 opencv随机数产生器
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;
    // 产生有噪声的数据
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++)
    {
        // x 在[0,1) 这个地方一开始写成了 i/100 应该是i/100.0
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }
    // 参数块
    double abc[3] = {ae, be, ce};
    // 构建最小二乘问题
    ceres::Problem problem;
    // 对于每一项数据 我们把误差项 加入问题
    for (int i = 0; i < N; i++)
    {

        problem.AddResidualBlock(
            // 向问题中添加误差项
            // 使用自动求导，参数模板：误差类型，输出维度，输入维度，维数要与前面struct中一致
            // 1 是误差项的维度 3是参数块的维度
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])
            ),
            nullptr,
            abc
        );
    }
    // 配置求解器
    ceres::Solver::Options options;
    // 有很多配置项可以填 增量方程求解
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    // 优化信息
    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // 开始优化
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    for (auto a : abc)
        cout << a << " ";
    cout << endl;
    return 0;
}
