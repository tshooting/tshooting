#include<iostream>
#include<ctime>
using namespace std;
//eigen 部分
#include</usr/include/eigen3/Eigen/Core>
//稠密矩阵的代数运算（逆 特征值）
#include</usr/include/eigen3/Eigen/Dense>
#define MATRIX_SIZE 50
int main(int argc,char ** argv)
{
    //模板类 数据类型 行 列
    Eigen::Matrix<float,2,3> matrix_23;
    // 同时，Eigen 通过 typedef 提供了许多内置类型，不过底层仍是 Eigen::Matrix
    // 例如 Vector3d 实质上是 Eigen::Matrix<double, 3, 1>
    Eigen::Vector3d v_3d;
    //double 3 3 初始化为0
    Eigen :: Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    //如果说不确定矩阵大小，可以使用动态大小的矩阵
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_dynamic;
    // 更简单的
    Eigen::MatrixXd matrix_x;
    
    //输入数据
    matrix_23 << 1,2,3,4,5,6;
    //输出数据
    cout<<matrix_23<<endl;
    //访问矩阵中的元素
    for(int i=0; i<2;i++){
        for(int j=0;j<3;j++){
            cout<<matrix_23(i,j)<<",";
        }
        cout<<endl;
    }
    v_3d << 3,2,1;
    //矩阵和向量相乘  其实还是矩阵和矩阵相乘 ，但是数据类型不能出错，都得是double或者float
    //if  matrix_23 * v_3d 就会报错
    Eigen::Matrix <double,2,1> result = matrix_23.cast<double>() * v_3d;
    cout<<result<<endl;

    matrix_33 = Eigen::Matrix3d ::Random();
    cout<<matrix_33<<endl;
    matrix_33<<1,2,3,1,2,3,4,5,6;

    //转置 各元素的和 迹 数乘 逆 行列式
    cout<<matrix_33.transpose()<<endl;
    cout<<matrix_33.sum()<<endl;
    cout<<matrix_33.trace()<<endl;
    cout<<10*matrix_33<<endl;
    cout<<matrix_33.inverse()<<endl;
    cout<<matrix_33.determinant()<<endl;

    //特征值
    // 实对称矩阵可以保证对角化成功
    Eigen::SelfAdjointEigenSolver<Eigen ::Matrix3d> eigen_solve (matrix_33.transpose()*matrix_33);
    cout<<"Eigen values = "<<eigen_solve.eigenvalues()<<endl;
    cout<<"Eigen vectors = "<<eigen_solve.eigenvectors()<<endl;
    //解方程 求解 Ax = b 
    //设置A
    Eigen::Matrix<double,MATRIX_SIZE,MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    //设置b
    Eigen::Matrix<double,MATRIX_SIZE,1> v_Nd;

    v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE,1);

    //计时
    clock_t time_stt = clock();
    //直接求逆
    Eigen::Matrix <double,MATRIX_SIZE,1> x = matrix_NN.inverse()*v_Nd;
    cout <<"time use in normal invers is " << 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms"<< endl;
    //通常用矩阵分解来求 例如QR分解
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of Qr decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;
    // 对于正定矩阵，还可以用cholesky分解来解方程
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
        << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;
    return 0;
}