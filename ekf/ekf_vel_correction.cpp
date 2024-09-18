#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>

using namespace std;
using namespace Eigen;


MatrixXd Q(5, 5);     
MatrixXd R(2, 2);     


MatrixXd INPUT_NOISE(2, 2);
MatrixXd GPS_NOISE(2, 2);

const double DT = 0.1;        
const double SIM_TIME = 50.0; 

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 1.0);

void initialize_matrices()
{
    Q = MatrixXd::Zero(5, 5);
    R = MatrixXd::Zero(2, 2);
    INPUT_NOISE = MatrixXd::Zero(2, 2);
    GPS_NOISE = MatrixXd::Zero(2, 2);

    VectorXd q_diag(5);
    q_diag << 0.1, 0.1, M_PI / 180.0, 0.4, 0.1;
    q_diag = q_diag.array().square();
    Q = q_diag.asDiagonal();

    VectorXd r_diag(2);
    r_diag << 0.1, 0.1;
    r_diag = r_diag.array().square();
    R = r_diag.asDiagonal();

    VectorXd input_noise_diag(2);
    input_noise_diag << 0.1, M_PI / 36.0; 
    input_noise_diag = input_noise_diag.array().square();
    INPUT_NOISE = input_noise_diag.asDiagonal();

    VectorXd gps_noise_diag(2);
    gps_noise_diag << 0.05, 0.05;
    gps_noise_diag = gps_noise_diag.array().square();
    GPS_NOISE = gps_noise_diag.asDiagonal();
}

Vector2d calc_input()
{
    double v = 1.0;    
    double yawrate = 0.1; 
    Vector2d u;
    u << v, yawrate;
    return u;
}

VectorXd motion_model(VectorXd x, VectorXd u)
{
    MatrixXd F = MatrixXd::Zero(5, 5);
    F(0, 0) = 1.0;
    F(1, 1) = 1.0;
    F(2, 2) = 1.0;
    F(4, 4) = 1.0;

    MatrixXd B = MatrixXd::Zero(5, 2);

    double yaw = x(2);
    double s = x(4);
    B(0, 0) = DT * cos(yaw) * s;
    B(1, 0) = DT * sin(yaw) * s;
    B(2, 1) = DT;
    B(3, 0) = 1.0;


    x = F * x + B * u;

    return x;
}

Vector2d observation_model(VectorXd x)
{
    MatrixXd H(2, 5);
    H << 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0;
    Vector2d z = H * x;
    return z;
}

VectorXd standard_normal_vector(int size)
{
    VectorXd vec(size);
    for (int i = 0; i < size; ++i)
    {
        vec(i) = distribution(generator);
    }
    return vec;
}

void observation(VectorXd& xTrue, VectorXd& xd, Vector2d u, Vector2d& z, Vector2d& ud)
{
  
    xTrue = motion_model(xTrue, u);

 
    Vector2d noise = GPS_NOISE.llt().matrixL() * standard_normal_vector(2);
    z = observation_model(xTrue) + noise;

    Vector2d input_noise = INPUT_NOISE.llt().matrixL() * standard_normal_vector(2);
    ud = u + input_noise;


    xd = motion_model(xd, ud);
}

MatrixXd jacob_f(VectorXd x, Vector2d u)
{
    double yaw = x(2);
    double v = u(0);
    double s = x(4);

    MatrixXd jF = MatrixXd::Zero(5, 5);

    jF(0, 0) = 1.0;
    jF(0, 2) = -DT * v * s * sin(yaw);
    jF(0, 3) = DT * s * cos(yaw);
    jF(0, 4) = DT * v * cos(yaw);

    jF(1, 1) = 1.0;
    jF(1, 2) = DT * v * s * cos(yaw);
    jF(1, 3) = DT * s * sin(yaw);
    jF(1, 4) = DT * v * sin(yaw);

    jF(2, 2) = 1.0;

    jF(3, 3) = 1.0;

    jF(4, 4) = 1.0;

    return jF;
}

MatrixXd jacob_h()
{
    MatrixXd jH(2, 5);
    jH << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0;
    return jH;
}

void ekf_estimation(VectorXd& xEst, MatrixXd& PEst, Vector2d z, Vector2d u)
{
  
    VectorXd xPred = motion_model(xEst, u);
    MatrixXd jF = jacob_f(xEst, u);
    MatrixXd PPred = jF * PEst * jF.transpose() + Q;


    MatrixXd jH = jacob_h();
    Vector2d zPred = observation_model(xPred);
    Vector2d y = z - zPred;
    MatrixXd S = jH * PPred * jH.transpose() + R;
    MatrixXd K = PPred * jH.transpose() * S.inverse();
    xEst = xPred + K * y;
    MatrixXd I = MatrixXd::Identity(xEst.size(), xEst.size());
    PEst = (I - K * jH) * PPred;
}

int main()
{
    initialize_matrices();

    double time = 0.0;

   
    VectorXd xEst = VectorXd::Zero(5);
    xEst(4) = 1.0;

    VectorXd xTrue = VectorXd::Zero(5);
    double true_scale_factor = 0.9; 
    xTrue(4) = true_scale_factor;

    VectorXd xDR = VectorXd::Zero(5); 

    MatrixXd PEst = MatrixXd::Identity(5, 5);

  
    vector<VectorXd> hxEst;
    vector<VectorXd> hxTrue;
    vector<VectorXd> hxDR;
    vector<Vector2d> hz;

   
    std::random_device rd;
    generator.seed(rd());

    while (SIM_TIME >= time)
    {
        time += DT;
        Vector2d u = calc_input();

        Vector2d z;
        Vector2d ud;
        observation(xTrue, xDR, u, z, ud);

        ekf_estimation(xEst, PEst, z, ud);

        
        hxEst.push_back(xEst);
        hxTrue.push_back(xTrue);
        hxDR.push_back(xDR);
        hz.push_back(z);

        double estimated_scale_factor = xEst(4);

      
        cout << "Time: " << time << endl;
        cout << "True Velocity Scale Factor: " << true_scale_factor << endl;
        cout << "Estimated Velocity Scale Factor: " << estimated_scale_factor << endl;
        cout << "------------------------------" << endl;
    }

  
    ofstream estFile("estimation.txt");
    ofstream trueFile("true.txt");
    ofstream drFile("dr.txt");
    ofstream zFile("observation.txt");

    for (size_t i = 0; i < hxEst.size(); ++i)
    {
        estFile << hxEst << " " << hxEst << endl;
        trueFile << hxTrue << " " << hxTrue << endl;
        drFile << hxDR << " " << hxDR << endl;
        zFile << hz << " " << hz << endl;
    }

    estFile.close();
    trueFile.close();
    drFile.close();
    zFile.close();

    return 0;
}