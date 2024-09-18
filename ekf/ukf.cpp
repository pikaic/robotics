#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

using namespace std;
using namespace Eigen;

MatrixXd Q(4, 4);  
MatrixXd R(2, 2);  


MatrixXd INPUT_NOISE(2, 2);
MatrixXd GPS_NOISE(2, 2);

const double DT = 0.1;        
const double SIM_TIME = 50.0; 


const double ALPHA = 0.001;
const double BETA = 2;
const double KAPPA = 0;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 1.0);


void initialize_matrices();
Vector2d calc_input();
VectorXd motion_model(const VectorXd& x, const VectorXd& u);
VectorXd observation_model(const VectorXd& x);
VectorXd standard_normal_vector(int size);
void observation(VectorXd& xTrue, VectorXd& xd, const VectorXd& u, VectorXd& z, VectorXd& ud);
void generate_sigma_points(const VectorXd& xEst, const MatrixXd& PEst, double gamma, MatrixXd& sigma);
void predict_sigma_motion(MatrixXd& sigma, const VectorXd& u);
void predict_sigma_observation(MatrixXd& sigma);
MatrixXd calc_sigma_covariance(const VectorXd& x, const MatrixXd& sigma, const RowVectorXd& wc, const MatrixXd& Pi);
MatrixXd calc_pxz(const MatrixXd& sigma, const VectorXd& x, const MatrixXd& z_sigma, const VectorXd& zb, const RowVectorXd& wc);
void ukf_estimation(VectorXd& xEst, MatrixXd& PEst, const VectorXd& z, const VectorXd& u, const RowVectorXd& wm, const RowVectorXd& wc, double gamma);
void setup_ukf(int nx, RowVectorXd& wm, RowVectorXd& wc, double& gamma);

int main()
{
    initialize_matrices();

    int nx = 4;  
    VectorXd xEst = VectorXd::Zero(nx);
    VectorXd xTrue = VectorXd::Zero(nx);
    MatrixXd PEst = MatrixXd::Identity(nx, nx);
    VectorXd xDR = VectorXd::Zero(nx); 
    RowVectorXd wm, wc;
    double gamma;
    setup_ukf(nx, wm, wc, gamma);

 
    vector<VectorXd> hxEst;
    vector<VectorXd> hxTrue;
    vector<VectorXd> hxDR;
    vector<VectorXd> hz;

    double time = 0.0;

    std::random_device rd;
    generator.seed(rd());

    while (SIM_TIME >= time)
    {
        time += DT;
        Vector2d u = calc_input();

        VectorXd z;
        VectorXd ud;
        observation(xTrue, xDR, u, z, ud);

        ukf_estimation(xEst, PEst, z, ud, wm, wc, gamma);


        hxEst.push_back(xEst);
        hxDR.push_back(xDR);
        hxTrue.push_back(xTrue);
        hz.push_back(z);
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

void initialize_matrices()
{
    VectorXd q_diag(4);
    q_diag << 0.1, 0.1, M_PI / 180.0, 1.0;
    q_diag = q_diag.array().square();
    Q = q_diag.asDiagonal();

    VectorXd r_diag(2);
    r_diag << 1.0, 1.0;
    r_diag = r_diag.array().square();
    R = r_diag.asDiagonal();

    VectorXd input_noise_diag(2);
    input_noise_diag << 1.0, M_PI / 6.0; 
    input_noise_diag = input_noise_diag.array().square();
    INPUT_NOISE = input_noise_diag.asDiagonal();

    VectorXd gps_noise_diag(2);
    gps_noise_diag << 0.5, 0.5;
    gps_noise_diag = gps_noise_diag.array().square();
    GPS_NOISE = gps_noise_diag.asDiagonal();
}

Vector2d calc_input()
{
    double v = 1.0;     
    double yawRate = 0.1; 
    Vector2d u;
    u << v, yawRate;
    return u;
}

VectorXd motion_model(const VectorXd& x, const VectorXd& u)
{
    VectorXd x_new = VectorXd::Zero(4);
    x_new(0) = x(0) + DT * cos(x(2)) * u(0);
    x_new(1) = x(1) + DT * sin(x(2)) * u(0);
    x_new(2) = x(2) + DT * u(1);
    x_new(3) = u(0); 
    return x_new;
}

VectorXd observation_model(const VectorXd& x)
{
    VectorXd z = VectorXd::Zero(2);
    z(0) = x(0);
    z(1) = x(1);
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

void observation(VectorXd& xTrue, VectorXd& xd, const VectorXd& u, VectorXd& z, VectorXd& ud)
{
 
    xTrue = motion_model(xTrue, u);

    
    VectorXd noise = GPS_NOISE.llt().matrixL() * standard_normal_vector(2);
    z = observation_model(xTrue) + noise;

  
    VectorXd input_noise = INPUT_NOISE.llt().matrixL() * standard_normal_vector(2);
    ud = u + input_noise;

 
    xd = motion_model(xd, ud);
}

void generate_sigma_points(const VectorXd& xEst, const MatrixXd& PEst, double gamma, MatrixXd& sigma)
{
    int n = xEst.size();
    sigma = MatrixXd(n, 2 * n + 1);
    sigma.col(0) = xEst;

    MatrixXd Psqrt = PEst.llt().matrixL();

    for (int i = 0; i < n; ++i)
    {
        sigma.col(i + 1) = xEst + gamma * Psqrt.col(i);
        sigma.col(i + 1 + n) = xEst - gamma * Psqrt.col(i);
    }
}

void predict_sigma_motion(MatrixXd& sigma, const VectorXd& u)
{
    for (int i = 0; i < sigma.cols(); ++i)
    {
        sigma.col(i) = motion_model(sigma.col(i), u);
    }
}

void predict_sigma_observation(MatrixXd& sigma)
{
    for (int i = 0; i < sigma.cols(); ++i)
    {
        VectorXd z = observation_model(sigma.col(i));
        sigma.col(i) = z;
    }
}

MatrixXd calc_sigma_covariance(const VectorXd& x, const MatrixXd& sigma, const RowVectorXd& wc, const MatrixXd& Pi)
{
    MatrixXd P = Pi;
    int nSigma = sigma.cols();
    for (int i = 0; i < nSigma; ++i)
    {
        VectorXd dx = sigma.col(i) - x;
        P += wc(i) * dx * dx.transpose();
    }
    return P;
}

MatrixXd calc_pxz(const MatrixXd& sigma, const VectorXd& x, const MatrixXd& z_sigma, const VectorXd& zb, const RowVectorXd& wc)
{
    int nSigma = sigma.cols();
    MatrixXd P = MatrixXd::Zero(x.size(), zb.size());
    for (int i = 0; i < nSigma; ++i)
    {
        VectorXd dx = sigma.col(i) - x;
        VectorXd dz = z_sigma.col(i) - zb;
        P += wc(i) * dx * dz.transpose();
    }
    return P;
}

void ukf_estimation(VectorXd& xEst, MatrixXd& PEst, const VectorXd& z, const VectorXd& u, const RowVectorXd& wm, const RowVectorXd& wc, double gamma)
{

    MatrixXd sigma;
    generate_sigma_points(xEst, PEst, gamma, sigma);
    predict_sigma_motion(sigma, u);
    VectorXd xPred = sigma * wm.transpose();
    MatrixXd PPred = calc_sigma_covariance(xPred, sigma, wc, Q);

 
    MatrixXd sigma_points;
    generate_sigma_points(xPred, PPred, gamma, sigma_points);
    MatrixXd z_sigma = sigma_points.topRows(4);
    predict_sigma_observation(z_sigma);
    VectorXd zb = z_sigma * wm.transpose();
    MatrixXd st = calc_sigma_covariance(zb, z_sigma, wc, R);
    MatrixXd Pxz = calc_pxz(sigma_points, xPred, z_sigma, zb, wc);
    MatrixXd K = Pxz * st.inverse();
    xEst = xPred + K * (z - zb);
    PEst = PPred - K * st * K.transpose();
}

void setup_ukf(int nx, RowVectorXd& wm, RowVectorXd& wc, double& gamma)
{
    double lambda = ALPHA * ALPHA * (nx + KAPPA) - nx;
    gamma = sqrt(nx + lambda);

    wm = RowVectorXd(2 * nx + 1);
    wc = RowVectorXd(2 * nx + 1);
    wm(0) = lambda / (nx + lambda);
    wc(0) = lambda / (nx + lambda) + (1 - ALPHA * ALPHA + BETA);
    for (int i = 1; i < 2 * nx + 1; ++i)
    {
        wm(i) = 1.0 / (2.0 * (nx + lambda));
        wc(i) = 1.0 / (2.0 * (nx + lambda));
    }
}
