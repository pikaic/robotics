#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <random>


const double DT = 0.1;      
const double SIM_TIME = 50.0; 
const double MAX_RANGE = 20.0; 
const int NP = 100;          
const double NTh = NP / 2.0;  


Eigen::MatrixXd Q(1, 1);      
Eigen::MatrixXd R(2, 2);      
Eigen::MatrixXd Q_sim(1, 1);  
Eigen::MatrixXd R_sim(2, 2);  
std::random_device rd;
std::mt19937 gen(rd());


Eigen::Vector2d calc_input();
void observation(const Eigen::VectorXd& x_true_in, Eigen::VectorXd& x_true_out,
                 Eigen::VectorXd& xd, const Eigen::Vector2d& u,
                 const Eigen::MatrixXd& rf_id, std::vector<Eigen::Vector3d>& z, Eigen::Vector2d& ud);
Eigen::VectorXd motion_model(const Eigen::VectorXd& x, const Eigen::Vector2d& u);
double gauss_likelihood(double x, double sigma);
Eigen::MatrixXd calc_covariance(const Eigen::VectorXd& x_est,
                                const Eigen::MatrixXd& px, const Eigen::RowVectorXd& pw);
void pf_localization(Eigen::MatrixXd& px, Eigen::RowVectorXd& pw,
                     const std::vector<Eigen::Vector3d>& z, const Eigen::Vector2d& u,
                     Eigen::VectorXd& x_est, Eigen::MatrixXd& p_est);
void re_sampling(Eigen::MatrixXd& px, Eigen::RowVectorXd& pw);

int main() {
    std::cout << "Particle Filter Localization start!!" << std::endl;


    Q(0, 0) = std::pow(0.2, 2);

    R.setZero(2, 2);
    R(0, 0) = std::pow(2.0, 2);
    R(1, 1) = std::pow(M_PI * 40.0 / 180.0, 2);

    Q_sim(0, 0) = std::pow(0.2, 2);

    R_sim.setZero(2, 2);
    R_sim(0, 0) = std::pow(1.0, 2);
    R_sim(1, 1) = std::pow(M_PI * 30.0 / 180.0, 2);

    double time = 0.0;


    Eigen::MatrixXd rf_id(4, 2);
    rf_id << 10.0, 0.0,
             10.0, 10.0,
              0.0, 15.0,
             -5.0, 20.0;


    Eigen::VectorXd x_est = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd x_true = Eigen::VectorXd::Zero(4);

    Eigen::MatrixXd px = Eigen::MatrixXd::Zero(4, NP);    
    Eigen::RowVectorXd pw = Eigen::RowVectorXd::Constant(NP, 1.0 / NP); 
    Eigen::VectorXd x_dr = Eigen::VectorXd::Zero(4);      


    std::vector<Eigen::VectorXd> h_x_est, h_x_true, h_x_dr;
    h_x_est.push_back(x_est);
    h_x_true.push_back(x_true);
    h_x_dr.push_back(x_dr);

    while (SIM_TIME >= time) {
        time += DT;
        Eigen::Vector2d u = calc_input();

        Eigen::VectorXd x_true_new;
        std::vector<Eigen::Vector3d> z;
        Eigen::Vector2d ud;
        observation(x_true, x_true_new, x_dr, u, rf_id, z, ud);

        x_true = x_true_new;

        Eigen::VectorXd p_est;
        pf_localization(px, pw, z, ud, x_est, p_est);

     
        h_x_est.push_back(x_est);
        h_x_dr.push_back(x_dr);
        h_x_true.push_back(x_true);

       
        std::cout << "Time: " << time << ", x_est: " << x_est.transpose() << std::endl;
    }

    return 0;
}

Eigen::Vector2d calc_input() {
    double v = 1.0;      
    double yaw_rate = 0.1;
    Eigen::Vector2d u;
    u << v, yaw_rate;
    return u;
}

void observation(const Eigen::VectorXd& x_true_in, Eigen::VectorXd& x_true_out,
                 Eigen::VectorXd& xd, const Eigen::Vector2d& u,
                 const Eigen::MatrixXd& rf_id, std::vector<Eigen::Vector3d>& z, Eigen::Vector2d& ud) {
    x_true_out = motion_model(x_true_in, u);

  
    z.clear();
    std::normal_distribution<> dist_Q_sim(0.0, std::sqrt(Q_sim(0, 0)));

    for (int i = 0; i < rf_id.rows(); ++i) {
        double dx = x_true_out(0) - rf_id(i, 0);
        double dy = x_true_out(1) - rf_id(i, 1);
        double d = std::hypot(dx, dy);
        if (d <= MAX_RANGE) {
    
            double dn = d + dist_Q_sim(gen);
            Eigen::Vector3d zi;
            zi << dn, rf_id(i, 0), rf_id(i, 1);
            z.push_back(zi);
        }
    }


    std::normal_distribution<> dist_ud1(0.0, std::sqrt(R_sim(0, 0)));
    std::normal_distribution<> dist_ud2(0.0, std::sqrt(R_sim(1, 1)));

    double ud1 = u(0) + dist_ud1(gen);
    double ud2 = u(1) + dist_ud2(gen);
    ud << ud1, ud2;

    xd = motion_model(xd, ud);
}

Eigen::VectorXd motion_model(const Eigen::VectorXd& x, const Eigen::Vector2d& u) {
    Eigen::MatrixXd F(4, 4);
    F << 1.0, 0, 0, 0,
         0, 1.0, 0, 0,
         0, 0, 1.0, 0,
         0, 0, 0, 0;

    double yaw = x(2);
    Eigen::MatrixXd B(4, 2);
    B << DT * std::cos(yaw), 0,
         DT * std::sin(yaw), 0,
         0.0, DT,
         1.0, 0.0;

    Eigen::VectorXd x_out = F * x + B * u;
    return x_out;
}

double gauss_likelihood(double x, double sigma) {
    double p = 1.0 / std::sqrt(2.0 * M_PI * sigma * sigma) *
               std::exp(-x * x / (2 * sigma * sigma));
    return p;
}

Eigen::MatrixXd calc_covariance(const Eigen::VectorXd& x_est,
                                const Eigen::MatrixXd& px, const Eigen::RowVectorXd& pw) {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(4, 4);
    int n_particle = px.cols();
    for (int i = 0; i < n_particle; ++i) {
        Eigen::VectorXd dx = px.col(i) - x_est;
        cov += pw(i) * dx * dx.transpose();
    }
    cov *= 1.0 / (1.0 - pw * pw.transpose());
    return cov;
}

void pf_localization(Eigen::MatrixXd& px, Eigen::RowVectorXd& pw,
                     const std::vector<Eigen::Vector3d>& z, const Eigen::Vector2d& u,
                     Eigen::VectorXd& x_est, Eigen::MatrixXd& p_est) {
    int NP = px.cols();

    std::normal_distribution<> dist_ud1(0.0, std::sqrt(R(0, 0)));
    std::normal_distribution<> dist_ud2(0.0, std::sqrt(R(1, 1)));

    for (int ip = 0; ip < NP; ++ip) {
        Eigen::VectorXd x = px.col(ip);
        double w = pw(ip);

     
        double ud1 = u(0) + dist_ud1(gen);
        double ud2 = u(1) + dist_ud2(gen);
        Eigen::Vector2d ud(ud1, ud2);

        x = motion_model(x, ud);

        for (const auto& zi : z) {
            double dx = x(0) - zi(1);
            double dy = x(1) - zi(2);
            double pre_z = std::hypot(dx, dy);
            double dz = pre_z - zi(0);
            w *= gauss_likelihood(dz, std::sqrt(Q(0, 0)));
        }

        px.col(ip) = x;
        pw(ip) = w;
    }

    double pw_sum = pw.sum();
    pw /= pw_sum;

  
    x_est = px * pw.transpose();

 
    p_est = calc_covariance(x_est, px, pw);

    double N_eff = 1.0 / pw.array().square().sum();

    if (N_eff < NTh) {
        re_sampling(px, pw);
    }
}

void re_sampling(Eigen::MatrixXd& px, Eigen::RowVectorXd& pw) {
 
    Eigen::VectorXd w_cum = pw.transpose().cumsum();
    double r = std::uniform_real_distribution<>(0.0, 1.0 / NP)(gen);
    Eigen::VectorXd positions = Eigen::VectorXd::LinSpaced(NP, 0.0, 1.0 - 1.0 / NP) + r;

    std::vector<int> indexes(NP);
    int i = 0, j = 0;
    while (i < NP) {
        if (positions(i) < w_cum(j)) {
            indexes[i] = j;
            ++i;
        } else {
            ++j;
        }
    }

    Eigen::MatrixXd px_resampled(4, NP);
    for (int i = 0; i < NP; ++i) {
        px_resampled.col(i) = px.col(indexes[i]);
    }
    px = px_resampled;

    pw = Eigen::RowVectorXd::Constant(NP, 1.0 / NP);
}
