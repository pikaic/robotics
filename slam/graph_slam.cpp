#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <iterator>
#include <random>

const double DT = 2.0;           
const double SIM_TIME = 100.0;  
const double MAX_RANGE = 30.0;   


const double C_SIGMA1 = 0.1;
const double C_SIGMA2 = 0.1;
const double C_SIGMA3 = M_PI / 180.0;  

const int MAX_ITR = 20;  


std::default_random_engine generator;
std::normal_distribution<double> dist(0.0, 1.0);


inline double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

inline double pi_2_pi(double angle) {
    while (angle >= M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

class Edge {
public:
    Eigen::Vector3d e;
    Eigen::Matrix3d omega;  
    double d1, d2;
    double yaw1, yaw2;
    double angle1, angle2;
    int id1, id2;
};

Eigen::Matrix3d cal_observation_sigma() {
    Eigen::Matrix3d sigma = Eigen::Matrix3d::Zero();
    sigma(0, 0) = C_SIGMA1 * C_SIGMA1;
    sigma(1, 1) = C_SIGMA2 * C_SIGMA2;
    sigma(2, 2) = C_SIGMA3 * C_SIGMA3;
    return sigma;
}

Eigen::Matrix3d calc_3d_rotational_matrix(double angle) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    double c = cos(angle);
    double s = sin(angle);
    R(0, 0) = c;
    R(0, 1) = -s;
    R(1, 0) = s;
    R(1, 1) = c;
    return R;
}

Edge calc_edge(double x1, double y1, double yaw1, double x2, double y2, double yaw2,
               double d1, double angle1, double d2, double angle2, int t1, int t2) {
    Edge edge;

    double tangle1 = pi_2_pi(yaw1 + angle1);
    double tangle2 = pi_2_pi(yaw2 + angle2);
    double tmp1 = d1 * cos(tangle1);
    double tmp2 = d2 * cos(tangle2);
    double tmp3 = d1 * sin(tangle1);
    double tmp4 = d2 * sin(tangle2);

    edge.e(0) = x2 - x1 - tmp1 + tmp2;
    edge.e(1) = y2 - y1 - tmp3 + tmp4;
    edge.e(2) = 0.0;

    Eigen::Matrix3d Rt1 = calc_3d_rotational_matrix(tangle1);
    Eigen::Matrix3d Rt2 = calc_3d_rotational_matrix(tangle2);

    Eigen::Matrix3d sig1 = cal_observation_sigma();
    Eigen::Matrix3d sig2 = cal_observation_sigma();

    edge.omega = (Rt1 * sig1 * Rt1.transpose() + Rt2 * sig2 * Rt2.transpose()).inverse();

    edge.d1 = d1;
    edge.d2 = d2;
    edge.yaw1 = yaw1;
    edge.yaw2 = yaw2;
    edge.angle1 = angle1;
    edge.angle2 = angle2;
    edge.id1 = t1;
    edge.id2 = t2;

    return edge;
}

std::vector<Edge> calc_edges(const Eigen::MatrixXd& x_list, const std::vector<Eigen::MatrixXd>& z_list) {
    std::vector<Edge> edges;
    double cost = 0.0;

    int n = z_list.size();
    for (int t1 = 0; t1 < n - 1; ++t1) {
        for (int t2 = t1 + 1; t2 < n; ++t2) {
            if (z_list[t1].cols() == 0 || z_list[t2].cols() == 0) {
                continue; 
            }

            double x1 = x_list(0, t1), y1 = x_list(1, t1), yaw1 = x_list(2, t1);
            double x2 = x_list(0, t2), y2 = x_list(1, t2), yaw2 = x_list(2, t2);

            for (int iz1 = 0; iz1 < z_list[t1].rows(); ++iz1) {
                for (int iz2 = 0; iz2 < z_list[t2].rows(); ++iz2) {
                    if (z_list[t1](iz1, 3) == z_list[t2](iz2, 3)) {
                        double d1 = z_list[t1](iz1, 0);
                        double angle1 = z_list[t1](iz1, 1);
                        double d2 = z_list[t2](iz2, 0);
                        double angle2 = z_list[t2](iz2, 1);

                        Edge edge = calc_edge(x1, y1, yaw1, x2, y2, yaw2, d1, angle1, d2, angle2, t1, t2);

                        edges.push_back(edge);
                        cost += edge.e.transpose() * edge.omega * edge.e;
                    }
                }
            }
        }
    }

    std::cout << "Cost: " << cost << ", Number of edges: " << edges.size() << std::endl;
    return edges;
}

void calc_jacobian(const Edge& edge, Eigen::MatrixXd& A, Eigen::MatrixXd& B) {
    double t1 = edge.yaw1 + edge.angle1;
    A = Eigen::MatrixXd(3, 3);
    A << -1.0, 0.0, edge.d1 * sin(t1),
         0.0, -1.0, -edge.d1 * cos(t1),
         0.0, 0.0, 0.0;

    double t2 = edge.yaw2 + edge.angle2;
    B = Eigen::MatrixXd(3, 3);
    B << 1.0, 0.0, -edge.d2 * sin(t2),
         0.0, 1.0, edge.d2 * cos(t2),
         0.0, 0.0, 0.0;
}

void fill_H_and_b(Eigen::MatrixXd& H, Eigen::VectorXd& b, const Edge& edge) {
    Eigen::MatrixXd A, B;
    calc_jacobian(edge, A, B);

    int id1 = edge.id1 * STATE_SIZE;
    int id2 = edge.id2 * STATE_SIZE;

    H.block(id1, id1, STATE_SIZE, STATE_SIZE) += A.transpose() * edge.omega * A;
    H.block(id1, id2, STATE_SIZE, STATE_SIZE) += A.transpose() * edge.omega * B;
    H.block(id2, id1, STATE_SIZE, STATE_SIZE) += B.transpose() * edge.omega * A;
    H.block(id2, id2, STATE_SIZE, STATE_SIZE) += B.transpose() * edge.omega * B;

    b.segment(id1, STATE_SIZE) += A.transpose() * edge.omega * edge.e;
    b.segment(id2, STATE_SIZE) += B.transpose() * edge.omega * edge.e;
}

Eigen::MatrixXd graph_based_slam(const Eigen::MatrixXd& x_init, const std::vector<Eigen::MatrixXd>& hz) {
    std::cout << "Start graph-based SLAM" << std::endl;

    std::vector<Eigen::MatrixXd> z_list = hz;
    Eigen::MatrixXd x_opt = x_init;
    int nt = x_opt.cols();
    int n = nt * STATE_SIZE;

    for (int itr = 0; itr < MAX_ITR; ++itr) {
        std::vector<Edge> edges = calc_edges(x_opt, z_list);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);

        for (const auto& edge : edges) {
            fill_H_and_b(H, b, edge);
        }


        H.block(0, 0, STATE_SIZE, STATE_SIZE) += Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);

        Eigen::VectorXd dx = -H.ldlt().solve(b);

        for (int i = 0; i < nt; ++i) {
            x_opt.col(i) += dx.segment(i * STATE_SIZE, STATE_SIZE);
        }

        double diff = dx.transpose() * dx;
        std::cout << "Iteration: " << itr + 1 << ", diff: " << diff << std::endl;
        if (diff < 1.0e-5) {
            break;
        }
    }

    return x_opt;
}

Eigen::Vector2d calc_input() {
    double v = 1.0;       // [m/s]
    double yaw_rate = 0.1;  // [rad/s]
    Eigen::Vector2d u;
    u << v, yaw_rate;
    return u;
}

void observation(Eigen::Vector3d& xTrue, Eigen::Vector3d& xd, const Eigen::Vector2d& u,
                 const Eigen::MatrixXd& RFID, Eigen::MatrixXd& z,
                 const Eigen::Matrix2d& Q_sim, const Eigen::Matrix2d& R_sim) {
    xTrue = xTrue + DT * Eigen::Vector3d(u(0) * cos(xTrue(2)), u(0) * sin(xTrue(2)), u(1));


    std::vector<Eigen::VectorXd> z_list;

    int n_landmarks = RFID.rows();
    for (int i = 0; i < n_landmarks; ++i) {
        double dx = RFID(i, 0) - xTrue(0);
        double dy = RFID(i, 1) - xTrue(1);
        double d = hypot(dx, dy);
        double angle = pi_2_pi(atan2(dy, dx)) - xTrue(2);
        double phi = pi_2_pi(atan2(dy, dx));
        if (d <= MAX_RANGE) {
            double dn = d + dist(generator) * sqrt(Q_sim(0, 0));  // add noise
            double angle_noise = dist(generator) * sqrt(Q_sim(1, 1));
            angle += angle_noise;
            phi += angle_noise;
            Eigen::VectorXd zi(4);
            zi << dn, angle, phi, i;
            z_list.push_back(zi);
        }
    }


    z = Eigen::MatrixXd::Zero(z_list.size(), 4);
    for (size_t i = 0; i < z_list.size(); ++i) {
        z.row(i) = z_list[i].transpose();
    }


    double ud1 = u(0) + dist(generator) * sqrt(R_sim(0, 0));
    double ud2 = u(1) + dist(generator) * sqrt(R_sim(1, 1));
    Eigen::Vector2d ud(ud1, ud2);

    xd = xd + DT * Eigen::Vector3d(ud(0) * cos(xd(2)), ud(0) * sin(xd(2)), ud(1));
}

Eigen::Vector3d motion_model(const Eigen::Vector3d& x, const Eigen::Vector2d& u) {
    Eigen::Vector3d x_new = x;
    x_new(0) += DT * u(0) * cos(x(2));
    x_new(1) += DT * u(0) * sin(x(2));
    x_new(2) += DT * u(1);
    return x_new;
}

int main() {
    std::cout << "Graph-based SLAM start!!" << std::endl;

    double time = 0.0;


    Eigen::MatrixXd RFID(5, 3);
    RFID << 10.0, -2.0, 0.0,
            15.0, 10.0, 0.0,
            3.0, 15.0, 0.0,
            -5.0, 20.0, 0.0,
            -5.0, 5.0, 0.0;


    Eigen::Vector3d xTrue = Eigen::Vector3d::Zero();
    Eigen::Vector3d xDR = Eigen::Vector3d::Zero();  // Dead reckoning


    std::vector<Eigen::Vector3d> hxTrue;
    std::vector<Eigen::Vector3d> hxDR;
    std::vector<Eigen::MatrixXd> hz;
    double d_time = 0.0;


    Eigen::Matrix2d Q_sim = Eigen::Matrix2d::Zero();
    Q_sim(0, 0) = 0.2 * 0.2;
    Q_sim(1, 1) = deg2rad(1.0) * deg2rad(1.0);

    Eigen::Matrix2d R_sim = Eigen::Matrix2d::Zero();
    R_sim(0, 0) = 0.1 * 0.1;
    R_sim(1, 1) = deg2rad(10.0) * deg2rad(10.0);


    while (SIM_TIME >= time) {
        hxDR.push_back(xDR);
        hxTrue.push_back(xTrue);

        time += DT;
        d_time += DT;
        Eigen::Vector2d u = calc_input();

        Eigen::MatrixXd z;
        observation(xTrue, xDR, u, RFID, z, Q_sim, R_sim);

        hz.push_back(z);

        if (d_time >= 20.0) {  
            int num_states = hxDR.size();
            Eigen::MatrixXd hxDR_mat(STATE_SIZE, num_states);
            for (int i = 0; i < num_states; ++i) {
                hxDR_mat.col(i) = hxDR[i];
            }

            Eigen::MatrixXd x_opt = graph_based_slam(hxDR_mat, hz);
            d_time = 0.0;

            
           
            std::cout << "Time: " << time << std::endl;
        }
    }

    return 0;
}
