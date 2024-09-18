#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <random>

#define _USE_MATH_DEFINES  
#include <cmath>


const double DT = 0.1;            
const double SIM_TIME = 50.0;     
const double MAX_RANGE = 20.0;    
const double M_DIST_TH = 2.0;     
const int STATE_SIZE = 3;         
const int LM_SIZE = 2;            
const int N_PARTICLE = 100;       
const double NTH = N_PARTICLE / 1.5;  
const double OFFSET_YAW_RATE_NOISE = 0.01;


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

class Particle {
public:
    double w;  
    double x, y, yaw;
    Eigen::MatrixXd lm;  
    Eigen::MatrixXd lmP; 

    Particle(int n_landmark) {
        w = 1.0 / N_PARTICLE;
        x = 0.0;
        y = 0.0;
        yaw = 0.0;
        lm = Eigen::MatrixXd::Zero(n_landmark, LM_SIZE);
        lmP = Eigen::MatrixXd::Zero(n_landmark * LM_SIZE, LM_SIZE);
    }
};

Eigen::Vector3d motion_model(Eigen::Vector3d x, Eigen::Vector2d u) {
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 2> B;
    B << DT * cos(x(2)), 0,
         DT * sin(x(2)), 0,
         0.0, DT;

    x = F * x + B * u;
    x(2) = pi_2_pi(x(2));
    return x;
}

Eigen::Vector2d calc_input(double time) {
    double v, yaw_rate;
    if (time <= 3.0) {
        v = 0.0;
        yaw_rate = 0.0;
    } else {
        v = 1.0;
        yaw_rate = 0.1;
    }
    Eigen::Vector2d u;
    u << v, yaw_rate;
    return u;
}

void normalize_weight(std::vector<Particle>& particles) {
    double sum_w = 0.0;
    for (const auto& p : particles) {
        sum_w += p.w;
    }
    if (sum_w == 0) {
        for (auto& p : particles) {
            p.w = 1.0 / N_PARTICLE;
        }
    } else {
        for (auto& p : particles) {
            p.w /= sum_w;
        }
    }
}

Eigen::Vector3d calc_final_state(std::vector<Particle>& particles) {
    Eigen::Vector3d x_est = Eigen::Vector3d::Zero();
    normalize_weight(particles);
    for (int i = 0; i < N_PARTICLE; i++) {
        x_est(0) += particles[i].w * particles[i].x;
        x_est(1) += particles[i].w * particles[i].y;
        x_est(2) += particles[i].w * particles[i].yaw;
    }
    x_est(2) = pi_2_pi(x_est(2));
    return x_est;
}

void predict_particles(std::vector<Particle>& particles, Eigen::Vector2d u, const Eigen::Matrix2d& R) {
    for (int i = 0; i < N_PARTICLE; i++) {
        Eigen::Vector3d px;
        px << particles[i].x, particles[i].y, particles[i].yaw;

        Eigen::Vector2d noise;
        noise(0) = dist(generator);
        noise(1) = dist(generator);

        Eigen::Vector2d ud = u + (R.llt().matrixL() * noise);

        px = motion_model(px, ud);

        particles[i].x = px(0);
        particles[i].y = px(1);
        particles[i].yaw = px(2);
    }
}

void compute_jacobians(const Particle& particle, const Eigen::Vector2d& xf, const Eigen::Matrix2d& Pf, const Eigen::Matrix2d& Q_cov,
                       Eigen::Vector2d& zp, Eigen::MatrixXd& Hv, Eigen::Matrix2d& Hf, Eigen::Matrix2d& Sf) {
    double dx = xf(0) - particle.x;
    double dy = xf(1) - particle.y;
    double d2 = dx * dx + dy * dy;
    double d = sqrt(d2);

    zp << d,
          pi_2_pi(atan2(dy, dx) - particle.yaw);

    Hv = Eigen::MatrixXd(2, STATE_SIZE);
    Hv << -dx / d, -dy / d, 0.0,
           dy / d2, -dx / d2, -1.0;

    Hf << dx / d, dy / d,
          -dy / d2, dx / d2;

    Sf = Hf * Pf * Hf.transpose() + Q_cov;
}

void update_kf_with_cholesky(const Eigen::Vector2d& xf, const Eigen::Matrix2d& Pf, const Eigen::Vector2d& v,
                             const Eigen::Matrix2d& Q_cov, const Eigen::Matrix2d& Hf,
                             Eigen::Vector2d& x, Eigen::Matrix2d& P) {
    Eigen::Matrix2d PHt = Pf * Hf.transpose();
    Eigen::Matrix2d S = Hf * PHt + Q_cov;

    S = (S + S.transpose()) * 0.5; 

    Eigen::LLT<Eigen::Matrix2d> lltOfS(S);
    Eigen::Matrix2d s_chol = lltOfS.matrixL();

    Eigen::Matrix2d s_chol_inv = s_chol.inverse();
    Eigen::Matrix2d W1 = PHt * s_chol_inv;
    Eigen::Matrix2d W = W1 * s_chol_inv.transpose();

    x = xf + W * v;
    P = Pf - W1 * W1.transpose();
}

void add_new_landmark(Particle& particle, Eigen::VectorXd z, const Eigen::Matrix2d& Q_cov) {
    double r = z(0);
    double b = z(1);
    int lm_id = static_cast<int>(z(2));

    double s = sin(pi_2_pi(particle.yaw + b));
    double c = cos(pi_2_pi(particle.yaw + b));

    particle.lm(lm_id, 0) = particle.x + r * c;
    particle.lm(lm_id, 1) = particle.y + r * s;

    
    double dx = r * c;
    double dy = r * s;
    double d2 = dx * dx + dy * dy;
    double d = sqrt(d2);

    Eigen::Matrix2d Gz;
    Gz << dx / d, dy / d,
         -dy / d2, dx / d2;

    Eigen::Matrix2d Gz_inv = Gz.inverse();
    Eigen::Matrix2d Pf = Gz_inv * Q_cov * Gz_inv.transpose();

    particle.lmP.block<2,2>(2 * lm_id, 0) = Pf;
}

void update_landmark(Particle& particle, const Eigen::VectorXd& z, const Eigen::Matrix2d& Q_cov) {
    int lm_id = static_cast<int>(z(2));
    Eigen::Vector2d xf = particle.lm.row(lm_id).transpose();
    Eigen::Matrix2d Pf = particle.lmP.block<2,2>(2 * lm_id, 0);

    Eigen::Vector2d zp;
    Eigen::MatrixXd Hv;
    Eigen::Matrix2d Hf, Sf;
    compute_jacobians(particle, xf, Pf, Q_cov, zp, Hv, Hf, Sf);

    Eigen::Vector2d dz = z.segment<2>(0) - zp;
    dz(1) = pi_2_pi(dz(1));

    Eigen::Vector2d xf_updated;
    Eigen::Matrix2d Pf_updated;
    update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf, xf_updated, Pf_updated);

    particle.lm.row(lm_id) = xf_updated.transpose();
    particle.lmP.block<2,2>(2 * lm_id, 0) = Pf_updated;
}

double compute_weight(const Particle& particle, const Eigen::VectorXd& z, const Eigen::Matrix2d& Q_cov) {
    int lm_id = static_cast<int>(z(2));
    Eigen::Vector2d xf = particle.lm.row(lm_id).transpose();
    Eigen::Matrix2d Pf = particle.lmP.block<2,2>(2 * lm_id, 0);

    Eigen::Vector2d zp;
    Eigen::MatrixXd Hv;
    Eigen::Matrix2d Hf, Sf;
    compute_jacobians(particle, xf, Pf, Q_cov, zp, Hv, Hf, Sf);

    Eigen::Vector2d dx = z.segment<2>(0) - zp;
    dx(1) = pi_2_pi(dx(1));

    double w = 1.0;
    double detSf = Sf.determinant();
    if (detSf > 0) {
        Eigen::Matrix2d invSf = Sf.inverse();
        double num = exp(-0.5 * dx.transpose() * invSf * dx);
        double den = 2.0 * M_PI * sqrt(detSf);
        w = num / den;
    } else {
        std::cout << "singular" << std::endl;
        w = 1.0;
    }
    return w;
}

void update_with_observation(std::vector<Particle>& particles, const Eigen::MatrixXd& z, const Eigen::Matrix2d& Q) {
    int num_observations = z.cols();
    for (int iz = 0; iz < num_observations; iz++) {
        int landmark_id = static_cast<int>(z(2, iz));
        Eigen::VectorXd zi = z.col(iz);
        for (int ip = 0; ip < N_PARTICLE; ip++) {
            // new landmark
            if (fabs(particles[ip].lm(landmark_id, 0)) <= 0.01) {
                add_new_landmark(particles[ip], zi, Q);
            } else {
                double w = compute_weight(particles[ip], zi, Q);
                particles[ip].w *= w;
                update_landmark(particles[ip], zi, Q);
            }
        }
    }
}

void resampling(std::vector<Particle>& particles) {
    normalize_weight(particles);

    
    double n_eff = 0.0;
    for (const auto& p : particles) {
        n_eff += p.w * p.w;
    }
    n_eff = 1.0 / n_eff;

    if (n_eff < NTH) {
        
        std::vector<double> cumulative_sum(N_PARTICLE);
        cumulative_sum[0] = particles[0].w;
        for (int i = 1; i < N_PARTICLE; i++) {
            cumulative_sum[i] = cumulative_sum[i - 1] + particles[i].w;
        }

        std::vector<Particle> new_particles;
        new_particles.reserve(N_PARTICLE);

        double step = 1.0 / N_PARTICLE;
        double r = ((double) rand() / (RAND_MAX)) * step;
        int index = 0;

        for (int i = 0; i < N_PARTICLE; i++) {
            double u = r + i * step;
            while (u > cumulative_sum[index]) {
                index++;
                if (index >= N_PARTICLE) index = N_PARTICLE - 1;
            }
            new_particles.push_back(particles[index]);
            new_particles.back().w = 1.0 / N_PARTICLE;
        }

        particles = new_particles;
    }
}

void observation(Eigen::Vector3d& x_true, Eigen::Vector3d& xd, const Eigen::Vector2d& u,
                 const Eigen::MatrixXd& rfid, Eigen::MatrixXd& z, const Eigen::Matrix2d& Q_sim,
                 const Eigen::Matrix2d& R_sim, double OFFSET_YAW_RATE_NOISE) {
    
    x_true = motion_model(x_true, u);

  
    std::vector<Eigen::VectorXd> z_list;

    int n_landmarks = rfid.rows();
    for (int i = 0; i < n_landmarks; i++) {
        double dx = rfid(i, 0) - x_true(0);
        double dy = rfid(i, 1) - x_true(1);
        double d = hypot(dx, dy);
        double angle = pi_2_pi(atan2(dy, dx) - x_true(2));
        if (d <= MAX_RANGE) {
            double dn = d + dist(generator) * sqrt(Q_sim(0, 0));  // add noise
            double angle_with_noise = angle + dist(generator) * sqrt(Q_sim(1, 1));  // add noise
            Eigen::VectorXd zi(3);
            zi << dn, pi_2_pi(angle_with_noise), i;
            z_list.push_back(zi);
        }
    }

    
    z = Eigen::MatrixXd(3, z_list.size());
    for (size_t i = 0; i < z_list.size(); i++) {
        z.col(i) = z_list[i];
    }

   
    double ud1 = u(0) + dist(generator) * sqrt(R_sim(0, 0));
    double ud2 = u(1) + dist(generator) * sqrt(R_sim(1, 1)) + OFFSET_YAW_RATE_NOISE;
    Eigen::Vector2d ud;
    ud << ud1, ud2;

    xd = motion_model(xd, ud);
}

int main() {
    std::cout << "FastSLAM 1.0 start!!" << std::endl;

    double time = 0.0;

    
    Eigen::MatrixXd rfid(8, 2);
    rfid << 10.0, -2.0,
            15.0, 10.0,
            15.0, 15.0,
            10.0, 20.0,
            3.0, 15.0,
            -5.0, 20.0,
            -5.0, 5.0,
            -10.0, 15.0;
    int n_landmark = rfid.rows();

    
    Eigen::Vector3d x_est = Eigen::Vector3d::Zero(); 
    Eigen::Vector3d x_true = Eigen::Vector3d::Zero();  
    Eigen::Vector3d x_dr = Eigen::Vector3d::Zero();  


    std::vector<Eigen::Vector3d> hist_x_est;
    std::vector<Eigen::Vector3d> hist_x_true;
    std::vector<Eigen::Vector3d> hist_x_dr;

    hist_x_est.push_back(x_est);
    hist_x_true.push_back(x_true);
    hist_x_dr.push_back(x_dr);

    std::vector<Particle> particles(N_PARTICLE, Particle(n_landmark));


    Eigen::Matrix2d Q;
    Q << pow(3.0, 2), 0,
         0, pow(deg2rad(10.0), 2);

    Eigen::Matrix2d R;
    R << pow(1.0, 2), 0,
         0, pow(deg2rad(20.0), 2);

    Eigen::Matrix2d Q_SIM;
    Q_SIM << pow(0.3, 2), 0,
             0, pow(deg2rad(2.0), 2);

    Eigen::Matrix2d R_SIM;
    R_SIM << pow(0.5, 2), 0,
             0, pow(deg2rad(10.0), 2);

    while (SIM_TIME >= time) {
        time += DT;
        Eigen::Vector2d u = calc_input(time);

        Eigen::MatrixXd z;
        observation(x_true, x_dr, u, rfid, z, Q_SIM, R_SIM, OFFSET_YAW_RATE_NOISE);

        predict_particles(particles, u, R);

        update_with_observation(particles, z, Q);

        resampling(particles);

        x_est = calc_final_state(particles);


        hist_x_est.push_back(x_est);
        hist_x_dr.push_back(x_dr);
        hist_x_true.push_back(x_true);

       
        std::cout << "Time: " << time << " x_est: " << x_est.transpose() << std::endl;
    }

 
    std::cout << "Final estimated position: " << x_est.transpose() << std::endl;

    return 0;
}
