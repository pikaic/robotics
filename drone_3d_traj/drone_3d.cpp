
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>


const double g = 9.81;
const double m = 0.2;
const double Ixx = 1.0;
const double Iyy = 1.0;
const double Izz = 1.0;
const double T = 5.0;


const double Kp_x = 1.0;
const double Kp_y = 1.0;
const double Kp_z = 1.0;
const double Kp_roll = 25.0;
const double Kp_pitch = 25.0;
const double Kp_yaw = 25.0;

const double Kd_x = 10.0;
const double Kd_y = 10.0;
const double Kd_z = 1.0;

class Quadrotor {
public:
    Quadrotor(double x, double y, double z, double roll, double pitch, double yaw, double size, bool show_animation) {
       
    }

    void update_pose(double x, double y, double z, double roll, double pitch, double yaw) {
        
        std::cout << "Position: (" << x << ", " << y << ", " << z << "), Orientation: ("
                  << roll << ", " << pitch << ", " << yaw << ")" << std::endl;
    }
};

double calculate_position(const std::vector<double>& c, double t) {
    return c[0] * pow(t, 5) + c[1] * pow(t, 4) + c[2] * pow(t, 3) + c[3] * pow(t, 2) + c[4] * t + c[5];
}

double calculate_velocity(const std::vector<double>& c, double t) {
    return 5 * c[0] * pow(t, 4) + 4 * c[1] * pow(t, 3) + 3 * c[2] * pow(t, 2) + 2 * c[3] * t + c[4];
}

double calculate_acceleration(const std::vector<double>& c, double t) {
    return 20 * c[0] * pow(t, 3) + 12 * c[1] * pow(t, 2) + 6 * c[2] * t + 2 * c[3];
}

Eigen::Matrix3d rotation_matrix(double roll, double pitch, double yaw) {
    Eigen::Matrix3d R;
    R(0, 0) = cos(yaw) * cos(pitch);
    R(0, 1) = -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll);
    R(0, 2) = sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll);
    R(1, 0) = sin(yaw) * cos(pitch);
    R(1, 1) = cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll);
    R(1, 2) = -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll);
    R(2, 0) = -sin(pitch);
    R(2, 1) = cos(pitch) * sin(roll);
    R(2, 2) = cos(pitch) * cos(roll);
    return R;
}

class TrajectoryGenerator {
public:
    TrajectoryGenerator(const std::vector<double>& start, const std::vector<double>& end, double T) {
        this->start = start;
        this->end = end;
        this->T = T;
    }

    void solve() {
        x_c = compute_coefficients(start[0], end[0]);
        y_c = compute_coefficients(start[1], end[1]);
        z_c = compute_coefficients(start[2], end[2]);
    }

    std::vector<double> getXCoefficients() const {
        return x_c;
    }

    std::vector<double> getYCoefficients() const {
        return y_c;
    }

    std::vector<double> getZCoefficients() const {
        return z_c;
    }

private:
    std::vector<double> start;
    std::vector<double> end;
    double T;
    std::vector<double> x_c;
    std::vector<double> y_c;
    std::vector<double> z_c;

    std::vector<double> compute_coefficients(double s0, double sT) {
      
        double v0 = 0.0;
        double vT = 0.0;
        double a0 = 0.0;
        double aT = 0.0;

        Eigen::MatrixXd A(6, 6);
        Eigen::VectorXd b(6);

        A(0, 0) = pow(0, 5);  A(0, 1) = pow(0, 4);  A(0, 2) = pow(0, 3);  A(0, 3) = pow(0, 2);  A(0, 4) = pow(0, 1);  A(0, 5) = 1;
        A(1, 0) = 5 * pow(0, 4);  A(1, 1) = 4 * pow(0, 3);  A(1, 2) = 3 * pow(0, 2);  A(1, 3) = 2 * pow(0, 1);  A(1, 4) = 1;  A(1, 5) = 0;
        A(2, 0) = 20 * pow(0, 3);  A(2, 1) = 12 * pow(0, 2);  A(2, 2) = 6 * pow(0, 1);  A(2, 3) = 2;  A(2, 4) = 0;  A(2, 5) = 0;

        
        A(3, 0) = pow(T, 5);  A(3, 1) = pow(T, 4);  A(3, 2) = pow(T, 3);  A(3, 3) = pow(T, 2);  A(3, 4) = T;  A(3, 5) = 1;
        A(4, 0) = 5 * pow(T, 4);  A(4, 1) = 4 * pow(T, 3);  A(4, 2) = 3 * pow(T, 2);  A(4, 3) = 2 * T;  A(4, 4) = 1;  A(4, 5) = 0;
        A(5, 0) = 20 * pow(T, 3);  A(5, 1) = 12 * pow(T, 2);  A(5, 2) = 6 * T;  A(5, 3) = 2;  A(5, 4) = 0;  A(5, 5) = 0;

        b << s0, v0, a0, sT, vT, aT;

       
        Eigen::VectorXd c = A.fullPivLu().solve(b);

        std::vector<double> coeffs(6);
        for (int i = 0; i < 6; ++i) {
            coeffs[i] = c(i);
        }

        return coeffs;
    }
};

void quad_sim(const std::vector<std::vector<double>>& x_c,
              const std::vector<std::vector<double>>& y_c,
              const std::vector<std::vector<double>>& z_c) {
    double x_pos = -5.0;
    double y_pos = -5.0;
    double z_pos = 5.0;
    double x_vel = 0.0;
    double y_vel = 0.0;
    double z_vel = 0.0;
    double x_acc = 0.0;
    double y_acc = 0.0;
    double z_acc = 0.0;
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    double roll_vel = 0.0;
    double pitch_vel = 0.0;
    double yaw_vel = 0.0;

    double des_yaw = 0.0;

    double dt = 0.1;
    double t = 0.0;

    bool show_animation = true;

    Quadrotor q(x_pos, y_pos, z_pos, roll, pitch, yaw, 1.0, show_animation);

    int i = 0;
    int n_run = 8;
    int irun = 0;

    while (true) {
        while (t <= T) {
            double des_z_pos = calculate_position(z_c[i], t);
            double des_z_vel = calculate_velocity(z_c[i], t);
            double des_x_acc = calculate_acceleration(x_c[i], t);
            double des_y_acc = calculate_acceleration(y_c[i], t);
            double des_z_acc = calculate_acceleration(z_c[i], t);

            double thrust = m * (g + des_z_acc + Kp_z * (des_z_pos - z_pos) + Kd_z * (des_z_vel - z_vel));

            double roll_torque = Kp_roll * (((des_x_acc * sin(des_yaw) - des_y_acc * cos(des_yaw)) / g) - roll);
            double pitch_torque = Kp_pitch * (((des_x_acc * cos(des_yaw) - des_y_acc * sin(des_yaw)) / g) - pitch);
            double yaw_torque = Kp_yaw * (des_yaw - yaw);

            roll_vel += roll_torque * dt / Ixx;
            pitch_vel += pitch_torque * dt / Iyy;
            yaw_vel += yaw_torque * dt / Izz;

            roll += roll_vel * dt;
            pitch += pitch_vel * dt;
            yaw += yaw_vel * dt;

            Eigen::Matrix3d R = rotation_matrix(roll, pitch, yaw);

            Eigen::Vector3d thrust_vector(0, 0, thrust);
            Eigen::Vector3d gravity_vector(0, 0, m * g);
            Eigen::Vector3d acc = (R * thrust_vector - gravity_vector) / m;
            x_acc = acc(0);
            y_acc = acc(1);
            z_acc = acc(2);

            x_vel += x_acc * dt;
            y_vel += y_acc * dt;
            z_vel += z_acc * dt;
            x_pos += x_vel * dt;
            y_pos += y_vel * dt;
            z_pos += z_vel * dt;

            q.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw);

            t += dt;
        }

        t = 0.0;
        i = (i + 1) % 4;
        irun += 1;
        if (irun >= n_run) {
            break;
        }
    }

    std::cout << "Done" << std::endl;
}

int main() {
    std::vector<std::vector<double>> x_coeffs(4);
    std::vector<std::vector<double>> y_coeffs(4);
    std::vector<std::vector<double>> z_coeffs(4);
    std::vector<std::vector<double>> waypoints = {
        { -5.0, -5.0, 5.0 },
        { 5.0, -5.0, 5.0 },
        { 5.0, 5.0, 5.0 },
        { -5.0, 5.0, 5.0 }
    };

    for (int i = 0; i < 4; ++i) {
        TrajectoryGenerator traj(waypoints[i], waypoints[(i + 1) % 4], T);
        traj.solve();
        x_coeffs[i] = traj.getXCoefficients();
        y_coeffs[i] = traj.getYCoefficients();
        z_coeffs[i] = traj.getZCoefficients();
    }

    quad_sim(x_coeffs, y_coeffs, z_coeffs);

    return 0;
}
