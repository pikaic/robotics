#include <vector>
#include <Eigen/Dense>

class TrajectoryGenerator {
public:
    TrajectoryGenerator(const std::vector<double>& start_pos, const std::vector<double>& des_pos, double T,
                        const std::vector<double>& start_vel = {0.0, 0.0, 0.0},
                        const std::vector<double>& des_vel = {0.0, 0.0, 0.0},
                        const std::vector<double>& start_acc = {0.0, 0.0, 0.0},
                        const std::vector<double>& des_acc = {0.0, 0.0, 0.0})
        : T(T)
    {
       
        start_x = start_pos[0];
        start_y = start_pos[1];
        start_z = start_pos[2];

        des_x = des_pos[0];
        des_y = des_pos[1];
        des_z = des_pos[2];

 
        start_x_vel = start_vel[0];
        start_y_vel = start_vel[1];
        start_z_vel = start_vel[2];

        des_x_vel = des_vel[0];
        des_y_vel = des_vel[1];
        des_z_vel = des_vel[2];

        start_x_acc = start_acc[0];
        start_y_acc = start_acc[1];
        start_z_acc = start_acc[2];

        des_x_acc = des_acc[0];
        des_y_acc = des_acc[1];
        des_z_acc = des_acc[2];
    }

    void solve() {

        Eigen::MatrixXd A(6, 6);
        A << 0,          0,          0,         0,       0, 1,
             pow(T, 5),  pow(T, 4),  pow(T, 3), pow(T, 2), T, 1,
             0,          0,          0,         0,       1, 0,
             5 * pow(T, 4), 4 * pow(T, 3), 3 * pow(T, 2), 2 * T, 1, 0,
             0,          0,          0,         2,       0, 0,
             20 * pow(T, 3), 12 * pow(T, 2), 6 * T, 2,   0, 0;

        Eigen::VectorXd b_x(6);
        b_x << start_x, des_x, start_x_vel, des_x_vel, start_x_acc, des_x_acc;

        Eigen::VectorXd b_y(6);
        b_y << start_y, des_y, start_y_vel, des_y_vel, start_y_acc, des_y_acc;

        Eigen::VectorXd b_z(6);

        x_c = A.colPivHouseholderQr().solve(b_x);
        y_c = A.colPivHouseholderQr().solve(b_y);
        z_c = A.colPivHouseholderQr().solve(b_z);
    }

    Eigen::VectorXd getXCoefficients() const {
        return x_c;
    }

    Eigen::VectorXd getYCoefficients() const {
        return y_c;
    }

    Eigen::VectorXd getZCoefficients() const {
        return z_c;
    }

private:
    double start_x, start_y, start_z;
    double des_x, des_y, des_z;
    double start_x_vel, start_y_vel, start_z_vel;
    double des_x_vel, des_y_vel, des_z_vel;
    double start_x_acc, start_y_acc, start_z_acc;
    double des_x_acc, des_y_acc, des_z_acc;
    double T;

    Eigen::VectorXd x_c;
    Eigen::VectorXd y_c;
    Eigen::VectorXd z_c;
};

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
        Eigen::VectorXd x_c = traj.getXCoefficients();
        Eigen::VectorXd y_c = traj.getYCoefficients();
        Eigen::VectorXd z_c = traj.getZCoefficients();

      
        x_coeffs[i] = std::vector<double>(x_c.data(), x_c.data() + x_c.size());
        y_coeffs[i] = std::vector<double>(y_c.data(), y_c.data() + y_c.size());
        z_coeffs[i] = std::vector<double>(z_c.data(), z_c.data() + z_c.size());
    }

    quad_sim(x_coeffs, y_coeffs, z_coeffs);

    return 0;
}