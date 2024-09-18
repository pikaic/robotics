

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <random>
#include <limits>
#include <ctime>

using namespace std;
using namespace Eigen;

const double EPS = 0.0001;
const int MAX_ITER = 100;
bool show_animation = false;  

MatrixXd update_homogeneous_matrix(const MatrixXd& Hin, const MatrixXd& R, const VectorXd& T) {
    int r_size = R.rows();
    MatrixXd H = MatrixXd::Zero(r_size + 1, r_size + 1);

    H.topLeftCorner(r_size, r_size) = R;
    H.topRightCorner(r_size, 1) = T;
    H(r_size, r_size) = 1.0;

    if (Hin.size() == 0) {
        return H;
    } else {
        return Hin * H;
    }
}

void nearest_neighbor_association(const MatrixXd& previous_points, const MatrixXd& current_points, VectorXi& indexes, double& error) {
    int n_prev = previous_points.cols();
    int n_curr = current_points.cols();

    error = 0.0;
    indexes.resize(n_curr);

    for (int i = 0; i < n_curr; ++i) {
        VectorXd curr_point = current_points.col(i);
      
        VectorXd distances(n_prev);
        for (int j = 0; j < n_prev; ++j) {
            VectorXd prev_point = previous_points.col(j);
            distances(j) = (curr_point - prev_point).norm();
        }
       
        double min_distance;
        distances.minCoeff(&indexes(i), &min_distance);
        error += min_distance;
    }
}

void svd_motion_estimation(const MatrixXd& previous_points, const MatrixXd& current_points, MatrixXd& R, VectorXd& t) {
    VectorXd pm = previous_points.rowwise().mean();
    VectorXd cm = current_points.rowwise().mean();

    MatrixXd p_shift = previous_points.colwise() - pm;
    MatrixXd c_shift = current_points.colwise() - cm;

    MatrixXd W = c_shift * p_shift.transpose();
    JacobiSVD<MatrixXd> svd(W, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    R = V * U.transpose();
    t = pm - R * cm;
}

void icp_matching(const MatrixXd& previous_points, MatrixXd& current_points, MatrixXd& R_final, VectorXd& T_final) {
    MatrixXd H;  
    double dError = std::numeric_limits<double>::infinity();
    double preError = std::numeric_limits<double>::infinity();
    int count = 0;

    while (dError >= EPS) {
        count++;

    

        VectorXi indexes;
        double error;
        nearest_neighbor_association(previous_points, current_points, indexes, error);

        MatrixXd Rt;
    
        MatrixXd previous_points_matched(previous_points.rows(), indexes.size());
        for (int i = 0; i < indexes.size(); ++i) {
            previous_points_matched.col(i) = previous_points.col(indexes(i));
        }

        svd_motion_estimation(previous_points_matched, current_points, Rt, Tt);

     
        current_points = (Rt * current_points).colwise() + Tt;

        dError = preError - error;
        cout << "Residual: " << error << endl;

        if (dError < 0) {
            cout << "Not Converge... " << preError << ", " << dError << ", " << count << endl;
            break;
        }

        preError = error;
        H = update_homogeneous_matrix(H, Rt, Tt);

        if (dError <= EPS) {
            cout << "Converge " << error << ", " << dError << ", " << count << endl;
            break;
        } else if (MAX_ITER <= count) {
            cout << "Not Converge... " << error << ", " << dError << ", " << count << endl;
            break;
        }
    }

    R_final = H.topLeftCorner(H.rows() - 1, H.cols() - 1);
    T_final = H.topRightCorner(H.rows() - 1, 1);
}

void main_icp_2d() {
    cout << "ICP 2D point set matching" << endl;


    int nPoint = 1000;
    double fieldLength = 50.0;
    vector<double> motion = {0.5, 2.0, -10.0 * M_PI / 180.0};  

    int nsim = 3;  

    for (int sim = 0; sim < nsim; ++sim) {

        MatrixXd previous_points(2, nPoint);
        for (int i = 0; i < nPoint; ++i) {
            previous_points(0, i) = ((double)rand() / RAND_MAX - 0.5) * fieldLength;
            previous_points(1, i) = ((double)rand() / RAND_MAX - 0.5) * fieldLength;
        }


        MatrixXd current_points(2, nPoint);
        double cos_theta = cos(motion[2]);
        double sin_theta = sin(motion[2]);
        for (int i = 0; i < nPoint; ++i) {
            double x = previous_points(0, i);
            double y = previous_points(1, i);
            current_points(0, i) = cos_theta * x - sin_theta * y + motion[0];
            current_points(1, i) = sin_theta * x + cos_theta * y + motion[1];
        }


        MatrixXd R;
        VectorXd T;
        icp_matching(previous_points, current_points, R, T);

        cout << "R:" << endl << R << endl;
        cout << "T:" << endl << T.transpose() << endl;
    }
}

void main_icp_3d() {
    cout << "ICP 3D point set matching" << endl;


    int nPoint = 1000;
    double fieldLength = 50.0;
    vector<double> motion = {0.5, 2.0, -5.0, -10.0 * M_PI / 180.0};  

    int nsim = 3;  
    for (int sim = 0; sim < nsim; ++sim) {
     
        MatrixXd previous_points(3, nPoint);
        for (int i = 0; i < nPoint; ++i) {
            previous_points(0, i) = ((double)rand() / RAND_MAX - 0.5) * fieldLength;
            previous_points(1, i) = ((double)rand() / RAND_MAX - 0.5) * fieldLength;
            previous_points(2, i) = ((double)rand() / RAND_MAX - 0.5) * fieldLength;
        }

        MatrixXd current_points(3, nPoint);
        double cos_theta = cos(motion[3]);
        double sin_theta = sin(motion[3]);
        for (int i = 0; i < nPoint; ++i) {
            double x = previous_points(0, i);
            double y = previous_points(1, i);
            double z = previous_points(2, i);
            current_points(0, i) = cos_theta * x - sin_theta * z + motion[0];
            current_points(1, i) = y + motion[1];
            current_points(2, i) = sin_theta * x + cos_theta * z + motion[2];
        }

        MatrixXd R;
        VectorXd T;
        icp_matching(previous_points, current_points, R, T);

        cout << "R:" << endl << R << endl;
        cout << "T:" << endl << T.transpose() << endl;
    }
}

int main() {
    srand(time(NULL)); 
    main_icp_2d();
    main_icp_3d();
    return 0;
}
