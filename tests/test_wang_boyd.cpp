#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

#include "ipm_admm_cg/admm_kkt_solver.hpp"

using namespace ipm_admm_cg;
//6 mass, 7 spring 3 actuator system
void compute_wang_boyd_discrete_matrices(double dt,
                                         Eigen::MatrixXd& A_sys,
                                         Eigen::MatrixXd& B_sys)
{
    const int n_masses = 6;
    const int nx = 2 * n_masses; //12 states
    const int nu = 3;            //3 actuators
    
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_masses, n_masses);
    for (int i = 0; i < n_masses; i++) {
        K(i, i) = 2.0;
        if (i > 0) K(i, i - 1) = -1.0;
        if (i < n_masses - 1) K(i, i + 1) = -1.0;
    }

    //damping
    double damping = 0.1;
    Eigen::MatrixXd D = damping * Eigen::MatrixXd::Identity(n_masses, n_masses);

    //tension forces on adjacent mass pairs
    Eigen::MatrixXd B_u = Eigen::MatrixXd::Zero(n_masses, nu);
    B_u(0, 0) =  1.0; B_u(1, 0) = -1.0;
    B_u(2, 1) =  1.0; B_u(3, 1) = -1.0;
    B_u(4, 2) =  1.0; B_u(5, 2) = -1.0;

    //state matrix
    Eigen::MatrixXd A_c = Eigen::MatrixXd::Zero(nx, nx);
    Eigen::MatrixXd B_c = Eigen::MatrixXd::Zero(nx, nu);

    for (int i = 0; i < n_masses; i++) {
        int pos_idx = 2 * i;
        int vel_idx = 2 * i + 1;

        A_c(pos_idx, vel_idx) = 1.0;

        for (int j = 0; j < n_masses; j++) {
            A_c(vel_idx, 2 * j) += -K(i, j);
            A_c(vel_idx, 2 * j + 1) += -D(i, j);
        }

        for (int k = 0; k < nu; k++) {
            B_c(vel_idx, k) = B_u(i, k);
        }
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nx + nu, nx + nu);
    M.block(0, 0, nx, nx) = A_c * dt;
    M.block(0, nx, nx, nu) = B_c * dt;

    Eigen::MatrixXd expM = M.exp();
    A_sys = expM.block(0, 0, nx, nx);
    B_sys = expM.block(0, nx, nx, nu);
}

int main() {
    Eigen::MatrixXd A_sys, B_sys;
    compute_wang_boyd_discrete_matrices(0.5, A_sys, B_sys);

    std::cout << "Wang-Boyd matrices computed successfully" << std::endl;
    std::cout << "A_sys dimensions: " << A_sys.rows() << "x" << A_sys.cols() << std::endl;
    std::cout << "B_sys dimensions: " << B_sys.rows() << "x" << B_sys.cols() << std::endl;
    std::cout << "A_sys (top 4x4):\n" << A_sys.block(0, 0, 4, 4) << std::endl;
    std::cout << "B_sys (top 4x3):\n" << B_sys.block(0, 0, 4, 3) << std::endl;

    return 0;
}
