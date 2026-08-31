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

// Computes exact ZOH discrete dynamics for 6-mass, 7-spring, 3-actuator system
void compute_wang_boyd_discrete_matrices(double dt,
                                         Eigen::MatrixXd& A_sys,
                                         Eigen::MatrixXd& B_sys)
{
    const int n_masses = 6;
    const int nx = 2 * n_masses; // 12 states: [p1, v1, p2, v2, p3, v3, p4, v4, p5, v5, p6, v6]
    const int nu = 3;            // 3 actuators (tensions between pairs of masses)

    // Stiffness matrix K (6x6 tridiagonal for 6 masses and 7 springs to walls)
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_masses, n_masses);
    for (int i = 0; i < n_masses; i++) {
        K(i, i) = 2.0;
        if (i > 0) K(i, i - 1) = -1.0;
        if (i < n_masses - 1) K(i, i + 1) = -1.0;
    }

    // Damping matrix D
    double damping = 0.1;
    Eigen::MatrixXd D = damping * Eigen::MatrixXd::Identity(n_masses, n_masses);

    // Actuator placement matrix B_u (6x3): tension forces on adjacent mass pairs
    // u1 acts between masses 1 and 2 (+u1 on mass 1, -u1 on mass 2)
    // u2 acts between masses 3 and 4 (+u2 on mass 3, -u2 on mass 4)
    // u3 acts between masses 5 and 6 (+u3 on mass 5, -u3 on mass 6)
    Eigen::MatrixXd B_u = Eigen::MatrixXd::Zero(n_masses, nu);
    B_u(0, 0) =  1.0; B_u(1, 0) = -1.0;
    B_u(2, 1) =  1.0; B_u(3, 1) = -1.0;
    B_u(4, 2) =  1.0; B_u(5, 2) = -1.0;

    // Continuous state matrix A_c and B_c for interleaved state [p1, v1, p2, v2, ...]
    Eigen::MatrixXd A_c = Eigen::MatrixXd::Zero(nx, nx);
    Eigen::MatrixXd B_c = Eigen::MatrixXd::Zero(nx, nu);

    for (int i = 0; i < n_masses; i++) {
        int pos_idx = 2 * i;
        int vel_idx = 2 * i + 1;

        // dp_i / dt = v_i
        A_c(pos_idx, vel_idx) = 1.0;

        // dv_i / dt = sum_j (-K_ij * p_j - D_ij * v_j) + sum_k B_u_ik * u_k
        for (int j = 0; j < n_masses; j++) {
            A_c(vel_idx, 2 * j) += -K(i, j);
            A_c(vel_idx, 2 * j + 1) += -D(i, j);
        }

        for (int k = 0; k < nu; k++) {
            B_c(vel_idx, k) = B_u(i, k);
        }
    }

    // Exact Van Loan matrix exponential discretization:
    // M = [ A_c   B_c ] * dt
    //     [  0     0  ]
    // exp(M) = [ A_sys  B_sys ]
    //          [   0      I   ]
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

    std::cout << "Wang-Boyd 6-mass 7-spring 3-actuator matrices computed successfully!" << std::endl;
    std::cout << "A_sys dimensions: " << A_sys.rows() << "x" << A_sys.cols() << std::endl;
    std::cout << "B_sys dimensions: " << B_sys.rows() << "x" << B_sys.cols() << std::endl;
    std::cout << "A_sys (top 4x4):\n" << A_sys.block(0, 0, 4, 4) << std::endl;
    std::cout << "B_sys (top 4x3):\n" << B_sys.block(0, 0, 4, 3) << std::endl;

    return 0;
}
