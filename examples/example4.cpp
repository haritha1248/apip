#include <iostream>
#include <limits>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <unsupported/Eigen/MatrixFunctions>
#include "ipm_admm_cg/admm_kkt_solver.hpp"

using namespace ipm_admm_cg;

static Eigen::VectorXd prev_x_4;
static Eigen::VectorXd prev_s_4;
static Eigen::VectorXd prev_y_4;
static Eigen::VectorXd prev_z_4;
static bool has_prev_4 = false;

//computes 6-mass, 7-spring, 3-actuator example
static void compute_wang_boyd_matrices(double dt, Eigen::MatrixXd& A_sys, Eigen::MatrixXd& B_sys)
{
    const int n_masses = 6;
    const int nx = 2 * n_masses;
    const int nu = 3;

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_masses, n_masses);
    for (int i = 0; i < n_masses; i++) {
        K(i, i) = 2.0;
        if (i > 0) K(i, i - 1) = -1.0;
        if (i < n_masses - 1) K(i, i + 1) = -1.0;
    }

    double damping = 0.1;
    Eigen::MatrixXd D = damping * Eigen::MatrixXd::Identity(n_masses, n_masses);

    Eigen::MatrixXd B_u = Eigen::MatrixXd::Zero(n_masses, nu);
    B_u(0, 0) =  1.0; B_u(1, 0) = -1.0;
    B_u(2, 1) =  1.0; B_u(3, 1) = -1.0;
    B_u(4, 2) =  1.0; B_u(5, 2) = -1.0;

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

//shift for horizon N
void run_mass_spring_for_N(int N, double& custom_time, int& custom_iters, int& total_cg_iters)
{
    const int nx = 12;
    const int nu = 3;
    const int n = (nx + nu) * N;
    const int p = nx * N;
    const int m = 2 * nu * N;

    Eigen::MatrixXd A_sys, B_sys;
    compute_wang_boyd_matrices(0.5, A_sys, B_sys);

    //setup P
    Eigen::SparseMatrix<double> P(n, n);
    std::vector<Eigen::Triplet<double>> P_triplets;
    for (int i = 0; i < N; i++) {
        int start = (nx + nu) * i;
        for (int mass = 0; mass < 6; mass++) {
            P_triplets.push_back(Eigen::Triplet<double>(start + 2 * mass + 0, start + 2 * mass + 0, 10.0));
            P_triplets.push_back(Eigen::Triplet<double>(start + 2 * mass + 1, start + 2 * mass + 1, 1.0));
        }
        for (int u = 0; u < nu; u++) {
            P_triplets.push_back(Eigen::Triplet<double>(start + nx + u, start + nx + u, 0.1));
        }
    }
    P.setFromTriplets(P_triplets.begin(), P_triplets.end());
    P.makeCompressed();

    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);

    Eigen::VectorXd x_init(nx);
    x_init << 1.0, 0.0, -0.5, 0.0, 0.5, 0.0, -0.2, 0.0, 0.3, 0.0, -0.1, 0.0;

    Eigen::SparseMatrix<double> A_eq(p, n);
    std::vector<Eigen::Triplet<double>> A_triplets;

    for (int r = 0; r < nx; r++) {
        A_triplets.push_back(Eigen::Triplet<double>(r, r, 1.0));
    }

    for (int k = 1; k < N; k++) {
        int row_offset = k * nx;
        int col_prev = (k - 1) * (nx + nu);
        int col_curr = k * (nx + nu);

        for (int r = 0; r < nx; r++) {
            A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_curr + r, 1.0));
        }
        for (int r = 0; r < nx; r++) {
            for (int col = 0; col < nx; col++) {
                A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + col, -A_sys(r, col)));
            }
        }
        for (int r = 0; r < nx; r++) {
            for (int col = 0; col < nu; col++) {
                A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + nx + col, -B_sys(r, col)));
            }
        }
    }
    A_eq.setFromTriplets(A_triplets.begin(), A_triplets.end());
    A_eq.makeCompressed();

    Eigen::VectorXd b_eq = Eigen::VectorXd::Zero(p);
    b_eq.head(nx) = x_init;

    Eigen::SparseMatrix<double> G_ineq(m, n);
    std::vector<Eigen::Triplet<double>> G_triplets;
    for (int i = 0; i < N; i++) {
        for (int u = 0; u < nu; u++) {
            int row_pos = i * (2 * nu) + u;
            int row_neg = i * (2 * nu) + nu + u;
            int col_u = i * (nx + nu) + nx + u;
            G_triplets.push_back(Eigen::Triplet<double>(row_pos, col_u,  1.0));
            G_triplets.push_back(Eigen::Triplet<double>(row_neg, col_u, -1.0));
        }
    }
    G_ineq.setFromTriplets(G_triplets.begin(), G_triplets.end());
    G_ineq.makeCompressed();

    Eigen::VectorXd h_ineq = Eigen::VectorXd::Constant(m, 0.5);

    //solve cold 
    ProximalIPMSolver solver(P, c, A_eq, b_eq, G_ineq, h_ineq);
    solver.set_settings(100, 1e-5, 0.15);
    solver.set_regularization(1e-8, 1e-8, 1e-8);
    solver.set_preconditioner_block_size(nx + nu);

    Eigen::VectorXd x_sol = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd s_sol = Eigen::VectorXd::Ones(m);
    Eigen::VectorXd y_sol = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd z_sol = Eigen::VectorXd::Ones(m);

    std::streambuf* orig_buf = std::cout.rdbuf();
    std::cout.rdbuf(NULL);

    int dummy_it = 0, dummy_cg = 0;
    solver.solve(x_sol, s_sol, y_sol, z_sol, dummy_it, dummy_cg, false, false);

    //shift init state
    Eigen::VectorXd b_eq2 = b_eq;
    Eigen::VectorXd x_init2 = 0.9 * x_init;
    b_eq2.head(nx) = x_init2;

    ProximalIPMSolver solver_warm(P, c, A_eq, b_eq2, G_ineq, h_ineq);
    solver_warm.set_settings(100, 1e-5, 0.15);
    solver_warm.set_regularization(1e-8, 1e-8, 1e-8);
    solver_warm.set_preconditioner_block_size(nx + nu);

    auto start_time = std::chrono::high_resolution_clock::now();
    bool status = solver_warm.solve(x_sol, s_sol, y_sol, z_sol, custom_iters, total_cg_iters, true, false);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    std::cout.rdbuf(orig_buf);

    if (status) {
        custom_time = duration.count();
    } else {
        custom_time = -1.0;
    }
}

void example4(bool warm_start = false)
{
    std::cout << "\nProximal IPM ADMM-CG QP Solver\n" << std::endl;
    std::cout << "Example 4\n" << std::endl;
    std::cout << "(6-Mass 7-Spring 3-Actuator MPC)\n" << std::endl;

    const int N = 20;
    const int nx = 12;
    const int nu = 3;
    const int n = (nx + nu) * N;
    const int p = nx * N;
    const int m = 2 * nu * N;

    std::cout << "Problem Dimensions:" << std::endl;
    std::cout << " Horizon Steps (N) = " << N << std::endl;
    std::cout << " Variables (n)     = " << n << std::endl;
    std::cout << " Equalities (p)    = " << p << std::endl;
    std::cout << " Inequalities (m)  = " << m << std::endl;

    Eigen::MatrixXd A_sys, B_sys;
    compute_wang_boyd_matrices(0.5, A_sys, B_sys);

    Eigen::SparseMatrix<double> P(n, n);
    std::vector<Eigen::Triplet<double>> P_triplets;
    for (int i = 0; i < N; i++) {
        int start = (nx + nu) * i;
        for (int mass = 0; mass < 6; mass++) {
            P_triplets.push_back(Eigen::Triplet<double>(start + 2 * mass + 0, start + 2 * mass + 0, 10.0));
            P_triplets.push_back(Eigen::Triplet<double>(start + 2 * mass + 1, start + 2 * mass + 1, 1.0));
        }
        for (int u = 0; u < nu; u++) {
            P_triplets.push_back(Eigen::Triplet<double>(start + nx + u, start + nx + u, 0.1));
        }
    }
    P.setFromTriplets(P_triplets.begin(), P_triplets.end());
    P.makeCompressed();

    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);

    Eigen::SparseMatrix<double> A_eq(p, n);
    std::vector<Eigen::Triplet<double>> A_triplets;

    Eigen::VectorXd x_init(nx);
    x_init << 1.0, 0.0, -0.5, 0.0, 0.5, 0.0, -0.2, 0.0, 0.3, 0.0, -0.1, 0.0;

    for (int r = 0; r < nx; r++) {
        A_triplets.push_back(Eigen::Triplet<double>(r, r, 1.0));
    }

    for (int k = 1; k < N; k++) {
        int row_offset = k * nx;
        int col_prev = (k - 1) * (nx + nu);
        int col_curr = k * (nx + nu);

        for (int r = 0; r < nx; r++) {
            A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_curr + r, 1.0));
        }
        for (int r = 0; r < nx; r++) {
            for (int col = 0; col < nx; col++) {
                A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + col, -A_sys(r, col)));
            }
        }
        for (int r = 0; r < nx; r++) {
            for (int col = 0; col < nu; col++) {
                A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + nx + col, -B_sys(r, col)));
            }
        }
    }
    A_eq.setFromTriplets(A_triplets.begin(), A_triplets.end());
    A_eq.makeCompressed();

    Eigen::VectorXd b_eq = Eigen::VectorXd::Zero(p);
    if (warm_start && has_prev_4) {
        b_eq.head(nx) = 0.9 * x_init;
    } else {
        b_eq.head(nx) = x_init;
    }

    Eigen::SparseMatrix<double> G_ineq(m, n);
    std::vector<Eigen::Triplet<double>> G_triplets;
    for (int i = 0; i < N; i++) {
        for (int u = 0; u < nu; u++) {
            int row_pos = i * (2 * nu) + u;
            int row_neg = i * (2 * nu) + nu + u;
            int col_u = i * (nx + nu) + nx + u;
            G_triplets.push_back(Eigen::Triplet<double>(row_pos, col_u,  1.0));
            G_triplets.push_back(Eigen::Triplet<double>(row_neg, col_u, -1.0));
        }
    }
    G_ineq.setFromTriplets(G_triplets.begin(), G_triplets.end());
    G_ineq.makeCompressed();

    Eigen::VectorXd h_ineq = Eigen::VectorXd::Constant(m, 0.5);

    ProximalIPMSolver solver(P, c, A_eq, b_eq, G_ineq, h_ineq);
    solver.set_settings(100, 1e-5, 0.15);
    solver.set_regularization(1e-8, 1e-8, 1e-8);
    solver.set_preconditioner_block_size(nx + nu);

    Eigen::VectorXd x_sol(n);
    Eigen::VectorXd s_sol(m);
    Eigen::VectorXd y_sol(p);
    Eigen::VectorXd z_sol(m);

    bool actual_warm = false;
    if (warm_start && has_prev_4) {
        x_sol = prev_x_4;
        s_sol = prev_s_4;
        y_sol = prev_y_4;
        z_sol = prev_z_4;
        actual_warm = true;
    } else {
        x_sol.setZero();
        s_sol.setOnes();
        y_sol.setZero();
        z_sol.setOnes();
    }

    int total_admm_iters = 0;
    int total_cg_iters = 0;

    std::cout << "\nRunning IPM-ADMM-CG solver..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    bool status = solver.solve(x_sol, s_sol, y_sol, z_sol, total_admm_iters, total_cg_iters, actual_warm, false);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    prev_x_4 = x_sol;
    prev_s_4 = s_sol;
    prev_y_4 = y_sol;
    prev_z_4 = z_sol;
    has_prev_4 = true;

    if (status) {
        std::cout << "Solver succeeded." << std::endl;
    } else {
        std::cout << "Solver failed." << std::endl;
    }

    std::cout << "Performance Metrics (IPM-ADMM-CG):" << std::endl;
    std::cout << " Total Inner ADMM Iterations = " << total_admm_iters << std::endl;
    std::cout << " Total Inner CG Iterations   = " << total_cg_iters << std::endl;
    std::cout << " Solve Execution Time        = " << duration.count() << " ms" << std::endl;

    if (warm_start)
    {
        std::cout << "\n SCALING TEST: SOLVER PERFORMANCE VS. HORIZON LENGTH (N) " << std::endl;
        std::cout << "  N   | Variables (n) | Custom Time (ms) | Inner CG Iters " << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;

        std::vector<int> N_values = {1000, 2000, 3000, 4000, 5000};
        for (int N_val : N_values) {
            double custom_t = 0.0;
            int custom_it = 0;
            int total_cg = 0;

            run_mass_spring_for_N(N_val, custom_t, custom_it, total_cg);

            if (custom_t > 0) {
                printf(" %4d |     %6d    |      %7.2f     |      %4d     \n",
                       N_val, (nx + nu) * N_val, custom_t, total_cg);
            } else {
                printf(" %4d |     %6d    |      FAILED      |      FAILED   \n",
                       N_val, (nx + nu) * N_val);
            }
        }
        std::cout << "--------------------------------------------------------------" << std::endl;
    }
}
