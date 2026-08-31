#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "osqp.h"

struct Result {
    int N;
    int n;
    int m;
    int cold_iters;
    double cold_setup_ms;
    double cold_solve_ms;
    double cold_total_ms;
    int warm_iters;
    double warm_solve_ms;
    bool cold_success;
    bool warm_success;
};

Result run_osqp_mass_spring(int N)
{
    Result res;
    res.N = N;

    int nx = 4;
    int nu = 1;
    int n = (nx + nu) * N; // Total primal variables
    int p = nx * N;        // Equality constraints
    int m_ineq = N;        // Inequality bounds (-1 <= u_i <= 1)
    int m = p + m_ineq;    // Total constraints

    res.n = n;
    res.m = m;

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  OSQP Mass-Spring MPC Solver Run: Horizon N = " << N << " (Variables: " << n << ")" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Setup P matrix (Diagonal, so strictly upper triangular part is just the diagonal)
    Eigen::SparseMatrix<double, Eigen::ColMajor, OSQPInt> P(n, n);
    std::vector<Eigen::Triplet<double>> P_triplets;
    for (int i = 0; i < N; i++) {
        int start = (nx + nu) * i;
        P_triplets.push_back(Eigen::Triplet<double>(start + 0, start + 0, 10.0));
        P_triplets.push_back(Eigen::Triplet<double>(start + 1, start + 1, 1.0));
        P_triplets.push_back(Eigen::Triplet<double>(start + 2, start + 2, 10.0));
        P_triplets.push_back(Eigen::Triplet<double>(start + 3, start + 3, 1.0));
        P_triplets.push_back(Eigen::Triplet<double>(start + 4, start + 4, 0.1));
    }
    P.setFromTriplets(P_triplets.begin(), P_triplets.end());
    P.makeCompressed();

    Eigen::VectorXd q = Eigen::VectorXd::Zero(n);

    // Dynamics
    Eigen::MatrixXd A_sys(nx, nx);
    A_sys <<  1.0,  0.1,  0.0,  0.0,
             -0.1,  0.95, 0.1,  0.05,
              0.0,  0.0,  1.0,  0.1,
              0.1,  0.05, -0.1, 0.95;

    Eigen::VectorXd B_sys(nx);
    B_sys << 0.0, 0.0, 0.0, 0.1;

    Eigen::VectorXd x_init(nx);
    x_init << 1.0, 0.0, -0.5, 0.0;

    // Build full A matrix (m x n) combining equality dynamics and input bounds
    Eigen::SparseMatrix<double, Eigen::ColMajor, OSQPInt> A(m, n);
    std::vector<Eigen::Triplet<double>> A_triplets;

    // Stage 0 equality: x_0 = x_init
    for (int r = 0; r < nx; r++) {
        A_triplets.push_back(Eigen::Triplet<double>(r, r, 1.0));
    }

    // Stage k dynamics equality: x_k - A_sys*x_{k-1} - B_sys*u_{k-1} = 0
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
            A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + nx, -B_sys(r)));
        }
    }

    // Input bounds: -1 <= u_i <= 1
    for (int i = 0; i < N; i++) {
        int row = p + i;
        int col_u = i * (nx + nu) + nx;
        A_triplets.push_back(Eigen::Triplet<double>(row, col_u, 1.0));
    }

    A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    A.makeCompressed();

    // Bounds l and u
    Eigen::VectorXd l(m);
    Eigen::VectorXd u(m);

    Eigen::VectorXd b_eq1 = Eigen::VectorXd::Zero(p);
    b_eq1.head(nx) = x_init;
    l.head(p) = b_eq1;
    u.head(p) = b_eq1;

    l.tail(m_ineq).setConstant(-1.0);
    u.tail(m_ineq).setConstant(1.0);

    // ----------------------------------------------------
    // COLD START RUN
    // ----------------------------------------------------
    std::cout << "\n>>> COLD START (Horizon N = " << N << ") <<<" << std::endl;
    
    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);
    settings->verbose = 0;
    settings->eps_abs = 1e-5;
    settings->eps_rel = 1e-5;

    OSQPCscMatrix* P_osqp = OSQPCscMatrix_new(n, n, P.nonZeros(), (OSQPFloat*)P.valuePtr(), (OSQPInt*)P.innerIndexPtr(), (OSQPInt*)P.outerIndexPtr());
    OSQPCscMatrix* A_osqp = OSQPCscMatrix_new(m, n, A.nonZeros(), (OSQPFloat*)A.valuePtr(), (OSQPInt*)A.innerIndexPtr(), (OSQPInt*)A.outerIndexPtr());

    OSQPSolver* solver = NULL;

    auto t_start_setup = std::chrono::high_resolution_clock::now();
    OSQPInt exitflag = osqp_setup(&solver, P_osqp, q.data(), A_osqp, l.data(), u.data(), m, n, settings);
    auto t_end_setup = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur_setup = t_end_setup - t_start_setup;

    auto t_start_solve = std::chrono::high_resolution_clock::now();
    if (!exitflag) {
        exitflag = osqp_solve(solver);
    }
    auto t_end_solve = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur_solve = t_end_solve - t_start_solve;

    res.cold_success = (exitflag == 0 && solver->info->status_val == OSQP_SOLVED);
    res.cold_iters = solver->info->iter;
    res.cold_setup_ms = dur_setup.count();
    res.cold_solve_ms = dur_solve.count();
    res.cold_total_ms = res.cold_setup_ms + res.cold_solve_ms;

    std::cout << "Cold Start Performance Metrics:" << std::endl;
    std::cout << "  Status          = " << (res.cold_success ? "SOLVED" : "FAILED") << std::endl;
    std::cout << "  ADMM Iterations = " << res.cold_iters << std::endl;
    std::cout << "  Setup Time      = " << std::fixed << std::setprecision(4) << res.cold_setup_ms << " ms" << std::endl;
    std::cout << "  Solve Time      = " << std::fixed << std::setprecision(4) << res.cold_solve_ms << " ms" << std::endl;
    std::cout << "  Setup+Solve Time= " << std::fixed << std::setprecision(4) << res.cold_total_ms << " ms" << std::endl;

    // Save primal and dual solution for warm start
    std::vector<double> x_sol(n);
    std::vector<double> y_sol(m);
    for (int i = 0; i < n; i++) x_sol[i] = solver->solution->x[i];
    for (int i = 0; i < m; i++) y_sol[i] = solver->solution->y[i];

    // ----------------------------------------------------
    // WARM START RUN
    // ----------------------------------------------------
    std::cout << "\n>>> WARM START (Horizon N = " << N << ") <<<" << std::endl;
    
    Eigen::VectorXd x_init2(nx);
    x_init2 << 0.9, 0.0, -0.45, 0.0;

    Eigen::VectorXd b_eq2 = b_eq1;
    b_eq2.head(nx) = x_init2;

    Eigen::VectorXd l2 = l;
    Eigen::VectorXd u2 = u;
    l2.head(p) = b_eq2;
    u2.head(p) = b_eq2;

    // Update bounds and warm start vectors
    osqp_update_data_vec(solver, NULL, l2.data(), u2.data());
    osqp_warm_start(solver, x_sol.data(), y_sol.data());

    auto t_start_warm = std::chrono::high_resolution_clock::now();
    OSQPInt exitflag_warm = osqp_solve(solver);
    auto t_end_warm = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur_warm = t_end_warm - t_start_warm;

    res.warm_success = (exitflag_warm == 0 && solver->info->status_val == OSQP_SOLVED);
    res.warm_iters = solver->info->iter;
    res.warm_solve_ms = dur_warm.count();

    std::cout << "Warm Start Performance Metrics:" << std::endl;
    std::cout << "  Status          = " << (res.warm_success ? "SOLVED" : "FAILED") << std::endl;
    std::cout << "  ADMM Iterations = " << res.warm_iters << std::endl;
    std::cout << "  Solve Time      = " << std::fixed << std::setprecision(4) << res.warm_solve_ms << " ms" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Cleanup
    osqp_cleanup(solver);
    OSQPCscMatrix_free(P_osqp);
    OSQPCscMatrix_free(A_osqp);
    free(settings);

    return res;
}

int main()
{
    std::vector<int> N_values = {20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 6000, 8000, 10000};
    std::vector<Result> results;

    for (int N_val : N_values) {
        results.push_back(run_osqp_mass_spring(N_val));
    }

    std::cout << "\n>>> OSQP SCALING SUMMARY TABLE <<<" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "  N   | Variables (n) | Cold Iters | Cold Setup (ms) | Cold Solve (ms) | Warm Iters | Warm Solve (ms)" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
    for (const auto& r : results) {
        std::cout << std::setw(4) << r.N << " | "
                  << std::setw(13) << r.n << " | "
                  << std::setw(10) << r.cold_iters << " | "
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.cold_setup_ms << " | "
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.cold_solve_ms << " | "
                  << std::setw(10) << r.warm_iters << " | "
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.warm_solve_ms << std::endl;
    }
    std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;

    return 0;
}
