#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "piqp/piqp.hpp"

struct PIQPResult {
    int N;
    int n;
    int iters;
    double setup_solve_time_ms;
    double internal_time_ms;
    int status;
};

PIQPResult run_piqp_mass_spring(int N)
{
    PIQPResult res;
    res.N = N;

    int nx = 4;
    int nu = 1;
    int n = (nx + nu) * N;
    int p = nx * N;
    int m = 2 * nu * N;

    res.n = n;

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  PIQP Mass-Spring MPC Solver Run: Horizon N = " << N << " (Variables: " << n << ")" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Setup P matrix
    Eigen::SparseMatrix<double> P(n, n);
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

    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);

    // Dynamics
    Eigen::MatrixXd A_sys(nx, nx);
    A_sys <<  1.0,  0.1,  0.0,  0.0,
             -0.1,  0.95, 0.1,  0.05,
              0.0,  0.0,  1.0,  0.1,
              0.1,  0.05, -0.1, 0.95;

    Eigen::VectorXd B_sys(nx);
    B_sys << 0.0, 0.0, 0.0, 0.1;

    Eigen::SparseMatrix<double> A_eq(p, n);
    std::vector<Eigen::Triplet<double>> A_triplets;

    Eigen::VectorXd x_init(nx);
    x_init << 1.0, 0.0, -0.5, 0.0;

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
            A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + nx, -B_sys(r)));
        }
    }
    A_eq.setFromTriplets(A_triplets.begin(), A_triplets.end());
    A_eq.makeCompressed();

    Eigen::VectorXd b_eq = Eigen::VectorXd::Zero(p);
    b_eq.head(nx) = x_init;

    // G matrix
    Eigen::SparseMatrix<double> G_ineq(m, n);
    std::vector<Eigen::Triplet<double>> G_triplets;
    for (int i = 0; i < N; i++) {
        int col_u = i * (nx + nu) + nx;
        G_triplets.push_back(Eigen::Triplet<double>(i, col_u, 1.0));
        G_triplets.push_back(Eigen::Triplet<double>(N + i, col_u, -1.0));
    }
    G_ineq.setFromTriplets(G_triplets.begin(), G_triplets.end());
    G_ineq.makeCompressed();

    Eigen::VectorXd h_ineq = Eigen::VectorXd::Ones(m);

    // PIQP Sparse Solver (Default settings)
    piqp::SparseSolver<double> solver;
    solver.settings().verbose = false;
    solver.settings().compute_timings = true;

    auto start_all = std::chrono::high_resolution_clock::now();
    solver.setup(P, c, A_eq, b_eq, G_ineq, piqp::nullopt, h_ineq, piqp::nullopt, piqp::nullopt);
    piqp::Status status = solver.solve();
    auto end_all = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur_all = end_all - start_all;

    res.status = (int)status;
    res.iters = solver.result().info.iter;
    res.setup_solve_time_ms = dur_all.count();
    res.internal_time_ms = (solver.result().info.setup_time + solver.result().info.solve_time) * 1000.0;

    std::cout << "PIQP Performance Metrics:" << std::endl;
    std::cout << "  Status          = " << res.status << std::endl;
    std::cout << "  IPM Iterations  = " << res.iters << std::endl;
    std::cout << "  Setup+Solve Time= " << std::fixed << std::setprecision(4) << res.setup_solve_time_ms << " ms" << std::endl;
    std::cout << "  Internal Time   = " << std::fixed << std::setprecision(4) << res.internal_time_ms << " ms" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return res;
}

int main()
{
    std::vector<int> N_values = {20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 6000, 8000, 10000};
    std::vector<PIQPResult> results;

    for (int N_val : N_values) {
        results.push_back(run_piqp_mass_spring(N_val));
    }

    std::cout << "\n>>> PIQP SCALING SUMMARY TABLE <<<" << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    std::cout << "  N  |  Variables (n)  |  Setup+Solve Time (ms)  |  Internal Time (ms)  |  Iters" << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    for (const auto& r : results) {
        std::cout << std::setw(4) << r.N << " | "
                  << std::setw(14) << r.n << " | "
                  << std::setw(23) << std::fixed << std::setprecision(4) << r.setup_solve_time_ms << " | "
                  << std::setw(20) << std::fixed << std::setprecision(4) << r.internal_time_ms << " | "
                  << std::setw(6) << r.iters << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------------" << std::endl;

    return 0;
}
