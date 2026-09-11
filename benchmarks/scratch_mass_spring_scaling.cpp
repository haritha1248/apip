#include <iostream>
#include <limits>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include "ipm_admm_cg/admm_kkt_solver.hpp"

using namespace ipm_admm_cg;

struct CustomResult {
    int N;
    int n;
    int cold_iters;
    int cold_cg;
    double cold_time_ms;
    int warm_iters;
    int warm_cg;
    double warm_time_ms;
};

void run_verbose_mass_spring(int N, CustomResult& res)
{
    int nx = 4;
    int nu = 1;
    int n = (nx + nu) * N;
    int p = nx * N;
    int m = 2 * nu * N;

    res.N = N;
    res.n = n;

    std::cout << "\n" << std::endl;
    std::cout << "IPM-ADMM-CG Mass-Spring MPC Run: Horizon N = " << N << " (Variables: " << n << ")" << std::endl;
    std::cout << "\n" << std::endl;

    //setup P
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

    Eigen::VectorXd b_eq1 = Eigen::VectorXd::Zero(p);
    b_eq1.head(nx) = x_init;

    //setup G
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

    //outputs
    Eigen::VectorXd x_sol = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd s_sol = Eigen::VectorXd::Ones(m);
    Eigen::VectorXd y_sol = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd z_sol = Eigen::VectorXd::Ones(m);

    // COLD START RUN
    std::cout << "\n Cold Start (Horizon N = " << N << ")" << std::endl;
    ProximalIPMSolver solver_cold(P, c, A_eq, b_eq1, G_ineq, h_ineq);
    solver_cold.set_settings(100, 1e-5, 0.15);
    solver_cold.set_regularization(1e-8, 1e-8, 1e-8);
    solver_cold.set_preconditioner_block_size(nx + nu);

    int cold_admm = 0;
    int cold_cg = 0;
    auto start_cold = std::chrono::high_resolution_clock::now();
    bool success_cold = solver_cold.solve(x_sol, s_sol, y_sol, z_sol, cold_admm, cold_cg, false, false);
    auto end_cold = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_cold = end_cold - start_cold;

    res.cold_iters = cold_admm;
    res.cold_cg = cold_cg;
    res.cold_time_ms = dur_cold.count();

    std::cout << "Cold Start Performance:" << std::endl;
    std::cout << " Status          = " << (success_cold ? "SOLVED" : "FAILED") << std::endl;
    std::cout << " IPM Iterations  = " << cold_admm << std::endl;
    std::cout << " CG Iterations   = " << cold_cg << std::endl;
    std::cout << " Solve Time      = " << std::fixed << std::setprecision(4) << dur_cold.count() << " ms" << std::endl;

    // WARM START RUN
    std::cout << "\n Warm Start (Horizon N = " << N << ") " << std::endl;
    Eigen::VectorXd b_eq2 = b_eq1;
    Eigen::VectorXd x_init2(nx);
    x_init2 << 0.9, 0.0, -0.45, 0.0;
    b_eq2.head(nx) = x_init2;

    ProximalIPMSolver solver_warm(P, c, A_eq, b_eq2, G_ineq, h_ineq);
    solver_warm.set_settings(100, 1e-5, 0.15);
    solver_warm.set_regularization(1e-8, 1e-8, 1e-8);
    solver_warm.set_preconditioner_block_size(nx + nu);

    int warm_admm = 0;
    int warm_cg = 0;
    auto start_warm = std::chrono::high_resolution_clock::now();
    bool success_warm = solver_warm.solve(x_sol, s_sol, y_sol, z_sol, warm_admm, warm_cg, true, false);
    auto end_warm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_warm = end_warm - start_warm;

    res.warm_iters = warm_admm;
    res.warm_cg = warm_cg;
    res.warm_time_ms = dur_warm.count();

    std::cout << "Warm Start Performance:" << std::endl;
    std::cout << " Status          = " << (success_warm ? "SOLVED" : "FAILED") << std::endl;
    std::cout << " IPM Iterations  = " << warm_admm << std::endl;
    std::cout << " CG Iterations   = " << warm_cg << std::endl;
    std::cout << " Solve Time      = " << std::fixed << std::setprecision(4) << dur_warm.count() << " ms" << std::endl;
    std::cout << "\n" << std::endl;
}

int main()
{
    std::vector<int> N_values = {20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 6000, 8000, 10000};
    std::vector<CustomResult> results;
    for (int N_val : N_values) {
        CustomResult r;
        run_verbose_mass_spring(N_val, r);
        results.push_back(r);
    }

    std::cout << "\nSCALING SUMMARY TABLE " << std::endl;
    std::cout << "  N   | Variables (n) | Cold Iters | Cold CG | Cold Time (ms) | Warm Iters | Warm CG | Warm Time (ms)" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
    for (const auto& r : results) {
        std::cout << std::setw(4) << r.N << " | "
                  << std::setw(13) << r.n << " | "
                  << std::setw(10) << r.cold_iters << " | "
                  << std::setw(7) << r.cold_cg << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.cold_time_ms << " | "
                  << std::setw(10) << r.warm_iters << " | "
                  << std::setw(7) << r.warm_cg << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.warm_time_ms << std::endl;
    }
    std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;

    return 0;
}
