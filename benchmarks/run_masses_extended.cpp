#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ipm_admm_cg/admm_kkt_solver.hpp"
#include "osqp.h"
#include "piqp/piqp.hpp"

using namespace ipm_admm_cg;

struct MassBenchmarkResult {
    int M;
    int n;
    
    //cold start
    double cold_custom_ms;
    int cold_custom_ipm_iters;
    int cold_custom_cg_iters;
    bool cold_custom_success;

    double cold_osqp_total_ms;
    double cold_osqp_solve_ms;
    int cold_osqp_iters;
    bool cold_osqp_success;

    double cold_piqp_ms;
    int cold_piqp_iters;
    bool cold_piqp_success;

    //warm start
    double warm_custom_ms;
    int warm_custom_ipm_iters;
    int warm_custom_cg_iters;
    bool warm_custom_success;

    double warm_osqp_solve_ms;
    int warm_osqp_iters;
    bool warm_osqp_success;

    double warm_piqp_solve_ms;
    int warm_piqp_iters;
    bool warm_piqp_success;
};

void build_multi_agent_platoon_problem(int M, int N,
                                        Eigen::SparseMatrix<double>& P,
                                        Eigen::VectorXd& c,
                                        Eigen::SparseMatrix<double>& A_eq,
                                        Eigen::VectorXd& b_eq,
                                        Eigen::SparseMatrix<double>& G_ineq,
                                        Eigen::VectorXd& h_ineq,
                                        int& nx_out, int& nu_out)
{
    int nx_agent = 4;
    int nu_agent = 1;
    nx_out = nx_agent;
    nu_out = nu_agent;

    int num_blocks = M * N;
    int block_size = nx_agent + nu_agent; // B = 5
    int n = block_size * num_blocks;       // Total decision variables
    int p = nx_agent * num_blocks;         // Total equality constraints
    int m_ineq = 2 * nu_agent * num_blocks;// Total inequality constraints

    double dt = 0.1;
    Eigen::MatrixXd A_sys(nx_agent, nx_agent);
    A_sys << 1.0,  dt, 0.5 * dt * dt, 0.0,
             0.0, 1.0,           dt, 0.0,
             0.0, 0.0,        0.90, 0.0,
            -dt,  0.0,         0.0, 0.95;

    Eigen::VectorXd B_sys(nx_agent);
    B_sys << 0.0, 0.0, dt, 0.0;

    Eigen::VectorXd x_init_single(nx_agent);
    x_init_single << 1.0, 0.2, -0.1, 0.5;

    P.resize(n, n);
    std::vector<Eigen::Triplet<double>> P_triplets;
    for (int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        P_triplets.push_back(Eigen::Triplet<double>(start + 0, start + 0, 10.0));
        P_triplets.push_back(Eigen::Triplet<double>(start + 1, start + 1, 1.0));
        P_triplets.push_back(Eigen::Triplet<double>(start + 2, start + 2, 0.5));
        P_triplets.push_back(Eigen::Triplet<double>(start + 3, start + 3, 5.0));
        P_triplets.push_back(Eigen::Triplet<double>(start + 4, start + 4, 0.1));
    }
    P.setFromTriplets(P_triplets.begin(), P_triplets.end());
    P.makeCompressed();

    c = Eigen::VectorXd::Zero(n);

    A_eq.resize(p, n);
    std::vector<Eigen::Triplet<double>> A_triplets;
    b_eq = Eigen::VectorXd::Zero(p);

    for (int b = 0; b < num_blocks; b++) {
        int row_offset = b * nx_agent;
        int col_curr = b * block_size;
        int col_prev = (b - 1) * block_size;

        for (int r = 0; r < nx_agent; r++) {
            A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_curr + r, 1.0));
        }

        if (b == 0) {
            b_eq.head(nx_agent) = x_init_single;
        } else {
            for (int r = 0; r < nx_agent; r++) {
                for (int col = 0; col < nx_agent; col++) {
                    if (std::abs(A_sys(r, col)) > 1e-12) {
                        A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + col, -A_sys(r, col)));
                    }
                }
                if (std::abs(B_sys(r)) > 1e-12) {
                    A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + nx_agent, -B_sys(r)));
                }
            }
        }
    }
    A_eq.setFromTriplets(A_triplets.begin(), A_triplets.end());
    A_eq.makeCompressed();

    G_ineq.resize(m_ineq, n);
    std::vector<Eigen::Triplet<double>> G_triplets;
    for (int b = 0; b < num_blocks; b++) {
        int col_u = b * block_size + nx_agent;
        int ineq_offset = b * 2;
        G_triplets.push_back(Eigen::Triplet<double>(ineq_offset + 0, col_u, 1.0));
        G_triplets.push_back(Eigen::Triplet<double>(ineq_offset + 1, col_u, -1.0));
    }
    G_ineq.setFromTriplets(G_triplets.begin(), G_triplets.end());
    G_ineq.makeCompressed();

    h_ineq = Eigen::VectorXd::Ones(m_ineq);
}

MassBenchmarkResult run_mass_benchmark(int M, int N)
{
    int nx, nu;
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd c;
    Eigen::SparseMatrix<double> A_eq;
    Eigen::VectorXd b_eq;
    Eigen::SparseMatrix<double> G_ineq;
    Eigen::VectorXd h_ineq;

    build_multi_agent_platoon_problem(M, N, P, c, A_eq, b_eq, G_ineq, h_ineq, nx, nu);

    int block_size = nx + nu; // B = 5
    int num_blocks = M * N;
    int n = block_size * num_blocks;
    int p = nx * num_blocks;
    int m_ineq = 2 * nu * num_blocks;

    MassBenchmarkResult res;
    res.M = M;
    res.n = n;

    std::cout << "\n" << std::endl;
    std::cout << "  MPC Benchmark: Masses M = " << M
              << " (Horizon N = " << N << ", Vars n = " << n << ")" << std::endl;
    std::cout << "==========================================================" << std::endl;

    //IPM-ADMM-CG Solver (Cold Start)
    Eigen::VectorXd x_sol = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd s_sol = Eigen::VectorXd::Ones(m_ineq);
    Eigen::VectorXd y_sol = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd z_sol = Eigen::VectorXd::Ones(m_ineq);

    ProximalIPMSolver custom_solver(P, c, A_eq, b_eq, G_ineq, h_ineq);
    custom_solver.set_settings(100, 1e-5, 0.15);
    custom_solver.set_regularization(1e-8, 1e-8, 1e-8);
    custom_solver.set_preconditioner_block_size(block_size);

    int custom_admm = 0;
    int custom_cg = 0;
    auto t1_custom = std::chrono::high_resolution_clock::now();
    res.cold_custom_success = custom_solver.solve(x_sol, s_sol, y_sol, z_sol, custom_admm, custom_cg, false, false);
    auto t2_custom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_custom = t2_custom - t1_custom;

    res.cold_custom_ipm_iters = custom_admm;
    res.cold_custom_cg_iters = custom_cg;
    res.cold_custom_ms = dur_custom.count();

    //OSQP Solver (Cold Start)
    int osqp_num_ineq = nu * num_blocks;
    int m_osqp = p + osqp_num_ineq;

    std::vector<Eigen::Triplet<double>> P_osqp_triplets;
    for (int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        P_osqp_triplets.push_back(Eigen::Triplet<double>(start + 0, start + 0, 10.0));
        P_osqp_triplets.push_back(Eigen::Triplet<double>(start + 1, start + 1, 1.0));
        P_osqp_triplets.push_back(Eigen::Triplet<double>(start + 2, start + 2, 0.5));
        P_osqp_triplets.push_back(Eigen::Triplet<double>(start + 3, start + 3, 5.0));
        P_osqp_triplets.push_back(Eigen::Triplet<double>(start + 4, start + 4, 0.1));
    }
    Eigen::SparseMatrix<double, Eigen::ColMajor, OSQPInt> P_osqp(n, n);
    P_osqp.setFromTriplets(P_osqp_triplets.begin(), P_osqp_triplets.end());
    P_osqp.makeCompressed();

    Eigen::SparseMatrix<double, Eigen::ColMajor, OSQPInt> A_osqp(m_osqp, n);
    std::vector<Eigen::Triplet<double>> A_osqp_triplets;

    for (int k = 0; k < A_eq.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_eq, k); it; ++it) {
            A_osqp_triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
        }
    }
    for (int b = 0; b < num_blocks; b++) {
        int col_u = b * block_size + nx;
        int osqp_ineq_offset = p + b;
        A_osqp_triplets.push_back(Eigen::Triplet<double>(osqp_ineq_offset, col_u, 1.0));
    }
    A_osqp.setFromTriplets(A_osqp_triplets.begin(), A_osqp_triplets.end());
    A_osqp.makeCompressed();

    Eigen::VectorXd l_osqp(m_osqp);
    Eigen::VectorXd u_osqp(m_osqp);
    l_osqp.head(p) = b_eq;
    u_osqp.head(p) = b_eq;
    l_osqp.tail(osqp_num_ineq).setConstant(-1.0);
    u_osqp.tail(osqp_num_ineq).setConstant(1.0);

    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);
    settings->verbose = 0;
    settings->eps_abs = 1e-5;
    settings->eps_rel = 1e-5;

    OSQPCscMatrix* P_csc = OSQPCscMatrix_new(n, n, P_osqp.nonZeros(), (OSQPFloat*)P_osqp.valuePtr(), (OSQPInt*)P_osqp.innerIndexPtr(), (OSQPInt*)P_osqp.outerIndexPtr());
    OSQPCscMatrix* A_csc = OSQPCscMatrix_new(m_osqp, n, A_osqp.nonZeros(), (OSQPFloat*)A_osqp.valuePtr(), (OSQPInt*)A_osqp.innerIndexPtr(), (OSQPInt*)A_osqp.outerIndexPtr());

    OSQPSolver* osqp_solver = NULL;
    auto t1_osqp_setup = std::chrono::high_resolution_clock::now();
    OSQPInt exitflag = osqp_setup(&osqp_solver, P_csc, c.data(), A_csc, l_osqp.data(), u_osqp.data(), m_osqp, n, settings);
    auto t2_osqp_setup = std::chrono::high_resolution_clock::now();

    auto t1_osqp_solve = std::chrono::high_resolution_clock::now();
    if (!exitflag) {
        exitflag = osqp_solve(osqp_solver);
    }
    auto t2_osqp_solve = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur_osqp_setup = t2_osqp_setup - t1_osqp_setup;
    std::chrono::duration<double, std::milli> dur_osqp_solve = t2_osqp_solve - t1_osqp_solve;

    res.cold_osqp_success = (exitflag == 0 && osqp_solver->info->status_val == OSQP_SOLVED);
    res.cold_osqp_iters = osqp_solver->info->iter;
    res.cold_osqp_solve_ms = dur_osqp_solve.count();
    res.cold_osqp_total_ms = dur_osqp_setup.count() + res.cold_osqp_solve_ms;

    // OSQP Warm Start
    Eigen::VectorXd b_eq_warm = b_eq;
    b_eq_warm(0) += 0.05; 
    l_osqp.head(p) = b_eq_warm;
    u_osqp.head(p) = b_eq_warm;
    osqp_update_data_vec(osqp_solver, nullptr, l_osqp.data(), u_osqp.data());

    auto t1_osqp_warm = std::chrono::high_resolution_clock::now();
    exitflag = osqp_solve(osqp_solver);
    auto t2_osqp_warm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_osqp_warm = t2_osqp_warm - t1_osqp_warm;

    res.warm_osqp_success = (exitflag == 0 && osqp_solver->info->status_val == OSQP_SOLVED);
    res.warm_osqp_iters = osqp_solver->info->iter;
    res.warm_osqp_solve_ms = dur_osqp_warm.count();

    osqp_cleanup(osqp_solver);
    OSQPCscMatrix_free(P_csc);
    OSQPCscMatrix_free(A_csc);
    free(settings);


    //PIQP Solver (Cold & Warm Start)
    piqp::SparseSolver<double> piqp_solver;
    piqp_solver.settings().verbose = false;
    piqp_solver.settings().compute_timings = true;

    auto t1_piqp = std::chrono::high_resolution_clock::now();
    piqp_solver.setup(P, c, A_eq, b_eq, G_ineq, piqp::nullopt, h_ineq, piqp::nullopt, piqp::nullopt);
    piqp::Status piqp_status = piqp_solver.solve();
    auto t2_piqp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_piqp = t2_piqp - t1_piqp;

    res.cold_piqp_success = (piqp_status == piqp::Status::PIQP_SOLVED);
    res.cold_piqp_iters = piqp_solver.result().info.iter;
    res.cold_piqp_ms = dur_piqp.count();

    //PIQP Warm start
    piqp_solver.update(piqp::nullopt, piqp::nullopt, piqp::nullopt, b_eq_warm, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt);
    auto t1_piqp_warm = std::chrono::high_resolution_clock::now();
    piqp_status = piqp_solver.solve();
    auto t2_piqp_warm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_piqp_warm = t2_piqp_warm - t1_piqp_warm;

    res.warm_piqp_success = (piqp_status == piqp::Status::PIQP_SOLVED);
    res.warm_piqp_iters = piqp_solver.result().info.iter;
    res.warm_piqp_solve_ms = dur_piqp_warm.count();

    // IPM-ADMM-CG Warm Start
    ProximalIPMSolver custom_solver_warm(P, c, A_eq, b_eq_warm, G_ineq, h_ineq);
    custom_solver_warm.set_settings(100, 1e-5, 0.15);
    custom_solver_warm.set_regularization(1e-8, 1e-8, 1e-8);
    custom_solver_warm.set_preconditioner_block_size(block_size);

    int warm_admm = 0;
    int warm_cg = 0;
    auto t1_custom_warm = std::chrono::high_resolution_clock::now();
    res.warm_custom_success = custom_solver_warm.solve(x_sol, s_sol, y_sol, z_sol, warm_admm, warm_cg, true, false);
    auto t2_custom_warm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_custom_warm = t2_custom_warm - t1_custom_warm;

    res.warm_custom_ipm_iters = warm_admm;
    res.warm_custom_cg_iters = warm_cg;
    res.warm_custom_ms = dur_custom_warm.count();

    std::cout << "  [Cold Start]  IPM-ADMM: " << res.cold_custom_ms << " ms (" << res.cold_custom_ipm_iters << " iters) | "
              << "OSQP Total: " << res.cold_osqp_total_ms << " ms (Solve: " << res.cold_osqp_solve_ms << " ms, " << res.cold_osqp_iters << " iters) | "
              << "PIQP: " << res.cold_piqp_ms << " ms (" << res.cold_piqp_iters << " iters)" << std::endl;

    std::cout << "  [Warm Start]  IPM-ADMM: " << res.warm_custom_ms << " ms (" << res.warm_custom_ipm_iters << " iters) | "
              << "OSQP Solve: " << res.warm_osqp_solve_ms << " ms (" << res.warm_osqp_iters << " iters) | "
              << "PIQP Solve: " << res.warm_piqp_solve_ms << " ms (" << res.warm_piqp_iters << " iters)" << std::endl;

    return res;
}

int main()
{
    int N_fixed = 20;
    std::vector<int> M_values = {50, 100, 150, 200, 250, 300, 350, 400, 450, 500};
    std::vector<MassBenchmarkResult> results;

    std::cout << "\n\n" << std::endl;
    std::cout << "  MASSES SCALING BENCHMARK (M = 50 to 500)" << std::endl;
    std::cout << "  Fixed Horizon N = " << N_fixed << " per unit | Variables n up to 50,000" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    for (int M_val : M_values) {
        results.push_back(run_mass_benchmark(M_val, N_fixed));
    }

    std::cout << "\n\n";
    std::cout << "         EXTENDED MASSES BENCHMARK RESULTS TABLE       " << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << " M   | Vars n | Cold IPM (ms) | Cold OSQP (ms) [Solve] | Cold PIQP (ms) | Warm IPM (ms) | Warm OSQP (ms) | Warm PIQP (ms) " << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    std::ofstream outfile("mass_scaling_extended_results.txt");
    outfile << "M\tVariables(n)\tCold_IPM(ms)\tCold_IPM_Iters\tCold_OSQP_Total(ms)\tCold_OSQP_Solve(ms)\tCold_OSQP_Iters\tCold_PIQP(ms)\tCold_PIQP_Iters\tWarm_IPM(ms)\tWarm_IPM_Iters\tWarm_OSQP_Solve(ms)\tWarm_OSQP_Iters\tWarm_PIQP_Solve(ms)\tWarm_PIQP_Iters\n";

    for (const auto& r : results) {
        std::cout << std::setw(4) << r.M << " | "
                  << std::setw(6) << r.n << " | "
                  << std::setw(13) << std::fixed << std::setprecision(3) << r.cold_custom_ms << " | "
                  << std::setw(10) << std::fixed << std::setprecision(3) << r.cold_osqp_total_ms << " [" << std::setw(8) << r.cold_osqp_solve_ms << "] | "
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.cold_piqp_ms << " | "
                  << std::setw(13) << std::fixed << std::setprecision(3) << r.warm_custom_ms << " | "
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.warm_osqp_solve_ms << " | "
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.warm_piqp_solve_ms << std::endl;

        outfile << r.M << "\t" << r.n << "\t"
                << r.cold_custom_ms << "\t" << r.cold_custom_ipm_iters << "\t"
                << r.cold_osqp_total_ms << "\t" << r.cold_osqp_solve_ms << "\t" << r.cold_osqp_iters << "\t"
                << r.cold_piqp_ms << "\t" << r.cold_piqp_iters << "\t"
                << r.warm_custom_ms << "\t" << r.warm_custom_ipm_iters << "\t"
                << r.warm_osqp_solve_ms << "\t" << r.warm_osqp_iters << "\t"
                << r.warm_piqp_solve_ms << "\t" << r.warm_piqp_iters << "\n";
    }
    outfile.close();

    return 0;
}
