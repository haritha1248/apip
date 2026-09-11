#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

#include "ipm_admm_cg/admm_kkt_solver.hpp"
#include "osqp.h"
#include "piqp/piqp.hpp"

using namespace ipm_admm_cg;

struct WarmResult {
    int N;
    int n;

    //IPM-ADMM-CG Warm
    int custom_ipm_iters;
    int custom_cg_iters;
    double custom_time_ms;
    bool custom_success;

    //OSQP Warm
    int osqp_iters;
    double osqp_solve_ms;
    bool osqp_success;

    //PIQP Warm
    int piqp_iters;
    double piqp_solve_ms;
    bool piqp_success;
};

//6-mass, 7-spring, 3-actuator example
void compute_wang_boyd_discrete_matrices(double dt,
                                         Eigen::MatrixXd& A_sys,
                                         Eigen::MatrixXd& B_sys)
{
    const int n_masses = 6;
    const int nx = 2 * n_masses; // 12 states
    const int nu = 3;            // 3 actuators

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_masses, n_masses);
    for (int i = 0; i < n_masses; i++) {
        K(i, i) = 2.0;
        if (i > 0) K(i, i - 1) = -1.0;
        if (i < n_masses - 1) K(i, i + 1) = -1.0;
    }

    //damping
    double damping = 0.1;
    Eigen::MatrixXd D = damping * Eigen::MatrixXd::Identity(n_masses, n_masses);

    //tension forces across masses
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

void build_wang_boyd_mass_spring_problem(int N,
                                         const Eigen::MatrixXd& A_sys,
                                         const Eigen::MatrixXd& B_sys,
                                         const Eigen::VectorXd& x_init,
                                         Eigen::SparseMatrix<double>& P,
                                         Eigen::VectorXd& c,
                                         Eigen::SparseMatrix<double>& A_eq,
                                         Eigen::VectorXd& b_eq,
                                         Eigen::SparseMatrix<double>& G_ineq,
                                         Eigen::VectorXd& h_ineq)
{
    const int nx = 12;
    const int nu = 3;
    const int n = (nx + nu) * N;
    const int p = nx * N;
    const int m = 2 * nu * N;

    //P matrix
    P.resize(n, n);
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

    c = Eigen::VectorXd::Zero(n);

    //A_eq matrix
    A_eq.resize(p, n);
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

    b_eq = Eigen::VectorXd::Zero(p);
    b_eq.head(nx) = x_init;

    //G_ineq matrix
    G_ineq.resize(m, n);
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

    double u_max = 0.5;
    h_ineq = Eigen::VectorXd::Constant(m, u_max);
}

WarmResult run_warm_benchmark_for_N(int N,
                                     const Eigen::MatrixXd& A_sys,
                                     const Eigen::MatrixXd& B_sys)
{
    const int nx = 12;
    const int nu = 3;
    const int n = (nx + nu) * N;
    const int p = nx * N;
    const int m_g = 2 * nu * N;

    WarmResult res;
    res.N = N;
    res.n = n;

    std::cout << "\n" << std::endl;
    std::cout << "  RUNNING WARM START BENCHMARK: Horizon N = " << N << " (Variables n = " << n << ")" << std::endl;
    std::cout << "  Wang-Boyd Classic: 6 Masses, 7 Springs, 3 Actuators" << std::endl;
    std::cout << "\n" << std::endl;

    Eigen::VectorXd x_init1(nx);
    x_init1 << 1.0, 0.0, -0.5, 0.0, 0.5, 0.0, -0.2, 0.0, 0.3, 0.0, -0.1, 0.0;

    Eigen::VectorXd x_init2 = 0.9 * x_init1; 

    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd c;
    Eigen::SparseMatrix<double> A_eq;
    Eigen::VectorXd b_eq1;
    Eigen::SparseMatrix<double> G_ineq;
    Eigen::VectorXd h_ineq;

    build_wang_boyd_mass_spring_problem(N, A_sys, B_sys, x_init1, P, c, A_eq, b_eq1, G_ineq, h_ineq);

    Eigen::VectorXd b_eq2 = b_eq1;
    b_eq2.head(nx) = x_init2;

    //IPM-ADMM-CG Warm Start
    std::cout << "Running IPM-ADMM-CG Warm Start" << std::endl;
    Eigen::VectorXd x_sol = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd s_sol = Eigen::VectorXd::Ones(m_g);
    Eigen::VectorXd y_sol = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd z_sol = Eigen::VectorXd::Ones(m_g);

    ProximalIPMSolver custom_solver1(P, c, A_eq, b_eq1, G_ineq, h_ineq);
    custom_solver1.set_settings(100, 1e-5, 0.15);
    custom_solver1.set_regularization(1e-8, 1e-8, 1e-8);
    custom_solver1.set_preconditioner_block_size(nx + nu);

    int dummy_admm = 0, dummy_cg = 0;
    custom_solver1.solve(x_sol, s_sol, y_sol, z_sol, dummy_admm, dummy_cg, false, false);

    //warm start solve
    ProximalIPMSolver custom_solver2(P, c, A_eq, b_eq2, G_ineq, h_ineq);
    custom_solver2.set_settings(100, 1e-5, 0.15);
    custom_solver2.set_regularization(1e-8, 1e-8, 1e-8);
    custom_solver2.set_preconditioner_block_size(nx + nu);

    int custom_admm = 0;
    int custom_cg = 0;
    auto t1_custom = std::chrono::high_resolution_clock::now();
    res.custom_success = custom_solver2.solve(x_sol, s_sol, y_sol, z_sol, custom_admm, custom_cg, true, false);
    auto t2_custom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_custom = t2_custom - t1_custom;

    res.custom_ipm_iters = custom_admm;
    res.custom_cg_iters = custom_cg;
    res.custom_time_ms = dur_custom.count();
    std::cout << "    [IPM-ADMM-CG Warm] Time: " << res.custom_time_ms << " ms | IPM Iters: " << custom_admm << " | CG Iters: " << custom_cg << std::endl;

    //OSQP Warm Start
    std::cout << "Running OSQP Warm Start" << std::endl;
    int m_ineq = nu * N;
    int m_osqp = p + m_ineq;

    std::vector<Eigen::Triplet<double>> P_osqp_triplets;
    for (int i = 0; i < N; i++) {
        int start = (nx + nu) * i;
        for (int mass = 0; mass < 6; mass++) {
            P_osqp_triplets.push_back(Eigen::Triplet<double>(start + 2 * mass + 0, start + 2 * mass + 0, 10.0));
            P_osqp_triplets.push_back(Eigen::Triplet<double>(start + 2 * mass + 1, start + 2 * mass + 1, 1.0));
        }
        for (int u = 0; u < nu; u++) {
            P_osqp_triplets.push_back(Eigen::Triplet<double>(start + nx + u, start + nx + u, 0.1));
        }
    }
    Eigen::SparseMatrix<double, Eigen::ColMajor, OSQPInt> P_osqp(n, n);
    P_osqp.setFromTriplets(P_osqp_triplets.begin(), P_osqp_triplets.end());
    P_osqp.makeCompressed();

    Eigen::SparseMatrix<double, Eigen::ColMajor, OSQPInt> A_osqp(m_osqp, n);
    std::vector<Eigen::Triplet<double>> A_osqp_triplets;
    for (int r = 0; r < nx; r++) {
        A_osqp_triplets.push_back(Eigen::Triplet<double>(r, r, 1.0));
    }
    for (int k = 1; k < N; k++) {
        int row_offset = k * nx;
        int col_prev = (k - 1) * (nx + nu);
        int col_curr = k * (nx + nu);
        for (int r = 0; r < nx; r++) {
            A_osqp_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_curr + r, 1.0));
        }
        for (int r = 0; r < nx; r++) {
            for (int col = 0; col < nx; col++) {
                A_osqp_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + col, -A_sys(r, col)));
            }
        }
        for (int r = 0; r < nx; r++) {
            for (int col = 0; col < nu; col++) {
                A_osqp_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_prev + nx + col, -B_sys(r, col)));
            }
        }
    }
    for (int i = 0; i < N; i++) {
        for (int u = 0; u < nu; u++) {
            int row = p + i * nu + u;
            int col_u = i * (nx + nu) + nx + u;
            A_osqp_triplets.push_back(Eigen::Triplet<double>(row, col_u, 1.0));
        }
    }
    A_osqp.setFromTriplets(A_osqp_triplets.begin(), A_osqp_triplets.end());
    A_osqp.makeCompressed();

    double u_max = 0.5;
    Eigen::VectorXd l1(m_osqp), u1(m_osqp);
    l1.head(p) = b_eq1; u1.head(p) = b_eq1;
    l1.tail(m_ineq).setConstant(-u_max); u1.tail(m_ineq).setConstant(u_max);

    Eigen::VectorXd l2(m_osqp), u2(m_osqp);
    l2.head(p) = b_eq2; u2.head(p) = b_eq2;
    l2.tail(m_ineq).setConstant(-u_max); u2.tail(m_ineq).setConstant(u_max);

    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);
    settings->verbose = 0;
    settings->eps_abs = 1e-5;
    settings->eps_rel = 1e-5;

    OSQPCscMatrix* P_csc = OSQPCscMatrix_new(n, n, P_osqp.nonZeros(), (OSQPFloat*)P_osqp.valuePtr(), (OSQPInt*)P_osqp.innerIndexPtr(), (OSQPInt*)P_osqp.outerIndexPtr());
    OSQPCscMatrix* A_csc = OSQPCscMatrix_new(m_osqp, n, A_osqp.nonZeros(), (OSQPFloat*)A_osqp.valuePtr(), (OSQPInt*)A_osqp.innerIndexPtr(), (OSQPInt*)A_osqp.outerIndexPtr());

    OSQPSolver* osqp_solver = NULL;
    osqp_setup(&osqp_solver, P_csc, c.data(), A_csc, l1.data(), u1.data(), m_osqp, n, settings);
    osqp_solve(osqp_solver);

    //warm start
    std::vector<double> osqp_x(n), osqp_y(m_osqp);
    for (int i = 0; i < n; i++) osqp_x[i] = osqp_solver->solution->x[i];
    for (int i = 0; i < m_osqp; i++) osqp_y[i] = osqp_solver->solution->y[i];

    osqp_update_data_vec(osqp_solver, NULL, l2.data(), u2.data());
    osqp_warm_start(osqp_solver, osqp_x.data(), osqp_y.data());

    auto t1_osqp_warm = std::chrono::high_resolution_clock::now();
    OSQPInt exitflag_warm = osqp_solve(osqp_solver);
    auto t2_osqp_warm = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur_osqp_warm = t2_osqp_warm - t1_osqp_warm;

    res.osqp_success = (exitflag_warm == 0 && osqp_solver->info->status_val == OSQP_SOLVED);
    res.osqp_iters = osqp_solver->info->iter;
    res.osqp_solve_ms = dur_osqp_warm.count();
    std::cout << "    [OSQP Warm] Time: " << res.osqp_solve_ms << " ms | Iterations: " << res.osqp_iters << std::endl;

    osqp_cleanup(osqp_solver);
    OSQPCscMatrix_free(P_csc);
    OSQPCscMatrix_free(A_csc);
    free(settings);

    //PIQP Warm Start
    std::cout << "Running PIQP Warm Start" << std::endl;
    piqp::SparseSolver<double> piqp_solver;
    piqp_solver.settings().verbose = false;
    piqp_solver.settings().compute_timings = true;

    //setup
    piqp_solver.setup(P, c, A_eq, b_eq1, G_ineq, piqp::nullopt, h_ineq, piqp::nullopt, piqp::nullopt);
    piqp_solver.solve();

    //update
    piqp_solver.update(piqp::nullopt, piqp::nullopt, piqp::nullopt, b_eq2, piqp::nullopt, piqp::nullopt, h_ineq, piqp::nullopt, piqp::nullopt);

    auto t1_piqp_warm = std::chrono::high_resolution_clock::now();
    piqp::Status piqp_status_warm = piqp_solver.solve();
    auto t2_piqp_warm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_piqp_warm = t2_piqp_warm - t1_piqp_warm;

    res.piqp_success = (piqp_status_warm == piqp::Status::PIQP_SOLVED);
    res.piqp_iters = piqp_solver.result().info.iter;
    res.piqp_solve_ms = dur_piqp_warm.count();
    std::cout << "    [PIQP Warm] Time: " << res.piqp_solve_ms << " ms | IPM Iterations: " << res.piqp_iters << std::endl;

    return res;
}

int main()
{
    Eigen::MatrixXd A_sys, B_sys;
    compute_wang_boyd_discrete_matrices(0.5, A_sys, B_sys);

    std::vector<int> N_values = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
    std::vector<WarmResult> results;

    std::cout << "\n" << std::endl;
    std::cout << "  WANG-BOYD 6-MASS 7-SPRING MPC BENCHMARK: WARM STARTS" << std::endl;
    std::cout << "  Solvers: IPM-ADMM-CG, OSQP, PIQP" << std::endl;
    std::cout << "  Horizons N: 1000 to 10000" << std::endl;
    std::cout << "\n" << std::endl;

    for (int N_val : N_values) {
        results.push_back(run_warm_benchmark_for_N(N_val, A_sys, B_sys));
    }

    std::cout << "\n\n";
    std::cout << "WANG-BOYD WARM-START COMPARISON SUMMARY TABLE\n" << std::endl;
    std::cout << "  N   | Variables (n) | IPM-ADMM-CG (ms) | IPM-ADMM Iters | OSQP Solve (ms) | OSQP Iters | PIQP Time (ms) | PIQP Iters " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(5) << r.N << " | "
                  << std::setw(13) << r.n << " | "
                  << std::setw(18) << std::fixed << std::setprecision(2) << r.custom_time_ms << " | "
                  << std::setw(14) << r.custom_ipm_iters << " | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.osqp_solve_ms << " | "
                  << std::setw(10) << r.osqp_iters << " | "
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.piqp_solve_ms << " | "
                  << std::setw(10) << r.piqp_iters << std::endl;
    }
    std::cout << "---------------------------------------------------------------------------------------------------------" << std::endl;

    //txt file
    std::ofstream outfile("warm_start_scaling_results.txt");
    if (outfile.is_open()) {
        outfile << "N\tVariables(n)\tIPM-ADMM-CG(ms)\tIPM-ADMM_Iters\tOSQP_Solve(ms)\tOSQP_Iters\tPIQP_Time(ms)\tPIQP_Iters\n";
        for (const auto& r : results) {
            outfile << r.N << "\t" << r.n << "\t"
                    << r.custom_time_ms << "\t" << r.custom_ipm_iters << "\t"
                    << r.osqp_solve_ms << "\t" << r.osqp_iters << "\t"
                    << r.piqp_solve_ms << "\t" << r.piqp_iters << "\n";
        }
        outfile.close();
    }

    return 0;
}
