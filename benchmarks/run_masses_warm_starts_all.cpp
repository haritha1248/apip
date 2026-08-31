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

struct MassWarmSolverResult {
    int M; // Number of agents / units in platoon
    int nx; // State dimension per agent block
    int nu; // Control dimension per agent block
    int N;  // Prediction steps
    int n;  // Total decision variables = (nx + nu) * M * N

    // IPM-ADMM-CG Warm Start
    int custom_ipm_iters;
    int custom_cg_iters;
    double custom_time_ms;
    bool custom_success;

    // OSQP Warm Start
    int osqp_iters;
    double osqp_solve_ms;
    bool osqp_success;

    // PIQP Warm Start
    int piqp_iters;
    double piqp_solve_ms;
    bool piqp_success;
};

// Scalable Multi-Agent Platoon MPC Problem Formulation
// Maintains constant stage sub-block size B = nx_agent + nu_agent = 5
void build_multi_agent_platoon_problem_warm(int M, int N,
                                             const Eigen::VectorXd& x_init,
                                             Eigen::SparseMatrix<double>& P,
                                             Eigen::VectorXd& c,
                                             Eigen::SparseMatrix<double>& A_eq,
                                             Eigen::VectorXd& b_eq,
                                             Eigen::SparseMatrix<double>& G_ineq,
                                             Eigen::VectorXd& h_ineq,
                                             int& nx_out, int& nu_out)
{
    int nx_agent = 4; // [position, velocity, acceleration, headway_error]
    int nu_agent = 1; // [jerk / thrust force]
    nx_out = nx_agent;
    nu_out = nu_agent;

    int num_blocks = M * N;
    int block_size = nx_agent + nu_agent; // B = 5
    int n = block_size * num_blocks;       // Total decision variables
    int p = nx_agent * num_blocks;         // Total equality constraints
    int m_ineq = 2 * nu_agent * num_blocks;// Total inequality constraints

    // 1. Agent Continuous & Discrete-Time System Dynamics (dt = 0.1s)
    double dt = 0.1;
    Eigen::MatrixXd A_sys(nx_agent, nx_agent);
    A_sys << 1.0,  dt, 0.5 * dt * dt, 0.0,
             0.0, 1.0,           dt, 0.0,
             0.0, 0.0,        0.90, 0.0,
            -dt,  0.0,         0.0, 0.95;

    Eigen::VectorXd B_sys(nx_agent);
    B_sys << 0.0, 0.0, dt, 0.0;

    // 2. Build QP Cost Matrix P (Stage block size B = 5)
    P.resize(n, n);
    std::vector<Eigen::Triplet<double>> P_triplets;
    for (int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        P_triplets.push_back(Eigen::Triplet<double>(start + 0, start + 0, 10.0)); // Position error
        P_triplets.push_back(Eigen::Triplet<double>(start + 1, start + 1, 1.0));  // Velocity error
        P_triplets.push_back(Eigen::Triplet<double>(start + 2, start + 2, 0.5));  // Acceleration
        P_triplets.push_back(Eigen::Triplet<double>(start + 3, start + 3, 5.0));  // Headway error
        P_triplets.push_back(Eigen::Triplet<double>(start + 4, start + 4, 0.1));  // Control effort u
    }
    P.setFromTriplets(P_triplets.begin(), P_triplets.end());
    P.makeCompressed();

    c = Eigen::VectorXd::Zero(n);

    // 3. Build Equality Dynamics Matrix A_eq * x = b_eq
    A_eq.resize(p, n);
    std::vector<Eigen::Triplet<double>> A_triplets;
    b_eq = Eigen::VectorXd::Zero(p);

    for (int b = 0; b < num_blocks; b++) {
        int row_offset = b * nx_agent;
        int col_curr = b * block_size;
        int col_prev = (b - 1) * block_size;

        // Identity for x_k
        for (int r = 0; r < nx_agent; r++) {
            A_triplets.push_back(Eigen::Triplet<double>(row_offset + r, col_curr + r, 1.0));
        }

        if (b == 0) {
            // Initial block constraint
            b_eq.head(nx_agent) = x_init;
        } else {
            // Transition from previous block: x_k - A_sys * x_{k-1} - B_sys * u_{k-1} = 0
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

    // 4. Build Inequality Constraints G_ineq * x <= h_ineq (-1.0 <= u <= 1.0)
    G_ineq.resize(m_ineq, n);
    std::vector<Eigen::Triplet<double>> G_triplets;
    for (int b = 0; b < num_blocks; b++) {
        int col_u = b * block_size + nx_agent;
        int ineq_offset = b * 2;
        G_triplets.push_back(Eigen::Triplet<double>(ineq_offset + 0, col_u, 1.0));  // u <= 1.0
        G_triplets.push_back(Eigen::Triplet<double>(ineq_offset + 1, col_u, -1.0)); // -u <= 1.0
    }
    G_ineq.setFromTriplets(G_triplets.begin(), G_triplets.end());
    G_ineq.makeCompressed();

    h_ineq = Eigen::VectorXd::Ones(m_ineq);
}

MassWarmSolverResult run_mass_warm_benchmark(int M, int N)
{
    int nx, nu;
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd c;
    Eigen::SparseMatrix<double> A_eq;
    Eigen::VectorXd b_eq1, b_eq2;
    Eigen::SparseMatrix<double> G_ineq;
    Eigen::VectorXd h_ineq;

    Eigen::VectorXd x_init1(4);
    x_init1 << 1.0, 0.2, -0.1, 0.5;

    Eigen::VectorXd x_init2(4);
    x_init2 << 0.9, 0.15, -0.05, 0.45; // Perturbed initial state for warm start

    build_multi_agent_platoon_problem_warm(M, N, x_init1, P, c, A_eq, b_eq1, G_ineq, h_ineq, nx, nu);
    b_eq2 = b_eq1;
    b_eq2.head(nx) = x_init2;

    int block_size = nx + nu; // B = 5
    int num_blocks = M * N;
    int n = block_size * num_blocks;
    int p = nx * num_blocks;
    int m_ineq = 2 * nu * num_blocks;

    MassWarmSolverResult res;
    res.M = M;
    res.nx = nx;
    res.nu = nu;
    res.N = N;
    res.n = n;

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  WARM START BENCHMARK: Platoon Units M = " << M
              << " (Horizon N = " << N << ", Vars n = " << n << ", Block B = " << block_size << ")" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // ----------------------------------------------------
    // 1. IPM-ADMM-CG Warm Start
    // ----------------------------------------------------
    std::cout << "--> Running IPM-ADMM-CG Warm Start..." << std::endl;
    Eigen::VectorXd x_sol = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd s_sol = Eigen::VectorXd::Ones(m_ineq);
    Eigen::VectorXd y_sol = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd z_sol = Eigen::VectorXd::Ones(m_ineq);

    // Initial solve to generate warm start primal-dual state
    ProximalIPMSolver custom_solver1(P, c, A_eq, b_eq1, G_ineq, h_ineq);
    custom_solver1.set_settings(100, 1e-5, 0.15);
    custom_solver1.set_regularization(1e-8, 1e-8, 1e-8);
    custom_solver1.set_preconditioner_block_size(block_size);
    int dump_admm = 0, dump_cg = 0;
    custom_solver1.solve(x_sol, s_sol, y_sol, z_sol, dump_admm, dump_cg, false, false);

    // Warm start solve with updated RHS b_eq2
    ProximalIPMSolver custom_solver_warm(P, c, A_eq, b_eq2, G_ineq, h_ineq);
    custom_solver_warm.set_settings(100, 1e-5, 0.15);
    custom_solver_warm.set_regularization(1e-8, 1e-8, 1e-8);
    custom_solver_warm.set_preconditioner_block_size(block_size);

    int warm_admm = 0;
    int warm_cg = 0;
    auto t1_custom = std::chrono::high_resolution_clock::now();
    res.custom_success = custom_solver_warm.solve(x_sol, s_sol, y_sol, z_sol, warm_admm, warm_cg, true, false);
    auto t2_custom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_custom = t2_custom - t1_custom;

    res.custom_ipm_iters = warm_admm;
    res.custom_cg_iters = warm_cg;
    res.custom_time_ms = dur_custom.count();
    std::cout << "    [IPM-ADMM-CG Warm] Time: " << res.custom_time_ms << " ms | IPM Iters: " << warm_admm << " | CG Iters: " << warm_cg << std::endl;

    // ----------------------------------------------------
    // 2. OSQP Warm Start
    // ----------------------------------------------------
    std::cout << "--> Running OSQP Warm Start..." << std::endl;
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

    Eigen::VectorXd l1(m_osqp), u1(m_osqp);
    l1.head(p) = b_eq1; u1.head(p) = b_eq1;
    l1.tail(osqp_num_ineq).setConstant(-1.0); u1.tail(osqp_num_ineq).setConstant(1.0);

    Eigen::VectorXd l2(m_osqp), u2(m_osqp);
    l2.head(p) = b_eq2; u2.head(p) = b_eq2;
    l2.tail(osqp_num_ineq).setConstant(-1.0); u2.tail(osqp_num_ineq).setConstant(1.0);

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

    // ----------------------------------------------------
    // 3. PIQP Warm Start
    // ----------------------------------------------------
    std::cout << "--> Running PIQP Warm Start..." << std::endl;
    piqp::SparseSolver<double> piqp_solver;
    piqp_solver.settings().verbose = false;
    piqp_solver.settings().compute_timings = true;

    piqp_solver.setup(P, c, A_eq, b_eq1, G_ineq, piqp::nullopt, h_ineq, piqp::nullopt, piqp::nullopt);
    piqp_solver.solve();

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
    int N_fixed = 20; // Fixed horizon per agent
    std::vector<int> M_values = {2, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150};
    std::vector<MassWarmSolverResult> results;

    std::cout << "==========================================================" << std::endl;
    std::cout << "  MULTI-AGENT PLATOON MPC SCALING BENCHMARK (WARM STARTS)" << std::endl;
    std::cout << "  Fixed Horizon N = " << N_fixed << " per unit" << std::endl;
    std::cout << "  Solvers: IPM-ADMM-CG, OSQP, PIQP (Warm Starts)" << std::endl;
    std::cout << "  Platoon Units M: 2 to 150 (Variables n up to 15,000)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    for (int M_val : M_values) {
        results.push_back(run_mass_warm_benchmark(M_val, N_fixed));
    }

    std::cout << "\n\n";
    std::cout << "=========================================================================================" << std::endl;
    std::cout << "               MULTI-AGENT SCALING BENCHMARK SUMMARY (WARM START)                        " << std::endl;
    std::cout << "=========================================================================================" << std::endl;
    std::cout << " M   | nx | nu | Vars (n) | IPM-ADMM-CG (ms) | IPM Iters | OSQP Solve (ms) | OSQP Iters | PIQP Solve (ms) | PIQP Iters " << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------";
    std::cout << "\n";

    std::ofstream outfile("mass_scaling_warm_start_results.txt");
    outfile << "M\tnx\tnu\tVariables(n)\tIPM-ADMM-CG(ms)\tIPM-ADMM_Iters\tOSQP_Solve(ms)\tOSQP_Iters\tPIQP_Solve(ms)\tPIQP_Iters\n";

    for (const auto& r : results) {
        std::cout << std::setw(4) << r.M << " | "
                  << std::setw(2) << r.nx << " | "
                  << std::setw(2) << r.nu << " | "
                  << std::setw(8) << r.n << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.custom_time_ms << " | "
                  << std::setw(9) << r.custom_ipm_iters << " | "
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.osqp_solve_ms << " | "
                  << std::setw(10) << r.osqp_iters << " | "
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.piqp_solve_ms << " | "
                  << std::setw(10) << r.piqp_iters << std::endl;

        outfile << r.M << "\t" << r.nx << "\t" << r.nu << "\t" << r.n << "\t"
                << r.custom_time_ms << "\t" << r.custom_ipm_iters << "\t"
                << r.osqp_solve_ms << "\t" << r.osqp_iters << "\t"
                << r.piqp_solve_ms << "\t" << r.piqp_iters << "\n";
    }
    std::cout << "=========================================================================================" << std::endl;
    outfile.close();

    return 0;
}
