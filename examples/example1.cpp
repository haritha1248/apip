#include <iostream>
#include <limits>
#include <chrono>
#include "ipm_admm_cg/admm_kkt_solver.hpp"

using namespace ipm_admm_cg;

void example1()
{
    std::cout << "==========================================================" << std::endl;
    std::cout << "              Proximal IPM ADMM-CG QP Solver              " << std::endl;
    std::cout << "                      Example 1                           " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Define dimensions
    int n = 2; // variables
    int p = 1; // equality constraints
    int m = 2; // inequality constraints

    // Setup P matrix (2x2 diagonal)
    Eigen::SparseMatrix<double> P(n, n);
    P.insert(0, 0) = 6.0;
    P.insert(1, 1) = 4.0;
    P.makeCompressed();

    // Setup c vector
    Eigen::VectorXd c(n);
    c << 2.0, -3.0;

    // Setup A matrix (1x2 sparse) - MPC dynamics: -u0 + x1 = 1.0
    Eigen::SparseMatrix<double> A(p, n);
    A.insert(0, 0) = -1.0;
    A.insert(0, 1) = 1.0;
    A.makeCompressed();

    // Setup b vector
    Eigen::VectorXd b(p);
    b << 1.0;

    // Setup G matrix (2x2 sparse)
    Eigen::SparseMatrix<double> G(m, n);
    G.insert(0, 0) = 1.0;
    G.insert(0, 1) = -1.0;
    G.insert(1, 0) = 2.0;
    G.makeCompressed();

    // Setup h vector
    Eigen::VectorXd h(m);
    h << 2.0, 3.0;

    // Initialize the IPM solver
    ProximalIPMSolver solver(P, c, A, b, G, h);

    // Set solver parameters
    solver.set_settings(100, 1e-6, 0.2); // max 100 iterations, 1e-6 tolerance, 0.2 centering parameter
    solver.set_regularization(1e-8, 1e-8, 1e-8); // PMM regularizers delta, x_reg, sigma_z

    // Outputs
    Eigen::VectorXd x_sol(n);
    Eigen::VectorXd s_sol(m);
    Eigen::VectorXd y_sol(p);
    Eigen::VectorXd z_sol(m);

    int total_admm_iters = 0;
    int total_cg_iters = 0;

    // Reset global FLOP counter
    global_flop_count = 0;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Solve
    bool success = solver.solve(x_sol, s_sol, y_sol, z_sol, total_admm_iters, total_cg_iters);

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // Print outputs
    if (success)
    {
        std::cout << "\nSolver succeeded!" << std::endl;
        std::cout << "Optimal x = " << x_sol.transpose() << std::endl;
        std::cout << "Slacks s  = " << s_sol.transpose() << std::endl;
        std::cout << "Duals y   = " << y_sol.transpose() << std::endl;
        std::cout << "Duals z   = " << z_sol.transpose() << std::endl;
        std::cout << "\nExpected analytical solution: x = [-0.3, 0.7]" << std::endl;
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "Performance Metrics (IPM-ADMM-CG):" << std::endl;
        std::cout << "Total Inner ADMM Iterations = " << total_admm_iters << std::endl;
        std::cout << "Total Inner CG Iterations   = " << total_cg_iters << std::endl;
        std::cout << "Solve Execution Time        = " << duration.count() << " ms" << std::endl;
        std::cout << "Actual Executed FLOPs       = " << global_flop_count << std::endl;
        std::cout << "Calculated Flop Rate        = " << (global_flop_count / (duration.count() / 1000.0)) / 1e6 << " MFLOPS" << std::endl;
        std::cout << "==========================================================" << std::endl;
    }
    else
    {
        std::cout << "\nSolver failed to converge." << std::endl;
    }

}
