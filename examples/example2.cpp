#include <iostream>
#include <limits>
#include <chrono>
#include "ipm_admm_cg/admm_kkt_solver.hpp"

using namespace ipm_admm_cg;

void example2()
{
    std::cout << "\nProximal IPM ADMM-CG QP Solver\n" << std::endl;
    std::cout << "Example 2\n" << std::endl;

    //dimensions
    int n = 3; // variables
    int p = 1; // equality constraints
    int m = 3; // inequality constraints

    //setup P
    Eigen::SparseMatrix<double> P(n, n);
    P.insert(0, 0) = 4.0;
    P.insert(1, 1) = 2.0;
    P.insert(2, 2) = 6.0;
    P.makeCompressed();

    //setup c
    Eigen::VectorXd c(n);
    c << 1.0, -2.0, 3.0;

    //setup A 
    Eigen::SparseMatrix<double> A(p, n);
    A.insert(0, 0) = 0.8;
    A.insert(0, 1) = 0.5;
    A.insert(0, 2) = -1.0;
    A.makeCompressed();

    //setup b
    Eigen::VectorXd b(p);
    b << 0.0;

    //setup G
    Eigen::SparseMatrix<double> G(m, n);
    G.insert(0, 0) = -1.0;
    G.insert(1, 1) = -1.0;
    G.insert(2, 0) = 1.0;  
    G.insert(2, 2) = 1.0;
    G.makeCompressed();

    //setup h
    Eigen::VectorXd h(m);
    h << 0.0, 0.0, 1.5;

    //init
    ProximalIPMSolver solver(P, c, A, b, G, h);

    //solver params
    solver.set_settings(100, 1e-6, 0.2); // max iters, tolerance, centering 
    solver.set_regularization(1e-8, 1e-8, 1e-8); // PMM regularizer delta, x_reg, sigma_z

    Eigen::VectorXd x_sol(n);
    Eigen::VectorXd s_sol(m);
    Eigen::VectorXd y_sol(p);
    Eigen::VectorXd z_sol(m);

    int total_admm_iters = 0;
    int total_cg_iters = 0;

    //reset flops
    global_flop_count = 0;

    //timer
    auto start_time = std::chrono::high_resolution_clock::now();

    //solve
    bool success = solver.solve(x_sol, s_sol, y_sol, z_sol, total_admm_iters, total_cg_iters);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    if (success)
    {
        std::cout << "\nSolver succeeded." << std::endl;
        std::cout << "Optimal x = " << x_sol.transpose() << std::endl;
        std::cout << "Slacks s  = " << s_sol.transpose() << std::endl;
        std::cout << "Duals y   = " << y_sol.transpose() << std::endl;
        std::cout << "Duals z   = " << z_sol.transpose() << std::endl;
        std::cout << "\n" << std::endl;
        std::cout << "Performance (IPM-ADMM-CG):" << std::endl;
        std::cout << "Total Inner ADMM Iterations = " << total_admm_iters << std::endl;
        std::cout << "Total Inner CG Iterations   = " << total_cg_iters << std::endl;
        std::cout << "Solve Execution Time        = " << duration.count() << " ms" << std::endl;
        std::cout << "Executed FLOPs              = " << global_flop_count << std::endl;
        std::cout << "Calculated Flop Rate        = " << (global_flop_count / (duration.count() / 1000.0)) / 1e6 << " MFLOPS" << std::endl;
        std::cout << "\n" << std::endl;
    }
    else
    {
        std::cout << "\nFailed to converge." << std::endl;
    }

}
