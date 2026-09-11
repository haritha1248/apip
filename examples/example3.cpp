#include <iostream>
#include <limits>
#include <chrono>
#include <cmath>
#include "ipm_admm_cg/admm_kkt_solver.hpp"

using namespace ipm_admm_cg;

// Saved state for warm-starting across multiple calls
static Eigen::VectorXd prev_x_3;
static Eigen::VectorXd prev_s_3;
static Eigen::VectorXd prev_y_3;
static Eigen::VectorXd prev_z_3;
static bool has_prev_3 = false;

void example3(bool warm_start = false)
{
    std::cout << "\nProximal IPM ADMM-CG QP Solver\n" << std::endl;
    std::cout << "Example 3\n" << std::endl;
    std::cout << "(Large-Scale Sparse QP)\n" << std::endl;

    //define dimension
    int n = 100; // variables
    int p = 50;  // equality constraints
    int m = 200; // inequality constraints

    std::cout << "Problem Dimensions:" << std::endl;
    std::cout << " Variables (n)     = " << n << std::endl;
    std::cout << " Equalities (p)    = " << p << std::endl;
    std::cout << " Inequalities (m)  = " << m << std::endl;

    //setup P 100 x 100 diag
    Eigen::SparseMatrix<double> P(n, n);
    std::vector<Eigen::Triplet<double>> P_triplets;
    for (int i = 0; i < n; i++) {
        P_triplets.push_back(Eigen::Triplet<double>(i, i, 2.0 + 0.1 * i));
    }
    P.setFromTriplets(P_triplets.begin(), P_triplets.end());
    P.makeCompressed();

    //setup c
    Eigen::VectorXd c(n);
    for (int i = 0; i < n; i++) {
        c(i) = std::sin(static_cast<double>(i));
    }

    //setup A
    Eigen::SparseMatrix<double> A(p, n);
    std::vector<Eigen::Triplet<double>> A_triplets;
    
    //init condition
    A_triplets.push_back(Eigen::Triplet<double>(0, 0, 1.0));
    
    double dynamic_a = 0.9;
    double dynamic_b = 0.5;
    for (int i = 1; i < p; i++) {
        A_triplets.push_back(Eigen::Triplet<double>(i, 2 * (i - 1), -dynamic_a));
        A_triplets.push_back(Eigen::Triplet<double>(i, 2 * (i - 1) + 1, -dynamic_b));
        A_triplets.push_back(Eigen::Triplet<double>(i, 2 * i, 1.0));
    }
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    A.makeCompressed();

    //setup b
    Eigen::VectorXd b(p);
    b.setZero();
    b(0) = (warm_start && has_prev_3) ? 0.9 : 1.0; // Initial state value

    //setup G 200 x 100
    Eigen::SparseMatrix<double> G(m, n);
    std::vector<Eigen::Triplet<double>> G_triplets;
    for (int i = 0; i < n; i++) {
        G_triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));     
        G_triplets.push_back(Eigen::Triplet<double>(n + i, i, -1.0));  
    }
    G.setFromTriplets(G_triplets.begin(), G_triplets.end());
    G.makeCompressed();

    //setup h
    Eigen::VectorXd h(m);
    for (int i = 0; i < n; i++) {
        h(i) = 1.0;       //upper bound
        h(n + i) = 0.0;   //lower bound
    }

    //init
    ProximalIPMSolver solver(P, c, A, b, G, h);

    //solver params
    solver.set_settings(100, 1e-5, 0.15); // max  iters, tolerance, centering
    solver.set_regularization(1e-8, 1e-8, 1e-8); // PMM regularizer delta, x_reg, sigma_z

    //output
    Eigen::VectorXd x_sol(n);
    Eigen::VectorXd s_sol(m);
    Eigen::VectorXd y_sol(p);
    Eigen::VectorXd z_sol(m);

    bool actual_warm = false;
    if (warm_start && has_prev_3)
    {
        x_sol = prev_x_3;
        s_sol = prev_s_3;
        y_sol = prev_y_3;
        z_sol = prev_z_3;
        actual_warm = true;
    }
    else
    {
        x_sol.setZero();
        s_sol.setOnes();
        y_sol.setZero();
        z_sol.setOnes();
    }

    int total_admm_iters = 0;
    int total_cg_iters = 0;

    //reset flops
    global_flop_count = 0;

    //timer
    auto start_time = std::chrono::high_resolution_clock::now();

    //solve
    bool success = solver.solve(x_sol, s_sol, y_sol, z_sol, total_admm_iters, total_cg_iters, actual_warm, false);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    //warm - start: save
    prev_x_3 = x_sol;
    prev_s_3 = s_sol;
    prev_y_3 = y_sol;
    prev_z_3 = z_sol;
    has_prev_3 = true;

    if (success)
    {
        std::cout << "\nSolver succeeded." << std::endl;
        std::cout << "Optimal x (first 5 elements): " << x_sol.head(5).transpose() << " ... " << std::endl;
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
