#pragma once
#ifndef IPM_ADMM_CG_ADMM_KKT_SOLVER_HPP
#define IPM_ADMM_CG_ADMM_KKT_SOLVER_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstddef>

namespace ipm_admm_cg {

//algorithm complexity analysis global flop counter
inline size_t global_flop_count = 0;

inline void reset_flop_count() {
    global_flop_count = 0;
}

inline size_t get_flop_count() {
    return global_flop_count;
}

class ADMMKKTSolver {
private:
    //problem matrixes
    Eigen::SparseMatrix<double> P;
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> G;

    //transposed
    Eigen::SparseMatrix<double> AT;
    Eigen::SparseMatrix<double> GT;

    //precomputed matrices
    Eigen::SparseMatrix<double> AtA;
    Eigen::VectorXd GtG_diag;
    Eigen::VectorXd diag_term;

    //regularization
    double delta;
    Eigen::VectorXd x_reg;
    Eigen::VectorXd z_reg_inv;

    //block thomas precondition
    int precond_block_size = 2;
    std::vector<int> block_szs;
    std::vector<int> block_starts;
    std::vector<Eigen::LLT<Eigen::MatrixXd>> llt_blocks;
    std::vector<Eigen::MatrixXd> C_prime;
    std::vector<Eigen::MatrixXd> U_blocks;

    //constant block parts
    std::vector<Eigen::MatrixXd> P_block;
    std::vector<Eigen::MatrixXd> AtA_block;
    std::vector<Eigen::MatrixXd> P_U_block;
    std::vector<Eigen::MatrixXd> AtA_U_block;

    std::vector<std::vector<int>> var_to_constraints;

    //admm parameters
    double rho = 1.0;
    int max_admm_iter = 200;
    double admm_tol = 1e-8;

    //CG parameters
    int max_cg_iter = 500;
    double cg_tol = 1e-4; //low tolerance for convergence

    mutable Eigen::VectorXd temp_n1;
    mutable Eigen::VectorXd cg_r;
    mutable Eigen::VectorXd cg_w;
    mutable Eigen::VectorXd cg_p;
    mutable Eigen::VectorXd cg_Ap;
    mutable Eigen::VectorXd cg_w_new;
    mutable Eigen::VectorXd cg_d_prime;
    mutable Eigen::VectorXd cg_d_prime_new;

public:
    ADMMKKTSolver(const Eigen::SparseMatrix<double>& P_in,
                  const Eigen::SparseMatrix<double>& A_in,
                  const Eigen::SparseMatrix<double>& G_in,
                  int precond_block_size_in = 2)
        : P(P_in), A(A_in), G(G_in), precond_block_size(precond_block_size_in)
    {
        AT = A.transpose();
        GT = G.transpose();

        //precompute
        AtA = AT * A;

        //thomas block structure
        int n = P.rows();
        int B = precond_block_size;
        int N = (n + B - 1) / B;
        block_szs.resize(N);
        block_starts.resize(N);
        for (int i = 0; i < N; i++)
        {
            block_starts[i] = i * B;
            block_szs[i] = std::min(B, n - block_starts[i]);
        }

        //pre extract blocks
        P_block.resize(N);
        AtA_block.resize(N);
        P_U_block.resize(N - 1);
        AtA_U_block.resize(N - 1);

        for (int i = 0; i < N; i++)
        {
            int gr = block_starts[i];
            int block_sz = block_szs[i];

            P_block[i] = Eigen::MatrixXd::Zero(block_sz, block_sz);
            AtA_block[i] = Eigen::MatrixXd::Zero(block_sz, block_sz);
            for (int r = 0; r < block_sz; r++)
            {
                for (int c = 0; c < block_sz; c++)
                {
                    P_block[i](r, c) = P.coeff(gr + r, gr + c);
                    AtA_block[i](r, c) = AtA.coeff(gr + r, gr + c);
                }
            }

            if (i < N - 1)
            {
                int block_sz_next = block_szs[i + 1];
                int gr_next = block_starts[i + 1];
                P_U_block[i] = Eigen::MatrixXd::Zero(block_sz, block_sz_next);
                AtA_U_block[i] = Eigen::MatrixXd::Zero(block_sz, block_sz_next);
                for (int r = 0; r < block_sz; r++)
                {
                    for (int c = 0; c < block_sz_next; c++)
                    {
                        P_U_block[i](r, c) = P.coeff(gr + r, gr_next + c);
                        AtA_U_block[i](r, c) = AtA.coeff(gr + r, gr_next + c);
                    }
                }
            }
        }

        //map constraints for G
        int m = G.rows();
        var_to_constraints.resize(n);
        for (int k = 0; k < G.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(G, k); it; ++it)
            {
                int row = it.row();
                int col = it.col();
                var_to_constraints[col].push_back(row);
            }
        }

        //resize
        temp_n1.resize(n);
        cg_r.resize(n);
        cg_w.resize(n);
        cg_p.resize(n);
        cg_Ap.resize(n);
        cg_w_new.resize(n);
        cg_d_prime.resize(n);
        llt_blocks.resize(N);
        C_prime.resize(N - 1);
        U_blocks.resize(N - 1);
    }

    void set_cg_tolerance(double tol)
    {
        cg_tol = tol;
    }

    void update(double delta_in, const Eigen::VectorXd& x_reg_in, const Eigen::VectorXd& z_reg_in, double rho_in)
    {
        delta = delta_in;
        x_reg = x_reg_in;
        rho = rho_in;

        //z reg inversion
        z_reg_inv.resize(z_reg_in.size());
        for (int i = 0; i < z_reg_in.size(); i++)
        {
            if (std::abs(z_reg_in(i)) > 1e-20)
            {
                z_reg_inv(i) = 1.0 / z_reg_in(i);
                global_flop_count += 1;
            }
            else
            {
                z_reg_inv(i) = 0.0;
            }
        }

        //GtG diagonal: compute
        int n = P.rows();
        GtG_diag.setZero(n);
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int row : var_to_constraints[i])
            {
                sum += z_reg_inv(row);
            }
            GtG_diag(i) = sum;
            global_flop_count += var_to_constraints[i].size();
        }

        diag_term = x_reg + GtG_diag;

        //precompute factorization of M_spd
        int B = precond_block_size;
        int N = (n + B - 1) / B;

        double delta_inv = 1.0 / delta;

        for (int i = 0; i < N; i++)
        {
            int gr = block_starts[i];
            int block_sz = block_szs[i];

            //build D i
            Eigen::MatrixXd D_i = P_block[i] + delta_inv * AtA_block[i];
            for (int r = 0; r < block_sz; r++)
            {
                D_i(r, r) += x_reg(gr + r) + rho + GtG_diag(gr + r);
            }

            //compute Gamma
            Eigen::MatrixXd Gamma_i = D_i;
            if (i > 0)
            {
                Gamma_i -= U_blocks[i - 1].transpose() * C_prime[i - 1];
            }

            //factor Gamma_i
            llt_blocks[i].compute(Gamma_i);
            
            global_flop_count += (block_sz * block_sz * block_sz) / 3;

            //find C prime
            if (i < N - 1)
            {
                Eigen::MatrixXd U_i = P_U_block[i] + delta_inv * AtA_U_block[i];
                U_blocks[i] = U_i;
                C_prime[i] = llt_blocks[i].solve(U_i);
            }
        }
    }

    //M_spd * x
    inline void apply_spd(const Eigen::VectorXd& x, Eigen::VectorXd& y) const
    {
        int n = P.rows();

        //y = P * x
        y = P * x;
        global_flop_count += 2 * P.nonZeros();

        //y += (diag_term + rho) * x 
        y.array() += (diag_term.array() + rho) * x.array();
        global_flop_count += 3 * n;

        //y += delta_inv * A^T * A * x
        if (A.rows() > 0)
        {
            temp_n1 = AtA * x;
            y += (1.0 / delta) * temp_n1;
            global_flop_count += 2 * AtA.nonZeros() + 2 * n + 1;
        }
    }

    //solve cg system
    int solve_cg(const Eigen::VectorXd& b, Eigen::VectorXd& x, bool warm_start = false) const
    {
        int size = b.size();
        if (warm_start)
        {
            apply_spd(x, cg_Ap);
            cg_r = b - cg_Ap;
            global_flop_count += 2 * size;
        }
        else
        {
            x.setZero(size);
            cg_r = b;
        }

        double r_norm2 = cg_r.dot(cg_r);
        global_flop_count += 2 * size;
        double cg_tol2 = cg_tol * cg_tol;
        if (r_norm2 < cg_tol2) return 0;

        int N_thomas = block_starts.size();
        
        // Forward sweep
        cg_d_prime.segment(block_starts[0], block_szs[0]) = llt_blocks[0].solve(cg_r.segment(block_starts[0], block_szs[0]));
        global_flop_count += 2 * block_szs[0] * block_szs[0];
        
        for (int i = 1; i < N_thomas; i++)
        {
            Eigen::VectorXd temp = cg_r.segment(block_starts[i], block_szs[i]) - U_blocks[i - 1].transpose() * cg_d_prime.segment(block_starts[i - 1], block_szs[i - 1]);
            cg_d_prime.segment(block_starts[i], block_szs[i]) = llt_blocks[i].solve(temp);
            global_flop_count += 2 * block_szs[i] * block_szs[i] + 2 * block_szs[i] * block_szs[i - 1];
        }
        
        // Backward sweep: cg_w
        cg_w.segment(block_starts[N_thomas - 1], block_szs[N_thomas - 1]) = cg_d_prime.segment(block_starts[N_thomas - 1], block_szs[N_thomas - 1]);
        for (int i = N_thomas - 2; i >= 0; i--)
        {
            cg_w.segment(block_starts[i], block_szs[i]) = cg_d_prime.segment(block_starts[i], block_szs[i]) - C_prime[i] * cg_w.segment(block_starts[i + 1], block_szs[i + 1]);
            global_flop_count += 2 * block_szs[i] * block_szs[i + 1];
        }

        cg_p = cg_w;
        double rw = cg_r.dot(cg_w);
        global_flop_count += 2 * size;

        int k = 0;
        for (; k < max_cg_iter; k++)
        {
            apply_spd(cg_p, cg_Ap);

            double alpha = rw / cg_p.dot(cg_Ap);
            global_flop_count += 2 * size + 1;

            x += alpha * cg_p;
            cg_r -= alpha * cg_Ap;
            global_flop_count += 4 * size;

            r_norm2 = cg_r.dot(cg_r);
            global_flop_count += 2 * size;
            if (r_norm2 < cg_tol2)
            {
                k++;
                break;
            }

            //forward sweep
            cg_d_prime_new.segment(block_starts[0], block_szs[0]) = llt_blocks[0].solve(cg_r.segment(block_starts[0], block_szs[0]));
            global_flop_count += 2 * block_szs[0] * block_szs[0];
            
            for (int i = 1; i < N_thomas; i++)
            {
                Eigen::VectorXd temp = cg_r.segment(block_starts[i], block_szs[i]) - U_blocks[i - 1].transpose() * cg_d_prime_new.segment(block_starts[i - 1], block_szs[i - 1]);
                cg_d_prime_new.segment(block_starts[i], block_szs[i]) = llt_blocks[i].solve(temp);
                global_flop_count += 2 * block_szs[i] * block_szs[i] + 2 * block_szs[i] * block_szs[i - 1];
            }
            
            //backward sweep
            cg_w_new.segment(block_starts[N_thomas - 1], block_szs[N_thomas - 1]) = cg_d_prime_new.segment(block_starts[N_thomas - 1], block_szs[N_thomas - 1]);
            for (int i = N_thomas - 2; i >= 0; i--)
            {
                cg_w_new.segment(block_starts[i], block_szs[i]) = cg_d_prime_new.segment(block_starts[i], block_szs[i]) - C_prime[i] * cg_w_new.segment(block_starts[i + 1], block_szs[i + 1]);
                global_flop_count += 2 * block_szs[i] * block_szs[i + 1];
            }

            double rw_new = cg_r.dot(cg_w_new);
            global_flop_count += 2 * size;

            double beta = rw_new / rw;
            global_flop_count += 1;

            cg_p = cg_w_new + beta * cg_p;
            global_flop_count += 2 * size;

            rw = rw_new;
        }
        return k;
    }

    //admm solve
    int solve(const Eigen::VectorXd& rhs_x,
              const Eigen::VectorXd& rhs_y,
              const Eigen::VectorXd& rhs_z,
              Eigen::VectorXd& dx,
              Eigen::VectorXd& dy,
              Eigen::VectorXd& dz,
              int& total_cg_iters,
              bool warm_start_cg = false) const
    {
        int n = P.rows();
        int m = G.rows();

        Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd temp(n);

        double delta_inv = 1.0 / delta;

        //precompute rhs constants
        Eigen::VectorXd rhs_total = rhs_x;
        if (A.rows() > 0)
        {
            rhs_total += delta_inv * (AT * rhs_y);
            global_flop_count += 2 * AT.nonZeros() + 2 * n;
        }
        if (G.rows() > 0)
        {
            rhs_total += GT * z_reg_inv.cwiseProduct(rhs_z);
            global_flop_count += m + 2 * GT.nonZeros() + n;
        }

        total_cg_iters = 0;
        int iter = 0;
        for (; iter < max_admm_iter; iter++)
        {
            //x-update
            temp = rhs_total + rho * (z - u);
            global_flop_count += 3 * n;

            int cg_iters = solve_cg(temp, dx, warm_start_cg);
            total_cg_iters += cg_iters;

            //z-update
            z = dx + u;
            global_flop_count += n;

            //u-update
            u += dx - z;
            global_flop_count += 2 * n;

            //check convergence
            double diff_norm2 = (dx - z).squaredNorm();
            if (diff_norm2 < admm_tol * admm_tol)
            {
                global_flop_count += 3 * n;
                iter++;
                break;
            }
            global_flop_count += 3 * n;
        }

        //recover dual steps
        if (A.rows() > 0)
        {
            dy = delta_inv * (A * dx - rhs_y);
            global_flop_count += 2 * A.nonZeros() + 2 * A.rows();
        }
        if (G.rows() > 0)
        {
            dz = z_reg_inv.cwiseProduct(G * dx - rhs_z);
            global_flop_count += 2 * G.nonZeros() + 2 * m;
        }

        return iter;
    }
};

class ProximalIPMSolver {
private:
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd c;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    Eigen::SparseMatrix<double> G;
    Eigen::VectorXd h;

    int n, p, m;

    //ipm parameters
    int max_ipm_iter = 100;
    double ipm_tol = 1e-6; 
    double centering_param = 0.2;
    double tau = 0.99; //must stay

    //pmm regularization terms
    double delta = 1e-8;
    double x_reg_val = 1e-8;
    double sigma_z = 1e-8;

    //precond set
    int precond_block_size = 2;
    double beta_rho = 0.1;

    //lower cg tol
    double cg_tol_val = 1e-4;

public:
    ProximalIPMSolver(const Eigen::SparseMatrix<double>& P_in,
                      const Eigen::VectorXd& c_in,
                      const Eigen::SparseMatrix<double>& A_in,
                      const Eigen::VectorXd& b_in,
                      const Eigen::SparseMatrix<double>& G_in,
                      const Eigen::VectorXd& h_in)
        : P(P_in), c(c_in), A(A_in), b(b_in), G(G_in), h(h_in)
    {
        n = P.rows();
        p = A.rows();
        m = G.rows();
    }

    void set_settings(int max_iter, double tol, double centering)
    {
        max_ipm_iter = max_iter;
        ipm_tol = tol;
        centering_param = centering;
    }

    void set_cg_tolerance(double tol)
    {
        cg_tol_val = tol;
    }

    void set_regularization(double delta_in, double x_reg_in, double sigma_z_in)
    {
        delta = delta_in;
        x_reg_val = x_reg_in;
        sigma_z = sigma_z_in;
    }

    void set_preconditioner_block_size(int size)
    {
        precond_block_size = size;
    }
    void set_beta_rho(double beta_in)
    {
        beta_rho = beta_in;
    }

    bool solve(Eigen::VectorXd& x_sol, Eigen::VectorXd& s_sol, Eigen::VectorXd& y_sol, Eigen::VectorXd& z_sol,
               int& total_admm_iters, int& total_cg_iters, bool warm_start = false, bool warm_start_cg = false)
    {
        //init variables /(warm start)
        if (!warm_start)
        {
            x_sol = Eigen::VectorXd::Zero(n);
            s_sol = Eigen::VectorXd::Ones(m);
            y_sol = Eigen::VectorXd::Zero(p);
            z_sol = Eigen::VectorXd::Ones(m);
        }
        else
        {
            if (x_sol.size() != n) x_sol = Eigen::VectorXd::Zero(n);
            if (s_sol.size() != m) s_sol = Eigen::VectorXd::Ones(m);
            if (y_sol.size() != p) y_sol = Eigen::VectorXd::Zero(p);
            if (z_sol.size() != m) z_sol = Eigen::VectorXd::Ones(m);

            //project slack / dual variables (interior shifted barrier protocol)
            double current_gap = (m > 0) ? s_sol.dot(z_sol) / m : 1e-4;
            double target_shift = std::max(1e-4, std::min(0.1, std::sqrt(current_gap)));

            for (int i = 0; i < m; i++)
            {
                if (s_sol(i) < 1e-4) s_sol(i) = target_shift;
                if (z_sol(i) < 1e-4) z_sol(i) = target_shift;
                if (s_sol(i) * z_sol(i) < 1e-6)
                {
                    s_sol(i) += 1e-3;
                    z_sol(i) += 1e-3;
                }
            }
        }

        //prox center
        Eigen::VectorXd x_bar = x_sol;
        Eigen::VectorXd y_bar = y_sol;
        Eigen::VectorXd z_bar = z_sol;

        //init kkt solver
        ADMMKKTSolver kkt_solver(P, A, G, precond_block_size);
        kkt_solver.set_cg_tolerance(cg_tol_val);

        Eigen::VectorXd dx = Eigen::VectorXd::Zero(n), dy = Eigen::VectorXd::Zero(p), dz = Eigen::VectorXd::Zero(m), ds = Eigen::VectorXd::Zero(m);
        Eigen::VectorXd r_dual(n), r_eq(p), r_ineq(m);

        total_admm_iters = 0;
        total_cg_iters = 0;

        for (int iter = 0; iter < max_ipm_iter; iter++)
        {
            //residuals
            r_dual = P * x_sol + c + (x_reg_val * (x_sol - x_bar));
            global_flop_count += 2 * P.nonZeros() + 4 * n;

            if (p > 0)
            {
                r_dual += A.transpose() * y_sol;
                r_eq = A * x_sol - b - delta * (y_sol - y_bar);
                global_flop_count += 2 * A.nonZeros() + n + 2 * A.nonZeros() + 4 * p;
            }
            else
            {
                r_eq = Eigen::VectorXd::Zero(0);
            }

            if (m > 0)
            {
                r_dual += G.transpose() * z_sol;
                r_ineq = G * x_sol + s_sol - h - sigma_z * (z_sol - z_bar);
                global_flop_count += 2 * G.nonZeros() + n + 2 * G.nonZeros() + 5 * m;
            }
            else
            {
                r_ineq = Eigen::VectorXd::Zero(0);
            }

            double duality_gap = (m > 0) ? s_sol.dot(z_sol) : 0.0;
            if (m > 0) global_flop_count += 2 * m;

            //check convergence
            double r_dual_norm = r_dual.lpNorm<Eigen::Infinity>();
            double r_eq_norm = (p > 0) ? r_eq.lpNorm<Eigen::Infinity>() : 0.0;
            double r_ineq_norm = (m > 0) ? r_ineq.lpNorm<Eigen::Infinity>() : 0.0;
            double relative_gap = (m > 0) ? duality_gap / m : 0.0;

            if (r_dual_norm < ipm_tol && r_eq_norm < ipm_tol && r_ineq_norm < ipm_tol && relative_gap < ipm_tol)
            {
                std::cout << "Iter " << iter << ": Dual Res = " << r_dual_norm 
                          << ", Eq Res = " << r_eq_norm << ", Ineq Res = " << r_ineq_norm 
                          << ", Gap = " << relative_gap << " (IPM Converged!)" << std::endl;
                return true;
            }

            //barrier mu & adaptive rho
            double mu_curr = (m > 0) ? (duality_gap / m) : 0.0;
            double r_eq_inf = (p > 0) ? r_eq.lpNorm<Eigen::Infinity>() : 0.0;
            double r_ineq_inf = (m > 0) ? r_ineq.lpNorm<Eigen::Infinity>() : 0.0;
            double adaptive_rho = std::max(1e-8, beta_rho * std::max(r_eq_inf, r_ineq_inf));

            //update regularization params
            Eigen::VectorXd x_reg_vec = Eigen::VectorXd::Constant(n, x_reg_val);
            Eigen::VectorXd z_reg_vec(m);
            for (int i = 0; i < m; i++)
            {
                z_reg_vec(i) = s_sol(i) / z_sol(i) + sigma_z;
                global_flop_count += 2;
            }
            kkt_solver.update(delta, x_reg_vec, z_reg_vec, adaptive_rho);

            Eigen::VectorXd rhs_x = -r_dual;
            Eigen::VectorXd rhs_y = -r_eq;
            Eigen::VectorXd rhs_z(m);

            //step
            double mu = (m > 0) ? centering_param * (duality_gap / m) : 0.0;
            for (int i = 0; i < m; i++)
            {
                rhs_z(i) = -r_ineq(i) + s_sol(i) - mu / z_sol(i);
            }

            int step_cg_iters = 0;
            int step_admm_iters = kkt_solver.solve(rhs_x, rhs_y, rhs_z, dx, dy, dz, step_cg_iters, warm_start_cg);

            for (int i = 0; i < m; i++)
            {
                ds(i) = -s_sol(i) + mu / z_sol(i) - (s_sol(i) / z_sol(i)) * dz(i);
            }

            total_admm_iters += step_admm_iters;
            total_cg_iters += step_cg_iters;

            std::cout << "Iter " << iter << ": Dual Res = " << r_dual_norm 
                      << ", Eq Res = " << r_eq_norm << ", Ineq Res = " << r_ineq_norm 
                      << ", Gap = " << relative_gap 
                      << ", Inner ADMM Iters = " << step_admm_iters 
                      << ", Inner CG Iters = " << step_cg_iters << std::endl;

            //step size
            double alpha_prim = 1.0;
            double alpha_dual = 1.0;

            for (int i = 0; i < m; i++)
            {
                if (ds(i) < 0)
                {
                    alpha_prim = std::min(alpha_prim, -tau * s_sol(i) / ds(i)); //fraction - to - boundary rule
                    global_flop_count += 3;
                }
                if (dz(i) < 0)
                {
                    alpha_dual = std::min(alpha_dual, -tau * z_sol(i) / dz(i));
                    global_flop_count += 3;
                }
            }

            //updates
            x_sol += alpha_prim * dx;
            s_sol += alpha_prim * ds;
            global_flop_count += 2 * n + 2 * m;

            if (p > 0)
            {
                y_sol += alpha_dual * dy;
                global_flop_count += 2 * p;
            }
            if (m > 0)
            {
                z_sol += alpha_dual * dz;
                global_flop_count += 2 * m;
            }

            //shift pmm centers
            x_bar = x_sol;
            y_bar = y_sol;
            z_bar = z_sol;
        }

        std::cout << "IPM failed converging within max iterations." << std::endl;
        return false;
    }
};

} // namespace ipm_admm_cg

#endif 
