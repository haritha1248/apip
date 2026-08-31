# Mathematical Formulation and Algorithmic Framework of the IPM-ADMM-CG Solver

This document details the mathematical derivation, algorithms, source citations, ADMM inner solver formulation, warm start protocols, and FLOP counting complexity analysis for the Proximal Interior Point Method (PIPM) [1], Alternating Direction Method of Multipliers (ADMM) inner solver [2, 3], and Block Thomas Preconditioned Conjugate Gradient (PCG) solver [4, 6], as implemented in admm_kkt_solver.hpp.

---

## I. Problem Formulation

The solver addresses general convex Quadratic Programs (QP) in standard form [1, 2, 7]:

Minimize over x in R^n:
  (1/2) * x^T * P * x + c^T * x

Subject to constraints:
  A * x = b                     (Equality constraints, A in R^(p x n)) [1, 7]
  G * x + s = h,  s >= 0        (Inequality constraints with slacks s in R^m, G in R^(m x n)) [1]

Where:
- P in R^(n x n) is a symmetric positive semi-definite cost matrix (P >= 0) [1, 2].
- c in R^n is the linear cost vector [1].
- A in R^(p x n) and b in R^p define linear equality constraints [1, 7].
- G in R^(m x n), h in R^m, and non-negative slack vector s in R^m define linear inequality constraints [1].
- y in R^p represents Lagrange multipliers (dual variables) for equality constraints [1, 3].
- z in R^m (z >= 0) represents Lagrange multipliers (dual variables) for inequality constraints [1].

### I.A Regularized Primal-Dual Residuals & Perturbed KKT Conditions

To handle ill-conditioned matrices, zero eigenvalues in P, or rank-deficient constraint matrices A or G, the solver applies the Primal-Dual Proximal Method of Multipliers (PMM) centered around reference point (x_bar, y_bar, z_bar) [1]:

1. Dual Residual [1]:
   r_dual = P * x + c + X_reg * (x - x_bar) + A^T * y + G^T * z
   where X_reg = x_reg_val * I_n (with x_reg_val > 0) is the primal proximal regularization matrix [1].

2. Equality Primal Residual [1]:
   r_eq = A * x - b - delta * (y - y_bar)
   where delta > 0 is the equality dual Tikhonov regularization parameter [1].

3. Inequality Primal Residual [1]:
   r_ineq = G * x + s - h - sigma_z * (z - z_bar)
   where sigma_z > 0 is the inequality dual regularization parameter [1].

4. Perturbed Complementarity Condition [1, 4]:
   S * Z * e = mu * e   =>   s_i * z_i = mu  for all i = 1, ..., m
   where S = diag(s), Z = diag(z), e = [1, ..., 1]^T in R^m, and mu = sigma * (s^T * z) / m is the barrier parameter with centering parameter sigma in (0, 1) [1, 4].

### I.B Condensed KKT System

Newton's method applied to regularized perturbed KKT conditions leads to search directions (dx, ds, dy, dz) [1]. Condensing ds yields [1]:

ds = -s + mu * Z^(-1) * e - Z^(-1) * S * dz [1]

Defining W_z = Z^(-1) * S + sigma_z * I_m = diag( s_i / z_i + sigma_z ) in R^(m x m) [1], the condensed 3-block saddle-point KKT system is [1]:

[ P + X_reg    A^T      G^T  ] [ dx ]   [ rhs_x ]
[     A     -delta*I     0   ] [ dy ] = [ rhs_y ]
[     G        0       -W_z  ] [ dz ]   [ rhs_z ]

where rhs_x = -r_dual, rhs_y = -r_eq, and rhs_z = -r_ineq + s - mu * Z^(-1) * e [1].

---

## II. ADMM Inner Solver Formulation

To solve the condensed 3-block KKT system without explicit indefinite factorizations, dual search directions dy and dz are expressed directly in terms of dx [1, 2]:

dy = (1 / delta) * (A * dx - rhs_y) [1]
dz = W_z^(-1) * (G * dx - rhs_z) [1]

Substituting dy and dz into the first block equation yields the unconstrained reduced symmetric positive-definite (SPD) system [1, 2]:

( P + X_reg + (1 / delta) * A^T * A + G^T * W_z^(-1) * G ) * dx = rhs_total [1, 2]

where rhs_total = rhs_x + (1 / delta) * A^T * rhs_y + G^T * W_z^(-1) * rhs_z [1].

### II.A Operator Splitting & Augmented Lagrangian

ADMM splits the reduced problem by introducing an auxiliary variable z_tilde in R^n and consensus constraint dx - z_tilde = 0 [2, 3]:

Minimize over dx, z_tilde in R^n:
  (1/2) * dx^T * ( P + X_reg + (1 / delta) * A^T * A + G^T * W_z^(-1) * G ) * dx - rhs_total^T * dx
Subject to constraint:
  dx - z_tilde = 0 [2, 3]

The Augmented Lagrangian with penalty parameter rho > 0 is defined as [2, 3]:

L_rho(dx, z_tilde, u) = (1/2) * dx^T * ( P + X_reg + (1 / delta) * A^T * A + G^T * W_z^(-1) * G ) * dx - rhs_total^T * dx + (rho / 2) * || dx - z_tilde + u ||_2^2 - (rho / 2) * || u ||_2^2 [2, 3]

where u in R^n is the scaled dual multiplier vector [2, 3].

#### Adaptive Penalty Parameter ($\rho$) Scheme [2, 3]
To balance primal and dual residual rates of convergence dynamically during IPM iterations [2, 3], $\rho$ is adaptively updated prior to each inner ADMM solve based on the current IPM primal infeasibility residuals [1, 2]:

$$\rho = \max\left( 10^{-8}, \, \beta_{\rho} \cdot \max\left( \| r_{\text{eq}} \|_{\infty}, \, \| r_{\text{ineq}} \|_{\infty} \right) \right)$$

where $\beta_{\rho} \in (0, 1)$ (default $\beta_{\rho} = 0.1$) scales the penalty parameter proportionally to constraint violation [1, 2].

### II.B ADMM Iteration Updates

Each inner ADMM iteration k updates the variables sequentially [2, 3]:

1. Primal dx-Step [2, 3]:
   Solve M_spd * dx^(k+1) = rhs_total + rho * (z_tilde^k - u^k)
   where M_spd = P + X_reg + rho * I_n + (1 / delta) * A^T * A + G^T * W_z^(-1) * G in R^(n x n) is symmetric positive definite [1, 2].

2. Auxiliary z_tilde-Step [2, 3]:
   z_tilde^(k+1) = dx^(k+1) + u^k

3. Dual Multiplier u-Step [2, 3]:
   u^(k+1) = u^k + dx^(k+1) - z_tilde^(k+1)

### II.C Stopping Criteria & Dual Recovery

In standard optimization literature [2, 3], ADMM iterates until the primal consensus residual satisfies:

$$\| \Delta x^{(k+1)} - \tilde{z}^{(k+1)} \|_2 < \text{admm\_tol}$$

*Implementation Note*: To avoid computing an expensive square-root (`std::sqrt`) at every inner iteration, the solver implementation evaluates the mathematically equivalent squared-norm condition [1, 2]:

$$\| \Delta x^{(k+1)} - \tilde{z}^{(k+1)} \|_2^2 < \text{admm\_tol}^2$$

Upon termination, full dual search directions $dy$ and $dz$ are recovered via [1]:
  $$dy = \frac{1}{\delta} (A \cdot \Delta x - \text{rhs}_y)$$
  $$dz = W_z^{-1} (G \cdot \Delta x - \text{rhs}_z)$$

---

## III. Block Thomas Preconditioned Conjugate Gradient (PCG)

The linear solve M_spd * dx = b_admm inside each ADMM iteration is performed using Preconditioned Conjugate Gradient (PCG) [4, 5].

For mass-spring dynamics and optimal control structures [7], M_spd exhibits a block-tridiagonal sparsity pattern [6]:

M_spd =
[ D_1    U_1     0     0  ]
[ U_1^T  D_2    U_2    0  ]
[  0    U_2^T  ...   U_(N-1) ]
[  0     0    U_(N-1)^T D_N ]

### III.A Block Thomas Preconditioner Setup

1. First Block [6]:
   Gamma_1 = D_1. Compute Cholesky factorization Gamma_1 = L_1 * L_1^T [6].

2. Intermediate Blocks (i = 1, ..., N-1) [6]:
   C'_i = Gamma_i^(-1) * U_i = (L_i * L_i^T)^(-1) * U_i [6]
   Gamma_(i+1) = D_(i+1) - U_i^T * C'_i. Compute Cholesky factorization Gamma_(i+1) = L_(i+1) * L_(i+1)^T [6].

### III.B Preconditioner Solve Step (M_precond * w = r)

- Forward Sweep [6]:
  d'_1 = Gamma_1^(-1) * r_1 [6]
  d'_i = Gamma_i^(-1) * (r_i - U_(i-1)^T * d'_(i-1))  for i = 2, ..., N [6]

- Backward Substitution [6]:
  w_N = d'_N [6]
  w_i = d'_i - C'_i * w_(i+1)  for i = N-1, ..., 1 [6]

---

## IV. Warm Starts & FLOP Counting Complexity Analysis

### IV.A Warm Start Protocols

#### 1. Interior-Shifted Barrier Protocol for Warm Starts [1]
When warm-starting sequential QPs (e.g. in MPC or sequential convex updates), dual or slack variables may lie near constraint boundaries (s_i -> 0 or z_i -> 0), causing numerical degeneration in logarithmic barrier terms [1].

The Interior-Shifted Barrier Protocol projects and shifts variables into the positive interior [1]:
- Calculate current duality gap average: current_gap = (s^T * z) / m [1].
- Determine adaptive target shift: target_shift = max(1e-4, min(0.1, sqrt(current_gap))) [1].
- For each constraint i = 1, ..., m [1]:
  If s_i < 1e-4  =>  s_i = target_shift [1]
  If z_i < 1e-4  =>  z_i = target_shift [1]
  If s_i * z_i < 1e-6  =>  s_i = s_i + 1e-3,  z_i = z_i + 1e-3 [1]

#### 2. CG Warm Starting [4, 5]
For linear system solves M_spd * dx = b_admm across ADMM iterations, warm starting CG with previous solution dx^0 = dx_prev initializes residual r^0 = b_admm - M_spd * dx^0 [4, 5], accelerating CG convergence significantly when direction steps change smoothly [4, 5].

### IV.B FLOP Counting Complexity Analysis

#### 1. General QP Dimension Definitions
In the general QP formulation [1, 2, 7]:
- $n$: Number of primal decision variables ($x \in \mathbb{R}^n$).
- $p$: Number of equality constraints ($A x = b$, $A \in \mathbb{R}^{p \times n}$).
- $m$: Number of inequality constraints ($G x + s = h$, $G \in \mathbb{R}^{m \times n}$, $s, z \in \mathbb{R}^m$).
- $W_z = Z^{-1} S + \sigma_z I_m = \text{diag}(s_i / z_i + \sigma_z)$: Inequality regularized slack-dual scaling matrix [1]. Its inverse $W_z^{-1} = \text{diag}\left(\frac{1}{s_i/z_i + \sigma_z}\right)$ requires $m$ floating-point divisions/inversions [1].

#### 2. MPC Problem Formulation & Exact Variable Mapping
For the linear Model Predictive Control (MPC) problem as defined:
- **System Dimensions**:
  - $n$: State vector dimension ($x_k \in \mathbb{R}^n$, $A \in \mathbb{R}^{n \times n}$)
  - $m$: Input vector dimension ($u_k \in \mathbb{R}^m$, $B \in \mathbb{R}^{n \times m}$)
  - $N$: Prediction horizon
- **Primal Decision Variable Vector ($x$)**:
  $$x = \begin{bmatrix} u_0 \\ x_1 \\ u_1 \\ \vdots \\ u_{N-1} \\ x_N \end{bmatrix} \in \mathbb{R}^{n_x}, \quad \text{where } n_x = (n + m) N$$
- **Equality Constraints ($E x = e$)**:
  State dynamic transition equations $x_{k+1} - A x_k - B u_k = 0$:
  - Dimension of equality constraints: $n_e = n N$.
  - Matrix $E \in \mathbb{R}^{n_e \times n_x}$ is block-lower-bidiagonal.
- **Inequality / Box Constraints ($x \in \mathcal{X}$)**:
  Upper and lower bounds $x_{\min} \le x_k \le x_{\max}$ and $u_{\min} \le u_k \le u_{\max}$:
  - Formulated as $G x + s = h$ with $G = \begin{bmatrix} I_{n_x} \\ -I_{n_x} \end{bmatrix} \in \mathbb{R}^{n_{\text{ineq}} \times n_x}$.
  - Total number of scalar inequality constraints: $n_{\text{ineq}} = 2 n_x = 2 (n + m) N$.

#### 3. MPC FLOP Counting Breakdown per Iteration

1. **Sparse Matrix & Precomputation Operations [1, 7]**:
   - **Cost matrix multiplication $P x$**: $2 n_x$ flops (if $Q, R$ diagonal) or $2 N (n^2 + m^2)$ flops (if block-diagonal).
   - **Equality SpMV $E x$ & $E^T y$**: $2 \cdot \text{nnz}(E) = 2 N (n + \text{nnz}(A) + \text{nnz}(B))$ flops per multiplication ($2 n_e$ for row dimension $n_e = n N$).
   - **Inequality SpMV $G x$ & $G^T z$**: $G = [I_{n_x}; -I_{n_x}] \implies G^T z = z_1 - z_2$, requiring $n_x$ subtractions ($n_x$ flops).
   - **$E^T E$ Precomputation**: $2 N n^2 (n + m)$ flops (forming block-tridiagonal matrix of block size $B = n + m$).

2. **Vector & Diagonal Updates [1]**:
   - **Inversion of $W_z$ ($W_z^{-1}$ calculation)**: $n_{\text{ineq}} = 2 n_x = 2 (n + m) N$ flops (divisions) [1].
   - **Diagonal matrix update $G^T W_z^{-1} G$**: $n_x$ flops ($W_{z,1}^{-1} + W_{z,2}^{-1}$).
   - **Dual step updates $dy$ and $dz$**: $2 \cdot \text{nnz}(E) + 2 n_e + 6 n_x$ flops [1].

3. **Block Thomas Preconditioner Setup [6, 7]**:
   - **Stage block size**: $B = n + m$, number of horizon blocks = $N$.
   - **Cholesky factorizations ($N$ blocks)**: $N \left( \frac{1}{3} B^3 + B^2 \right) = N \left( \frac{1}{3} (n + m)^3 + (n + m)^2 \right)$ flops [6].
   - **Off-diagonal block elimination**: $(N-1) B^3 = (N-1) (n + m)^3$ flops [6].

4. **PCG Iteration Complexity [4, 6]**:
   - **Matrix-vector multiply $M_{\text{spd}} p$**: $2 \cdot \text{nnz}(M_{\text{spd}}) + 5 n_x = 2 N \left( (n + m)^2 + 2 n (n + m) \right) + 5 (n + m) N$ flops [1, 7].
   - **Block Thomas forward/backward solve per CG iteration**: $4 N (n + m)^2 - 2 (n + m)^2$ flops [6].
   - **Vector dot products & AXPY updates**: $10 n_x = 10 (n + m) N$ flops per CG step [4].

---

## V. Line Search & Fraction-to-the-Boundary Rule

To maintain strict interior positivity of slacks s > 0 and duals z > 0 [1, 4]:

alpha_prim = min( 1, tau * min_{i: ds_i < 0} ( -s_i / ds_i ) ) [1, 4]
alpha_dual = min( 1, tau * min_{i: dz_i < 0} ( -z_i / dz_i ) ) [1, 4]

where tau = 0.99 is the boundary buffer fraction [1].

Variable updates [1]:
  x^(k+1) = x^k + alpha_prim * dx
  s^(k+1) = s^k + alpha_prim * ds
  y^(k+1) = y^k + alpha_dual * dy
  z^(k+1) = z^k + alpha_dual * dz

---

## References

[1] Schwan, R., Jiang, Y., Kuhn, D., & Jones, C. N. (2023). PIQP: A Proximal Interior-Point Quadratic Programming Solver. IEEE Transactions on Automatic Control. (Cited in Sections I, I.A, I.B, II, II.A, II.B, II.C, IV.A, IV.B, V)

[2] Stellato, B., Banerjee, A., Goulart, P., Bemporad, A., & Boyd, S. (2020). OSQP: An operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4), 637-672. (Cited in Sections I, II, II.A, II.B, II.C, IV.B)

[3] Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine Learning, 3(1), 1-122. (Cited in Sections I, II, II.A, II.B, II.C)

[4] Gondzio, J. (2012). Interior point methods 25 years later. European Journal of Operational Research, 218(3), 587-601. (Cited in Sections I.A, III, IV.A, IV.B, V)

[5] Al-Jeiroudi, E., & Gondzio, J. (2009). Convergence analysis of inexact infeasible interior point method with preconditioner update. Optimization Methods and Software, 24(3), 357-376. (Cited in Sections III, IV.A)

[6] Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press. (Section 4.5: Block Tridiagonal Systems). (Cited in Sections III, III.A, III.B, IV.B)

[7] Frison, G., & Diehl, M. (2020). HPIPM: a high-performance interior-point method solver for quadratic programming. IFAC-PapersOnLine, 53(2), 6563-6569. (Cited in Sections I, III, IV.B)

[8] Mehrotra, S. (1992). On the implementation of a primal-dual interior point method. SIAM Journal on Optimization, 2(4), 575-601. (Barrier parameter update $\mu = \sigma \frac{s^T z}{m}$ origin).

[9] Wright, S. J. (1997). Primal-Dual Interior-Point Methods. Society for Industrial and Applied Mathematics (SIAM). (Primal-Dual interior point parameter theory).

