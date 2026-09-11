# APIP: A Proximal  Interior Point-ADMM Based QP Solver

APIP is a high performance, header-only C++17 solver for generic convex QP problems arising within Model Predictive Control (MPC). 
It combines the Proximal Interior Point and an ADMM operator splitting inner solver with a Block Thomas Preconditioned Conjugate Gradient (CG) to achieve fast solve times with zero dynamic memory allocations within iterations.

## Key Features

Header only and Dependency Free: Depends only on standard C++17 and Eigen 3.4+.  

Block Thomas Preconditioning: Exploits tridiagonal and banded structure of MPC problems for $O(N)$ linear scaling per iterative loop. 

Zero Dynamic Allocations: All vectors, Cholesky factorization buffers, and sparse matrices are preallocated. 

Interior Shifted Warm Starting: dual/slack variables are interior projected to prevent boundary sticking. 

Built In FLOP count: explicity counts FLOPs per solve. 

## Mathematical Formulation

The solver addresses standard convex QPs: 

$$\begin {aligned}
\min_{x \in \mathbb{R}^n} \quad & \frac{1}{2}  x^T P x + c^T x  \\
\text{subject to} \quad & E x = e \quad (\text{Equality constraints}, \, E \in \mathbb{R} ^ {p \times n}) \\
& G x + s = h, \quad s \ge 0 \quad (\text{Inequality constraints}, \, G \in \mathbb{R} ^ {m \times n})
\end {aligned} $$

The solver uses the Proximal Interior Point framework to structure the problem as a modified Lagrangian equation:

$$\begin{align}
\mathcal{L}_{\delta}(x, s, \lambda, \nu) &= \frac{1}{2} x^T P x + q^T x + \lambda^T (Ex - e) + \frac{1}{2\delta} \| Ex - e \|_2^2 + \nu^T (Gx - h + s) + \frac{1}{2\delta} \| Gx - h + s \|_2^2,
\end{align}$$

Which then uses proximal multipliers and barrier parameters to structure the problem into a KKT structure matrix. We rewrite Newton's equations as one equation by elimination of $\delta s, \delta z,$ and $\delta y $, (slack parameters and y), which yields:

$$\begin{align}
    \left(P + \rho^k I_n + \frac{1}{\delta^k} E^T E + G^T  \mathcal{W}^{-k}  G \right)  \Delta x = \overline{r}^k
\end{align}$$

By defining:  

$$\begin{align} \Phi = P + \rho^k I_n + \frac{1}{\delta^k} E^T E + G^T \mathcal{W}^{-k} G. \end{align}$$

ADMM operator splitting can be applied to the system $$\Phi \delta x = \overline{r}^k$$.The resulting updates are:

$$\begin{align}
\begin{cases}
    \Delta x^{k+1} = \arg\min\limits_{\Delta x}   \mathcal{L}_{\beta}(\Delta x, \Delta w^k, \gamma^k), \\
    \Delta w^{k+1} = \arg\min\limits_{\Delta w}   \mathcal{L}_{\beta}(\Delta x^{k+1}, \Delta w, \gamma^k), \\
    \gamma^{k+1} = \gamma^k + (\Delta x^{k+1} - \Delta w^{k+1}).
\end{cases}
\end{align}$$

The first ADMM update is then solved using a Block PCG. This is equivalent to 

$$\begin{align} \Psi \Delta x^{k+1}=\overline{r}^k +\beta\gamma^k.\end{align}$$ 

where
$\Psi= \Phi+\beta I$.

The forward and backward substitutions are used to solve for the subsequent missing variables. 

See APIP: ADMM-based Proximal Interior Point Solver for Linear Model Predictive Control for full solver details.

## Benchmark & Scaling 

The solver was tested on mass-spring-damper example benchmarks. The problem was scaled across horizons $N = 1000$ to $10000$, as shown below:

| Horizon | Variables | Solve Time (ms) | Inner CG iterations |
| :--- | :--- | :--- | :--- |
| 1000 | 15,000 | 68.4 | 5 |
| 2000 | 30,000 | 137.8 | 5 |
| 3000 | 45,000 | 205.3 | 5 |
| 4000 | 60,000 | 274.9 | 5 |
| 5000 | 75,000 | 340.8 | 5 |
| ... | ... | ... | ... |


This has minimal PCG iterations with consistent $O(N)$ runtime. 

## Integration 

### Using CMake FetchContent

Adding to CMake projects directly:

    include (FetchContent)
    FetchContent_Declare(
        ipm_admm_cg
        GIT_REPOSITORY https://github.com/haritha1248/apip
        GIT_TAG MAIN
    )
    FetchContent_MakeAvailable(ipm_admm_cg)

    target_link_libraries (my_target PRIVATE ipm_admm_cg)

Or, add /include/ipm_admm_cg/admm_kkt_solver.hpp directly and ensure Eigen 3.4+ is available.

## Building Examples and Tests

#Clone the repository:

    git clone https://github.com/haritha1248/apip
    cd ipm_admm_cg

    cmake -B build -DIPM_ADMM_CG_BUILD_EXAMPLES=ON  -DIPM_ADMM_CG_BUILD_TESTS=ON

    cmake –build build –config Release

    Ctest –test-dir build –output-on-failure -C Release

    #run examples
    ./build/examples/Release/ipm_admm_cg_examples

## License 

This project is licensed under the MIT license. See LICENSE for details.

## References

1. Y. Wang and S. Boyd, “Fast model predictive control using online
optimization,” in IEEE Transactions on control systems technology.,
vol. 18, no. 2, pp. 267–278, 2010.
2. A. Malyshev, R. Quirynen, A. Knyazev and S. D. Cairano, “A
regularized Newton solver for linear model predictive control,” in
European Control Conference., Limassol, Cyprus, 2018, pp. 1393-
1398.
3. R. Schwan, Y. Jiang, D. Kuhn and C. N. Jones, “PIQP: A Proximal
Interior-Point Quadratic Programming Solver,” in IEEE Conference on
Decision and Control., Singapore, Singapore, 2023, pp. 1088–1093.
4. R. Schwan, D. Kuhn and C. N. Jones, “Exploiting multistage optimiza-
tion structure in proximal solvers,” in IEEE Conference on Decision
and Control., Rio de Janeiro, Brazil, 2025, pp. 4677-4683.
5. G. Frison, and M. Diehl, “HPIPM: a high-performance interior-point
method solver for quadratic programming,” in IFAC-PapersOnLine.,
vol. 53, no. 2, pp. 6563–6569, 2020.
6. B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, “OSQP:
An operator splitting solver for quadratic programs,” in Mathematical
Programming Computation., vol. 12, no. 4, pp. 637-672, 2020.
7. J. Nocedal and S. J. Wright, “Numerical Optimization,” Springer,
2006.
8. S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, “Distributed
optimization and statistical learning via the alternating direction
method of multipliers,” in Foundations and Trends® in Machine
learning., vol. 3, no. 1, pp. 1-122, 2011.
9. K. Nguyen, S. Schoedel, A. Alavilli, B. Plancher and Z. Manchester,
”TinyMPC: Model-Predictive Control on Resource-Constrained Mi-
crocontrollers,” in IEEE International Conference on Robotics and
Automation (ICRA), Yokohama, Japan, 2024, pp. 1-7.
10. K. F. Løwenstein, D. Bernardini and P. Patrinos, “QPALM-OCP: A
Newton-Type Proximal Augmented Lagrangian Solver Tailored for
Quadratic Programs Arising in Model Predictive Control,” in IEEE
Control Systems Letters., vol. 8, pp. 1349-1354, 2024.
11. H. J. Ferreau, C. Kirches, A. Potschka, H. G. Bock and M. Diehl,
“qpOASES: A parametric active-set algorithm for quadratic program-
ming,” in Mathematical Programming Computation., vol. 6, no. 4, pp.
327-363, 2014.
12. A. Bemporad, “A Numerically Stable Solver for Positive Semidefinite
Quadratic Programs Based on Nonnegative Least Squares,” in IEEE
Transactions on Automatic Control., vol. 63, no. 2, pp. 525–531, 2018.
13. J. Gondzio, “Interior point methods 25 years later,” in European
Journal of Operational Research., vol. 218, no. 3, pp. 587–601, 2012.
