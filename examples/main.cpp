#include <iostream>

// Forward declarations of example functions
void example1();
void example2();
void example3(bool warm_start = false);
void example4(bool warm_start = false);

int main(int argc, char** argv)
{
    std::cout << "==========================================================" << std::endl;
    std::cout << "          IPM-ADMM-CG QP Solver Demonstration             " << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    // Run Example 1 (Basic 2D QP with equality & inequality constraints)
    std::cout << "\n>>> RUNNING EXAMPLE 1 <<<\n" << std::endl;
    example1();

    // Run Example 2 (3D QP)
    std::cout << "\n>>> RUNNING EXAMPLE 2 <<<\n" << std::endl;
    example2();

    // Run Example 3 (Large-Scale Sparse QP - Cold and Warm Start)
    std::cout << "\n>>> RUNNING EXAMPLE 3 (Cold Start) <<<\n" << std::endl;
    example3(false);

    std::cout << "\n>>> RUNNING EXAMPLE 3 (Warm Start) <<<\n" << std::endl;
    example3(true);

    // Run Example 4 (Wang-Boyd Mass-Spring System - Cold and Warm Start)
    std::cout << "\n>>> RUNNING EXAMPLE 4 (Cold Start) <<<\n" << std::endl;
    example4(false);

    std::cout << "\n>>> RUNNING EXAMPLE 4 (Warm Start) <<<\n" << std::endl;
    example4(true);

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "           All Examples Completed Successfully!           " << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}

