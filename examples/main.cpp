#include <iostream>

// Forward declarations of example functions
void example1();
void example2();
void example3(bool warm_start = false);
void example4(bool warm_start = false);

int main(int argc, char** argv)
{
    std::cout << "IPM-ADMM-CG QP Solver Demonstration" << std::endl;

    // Run Example 1 (Basic 2D QP with equality & inequality constraints)
    std::cout << "\n Running Example 1 \n" << std::endl;
    example1();

    // Run Example 2 (3D QP)
    std::cout << "\n Running Example 2 \n" << std::endl;
    example2();

    // Run Example 3 (Large-Scale Sparse QP - Cold and Warm Start)
    std::cout << "\n Running Example 3 (Cold Start) \n" << std::endl;
    example3(false);

    std::cout << "\n Running Example 3 (Warm Start) \n" << std::endl;
    example3(true);

    // Run Example 4 (Wang-Boyd Mass-Spring System - Cold and Warm Start)
    std::cout << "\n Running Example 4 (Cold Start) \n" << std::endl;
    example4(false);

    std::cout << "\n Running Example 4 (Warm Start) \n" << std::endl;
    example4(true);

    std::cout << "All Examples Completed" << std::endl;


    return 0;
}

