# Standalone benchmark executable
add_executable(benchmark_mass_spring_scratch scratch_mass_spring_scaling.cpp)
target_link_libraries(benchmark_mass_spring_scratch PRIVATE ipm_admm_cg)

# Check for optional external solvers (OSQP, PIQP)
find_package(osqp QUIET)
find_package(piqp QUIET)

if(TARGET osqp::osqp OR OSQP_FOUND)
    message(STATUS "Found OSQP for benchmarks")
endif()

if(TARGET piqp::piqp OR PIQP_FOUND)
    message(STATUS "Found PIQP for benchmarks")
endif()
