add_executable(ipm_admm_cg_examples
    main.cpp
    example1.cpp
    example2.cpp
    example3.cpp
    example4.cpp
)

target_link_libraries(ipm_admm_cg_examples PRIVATE ipm_admm_cg)
