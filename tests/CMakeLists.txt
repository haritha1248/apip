add_executable(test_sparse test_sparse.cpp)
target_link_libraries(test_sparse PRIVATE ipm_admm_cg)
add_test(NAME test_sparse COMMAND test_sparse)

add_executable(test_wang_boyd test_wang_boyd.cpp)
target_link_libraries(test_wang_boyd PRIVATE ipm_admm_cg)
add_test(NAME test_wang_boyd COMMAND test_wang_boyd)
