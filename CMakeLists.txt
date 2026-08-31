cmake_minimum_required(VERSION 3.16)
project(ipm_admm_cg VERSION 1.0.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# User-configurable build options
option(IPM_ADMM_CG_BUILD_EXAMPLES "Build example applications" ON)
option(IPM_ADMM_CG_BUILD_TESTS "Build test suite" ON)
option(IPM_ADMM_CG_BUILD_BENCHMARKS "Build comparative benchmark suite" OFF)

# Dependency Management: Find or Fetch Eigen3
find_package(Eigen3 3.3 QUIET CONFIG)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found on system. Fetching Eigen 3.4.0 via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    # Avoid building Eigen's own tests/examples
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
else()
    message(STATUS "Found Eigen3: ${Eigen3_VERSION}")
endif()

# Define header-only INTERFACE library target
add_library(ipm_admm_cg INTERFACE)
add_library(ipm_admm_cg::ipm_admm_cg ALIAS ipm_admm_cg)

target_include_directories(ipm_admm_cg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ipm_admm_cg INTERFACE Eigen3::Eigen)
target_compile_features(ipm_admm_cg INTERFACE cxx_std_17)

# Subdirectories
if(IPM_ADMM_CG_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(IPM_ADMM_CG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(IPM_ADMM_CG_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation and CMake Package Configuration
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ipm_admm_cg
    EXPORT ipm_admm_cgTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

set(IPM_ADMM_CG_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/ipm_admm_cg)

install(EXPORT ipm_admm_cgTargets
    FILE ipm_admm_cgTargets.cmake
    NAMESPACE ipm_admm_cg::
    DESTINATION ${IPM_ADMM_CG_CMAKE_DIR}
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ipm_admm_cgConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ipm_admm_cgConfig.cmake
    INSTALL_DESTINATION ${IPM_ADMM_CG_CMAKE_DIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ipm_admm_cgConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ipm_admm_cgConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ipm_admm_cgConfigVersion.cmake
    DESTINATION ${IPM_ADMM_CG_CMAKE_DIR}
)
