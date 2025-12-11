cmake_minimum_required(VERSION 3.18)
project(TurboCat VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# TurboCat: Next-Generation Gradient Boosting Framework
# Designed to outperform CatBoost with:
# - GradTree: Gradient-based global tree optimization
# - Advanced loss functions (Robust Focal, LDAM, Logit-adjusted)
# - Tsallis entropy splitting criteria
# - CPU SIMD optimizations (AVX2/AVX-512)
# - Future: CUDA and Metal support
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# Compiler Flags
# ============================================================================

# Detect CPU features for SIMD
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
#check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
set(COMPILER_SUPPORTS_AVX512 FALSE)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

set(TURBOCAT_SIMD_FLAGS "")

if(COMPILER_SUPPORTS_AVX512)
    set(TURBOCAT_SIMD_FLAGS "-mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl")
    add_compile_definitions(TURBOCAT_AVX512)
    message(STATUS "TurboCat: AVX-512 support enabled")
elseif(COMPILER_SUPPORTS_AVX2)
    set(TURBOCAT_SIMD_FLAGS "-mavx2")
    add_compile_definitions(TURBOCAT_AVX2)
    message(STATUS "TurboCat: AVX2 support enabled")
endif()

if(COMPILER_SUPPORTS_FMA)
    set(TURBOCAT_SIMD_FLAGS "${TURBOCAT_SIMD_FLAGS} -mfma")
    add_compile_definitions(TURBOCAT_FMA)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${TURBOCAT_SIMD_FLAGS} -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Warnings
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# ============================================================================
# Options
# ============================================================================

option(TURBOCAT_BUILD_PYTHON "Build Python bindings" ON)
option(TURBOCAT_BUILD_TESTS "Build tests" ON)
option(TURBOCAT_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TURBOCAT_USE_OPENMP "Use OpenMP for parallelization" ON)

# ============================================================================
# Dependencies
# ============================================================================

# OpenMP
if(TURBOCAT_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "TurboCat: OpenMP support enabled")
    else()
        message(WARNING "TurboCat: OpenMP not found, parallel execution disabled")
    endif()
endif()

# Eigen (header-only)
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "TurboCat: Eigen3 not found, will download")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

# ============================================================================
# Core Library
# ============================================================================

set(TURBOCAT_CORE_SOURCES
    src/core/types.cpp
    src/core/config.cpp
    src/core/dataset.cpp
    src/tree/histogram.cpp
    src/tree/splitter.cpp
    src/tree/tree.cpp
    src/tree/gradtree.cpp
    src/boosting/loss.cpp
    src/boosting/booster.cpp
    src/utils/simd.cpp
    src/utils/threading.cpp
)

add_library(turbocat_core STATIC ${TURBOCAT_CORE_SOURCES})

target_include_directories(turbocat_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(Eigen3_FOUND)
    target_link_libraries(turbocat_core PUBLIC Eigen3::Eigen)
else()
    target_link_libraries(turbocat_core PUBLIC eigen)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(turbocat_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Python Bindings
# ============================================================================

if(TURBOCAT_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    
    pybind11_add_module(_turbocat python/bindings.cpp)
    target_link_libraries(_turbocat PRIVATE turbocat_core)
    
    # Install Python module
    install(TARGETS _turbocat LIBRARY DESTINATION python/turbocat)
endif()

# ============================================================================
# Tests
# ============================================================================

if(TURBOCAT_BUILD_TESTS)
    enable_testing()
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    
    add_executable(turbocat_tests
        tests/test_histogram.cpp
        tests/test_tree.cpp
        tests/test_gradtree.cpp
        tests/test_booster.cpp
    )
    
    target_link_libraries(turbocat_tests PRIVATE turbocat_core GTest::gtest_main)
    
    include(GoogleTest)
    gtest_discover_tests(turbocat_tests)
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(TURBOCAT_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
    
    add_executable(turbocat_bench
        benchmarks/bench_histogram.cpp
        benchmarks/bench_tree.cpp
    )
    
    target_link_libraries(turbocat_bench PRIVATE turbocat_core benchmark::benchmark)
endif()

# ============================================================================
# Install
# ============================================================================

install(TARGETS turbocat_core
    EXPORT TurboCatTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/turbocat DESTINATION include)

install(EXPORT TurboCatTargets
    FILE TurboCatTargets.cmake
    NAMESPACE TurboCat::
    DESTINATION lib/cmake/TurboCat
)

message(STATUS "")
message(STATUS "TurboCat Configuration Summary:")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD:         ${TURBOCAT_SIMD_FLAGS}")
message(STATUS "  OpenMP:       ${OpenMP_CXX_FOUND}")
message(STATUS "  Python:       ${TURBOCAT_BUILD_PYTHON}")
message(STATUS "  Tests:        ${TURBOCAT_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${TURBOCAT_BUILD_BENCHMARKS}")
message(STATUS "")
