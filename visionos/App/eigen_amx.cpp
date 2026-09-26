// bench_eigen, Eigen with the Apple AMX GEMM, one thread.
#define Eigen EigenAmx
#define main eigen_amx_main
#define BENCH_EIGEN_NEON
#define EIGEN_ARM64_USE_APPLE_AMX
#define BENCH_EIGEN_NAME "eigen-amx"
#include "../../bench/bench_eigen.cpp"
