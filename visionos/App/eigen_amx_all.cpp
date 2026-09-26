// bench_eigen, Eigen with the Apple AMX GEMM at every size (no crossover to NEON), one thread.
#define Eigen EigenAmxAll
#define main eigen_amx_all_main
#define BENCH_EIGEN_NEON
#define EIGEN_ARM64_USE_APPLE_AMX
#define EIGEN_APPLE_AMX_MIN_RESULT_SIZE 0
#define EIGEN_APPLE_AMX_MIN_WORK_BYTES 0
#define BENCH_EIGEN_NAME "eigen-amx-all"
#include "../../bench/bench_eigen.cpp"
