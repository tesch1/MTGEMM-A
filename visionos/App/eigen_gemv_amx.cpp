// bench_eigen_gemv, Eigen with the Apple AMX GEMV and its default crossover.
#define Eigen EigenGemvAmx
#define main eigen_gemv_amx_main
#define EIGEN_ARM64_USE_APPLE_AMX
#define BENCH_EIGEN_NAME "eigen-amx"
#include "../../bench/bench_eigen_gemv.cpp"
