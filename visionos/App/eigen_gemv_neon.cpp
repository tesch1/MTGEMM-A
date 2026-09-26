// bench_eigen_gemv, Eigen on NEON.
#define Eigen EigenGemvNeon
#define main eigen_gemv_neon_main
#define BENCH_EIGEN_NAME "eigen-neon"
#include "../../bench/bench_eigen_gemv.cpp"
