// bench_eigen_gemv, Eigen with the Apple AMX GEMV at every size.
#define Eigen EigenGemvAmxAll
#define main eigen_gemv_amx_all_main
#define EIGEN_ARM64_USE_APPLE_AMX
#define EIGEN_APPLE_AMX_GEMV_MIN_ROW_BYTES 0
#define EIGEN_APPLE_AMX_GEMV_MIN_BYTES 0
#define BENCH_EIGEN_NAME "eigen-amx-all"
#include "../../bench/bench_eigen_gemv.cpp"
