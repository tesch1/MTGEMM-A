// bench_eigen_gemv, Eigen with the Apple AMX GEMV at every size and without its prefetch.
#define Eigen EigenGemvAmxNoPf
#define main eigen_gemv_amx_nopf_main
#define EIGEN_ARM64_USE_APPLE_AMX
#define EIGEN_APPLE_AMX_GEMV_MIN_ROW_BYTES 0
#define EIGEN_APPLE_AMX_GEMV_MIN_BYTES 0
#define EIGEN_APPLE_AMX_GEMV_PREFETCH_MIN_BYTES 1e30
#define BENCH_EIGEN_NAME "eigen-amx-nopf"
#include "../../bench/bench_eigen_gemv.cpp"
