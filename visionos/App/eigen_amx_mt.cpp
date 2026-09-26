// bench_eigen, Eigen with the Apple AMX GEMM, on a thread pool of one thread per core.
#define Eigen EigenAmxMt
#define main eigen_amx_mt_main
#define BENCH_EIGEN_NEON
#define EIGEN_ARM64_USE_APPLE_AMX
#define EIGEN_GEMM_THREADPOOL
#define BENCH_EIGEN_NAME "eigen-amx-mt"
#include "../../bench/bench_eigen.cpp"
