// bench_eigen, Eigen on NEON, on a thread pool of one thread per core.
#define Eigen EigenNeonMt
#define main eigen_neon_mt_main
#define BENCH_EIGEN_NEON
#define EIGEN_GEMM_THREADPOOL
#define BENCH_EIGEN_NAME "eigen-neon-mt"
#include "../../bench/bench_eigen.cpp"
