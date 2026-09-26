// bench_eigen, Eigen on NEON, one thread; each variant has its own Eigen namespace, so none shares inline definitions.
#define Eigen EigenNeon
#define main eigen_neon_main
#define BENCH_EIGEN_NEON
#define BENCH_EIGEN_NAME "eigen-neon"
#include "../../bench/bench_eigen.cpp"
