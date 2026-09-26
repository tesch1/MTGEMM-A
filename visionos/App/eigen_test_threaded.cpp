// Eigen's product_threaded test with the Apple AMX GEMM as the runner program eigen_test_threaded.
#define Eigen EigenAmxTest_threaded
#define main eigen_test_threaded_main
#define EIGEN_TEST_PART_ALL 1
#define EIGEN_ARM64_USE_APPLE_AMX
#include "../../third_party/src/eigen-amx/test/product_threaded.cpp"
