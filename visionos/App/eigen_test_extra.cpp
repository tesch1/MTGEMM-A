// Eigen's product_extra test with the Apple AMX GEMM as the runner program eigen_test_extra.
#define Eigen EigenAmxTest_extra
#define main eigen_test_extra_main
#define EIGEN_TEST_PART_ALL 1
#define EIGEN_ARM64_USE_APPLE_AMX
#include "../../third_party/src/eigen-amx/test/product_extra.cpp"
