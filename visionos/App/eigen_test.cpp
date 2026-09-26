// Eigen's product_apple_amx test (third_party/src/eigen-amx) as the runner program eigen_test.
#define Eigen EigenAmxTest
#define main eigen_test_main
#define EIGEN_TEST_PART_ALL 1
#include "../../third_party/src/eigen-amx/test/product_apple_amx.cpp"
