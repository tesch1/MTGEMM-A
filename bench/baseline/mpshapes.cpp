// The MpGEMM paper's 24 FP32 workloads (arXiv:2512.21473, Table), column-major C += A*B, single thread.
// Mode "eigen" or "accel"; min over trials of >= 50 ms.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <Eigen/Core>
#include <chrono>
#include <cstdio>
#include <cstring>
using namespace Eigen;
static const int W[24][3] = {{64,2112,7168},{64,24576,1536},{64,32768,512},{64,7168,16384},{64,4096,7168},{64,7168,2048},
 {128,2112,7168},{128,24576,1536},{128,32768,512},{128,7168,16384},{128,4096,7168},{128,7168,2048},
 {4096,2112,7168},{4096,24576,1536},{4096,32768,512},{4096,7168,16384},{4096,4096,7168},{4096,7168,2048},
 {4096,256,4096},{11008,256,4096},{4096,256,11008},{5120,256,5120},{13824,256,5120},{5120,256,13824}};
int main(int argc, char** argv) {
  const bool accel = argc > 1 && !std::strcmp(argv[1], "accel");
  for (int id = 0; id < 24; ++id) {
    const int m = W[id][0], n = W[id][1], k = W[id][2];
    MatrixXf A = MatrixXf::Random(m, k), B = MatrixXf::Random(k, n), C = MatrixXf::Zero(m, n);
    auto run = [&] {
      if (accel) cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A.data(), m, B.data(), k, 1.f, C.data(), m);
      else C.noalias() += A * B;
    };
    run();
    double best = 1e30;
    for (int t = 0; t < 5; ++t) {
      int reps = 0; auto t0 = std::chrono::steady_clock::now(); double s = 0;
      do { run(); ++reps; s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (s < 0.05);
      best = std::min(best, s / reps);
    }
    std::printf("%d %d %d %d %.0f\n", id + 1, m, n, k, 2.0 * m * n * k / best / 1e9);
    std::fflush(stdout);
  }
}
