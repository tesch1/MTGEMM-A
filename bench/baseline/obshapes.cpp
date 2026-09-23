// OpenBLAS / Accelerate sgemm, single thread, on square sizes and the MpGEMM paper's 24 shapes.
// argv[1]: "row" (RowMajor, beta 0: OpenBLAS's SME direct path) or "col" (ColMajor, beta 0).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#ifdef USE_ACCEL
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
extern "C" char* openblas_get_corename();
#endif
static const int W[][3] = {{512,512,512},{1000,1000,1000},{1024,1024,1024},{2048,2048,2048},{3000,3000,3000},{4096,4096,4096},
 {64,2112,7168},{64,24576,1536},{64,32768,512},{64,7168,16384},{64,4096,7168},{64,7168,2048},
 {128,2112,7168},{128,24576,1536},{128,32768,512},{128,7168,16384},{128,4096,7168},{128,7168,2048},
 {4096,2112,7168},{4096,24576,1536},{4096,32768,512},{4096,7168,16384},{4096,4096,7168},{4096,7168,2048},
 {4096,256,4096},{11008,256,4096},{4096,256,11008},{5120,256,5120},{13824,256,5120},{5120,256,13824}};
int main(int argc, char** argv) {
  const bool row = argc > 1 && !std::strcmp(argv[1], "row");
#ifndef USE_ACCEL
  std::printf("# core %s\n", openblas_get_corename());
#endif
  for (auto& w : W) {
    const int m = w[0], n = w[1], k = w[2];
    std::vector<float> A(size_t(m) * k), B(size_t(k) * n), C(size_t(m) * n);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 97) * 1e-3f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 89) * 1e-3f;
    auto run = [&] {
      if (row) cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A.data(), k, B.data(), n, 0.f, C.data(), n);
      else cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A.data(), m, B.data(), k, 0.f, C.data(), m);
    };
    run();
    double best = 1e30;
    for (int t = 0; t < 5; ++t) {
      int reps = 0; auto t0 = std::chrono::steady_clock::now(); double s = 0;
      do { run(); ++reps; s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (s < 0.05);
      best = std::min(best, s / reps);
    }
    std::printf("%dx%dx%d %.0f\n", m, n, k, 2.0 * m * n * k / best / 1e9);
    std::fflush(stdout);
  }
}
