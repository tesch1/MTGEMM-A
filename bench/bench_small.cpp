// Small GEMMs, timed as the minimum over many short batches (the per-call cost matters here, and the 50 ms
// trials of bench/bench_openblas are too coarse below about 64^3). Links against OpenBLAS (cblas_?gemm) or, with
// -DUSE_MT, against MTGEMM-A. Same conventions as bench: row-major beta 0, column-major beta 1, alpha 1.
// usage: bench_small <row|col> <f32|f64> <threads (1, or 0: default; MTGEMM-A: threads=1 or 0)> <n> [n ...]
#ifdef USE_MT
#include "mtgemm.h"
#else
#include <cblas.h>
#endif
#include <pthread/qos.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static int g_threads = 1;

template <class T>
static void call(bool row, int n, const T* A, const T* B, T* C) {
  const T beta = row ? T(0) : T(1);
#ifdef USE_MT
  mt_options o;
  o.threads = g_threads == 1 ? 1 : 0;  // the same thread count as the OpenBLAS runs
  if constexpr (sizeof(T) == 4) mt_sgemm(row ? MtRowMajor : MtColMajor, n, n, n, 1.f, A, n, B, n, beta, C, n, &o);
  else mt_dgemm(row ? MtRowMajor : MtColMajor, n, n, n, 1.0, A, n, B, n, beta, C, n, &o);
#else
  const auto o = row ? CblasRowMajor : CblasColMajor;
  if constexpr (sizeof(T) == 4) cblas_sgemm(o, CblasNoTrans, CblasNoTrans, n, n, n, 1.f, A, n, B, n, beta, C, n);
  else cblas_dgemm(o, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, beta, C, n);
#endif
}

// Returns the best time per call in seconds.
template <class T>
static double run(bool row, int n) {
  std::vector<T> A(size_t(n) * n), B(A.size()), C(A.size());
  for (size_t i = 0; i < A.size(); ++i) { A[i] = T(0.001) * T(i % 7); B[i] = T(0.002) * T(i % 5); C[i] = 0; }
  const int batch = n <= 32 ? 64 : (n <= 128 ? 8 : 1);
  double best = 1e30;
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
  while (std::chrono::steady_clock::now() < end) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < batch; ++i) call<T>(row, n, A.data(), B.data(), C.data());
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / batch;
    if (t < best) best = t;
  }
  return best;
}

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: bench_small <row|col> <f32|f64> <threads> <n> [n ...]\n"); return 2; }
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  const bool row = !std::strcmp(argv[1], "row"), f64 = !std::strcmp(argv[2], "f64");
  g_threads = std::atoi(argv[3]);
#ifndef USE_MT
  if (g_threads > 0) openblas_set_num_threads(g_threads);
  std::this_thread::sleep_for(std::chrono::seconds(12));  // idle pool threads spin for about 11 s after start-up
#endif
  std::printf("# %s %s %s\n", argv[1], argv[2],
#ifdef USE_MT
              "mtgemm"
#else
              "openblas"
#endif
  );
  // Three passes over all sizes, minimum per size: a pass can start on an efficiency core, whose SME unit is
  // several times slower, until macOS moves the thread.
  std::vector<double> best(argc, 1e30);
  for (int pass = 0; pass < 3; ++pass)
    for (int i = 4; i < argc; ++i) {
      const double t = f64 ? run<double>(row, std::atoi(argv[i])) : run<float>(row, std::atoi(argv[i]));
      if (t < best[i]) best[i] = t;
    }
  for (int i = 4; i < argc; ++i) {
    const double n = std::atoi(argv[i]);
    std::printf("%5d %9.0f ns %7.1f GFLOPS\n", (int)n, best[i] * 1e9, 2.0 * n * n * n / best[i] / 1e9);
  }
}
