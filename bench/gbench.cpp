// Cross-check of the chrono harness with Google Benchmark on a few shapes; reports max GFLOPS over repetitions.
#include "mtgemm.h"
#include <Accelerate/Accelerate.h>
#include <benchmark/benchmark.h>
#include <pthread/qos.h>
#include <algorithm>
#include <random>
#include <vector>

static void gemm(benchmark::State& st, bool accel, bool row) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  const int m = st.range(0), n = st.range(1), k = st.range(2);
  const int lda = row ? k : m, ldb = row ? n : k, ldc = row ? n : m;
  const float beta = row ? 0.f : 1.f;
  std::vector<float> A(size_t(m) * k), B(size_t(k) * n), C(size_t(m) * n);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-1, 1);
  for (auto& x : A) x = d(rng);
  for (auto& x : B) x = d(rng);
  for (auto& x : C) x = d(rng) * 0.01f;
  for (auto _ : st) {
    if (accel)
      cblas_sgemm(row ? CblasRowMajor : CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A.data(), lda,
                  B.data(), ldb, beta, C.data(), ldc);
    else
      mt_sgemm(row ? MtRowMajor : MtColMajor, m, n, k, 1.f, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    benchmark::ClobberMemory();
  }
  st.counters["GFLOPS"] = benchmark::Counter(2.0 * m * n * k * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
}

static void shapes(benchmark::internal::Benchmark* b) {
  for (auto s : std::vector<std::vector<int64_t>>{{4096, 4096, 4096}, {64, 2112, 7168}, {128, 7168, 2048},
                                                  {4096, 7168, 2048}, {4096, 256, 4096}, {5120, 256, 13824}})
    b->Args(s);
  b->MinTime(0.05)->Repetitions(5)->ComputeStatistics("max", [](const std::vector<double>& v) {
    return *std::max_element(v.begin(), v.end());
  });
  b->Unit(benchmark::kMillisecond);
}

BENCHMARK_CAPTURE(gemm, accel_row, true, true)->Apply(shapes);
BENCHMARK_CAPTURE(gemm, mt_row, false, true)->Apply(shapes);
BENCHMARK_CAPTURE(gemm, accel_col, true, false)->Apply(shapes);
BENCHMARK_CAPTURE(gemm, mt_col, false, false)->Apply(shapes);

BENCHMARK_MAIN();
