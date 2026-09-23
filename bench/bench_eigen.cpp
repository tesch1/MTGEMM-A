// Eigen (master or the SME branch, from third_party/) on the same shapes: column-major C += A*B, row-major C = A*B;
// one thread, or a thread pool of one thread per core when built with EIGEN_GEMM_THREADPOOL.
// usage: bench_eigen <squares|paper|irr|all|small|thin|MxNxK> <row|col> [ids=a-b] [ms=50] [trials=5] [check]
#include <Accelerate/Accelerate.h>
#include <Eigen/Core>
#ifdef EIGEN_GEMM_THREADPOOL
#include <Eigen/ThreadPool>
#include <thread>
#endif
#include <pthread.h>
#include <pthread/qos.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "shapes.h"

#ifndef EIGEN_VECTORIZE_SME
#error "bench_eigen measures Eigen's SME backend; build with -march=...+sme2"
#endif
#ifndef BENCH_EIGEN_NAME
#define BENCH_EIGEN_NAME "eigen"
#endif
static const char* kBackend = BENCH_EIGEN_NAME;
static int g_argc;
static char** g_argv;

template <int Order>
static void product(int m, int n, int k, const float* A, const float* B, float* C, bool accumulate) {
  using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Order>;
  Eigen::Map<const Mat> a(A, m, k), b(B, k, n);
  Eigen::Map<Mat> c(C, m, n);
  if (accumulate) c.noalias() += a * b;
  else c.noalias() = a * b;
}

static void* body(void*) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (g_argc < 3) { std::fprintf(stderr, "usage: bench_eigen <set> <row|col> [opts]\n"); std::exit(2); }
  const std::string set = g_argv[1];
  const bool row = !std::strcmp(g_argv[2], "row");
  double ms = 50;
  int trials = 5, id_lo = 1, id_hi = 1000;
  bool check_only = false;
  for (int i = 3; i < g_argc; ++i) {
    const char* a = g_argv[i];
    double x;
    int v;
    if (std::sscanf(a, "ms=%lf", &x) == 1) ms = x;
    else if (std::sscanf(a, "trials=%d", &v) == 1) trials = v;
    else if (std::sscanf(a, "ids=%d-%d", &id_lo, &id_hi) == 2) {}
    else if (!std::strcmp(a, "check")) check_only = true;
    else { std::fprintf(stderr, "bad option %s\n", a); std::exit(2); }
  }
  const std::vector<Shape> shapes = make_shapes(set, id_lo, id_hi);
  const double beta = row ? 0.0 : 1.0;
#ifdef EIGEN_GEMM_THREADPOOL
  // A pool of one thread per core, as an application would create; the SME branch caps products at its unit count.
  static Eigen::ThreadPool pool(static_cast<int>(std::thread::hardware_concurrency()));
  Eigen::setGemmThreadPool(&pool);
  const int threads = static_cast<int>(std::thread::hardware_concurrency());
#else
  const int threads = 1;
#endif
#ifdef BENCH_EIGEN_HAS_SME_UNITS
  const int units = Eigen::nbSmeUnits();
#else
  const int units = -1;
#endif
  std::printf("# %s %s %s f32 beta=%g threads=%d sme_units=%d\n", set.c_str(), row ? "row" : "col", kBackend, beta, threads,
              units);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-1, 1);
  int failures = 0;
  for (const Shape& s : shapes) {
    const int m = s.m, n = s.n, k = s.k;
    const int lda = row ? k : m, ldb = row ? n : k, ldc = row ? n : m;
    std::vector<float> A(size_t(m) * k), B(size_t(k) * n), C(size_t(m) * n);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    for (auto& x : C) x = d(rng) * 0.01f;
    auto call = [&] {
      if (row) product<Eigen::RowMajor>(m, n, k, A.data(), B.data(), C.data(), false);
      else product<Eigen::ColMajor>(m, n, k, A.data(), B.data(), C.data(), true);
    };
    std::vector<float> C0 = C, R = C;
    cblas_sgemm(row ? CblasRowMajor : CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A.data(), lda, B.data(),
                ldb, float(beta), R.data(), ldc);
    call();
    double err = 0, moved = 0;
    for (size_t i = 0; i < C.size(); ++i) {
      err = std::max(err, double(std::fabs(C[i] - R[i])));
      moved = std::max(moved, double(std::fabs(C[i] - C0[i])));
    }
    err /= std::sqrt(double(k));
    // Eigen's summation order differs from Accelerate's, so this bound is looser than bench_ext's.
    const bool ok = err < 1e-4 && moved > 0;
    if (!ok) ++failures;
    C = C0;
    if (check_only) { std::printf("%2d %5d %5d %5d  err/sqrtK=%.1e %s\n", s.id, m, n, k, err, ok ? "ok" : "FAIL"); continue; }
    call();
    double best = 1e30;
    for (int t = 0; t < trials; ++t) {
      int reps = 0;
      double sec = 0;
      const auto t0 = std::chrono::steady_clock::now();
      do {
        call();
        ++reps;
        sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      } while (sec < ms * 1e-3);
      best = std::min(best, sec / reps);
    }
    std::printf("%2d %5d %5d %5d %6.0f  err/sqrtK=%.1e\n", s.id, m, n, k, 2.0 * m * n * k / best / 1e9, err);
    std::fflush(stdout);
  }
  if (failures) std::printf("# %d shape(s) FAILED the check against Accelerate\n", failures);
  std::exit(failures ? 1 : 0);
}

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_setstacksize(&at, 64 << 20);
  pthread_t t;
  pthread_create(&t, &at, body, nullptr);
  pthread_join(t, nullptr);
}
