// OpenBLAS (develop, SME kernels) on the same shapes as bench; one checked call against MTGEMM-A (one thread), which
// bench checks against Accelerate. A separate binary because OpenBLAS and Accelerate both export cblas_*.
// usage: bench_openblas <squares|paper|irr|all|small|thin|MxNxK> <row|col> [f64] [beta=x] [threads=n] [ids=a-b]
//        [ms=50] [trials=5] [check]
// Row-major runs use beta 0 and column-major runs beta 1, as in bench. threads=0 leaves OpenBLAS at its default.
// Timed runs first wait 12 s so that the idle pool threads have stopped spinning.
#include "mtgemm.h"
#include <cblas.h>
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
#include <thread>
#include <vector>

#include "shapes.h"

static int g_argc;
static char** g_argv;

template <class T>
static void ob(bool row, int m, int n, int k, T al, const T* A, int lda, const T* B, int ldb, T be, T* C, int ldc) {
  const auto o = row ? CblasRowMajor : CblasColMajor;
  if constexpr (sizeof(T) == 4) cblas_sgemm(o, CblasNoTrans, CblasNoTrans, m, n, k, al, A, lda, B, ldb, be, C, ldc);
  else cblas_dgemm(o, CblasNoTrans, CblasNoTrans, m, n, k, al, A, lda, B, ldb, be, C, ldc);
}
template <class T>
static void ref(bool row, int m, int n, int k, T al, const T* A, int lda, const T* B, int ldb, T be, T* C, int ldc) {
  const auto ord = row ? MtRowMajor : MtColMajor;
  if constexpr (sizeof(T) == 4) mt_sgemm(ord, m, n, k, al, A, lda, B, ldb, be, C, ldc);
  else mt_dgemm(ord, m, n, k, al, A, lda, B, ldb, be, C, ldc);
}

template <class T>
static int run(const std::vector<Shape>& shapes, bool row, double beta, double ms, int trials, bool check_only) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-1, 1);
  const double tol = sizeof(T) == 4 ? 1e-5 : 1e-13;
  int failures = 0;
  for (const Shape& s : shapes) {
    const int m = s.m, n = s.n, k = s.k;
    const int lda = row ? k : m, ldb = row ? n : k, ldc = row ? n : m;
    std::vector<T> A(size_t(m) * k), B(size_t(k) * n), C(size_t(m) * n);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    for (auto& x : C) x = d(rng) * 0.01f;
    const T be = T(beta);
    auto call = [&] { ob<T>(row, m, n, k, T(1), A.data(), lda, B.data(), ldb, be, C.data(), ldc); };
    std::vector<T> C0 = C, R = C;
    ref<T>(row, m, n, k, T(1), A.data(), lda, B.data(), ldb, be, R.data(), ldc);
    call();
    double err = 0;
    for (size_t i = 0; i < C.size(); ++i) err = std::max(err, double(std::fabs(C[i] - R[i])));
    err /= std::sqrt(double(k));
    const bool ok = err < tol;
    failures += !ok;
    C = C0;
    if (check_only) { std::printf("%2d %5d %5d %5d  err/sqrtK=%.1e %s\n", s.id, m, n, k, err, ok ? "ok" : "FAIL"); continue; }
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
    std::printf("%2d %5d %5d %5d %6.0f  err/sqrtK=%.1e%s\n", s.id, m, n, k, 2.0 * m * n * k / best / 1e9, err,
                ok ? "" : " FAIL");
    std::fflush(stdout);
  }
  return failures;
}

static void* body(void*) {
  if (g_argc < 3) {
    std::fprintf(stderr, "usage: bench_openblas <squares|paper|irr|all|small|thin|MxNxK> <row|col> [options]\n");
    std::exit(2);
  }
  const std::string set = g_argv[1];
  const bool row = !std::strcmp(g_argv[2], "row");
  bool f64 = false, check_only = false;
  double beta = row ? 0.0 : 1.0, ms = 50;
  int trials = 5, id_lo = 1, id_hi = 1000, threads = 1;
  for (int i = 3; i < g_argc; ++i) {
    const char* a = g_argv[i];
    int v;
    double x;
    if (!std::strcmp(a, "f64")) f64 = true;
    else if (!std::strcmp(a, "check")) check_only = true;
    else if (std::sscanf(a, "beta=%lf", &x) == 1) beta = x;
    else if (std::sscanf(a, "ms=%lf", &x) == 1) ms = x;
    else if (std::sscanf(a, "trials=%d", &v) == 1) trials = v;
    else if (std::sscanf(a, "threads=%d", &v) == 1) threads = v;
    else if (std::sscanf(a, "ids=%d-%d", &id_lo, &id_hi) == 2) {}
    else { std::fprintf(stderr, "bad option %s\n", a); std::exit(2); }
  }
  if (threads > 0) openblas_set_num_threads(threads);
  // The pool workers spin for 2^28 timer ticks after start-up (11 s at Apple's 24 MHz) and slow the first shapes.
  if (!check_only) std::this_thread::sleep_for(std::chrono::seconds(12));
  std::printf("# %s %s openblas %s beta=%g core=%s threads=%d config=%s\n", set.c_str(), row ? "row" : "col",
              f64 ? "f64" : "f32", beta, openblas_get_corename(), openblas_get_num_threads(), openblas_get_config());
  const std::vector<Shape> shapes = make_shapes(set, id_lo, id_hi);
  const int failures = f64 ? run<double>(shapes, row, beta, ms, trials, check_only)
                           : run<float>(shapes, row, beta, ms, trials, check_only);
  if (failures) std::printf("# %d shape(s) FAILED the check against MTGEMM-A\n", failures);
  std::exit(failures ? 1 : 0);
}

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_setstacksize(&at, size_t(64) << 20);
  pthread_attr_set_qos_class_np(&at, QOS_CLASS_USER_INTERACTIVE, 0);
  pthread_t th;
  pthread_create(&th, &at, body, nullptr);
  pthread_join(th, nullptr);
}
