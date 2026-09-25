// Accelerate vs MTGEMM-A on squares, the paper's 24 workloads and its irregular shapes.
// usage: bench <squares|paper|irr|all|small|thin|MxNxK> <row|col> <accel|mt> [f64] [paper] [beta=x] [online=0 x4=0 heap=0 model=0
//        shape=1 cdirect=1 threads=2 mc=.. nc=.. kc=..] [ids=a-b] [ms=50] [trials=5]
// Row-major runs use beta 0 and column-major runs beta 1, as in the paper (Sec. 5.1.3). Min over trials.
#include "mtgemm.h"
#include <Accelerate/Accelerate.h>
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

static int g_argc;
static char** g_argv;

#ifdef MT_BENCH_AMX
#include "../src/amx.h"
// AMX fma peak of this core right now (fma32 x4 or fma64 x8, best of 3 x 5 ms): the clock reference for eff=.
template <class T>
static double amx_peak() {
  using namespace mt::amx;
  alignas(256) static float z[128] = {};
  AMX_SET();
  AMX_LDX(xy(z, 0, kQuad)); AMX_LDX(xy(z, 4, kQuad)); AMX_LDY(xy(z, 0, kQuad)); AMX_LDY(xy(z, 4, kQuad));
  double best = 0;
  for (int t = 0; t < 3; ++t) {
    long n = 0;
    double s;
    const auto t0 = std::chrono::steady_clock::now();
    do {
      for (int i = 0; i < 256; ++i) {
        if constexpr (sizeof(T) == 4) {
          AMX_FMA32(fma_op(0, 0, 0)); AMX_FMA32(fma_op(1, 64, 64)); AMX_FMA32(fma_op(2, 128, 128)); AMX_FMA32(fma_op(3, 192, 192));
        } else {
          AMX_FMA64(fma_op(0, 0, 0)); AMX_FMA64(fma_op(1, 64, 64)); AMX_FMA64(fma_op(2, 128, 128)); AMX_FMA64(fma_op(3, 192, 192));
          AMX_FMA64(fma_op(4, 256, 256)); AMX_FMA64(fma_op(5, 320, 320)); AMX_FMA64(fma_op(6, 384, 384)); AMX_FMA64(fma_op(7, 448, 448));
        }
      }
      n += 256;
      s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } while (s < 5e-3);
    best = std::max(best, n * (sizeof(T) == 4 ? 4 * 512.0 : 8 * 128.0) / s * 1e-9);
  }
  AMX_CLR();
  return best;
}
#endif

template <class T>
static void accel(bool row, int m, int n, int k, T al, const T* A, int lda, const T* B, int ldb, T be, T* C, int ldc) {
  const auto o = row ? CblasRowMajor : CblasColMajor;
  if constexpr (sizeof(T) == 4) cblas_sgemm(o, CblasNoTrans, CblasNoTrans, m, n, k, al, A, lda, B, ldb, be, C, ldc);
  else cblas_dgemm(o, CblasNoTrans, CblasNoTrans, m, n, k, al, A, lda, B, ldb, be, C, ldc);
}
template <class T>
static void mine(bool row, int m, int n, int k, T al, const T* A, int lda, const T* B, int ldb, T be, T* C, int ldc,
                 const mt_options& o) {
  const auto ord = row ? MtRowMajor : MtColMajor;
  if constexpr (sizeof(T) == 4) mt_sgemm(ord, m, n, k, al, A, lda, B, ldb, be, C, ldc, &o);
  else mt_dgemm(ord, m, n, k, al, A, lda, B, ldb, be, C, ldc, &o);
}

template <class T>
static void run(const std::vector<Shape>& shapes, bool row, bool use_accel, double beta, const mt_options& o, double ms,
                int trials) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-1, 1);
  for (const Shape& s : shapes) {
    const int m = s.m, n = s.n, k = s.k;
    const int lda = row ? k : m, ldb = row ? n : k, ldc = row ? n : m;
    std::vector<T> A(size_t(m) * k), B(size_t(k) * n), C(size_t(m) * n), R;
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    for (auto& x : C) x = d(rng) * 0.01f;
    const T be = T(beta);
    auto call = [&] {
      if (use_accel) accel<T>(row, m, n, k, T(1), A.data(), lda, B.data(), ldb, be, C.data(), ldc);
      else mine<T>(row, m, n, k, T(1), A.data(), lda, B.data(), ldb, be, C.data(), ldc, o);
    };
    double err = 0;
    if (!use_accel) {  // one checked call against Accelerate
      std::vector<T> C0 = C;
      R = C;
      accel<T>(row, m, n, k, T(1), A.data(), lda, B.data(), ldb, be, R.data(), ldc);
      call();
      for (size_t i = 0; i < C.size(); ++i) err = std::max(err, double(std::fabs(C[i] - R[i])));
      err /= std::sqrt(double(k));
      C = C0;
    }
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
    std::printf("%2d %5d %5d %5d %6.0f", s.id, m, n, k, 2.0 * m * n * k / best / 1e9);
    if (!use_accel) std::printf("  err/sqrtK=%.1e", err);
#ifdef MT_BENCH_AMX
    const double pk = amx_peak<T>();
    std::printf("  peak=%.0f eff=%.3f", pk, 2.0 * m * n * k / best / 1e9 / pk);
#endif
    std::printf("\n");
    std::fflush(stdout);
  }
}

static void* body(void*) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (g_argc < 4) {
    std::fprintf(stderr, "usage: bench <squares|paper|irr|all|small|thin|MxNxK> <row|col> <accel|mt> [options]\n");
    std::exit(2);
  }
  const std::string set = g_argv[1];
  const bool row = !std::strcmp(g_argv[2], "row");
  const bool use_accel = !std::strcmp(g_argv[3], "accel");
  bool f64 = false;
  double beta = row ? 0.0 : 1.0, ms = 50;
  int trials = 5, id_lo = 1, id_hi = 1000;
  mt_options o;
  std::vector<std::string> opts;  // options may also arrive space-separated in one argument
  for (int i = 4; i < g_argc; ++i) {
    std::string w, all = g_argv[i];
    for (char ch : all + " ")
      if (ch == ' ') { if (!w.empty()) opts.push_back(w); w.clear(); } else w += ch;
  }
  for (const std::string& os : opts) {
    const char* a = os.c_str();
    int v;
    double x;
    if (!std::strcmp(a, "f64")) f64 = true;
    else if (!std::strcmp(a, "paper")) { o.cdirect = 0; o.pack4 = 0; o.prefetch = 0; }
    else if (std::sscanf(a, "beta=%lf", &x) == 1) beta = x;
    else if (std::sscanf(a, "ms=%lf", &x) == 1) ms = x;
    else if (std::sscanf(a, "trials=%d", &v) == 1) trials = v;
    else if (std::sscanf(a, "ids=%d-%d", &id_lo, &id_hi) == 2) {}
    else if (std::sscanf(a, "online=%d", &v) == 1) o.online = v;
    else if (std::sscanf(a, "x4=%d", &v) == 1) o.x4 = v;
    else if (std::sscanf(a, "heap=%d", &v) == 1) o.heap = v;
    else if (std::sscanf(a, "model=%d", &v) == 1) o.model = v;
    else if (std::sscanf(a, "shape=%d", &v) == 1) o.shape = v;
    else if (std::sscanf(a, "cdirect=%d", &v) == 1) o.cdirect = v;
    else if (std::sscanf(a, "threads=%d", &v) == 1) o.threads = v;
    else if (std::sscanf(a, "prof=%d", &v) == 1) o.prof = v;
    else if (std::sscanf(a, "pack4=%d", &v) == 1) o.pack4 = v;
    else if (std::sscanf(a, "pf=%d", &v) == 1) o.prefetch = v;
    else if (std::sscanf(a, "mc=%d", &v) == 1) o.mc = v;
    else if (std::sscanf(a, "nc=%d", &v) == 1) o.nc = v;
    else if (std::sscanf(a, "kc=%d", &v) == 1) o.kc = v;
    else { std::fprintf(stderr, "bad option %s\n", a); std::exit(2); }
  }
  std::vector<Shape> shapes = make_shapes(set, id_lo, id_hi);
  std::printf("# %s %s %s %s beta=%g", set.c_str(), row ? "row" : "col", use_accel ? "accel" : "mt", f64 ? "f64" : "f32", beta);
  if (!use_accel)
    std::printf(" online=%d x4=%d heap=%d model=%d shape=%d cdirect=%d pack4=%d pf=%d threads=%d", o.online, o.x4, o.heap,
                o.model, o.shape, o.cdirect, o.pack4, o.prefetch, o.threads);
  std::printf("\n");
  if (f64) run<double>(shapes, row, use_accel, beta, o, ms, trials);
  else run<float>(shapes, row, use_accel, beta, o, ms, trials);
  return nullptr;
}

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  pthread_attr_t at;  // large stack so that heap=0 can put the packed buffers on it
  pthread_attr_init(&at);
  pthread_attr_setstacksize(&at, size_t(512) << 20);
  pthread_attr_set_qos_class_np(&at, QOS_CLASS_USER_INTERACTIVE, 0);
  pthread_t th;
  pthread_create(&th, &at, body, nullptr);
  pthread_join(th, nullptr);
  return 0;
}
