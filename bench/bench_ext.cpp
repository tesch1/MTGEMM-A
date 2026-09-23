// LIBXSMM and KleidiAI on the same shapes as bench, set up as in the paper (Sec. 5.1.3): LIBXSMM column-major
// C += A*B (one JIT kernel for the whole problem), KleidiAI row-major C = A*B with both operands packed per call.
// usage: bench_ext <squares|paper|irr|all|MxNxK> <libxsmm|kleidiai> [ids=a-b] [ms=50] [trials=5] [check]
#include <Accelerate/Accelerate.h>
#include <libxsmm.h>
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include <pthread.h>
#include <pthread/qos.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

struct Shape { int id, m, n, k; };
static const int W[24][3] = {{64, 2112, 7168},   {64, 24576, 1536},  {64, 32768, 512},   {64, 7168, 16384},
                             {64, 4096, 7168},   {64, 7168, 2048},   {128, 2112, 7168},  {128, 24576, 1536},
                             {128, 32768, 512},  {128, 7168, 16384}, {128, 4096, 7168},  {128, 7168, 2048},
                             {4096, 2112, 7168}, {4096, 24576, 1536}, {4096, 32768, 512}, {4096, 7168, 16384},
                             {4096, 4096, 7168}, {4096, 7168, 2048}, {4096, 256, 4096},  {11008, 256, 4096},
                             {4096, 256, 11008}, {5120, 256, 5120},  {13824, 256, 5120}, {5120, 256, 13824}};
static int g_argc;
static char** g_argv;

// Returns the timed call for one shape; A, B, C use the library's storage order.
static std::function<void()> make_call(bool xsmm, int m, int n, int k, float* A, float* B, float* C,
                                       std::vector<float>& lp, std::vector<float>& rp, std::vector<float>& bias) {
  if (xsmm) {
    const libxsmm_gemm_shape sh = libxsmm_create_gemm_shape(m, n, k, m, k, m, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const libxsmm_gemmfunction f = libxsmm_dispatch_gemm(sh, LIBXSMM_GEMM_FLAG_NONE, LIBXSMM_GEMM_PREFETCH_NONE);
    if (f == nullptr) { std::fprintf(stderr, "libxsmm: no kernel for %dx%dx%d\n", m, n, k); std::exit(1); }
    return [=] {
      libxsmm_gemm_param p;
      std::memset(&p, 0, sizeof(p));
      p.a.primary = A;
      p.b.primary = B;
      p.c.primary = C;
      f(&p);
    };
  }
  const size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
  const size_t nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
  const size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
  const size_t sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
  lp.resize(kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr) / sizeof(float) + 16);
  rp.resize(kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k) / sizeof(float) + 16);
  bias.assign(n, 0.f);
  float *L = lp.data(), *R = rp.data(), *b = bias.data();
  return [=] {
    kai_run_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr, 0, A, size_t(k) * sizeof(float), L);
    kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(1, n, k, nr, kr, sr, size_t(n) * sizeof(float), B, b, nullptr, R,
                                                     0, nullptr);
    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(m, n, k, L, R, C, size_t(n) * sizeof(float),
                                                                  sizeof(float), -FLT_MAX, FLT_MAX);
  };
}

static void* body(void*) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (g_argc < 3) { std::fprintf(stderr, "usage: bench_ext <squares|paper|irr|all|MxNxK> <libxsmm|kleidiai> [opts]\n"); std::exit(2); }
  const std::string set = g_argv[1];
  const bool xsmm = !std::strcmp(g_argv[2], "libxsmm");
  if (!xsmm && std::strcmp(g_argv[2], "kleidiai")) { std::fprintf(stderr, "unknown library %s\n", g_argv[2]); std::exit(2); }
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
  std::vector<Shape> shapes;
  if (set == "squares" || set == "all")
    for (int s : {512, 1000, 1024, 2048, 3000, 4096}) shapes.push_back({0, s, s, s});
  if (set == "paper" || set == "all")
    for (int i = 0; i < 24; ++i)
      if (i + 1 >= id_lo && i + 1 <= id_hi) shapes.push_back({i + 1, W[i][0], W[i][1], W[i][2]});
  if (set == "irr")
    for (int s = 80; s <= 200; s += 30) shapes.push_back({0, s, s, 25600});
  int m, n, k;
  if (std::sscanf(set.c_str(), "%dx%dx%d", &m, &n, &k) == 3) shapes.push_back({0, m, n, k});
  const bool row = !xsmm;
  const double beta = xsmm ? 1.0 : 0.0;
  std::printf("# %s %s %s f32 beta=%g\n", set.c_str(), row ? "row" : "col", xsmm ? "libxsmm" : "kleidiai", beta);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-1, 1);
  int failures = 0;
  for (const Shape& s : shapes) {
    m = s.m; n = s.n; k = s.k;
    const int lda = row ? k : m, ldb = row ? n : k, ldc = row ? n : m;
    std::vector<float> A(size_t(m) * k), B(size_t(k) * n), C(size_t(m) * n), lp, rp, bias;
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    for (auto& x : C) x = d(rng) * 0.01f;
    auto call = make_call(xsmm, m, n, k, A.data(), B.data(), C.data(), lp, rp, bias);
    std::vector<float> C0 = C, R = C;
    cblas_sgemm(row ? CblasRowMajor : CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A.data(), lda, B.data(),
                ldb, float(beta), R.data(), ldc);
    call();
    double err = 0;
    for (size_t i = 0; i < C.size(); ++i) err = std::max(err, double(std::fabs(C[i] - R[i])));
    err /= std::sqrt(double(k));
    double moved = 0;  // guards against a call that writes nothing
    for (size_t i = 0; i < C.size(); ++i) moved = std::max(moved, double(std::fabs(C[i] - C0[i])));
    if (!(err < 1e-5) || !(moved > 0)) ++failures;
    C = C0;
    if (check_only) { std::printf("%2d %5d %5d %5d  err/sqrtK=%.1e max|dC|=%.1f %s\n", s.id, m, n, k, err, moved, err < 1e-5 && moved > 0 ? "ok" : "FAIL"); continue; }
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
  libxsmm_init();
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_setstacksize(&at, 64 << 20);
  pthread_t t;
  pthread_create(&t, &at, body, nullptr);
  pthread_join(t, nullptr);
}
