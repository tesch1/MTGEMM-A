// Correctness against a naive reference: random sizes with edge tails, leading-dimension padding, alpha/beta.
#include "mtgemm.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::mt19937_64 rng(12345);
static int rnd(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }

template <class T>
static void gemm(mt_order o, int M, int N, int K, T al, const T* A, int lda, const T* B, int ldb, T be, T* C, int ldc,
                 const mt_options* op) {
  if constexpr (sizeof(T) == 4) mt_sgemm(o, M, N, K, al, A, lda, B, ldb, be, C, ldc, op);
  else mt_dgemm(o, M, N, K, al, A, lda, B, ldb, be, C, ldc, op);
}

template <class T>
static bool check(mt_order o, int M, int N, int K, T al, T be, int pad, const mt_options& op) {
  const bool row = o == MtRowMajor;
  const int lda = (row ? K : M) + pad, ldb = (row ? N : K) + pad, ldc = (row ? N : M) + pad;
  const size_t sa = size_t(lda) * (row ? M : K), sb = size_t(ldb) * (row ? K : N), sc = size_t(ldc) * (row ? M : N);
  std::vector<T> A(sa + 1), B(sb + 1), C(sc + 1);
  std::uniform_real_distribution<double> d(-1, 1);
  for (auto& x : A) x = T(d(rng));
  for (auto& x : B) x = T(d(rng));
  for (auto& x : C) x = T(d(rng));
  if (be == T(0))
    for (auto& x : C) x = NAN;  // beta = 0 must not read C
  auto a = [&](int i, int k) { return row ? A[size_t(i) * lda + k] : A[size_t(k) * lda + i]; };
  auto b = [&](int k, int j) { return row ? B[size_t(k) * ldb + j] : B[size_t(j) * ldb + k]; };
  auto ci = [&](int i, int j) { return row ? size_t(i) * ldc + j : size_t(j) * ldc + i; };
  std::vector<T> C0 = C;
  gemm<T>(o, M, N, K, al, A.data(), lda, B.data(), ldb, be, C.data(), ldc, &op);
  const double eps = sizeof(T) == 4 ? 1.2e-7 : 2.3e-16;
  double worst = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0, sa = 0;
      for (int k = 0; k < K; ++k) {
        s += double(a(i, k)) * double(b(k, j));
        sa += std::fabs(double(a(i, k)) * double(b(k, j)));
      }
      const double c0 = be == T(0) ? 0.0 : double(C0[ci(i, j)]);
      const double ref = double(al) * s + double(be) * c0;
      const double tol = 4 * eps * (std::fabs(double(al)) * sa * (1 + std::sqrt(double(K))) + std::fabs(double(be) * c0) + 1e-30);
      const double err = std::fabs(double(C[ci(i, j)]) - ref);
      if (!(err <= tol)) {
        std::printf("FAIL %s %s M=%d N=%d K=%d alpha=%g beta=%g pad=%d opt(on=%d x4=%d heap=%d model=%d shape=%d cd=%d thr=%d) at (%d,%d): %g vs %g\n",
                    sizeof(T) == 4 ? "f32" : "f64", row ? "row" : "col", M, N, K, double(al), double(be), pad, op.online,
                    op.x4, op.heap, op.model, op.shape, op.cdirect, op.threads, i, j, double(C[ci(i, j)]), ref);
        return false;
      }
      worst = std::max(worst, err / tol);
    }
  // Padding columns and the guard element must be untouched.
  for (int i = 0; i < (row ? M : N); ++i)
    for (int j = (row ? N : M); j < ldc; ++j) {
      const size_t idx = size_t(i) * ldc + j;
      if (!(C[idx] == C0[idx] || (std::isnan(C[idx]) && std::isnan(C0[idx])))) {
        std::printf("FAIL: padding written at %zu\n", idx);
        return false;
      }
    }
  if (!(C[sc] == C0[sc] || (std::isnan(C[sc]) && std::isnan(C0[sc])))) {
    std::printf("FAIL: guard written\n");
    return false;
  }
  return true;
}

template <class T>
static int run(int trials) {
  int fails = 0, n = 0;
  const T alphas[] = {T(1), T(-0.75), T(2)};
  const T betas[] = {T(0), T(1), T(0.5)};
  for (int t = 0; t < trials; ++t) {
    mt_options op;
    const int v = t % 9;
    if (v == 1) op.online = 0;
    if (v == 2) op.x4 = 0;
    if (v == 3) op.heap = 0;
    if (v == 4) op.model = 0;
    if (v == 5) op.shape = 1;
    if (v == 6) op.cdirect = 1;
    if (v == 7) op.threads = 2;
    if (v == 8) { op.mc = 16 * rnd(1, 6); op.nc = 64 * rnd(1, 4); op.kc = rnd(1, 150); }
    const bool big = t % 17 == 0;
    const int M = big ? rnd(100, 400) : rnd(1, 150), N = big ? rnd(100, 400) : rnd(1, 150), K = big ? rnd(200, 700) : rnd(1, 130);
    const mt_order o = t % 2 ? MtColMajor : MtRowMajor;
    const T al = alphas[rnd(0, 2)], be = betas[rnd(0, 2)];
    const int pad = rnd(0, 1) ? rnd(1, 20) : 0;
    ++n;
    if (!check<T>(o, M, N, K, al, be, pad, op)) ++fails;
  }
  std::printf("%s: %d / %d passed\n", sizeof(T) == 4 ? "sgemm" : "dgemm", n - fails, n);
  return fails;
}

int main(int argc, char** argv) {
  const int trials = argc > 1 ? std::atoi(argv[1]) : 600;
  int f = 0;
  // Fixed shapes: exact multiples, one-off tails and degenerate sizes.
  const int fixed[][3] = {{16, 64, 4}, {64, 64, 64}, {17, 65, 5}, {1, 1, 1}, {15, 63, 3}, {64, 16, 7}, {80, 200, 25},
                          {200, 80, 131}, {128, 256, 512}, {33, 17, 1}, {5, 300, 2}};
  for (auto& s : fixed)
    for (int o = 0; o < 2; ++o) {
      mt_options op;
      f += !check<float>(o ? MtColMajor : MtRowMajor, s[0], s[1], s[2], 1.f, 0.f, 0, op);
      f += !check<double>(o ? MtColMajor : MtRowMajor, s[0], s[1], s[2], 1.0, 1.0, 3, op);
    }
  f += run<float>(trials);
  f += run<double>(trials / 2);
  std::printf(f ? "FAILED (%d)\n" : "ALL PASSED\n", f);
  return f != 0;
}
