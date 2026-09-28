// Checks cblas_sgemm / cblas_dgemm of an OpenBLAS build against a long-double reference: both orders, all four
// transpose cases, several alpha/beta values, odd sizes and padded leading dimensions, 1 and default threads.
// usage: [VERBOSE=1] test_openblas_gemm [quick]   (VERBOSE traces each call before it runs)
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rs = 88172645463325252ULL;
static double urand(void) {
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
  return (double)(rs >> 11) / 9007199254740992.0 * 2 - 1;
}

#define DEFINE_CHECK(T, NAME, GEMM, EPS)                                                                          \
  static int NAME(int row, int ta, int tb, int m, int n, int k, T alpha, T beta, int pad) {                      \
    const int ar = ta ? k : m, ac = ta ? m : k, br = tb ? n : k, bc = tb ? k : n;                                \
    const int lda = (row ? ac : ar) + pad, ldb = (row ? bc : br) + pad, ldc = (row ? n : m) + pad;              \
    const size_t sa = (size_t)lda * (row ? ar : ac), sb = (size_t)ldb * (row ? br : bc),                         \
                 sc = (size_t)ldc * (row ? m : n);                                                               \
    T *A = malloc(sa * sizeof(T)), *B = malloc(sb * sizeof(T)), *C = malloc(sc * sizeof(T));                     \
    long double *R = malloc(sc * sizeof(long double));                                                           \
    for (size_t i = 0; i < sa; ++i) A[i] = (T)urand();                                                           \
    for (size_t i = 0; i < sb; ++i) B[i] = (T)urand();                                                           \
    for (size_t i = 0; i < sc; ++i) R[i] = C[i] = (T)urand();                                                    \
    for (int i = 0; i < m; ++i)                                                                                  \
      for (int j = 0; j < n; ++j) {                                                                              \
        long double s = 0;                                                                                       \
        for (int p = 0; p < k; ++p) {                                                                            \
          const int ai = ta ? p : i, aj = ta ? i : p, bi = tb ? j : p, bj = tb ? p : j;                          \
          const T a = row ? A[(size_t)ai * lda + aj] : A[ai + (size_t)aj * lda];                                 \
          const T b = row ? B[(size_t)bi * ldb + bj] : B[bi + (size_t)bj * ldb];                                 \
          s += (long double)a * b;                                                                               \
        }                                                                                                        \
        const size_t ci = row ? (size_t)i * ldc + j : i + (size_t)j * ldc;                                       \
        R[ci] = alpha * s + (beta == 0 ? 0 : beta * R[ci]);                                                      \
      }                                                                                                          \
    if (getenv("VERBOSE"))                                                                                       \
      fprintf(stderr, "%s row=%d ta=%d tb=%d m=%d n=%d k=%d pad=%d\n", #T, row, ta, tb, m, n, k, pad);           \
    GEMM(row ? CblasRowMajor : CblasColMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, m, \
         n, k, alpha, A, lda, B, ldb, beta, C, ldc);                                                             \
    double err = 0;                                                                                              \
    for (size_t i = 0; i < sc; ++i) err = fmax(err, fabs((double)(C[i] - R[i])));                               \
    const double tol = EPS * (k + 4) * 4;                                                                        \
    const int bad = !(err <= tol);                                                                               \
    if (bad)                                                                                                     \
      printf("FAIL %s %s ta=%d tb=%d m=%d n=%d k=%d alpha=%g beta=%g pad=%d err=%.3g tol=%.3g\n", #T,            \
             row ? "row" : "col", ta, tb, m, n, k, (double)alpha, (double)beta, pad, err, tol);                   \
    free(A); free(B); free(C); free(R);                                                                          \
    return bad;                                                                                                  \
  }
DEFINE_CHECK(float, check_s, cblas_sgemm, 1.2e-7)
DEFINE_CHECK(double, check_d, cblas_dgemm, 2.3e-16)

int main(int argc, char **argv) {
  const int quick = argc > 1 && !strcmp(argv[1], "quick");
  static const int sizes[] = {1, 3, 7, 16, 17, 31, 32, 33, 48, 63, 64, 65, 97, 128, 130, 200};
  static const int big[][3] = {{256, 300, 260}, {513, 129, 700}, {70, 1100, 300}, {1030, 40, 1500}, {600, 700, 1300}};
  const int ns = quick ? 8 : (int)(sizeof(sizes) / sizeof(sizes[0]));
  const double ab[][2] = {{1, 0}, {1, 1}, {-0.5, 0.25}, {2, -1}};
  int fails = 0, runs = 0;
  for (int th = 0; th < 2; ++th) {
    openblas_set_num_threads(th == 0 ? 1 : openblas_get_num_procs());
    for (int row = 0; row < 2; ++row)
      for (int ta = 0; ta < 2; ++ta)
        for (int tb = 0; tb < 2; ++tb) {
          for (int x = 0; x < ns; ++x)
            for (int y = 0; y < ns; y += 3)
              for (int z = 0; z < ns; z += 5) {
                const int m = sizes[x], n = sizes[(x + y) % ns], k = sizes[(y + z) % ns];
                const double *s = ab[(x + y + z) % 4];
                const int pad = (x + z) % 3;
                fails += check_s(row, ta, tb, m, n, k, (float)s[0], (float)s[1], pad);
                fails += check_d(row, ta, tb, m, n, k, s[0], s[1], pad);
                runs += 2;
              }
          for (int b = 0; b < 5; ++b) {
            const double *s = ab[b % 4];
            fails += check_s(row, ta, tb, big[b][0], big[b][1], big[b][2], (float)s[0], (float)s[1], b % 2);
            fails += check_d(row, ta, tb, big[b][0], big[b][1], big[b][2], s[0], s[1], b % 2);
            runs += 2;
          }
        }
  }
  printf("%s: %d of %d checks failed (core %s)\n", fails ? "FAILED" : "passed", fails, runs, openblas_get_corename());
  return fails != 0;
}
