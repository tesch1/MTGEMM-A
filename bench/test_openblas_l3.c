// Checks cblas_?symm, ?syrk, ?syr2k, ?trmm and ?trsm of an OpenBLAS build against long-double references: both
// orders, every side / uplo / trans / diag, several alpha and beta values, sizes around the recursion thresholds,
// padded leading dimensions, 1 and all threads. Unused triangles of A (and a unit diagonal) hold NaN, and every
// element of C or B outside the result must come back bit for bit, so reads or writes outside the BLAS contract fail.
// usage: [VERBOSE=1] test_openblas_l3 [quick]
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long double R;
static unsigned long long rs = 88172645463325252ULL;
static double urand(void) {
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
  return (double)(rs >> 11) / 9007199254740992.0 * 2 - 1;
}

// Element (i, j) of a matrix stored in the given order with leading dimension ld.
#define IDX(row, ld, i, j) ((row) ? (size_t)(i) * (ld) + (j) : (i) + (size_t)(j) * (ld))

#define DEFINE_TESTS(T, P, EPS)                                                                                    \
  static T *P##_alloc(size_t n) { return malloc((n ? n : 1) * sizeof(T)); }                                      \
  /* A symmetric or triangular t x t matrix: the unused triangle (and a unit diagonal) is NaN. */                  \
  static T *P##_tri(int row, int lower, int unit, int t, int ld, int dom) {                                      \
    T *a = P##_alloc((size_t)ld * t);                                                                            \
    for (size_t x = 0; x < (size_t)ld * t; ++x) a[x] = (T)NAN;                                                     \
    for (int i = 0; i < t; ++i)                                                                                  \
      for (int j = 0; j < t; ++j) {                                                                              \
        if (lower ? i < j : i > j) continue;                                                                     \
        if (i == j && unit) continue;                                                                            \
        a[IDX(row, ld, i, j)] = (T)(i == j && dom ? 2 + urand() : urand() / (dom ? t : 1));                       \
      }                                                                                                          \
    return a;                                                                                                    \
  }                                                                                                              \
  static T *P##_gen(size_t n) { T *x = P##_alloc(n); for (size_t i = 0; i < n; ++i) x[i] = (T)urand(); return x; } \
  /* op(A)(i, j) of a triangular A (zero outside the triangle, one on a unit diagonal) */                          \
  static R P##_top(const T *a, int row, int ld, int lower, int tr, int unit, int i, int j) {                      \
    if (tr) { int s = i; i = j; j = s; }                                                                         \
    if (i == j) return unit ? 1 : a[IDX(row, ld, i, i)];                                                         \
    return (lower ? i > j : i < j) ? a[IDX(row, ld, i, j)] : 0;                                                   \
  }                                                                                                              \
  static R P##_sym(const T *a, int row, int ld, int lower, int i, int j) {                                       \
    if (lower ? i < j : i > j) { int s = i; i = j; j = s; }                                                      \
    return a[IDX(row, ld, i, j)];                                                                                \
  }                                                                                                              \
  /* max |got - ref| over the result, scaled; out-of-result elements of got must equal orig bit for bit */        \
  static int P##_cmp(const char *what, const T *got, const T *orig, const R *ref, const unsigned char *in,         \
                     size_t n, R scale) {                                                                        \
    R err = 0;                                                                                                   \
    for (size_t i = 0; i < n; ++i) {                                                                             \
      if (!in[i]) {                                                                                              \
        if (memcmp(&got[i], &orig[i], sizeof(T))) { printf("FAIL %s: element %zu outside the result changed\n", what, i); return 1; } \
        continue;                                                                                                \
      }                                                                                                          \
      R e = fabsl((R)got[i] - ref[i]);                                                                           \
      if (!(e == e)) e = INFINITY;                                                                               \
      if (e > err) err = e;                                                                                      \
    }                                                                                                            \
    if (!(err <= scale * EPS)) { printf("FAIL %s: err %.3Lg > %.3Lg\n", what, err, scale * (R)EPS); return 1; }   \
    return 0;                                                                                                    \
  }                                                                                                              \
  static int P##_symm(int row, int side, int lower, int m, int n, T alpha, T beta, int pad) {                   \
    const int t = side ? n : m, lda = t + pad, ldb = (row ? n : m) + pad, ldc = ldb;                             \
    const size_t sb = (size_t)ldb * (row ? m : n);                                                               \
    T *a = P##_tri(row, lower, 0, t, lda, 0), *b = P##_gen(sb), *c = P##_gen(sb), *c0 = P##_alloc(sb);           \
    R *ref = malloc(sb * sizeof(R)); unsigned char *in = calloc(sb, 1);                                          \
    memcpy(c0, c, sb * sizeof(T));                                                                               \
    for (int i = 0; i < m; ++i)                                                                                  \
      for (int j = 0; j < n; ++j) {                                                                              \
        R s = 0;                                                                                                 \
        for (int p = 0; p < t; ++p)                                                                              \
          s += side ? (R)b[IDX(row, ldb, i, p)] * P##_sym(a, row, lda, lower, p, j)                              \
                    : P##_sym(a, row, lda, lower, i, p) * (R)b[IDX(row, ldb, p, j)];                             \
        size_t x = IDX(row, ldc, i, j); in[x] = 1;                                                               \
        ref[x] = alpha * s + (beta == 0 ? 0 : beta * (R)c0[x]);                                                  \
      }                                                                                                          \
    cblas_##P##symm(row ? CblasRowMajor : CblasColMajor, side ? CblasRight : CblasLeft, lower ? CblasLower : CblasUpper, \
                    m, n, alpha, a, lda, b, ldb, beta, c, ldc);                                                  \
    char what[160]; snprintf(what, sizeof what, "%ssymm row=%d side=%d lower=%d m=%d n=%d alpha=%g beta=%g pad=%d", #P, row, side, lower, m, n, (double)alpha, (double)beta, pad); \
    int bad = P##_cmp(what, c, c0, ref, in, sb, 4 * (t + 4));                                                   \
    free(a); free(b); free(c); free(c0); free(ref); free(in); return bad;                                        \
  }                                                                                                              \
  static int P##_syrk(int two, int row, int lower, int tr, int n, int k, T alpha, T beta, int pad) {            \
    const int ar = tr ? k : n, ac = tr ? n : k, lda = (row ? ac : ar) + pad, ldc = n + pad;                      \
    const size_t sa = (size_t)lda * (row ? ar : ac), sc = (size_t)ldc * n;                                       \
    T *a = P##_gen(sa), *b = P##_gen(sa), *c = P##_gen(sc), *c0 = P##_alloc(sc);                                 \
    R *ref = malloc(sc * sizeof(R)); unsigned char *in = calloc(sc, 1);                                          \
    memcpy(c0, c, sc * sizeof(T));                                                                               \
    for (int i = 0; i < n; ++i)                                                                                  \
      for (int j = lower ? 0 : i; j <= (lower ? i : n - 1); ++j) {                                               \
        R s = 0;                                                                                                 \
        for (int p = 0; p < k; ++p) {                                                                            \
          R ai = tr ? a[IDX(row, lda, p, i)] : a[IDX(row, lda, i, p)], aj = tr ? a[IDX(row, lda, p, j)] : a[IDX(row, lda, j, p)]; \
          R bi = tr ? b[IDX(row, lda, p, i)] : b[IDX(row, lda, i, p)], bj = tr ? b[IDX(row, lda, p, j)] : b[IDX(row, lda, j, p)]; \
          s += two ? ai * bj + bi * aj : ai * aj;                                                                \
        }                                                                                                        \
        size_t x = IDX(row, ldc, i, j); in[x] = 1;                                                               \
        ref[x] = alpha * s + (beta == 0 ? 0 : beta * (R)c0[x]);                                                  \
      }                                                                                                          \
    const enum CBLAS_ORDER o = row ? CblasRowMajor : CblasColMajor;                                              \
    const enum CBLAS_UPLO u = lower ? CblasLower : CblasUpper;                                                   \
    const enum CBLAS_TRANSPOSE ct = tr ? CblasTrans : CblasNoTrans;                                              \
    if (two) cblas_##P##syr2k(o, u, ct, n, k, alpha, a, lda, b, lda, beta, c, ldc);                              \
    else cblas_##P##syrk(o, u, ct, n, k, alpha, a, lda, beta, c, ldc);                                           \
    char what[160]; snprintf(what, sizeof what, "%ssyr%sk row=%d lower=%d trans=%d n=%d k=%d alpha=%g beta=%g pad=%d", #P, two ? "2" : "", row, lower, tr, n, k, (double)alpha, (double)beta, pad); \
    int bad = P##_cmp(what, c, c0, ref, in, sc, 4 * (2 * k + 4));                                               \
    free(a); free(b); free(c); free(c0); free(ref); free(in); return bad;                                        \
  }                                                                                                              \
  /* TRMM (solve 0): B := alpha op(A) B or alpha B op(A); TRSM (solve 1): residual of op(A) X = alpha B */        \
  static int P##_trxm(int solve, int row, int side, int lower, int tr, int unit, int m, int n, T alpha, int pad) { \
    const int t = side ? n : m, lda = t + pad, ldb = (row ? n : m) + pad;                                         \
    const size_t sb = (size_t)ldb * (row ? m : n);                                                               \
    T *a = P##_tri(row, lower, unit, t, lda, solve), *b = P##_gen(sb), *b0 = P##_alloc(sb);                       \
    R *ref = malloc(sb * sizeof(R)); unsigned char *in = calloc(sb, 1);                                          \
    memcpy(b0, b, sb * sizeof(T));                                                                               \
    const enum CBLAS_ORDER o = row ? CblasRowMajor : CblasColMajor;                                              \
    const enum CBLAS_SIDE sd = side ? CblasRight : CblasLeft;                                                    \
    const enum CBLAS_UPLO u = lower ? CblasLower : CblasUpper;                                                   \
    const enum CBLAS_TRANSPOSE ct = tr ? CblasTrans : CblasNoTrans;                                              \
    const enum CBLAS_DIAG dg = unit ? CblasUnit : CblasNonUnit;                                                  \
    if (solve) cblas_##P##trsm(o, sd, u, ct, dg, m, n, alpha, a, lda, b, ldb);                                   \
    else cblas_##P##trmm(o, sd, u, ct, dg, m, n, alpha, a, lda, b, ldb);                                         \
    /* TRMM: ref = alpha op(A) B0; TRSM: ref = op(A) X compared with alpha B0 */                                   \
    const T *x = solve ? b : b0;                                                                                 \
    for (int i = 0; i < m; ++i)                                                                                  \
      for (int j = 0; j < n; ++j) {                                                                              \
        R s = 0;                                                                                                 \
        for (int p = 0; p < t; ++p)                                                                              \
          s += side ? (R)x[IDX(row, ldb, i, p)] * P##_top(a, row, lda, lower, tr, unit, p, j)                     \
                    : P##_top(a, row, lda, lower, tr, unit, i, p) * (R)x[IDX(row, ldb, p, j)];                    \
        size_t q = IDX(row, ldb, i, j); in[q] = 1;                                                               \
        ref[q] = solve ? s : alpha * s;                                                                          \
      }                                                                                                          \
    if (solve) /* compare op(A) X (in ref) with alpha B0: swap roles so that P##_cmp sees ref vs got */            \
      for (size_t q = 0; q < sb; ++q) if (in[q]) { R r = ref[q]; ref[q] = alpha * (R)b0[q]; b0[q] = b[q]; b[q] = (T)r; } \
    char what[180]; snprintf(what, sizeof what, "%str%sm row=%d side=%d lower=%d trans=%d unit=%d m=%d n=%d alpha=%g pad=%d", #P, solve ? "s" : "m", row, side, lower, tr, unit, m, n, (double)alpha, pad); \
    int bad = P##_cmp(what, b, b0, ref, in, sb, 4 * (t + 4) * (solve ? 4 : 1));                                  \
    free(a); free(b); free(b0); free(ref); free(in); return bad;                                                 \
  }

DEFINE_TESTS(float, s, 1.2e-7)
DEFINE_TESTS(double, d, 2.3e-16)

int main(int argc, char **argv) {
  const int quick = argc > 1 && !strcmp(argv[1], "quick");
  static const int dims[][2] = {{150, 130}, {257, 90}, {90, 300}, {64, 64}, {65, 200}, {200, 33}, {513, 70}, {128, 128}};
  const int nd = quick ? 3 : (int)(sizeof(dims) / sizeof(dims[0]));
  static const double ab[][2] = {{1, 0}, {1, 1}, {-0.5, 0.25}, {2, -1}};
  int fails = 0, runs = 0;
#define RUN(expr) do { if (getenv("VERBOSE")) fprintf(stderr, "%s\n", #expr); fails += (expr); ++runs; } while (0)
  for (int th = 0; th < 2; ++th) {
    openblas_set_num_threads(th == 0 ? 1 : openblas_get_num_procs());
    for (int d = 0; d < nd; ++d) {
      const int m = dims[d][0], n = dims[d][1];
      for (int row = 0; row < 2; ++row)
        for (int side = 0; side < 2; ++side)
          for (int lower = 0; lower < 2; ++lower) {
            const double *s = ab[(d + row + side + lower) % 4];
            const int pad = (d + lower) % 3;
            RUN(s_symm(row, side, lower, m, n, (float)s[0], (float)s[1], pad));
            RUN(d_symm(row, side, lower, m, n, s[0], s[1], pad));
            for (int tr = 0; tr < 2; ++tr) {
              if (side == 0) {
                RUN(s_syrk(0, row, lower, tr, m, n, (float)s[0], (float)s[1], pad));
                RUN(d_syrk(0, row, lower, tr, m, n, s[0], s[1], pad));
                RUN(s_syrk(1, row, lower, tr, m, n, (float)s[0], (float)s[1], pad));
                RUN(d_syrk(1, row, lower, tr, m, n, s[0], s[1], pad));
              }
              for (int unit = 0; unit < 2; ++unit)
                for (int solve = 0; solve < 2; ++solve) {
                  RUN(s_trxm(solve, row, side, lower, tr, unit, m, n, (float)s[0], pad));
                  RUN(d_trxm(solve, row, side, lower, tr, unit, m, n, s[0], pad));
                }
            }
          }
    }
  }
  printf("%s: %d of %d checks failed (core %s)\n", fails ? "FAILED" : "passed", fails, runs, openblas_get_corename());
  return fails != 0;
}
