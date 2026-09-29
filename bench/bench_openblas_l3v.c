// GFLOPS of every variant of ?symm (side, uplo), ?syrk and ?syr2k (uplo, trans), ?trmm and ?trsm (side, uplo,
// trans, diag), column-major, n x n. Built against OpenBLAS, or against Accelerate with -DUSE_ACCEL.
// usage: bench_openblas_l3v <threads (0: library default)> <n>
#ifdef USE_ACCEL
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <pthread/qos.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
#define TIME(expr, flops)                                                                                          \
  ({                                                                                                               \
    double best_ = 1e9;                                                                                            \
    for (int r_ = 0; r_ < 5; r_++) {                                                                               \
      int n_ = 0;                                                                                                  \
      double t0_ = now(), t_;                                                                                      \
      do { expr; n_++; t_ = now() - t0_; } while (t_ < 0.05);                                                       \
      if (t_ / n_ < best_) best_ = t_ / n_;                                                                        \
    }                                                                                                              \
    (flops) / best_ / 1e9;                                                                                         \
  })

#define VARIANTS(T, P)                                                                                             \
  static void P##_run(int n) {                                                                                     \
    const size_t nn = (size_t)n * n;                                                                               \
    T *A = malloc(nn * sizeof(T)), *B = malloc(nn * sizeof(T)), *C = malloc(nn * sizeof(T));                       \
    for (size_t i = 0; i < nn; i++) { A[i] = (T)0.001 * (i % 7) / n; B[i] = C[i] = (T)0.001 * (i % 7); }          \
    for (int i = 0; i < n; i++) A[i + (size_t)i * n] = 1;                                                          \
    const double N3 = (double)n * n * n;                                                                           \
    const char *pr = sizeof(T) == 4 ? "f32" : "f64";                                                               \
    for (int s = 0; s < 2; ++s)                                                                                    \
      for (int u = 0; u < 2; ++u)                                                                                  \
        printf("%s symm  side=%c uplo=%c              %6.0f\n", pr, "LR"[s], "UL"[u],                             \
               TIME(cblas_##P##symm(CblasColMajor, s ? CblasRight : CblasLeft, u ? CblasLower : CblasUpper, n, n, 1, A, n, B, n, 1, C, n), 2 * N3)); \
    for (int u = 0; u < 2; ++u)                                                                                    \
      for (int t = 0; t < 2; ++t) {                                                                                \
        printf("%s syrk  uplo=%c trans=%c             %6.0f\n", pr, "UL"[u], "NT"[t],                             \
               TIME(cblas_##P##syrk(CblasColMajor, u ? CblasLower : CblasUpper, t ? CblasTrans : CblasNoTrans, n, n, 1, A, n, 1, C, n), N3)); \
        printf("%s syr2k uplo=%c trans=%c             %6.0f\n", pr, "UL"[u], "NT"[t],                             \
               TIME(cblas_##P##syr2k(CblasColMajor, u ? CblasLower : CblasUpper, t ? CblasTrans : CblasNoTrans, n, n, 1, A, n, B, n, 1, C, n), 2 * N3)); \
      }                                                                                                            \
    for (int m = 0; m < 2; ++m)                                                                                    \
      for (int s = 0; s < 2; ++s)                                                                                  \
        for (int u = 0; u < 2; ++u)                                                                                \
          for (int t = 0; t < 2; ++t)                                                                              \
            for (int d = 0; d < 2; ++d) {                                                                          \
              const enum CBLAS_SIDE sd = s ? CblasRight : CblasLeft;                                               \
              const enum CBLAS_UPLO ul = u ? CblasLower : CblasUpper;                                              \
              const enum CBLAS_TRANSPOSE tr = t ? CblasTrans : CblasNoTrans;                                       \
              const enum CBLAS_DIAG dg = d ? CblasUnit : CblasNonUnit;                                             \
              printf("%s %s side=%c uplo=%c trans=%c diag=%c %6.0f\n", pr, m ? "trsm " : "trmm ", "LR"[s], "UL"[u], \
                     "NT"[t], "NU"[d],                                                                             \
                     m ? TIME(cblas_##P##trsm(CblasColMajor, sd, ul, tr, dg, n, n, 1, A, n, B, n), N3)               \
                       : TIME(cblas_##P##trmm(CblasColMajor, sd, ul, tr, dg, n, n, 1, A, n, B, n), N3));              \
              fflush(stdout);                                                                                      \
            }                                                                                                      \
    free(A); free(B); free(C);                                                                                     \
  }
VARIANTS(float, s)
VARIANTS(double, d)

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: bench_openblas_l3v <threads> <n>\n"); return 2; }
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#ifndef USE_ACCEL
  if (atoi(argv[1]) > 0) openblas_set_num_threads(atoi(argv[1]));
  struct timespec settle = {12, 0};  // idle pool threads spin for about 11 s after start-up
  nanosleep(&settle, 0);
#endif
  const int n = atoi(argv[2]);
  s_run(n);
  d_run(n);
  return 0;
}
