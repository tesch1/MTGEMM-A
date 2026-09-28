// GFLOPS of ?gemm, ?symm, ?syrk, ?syr2k, ?trmm and ?trsm on n x n problems (column-major; lower and left where
// they apply; beta 1). Built against OpenBLAS, or against Accelerate with -DUSE_ACCEL.
// usage: bench_openblas_l3 <threads (0: library default)> <n> [n ...]
// Flops: 2 n^3 for gemm, symm and syr2k; n^3 for syrk, trmm and trsm. Minimum over five trials of at least 50 ms.
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

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: bench_openblas_l3 <threads> <n> [n ...]\n"); return 2; }
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#ifndef USE_ACCEL
  if (atoi(argv[1]) > 0) openblas_set_num_threads(atoi(argv[1]));
  struct timespec settle = {12, 0};  // idle pool threads spin for about 11 s after start-up
  nanosleep(&settle, 0);
  printf("# openblas %s threads=%d\n", openblas_get_config(), openblas_get_num_threads());
#else
  printf("# accelerate\n");
#endif
  printf("#   n prec   gemm   symm   syrk  syr2k   trmm   trsm\n");
  for (int ni = 2; ni < argc; ++ni) {
    const int n = atoi(argv[ni]);
    const size_t nn = (size_t)n * n;
    float *A = malloc(nn * 4), *B = malloc(nn * 4), *C = malloc(nn * 4);
    double *dA = malloc(nn * 8), *dB = malloc(nn * 8), *dC = malloc(nn * 8);
    // unit diagonal and off-diagonal entries of order 1/n: trmm and trsm, which overwrite B on every call, change it
    // by a factor near 1, so that repeated calls neither overflow nor reach subnormal numbers
    for (size_t i = 0; i < nn; i++) {
      A[i] = 0.001f * (i % 7) / n; dA[i] = 0.001 * (i % 7) / n;
      B[i] = C[i] = 0.001f * (i % 7); dB[i] = dC[i] = 0.001 * (i % 7);
    }
    for (int i = 0; i < n; i++) { A[i + (size_t)i * n] = 1; dA[i + (size_t)i * n] = 1; }
    const double N3 = (double)n * n * n;
    printf("%5d f32 %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f\n", n,
           TIME(cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 1, C, n), 2 * N3),
           TIME(cblas_ssymm(CblasColMajor, CblasLeft, CblasLower, n, n, 1, A, n, B, n, 1, C, n), 2 * N3),
           TIME(cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n, 1, A, n, 1, C, n), N3),
           TIME(cblas_ssyr2k(CblasColMajor, CblasLower, CblasNoTrans, n, n, 1, A, n, B, n, 1, C, n), 2 * N3),
           TIME(cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, n, 1, A, n, B, n), N3),
           TIME(cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, n, 1, A, n, B, n), N3));
    printf("%5d f64 %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f\n", n,
           TIME(cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, dA, n, dB, n, 1, dC, n), 2 * N3),
           TIME(cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, n, n, 1, dA, n, dB, n, 1, dC, n), 2 * N3),
           TIME(cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n, 1, dA, n, 1, dC, n), N3),
           TIME(cblas_dsyr2k(CblasColMajor, CblasLower, CblasNoTrans, n, n, 1, dA, n, dB, n, 1, dC, n), 2 * N3),
           TIME(cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, n, 1, dA, n, dB, n), N3),
           TIME(cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, n, 1, dA, n, dB, n), N3));
    fflush(stdout);
    free(A); free(B); free(C); free(dA); free(dB); free(dC);
  }
  return 0;
}
