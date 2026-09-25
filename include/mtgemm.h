// MTGEMM-A: SME GEMM after Deng et al., "Demystifying ARM SME to Optimize General Matrix Multiplications".
// libmtgemm_amx.a implements the same API with Apple AMX (M1-M3 class, Vision Pro M2); see the notes per option.
#pragma once
#include <cstddef>

enum mt_order { MtRowMajor = 101, MtColMajor = 102 };

// Switches for the ablation study. Defaults: the paper's design plus our fixes (cdirect=2, pack4=1, prefetch=7);
// the paper's design as published is cdirect=0, pack4=0, prefetch=0.
struct mt_options {
  int online = 1;   // first-round online packing of B (AMX: 0 = up front, 1 = auto, 2 = by AMX, 3 = by the core, 4 = not packed)
  int x4 = 1;       // four-vector ld1/st1 in packing and micro-kernel (0: one vector per instruction)
  int heap = 1;     // packed buffers on the heap (0: on the stack of the calling thread)
  int model = 1;    // mc/nc/kc from the analytical model (0: fixed kc=mc=256, nc=1024; 2: no blocking)
  int shape = 0;    // main micro-kernel: 0 = 16x64 (fp64: 8x64), 1 = 32x32 (fp64: 16x32)
  int cdirect = 2;  // C tiles: 0 = x4 ld/st + one MOVA per slice (paper), 1 = ld1w/st1w ZA slices, 2 = 4-row MOVA vg4
  int pack4 = 1;    // A transposition in 4-row groups with MOVA vg4 (0: paper, one row and four MOVAs; AMX: 1 = through Z, 0 = NEON)
  int prefetch = 7;  // core prfm into L2, bit mask: 1 = A rows being transposed, 2 = B strips, 4 = next C tile
                     // (AMX: 1 also prefetches the next A panel during a kernel; 8 turns that part off)
  int threads = 1;  // 1, 2 (one thread per performance-cluster SME unit), or 0: 2 from 2^22 multiply-adds, else 1
                    // (AMX: 0 means 1, since the M2 cores share one P-cluster AMX unit)
  int mc = 0, nc = 0, kc = 0;  // nonzero: override the blocking
  int pfdist = 0;   // AMX backend: B source rows prefetched ahead during online packing (0: default)
  int eshare = 10;  // AMX backend, threads = 3: percent of the work for a thread on the efficiency cluster
  int prof = 0;     // profiling only, gives wrong results: 1 skips A packing, 2 skips micro-kernels, 4 skips B packing
};

void mt_sgemm(mt_order order, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
              float beta, float* C, int ldc, const mt_options* opt = nullptr);
void mt_dgemm(mt_order order, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb,
              double beta, double* C, int ldc, const mt_options* opt = nullptr);

struct mt_blocking { int mc, nc, kc; };
// Blocking chosen by the model for a row-major problem (col-major problems are solved as C^T = B^T A^T).
mt_blocking mt_model_blocking(int M, int N, int K, int elem_size, int mr, int nr);
