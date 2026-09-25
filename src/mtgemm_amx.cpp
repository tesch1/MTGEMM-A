// AMX GEMM (Apple M1-M3 class, target Vision Pro M2): the blocking of the SME version with an AMX micro-kernel.
// fp32: 32x32 C block in the four 16x16 Z accumulators; fp64: 16x32 in the eight 8x8 ones. Row-major core,
// column-major solved as C^T = B^T A^T. A is packed k-major into Y-side panels, B into X-side panels.
#include "internal.h"
#include "amx.h"
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

#define MT_INL __attribute__((always_inline)) inline
#define MT_NOINL __attribute__((noinline))

namespace mt {
namespace {

using namespace amx;

template <class T> struct Ak;
template <> struct Ak<float> { static constexpr int MR = 32, NR = 32; };
template <> struct Ak<double> { static constexpr int MR = 16, NR = 32; };

constexpr int kKU = 4;  // packed depth is padded with zeros to a multiple of this

// Elements from one packed B panel to the next: 256 bytes more than the panel, so that the panels of a block
// (written a row at a time) do not all fall into the same L1 sets when kp * NR is a power of two.
template <class T>
constexpr long bstride(int kp) { return long(kp) * Ak<T>::NR + 256 / long(sizeof(T)); }

MT_INL void pf_l2(const void* p, int bytes) {
  for (int l = 0; l < bytes; l += 128) __builtin_prefetch(static_cast<const char*>(p) + l, 0, 2);
}

// Z <-> C. fp32 row m = 16p + j lives in Z rows 4j + 2p + {0,1}; fp64 row m = 8p + j in Z rows 8j + 4p + {0..3}.
template <class T, bool STORE>
MT_INL void z_io(T* C, long ldc) {
  const bool pair = ((reinterpret_cast<uintptr_t>(C) | uintptr_t(ldc * sizeof(T))) & 127) == 0;
  if constexpr (sizeof(T) == 4) {
    for (int p = 0; p < 2; ++p)
      for (int j = 0; j < 16; ++j) {
        T* row = C + long(16 * p + j) * ldc;
        const int z = 4 * j + 2 * p;
        if (pair) {
          if (STORE) AMX_STZ(zr(row, z, true)); else AMX_LDZ(zr(row, z, true));
        } else {
          if (STORE) { AMX_STZ(zr(row, z)); AMX_STZ(zr(row + 16, z + 1)); }
          else { AMX_LDZ(zr(row, z)); AMX_LDZ(zr(row + 16, z + 1)); }
        }
      }
  } else {
    for (int p = 0; p < 2; ++p)
      for (int j = 0; j < 8; ++j) {
        T* row = C + long(8 * p + j) * ldc;
        const int z = 8 * j + 4 * p;
        if (pair) {
          if (STORE) { AMX_STZ(zr(row, z, true)); AMX_STZ(zr(row + 16, z + 2, true)); }
          else { AMX_LDZ(zr(row, z, true)); AMX_LDZ(zr(row + 16, z + 2, true)); }
        } else {
          for (int q = 0; q < 4; ++q) {
            if (STORE) AMX_STZ(zr(row + 8 * q, z + q)); else AMX_LDZ(zr(row + 8 * q, z + q));
          }
        }
      }
  }
}

// B rows straight from the source (online = 2): single-register X loads (no alignment needed), and the rows are
// written to the packed panel with X stores for the later rows of kernels; rows from kb up to kp read zeros.
template <class T>
struct SrcB {
  const T* src;
  long ldb;
  int kb, pfrows;
  bool store;  // write the packed copy for later rows of kernels
};
alignas(256) const float kZeros[64] = {};

template <class T>
MT_INL void src_rows(const SrcB<T>& s, int r0, int nrows, char* dst) {
  constexpr int RB = Ak<T>::NR * int(sizeof(T));  // bytes per B row: 128 (fp32) or 256 (fp64)
  for (int u = 0; u < nrows; ++u) {
    const int r = r0 + u;
    const char* p = r < s.kb ? reinterpret_cast<const char*>(s.src + long(r) * s.ldb) : reinterpret_cast<const char*>(kZeros);
    if (s.pfrows && r + s.pfrows < s.kb) pf_l2(s.src + long(r + s.pfrows) * s.ldb, RB);
    const int x0 = u * RB / 64;
    for (int q = 0; q < RB / 64; ++q) AMX_LDX(xy(p + 64 * q, x0 + q));
    if (s.store)
      for (int q = 0; q < RB / 128; ++q) AMX_STX(xy(dst + long(r) * RB + 128 * q, x0 + 2 * q, kPair));
  }
}

// Four depth steps of fp32 from 512-byte slabs of the A and B panels: 2 + 2 quad loads, 16 fma32.
template <bool SKIPZ, bool SRC = false>
MT_INL void step4_f32(const char* a, const char* b, const SrcB<float>* s = nullptr, int k = 0, char* bp = nullptr) {
  if constexpr (SRC) src_rows(*s, k, 4, bp);
  else { AMX_LDX(xy(b, 0, kQuad)); AMX_LDX(xy(b + 256, 4, kQuad)); }
  AMX_LDY(xy(a, 0, kQuad)); AMX_LDY(xy(a + 256, 4, kQuad));
#define MT_K(k, S) \
  AMX_FMA32(fma(0, 128 * k, 128 * k, S)); AMX_FMA32(fma(1, 128 * k + 64, 128 * k, S)); \
  AMX_FMA32(fma(2, 128 * k, 128 * k + 64, S)); AMX_FMA32(fma(3, 128 * k + 64, 128 * k + 64, S));
  MT_K(0, SKIPZ) MT_K(1, false) MT_K(2, false) MT_K(3, false)
#undef MT_K
}

// Four depth steps of fp64: A slab 4 x 16 (512 B, all of Y), B slab 4 x 32 (1 KB, X holds two steps at a time).
template <bool SKIPZ, bool SRC = false>
MT_INL void step4_f64(const char* a, const char* b, const SrcB<double>* s = nullptr, int k = 0, char* bp = nullptr) {
  AMX_LDY(xy(a, 0, kQuad)); AMX_LDY(xy(a + 256, 4, kQuad));
#define MT_K(kx, ky, S) \
  AMX_FMA64(fma(0, 256 * kx, 128 * ky, S)); AMX_FMA64(fma(1, 256 * kx + 64, 128 * ky, S)); \
  AMX_FMA64(fma(2, 256 * kx + 128, 128 * ky, S)); AMX_FMA64(fma(3, 256 * kx + 192, 128 * ky, S)); \
  AMX_FMA64(fma(4, 256 * kx, 128 * ky + 64, S)); AMX_FMA64(fma(5, 256 * kx + 64, 128 * ky + 64, S)); \
  AMX_FMA64(fma(6, 256 * kx + 128, 128 * ky + 64, S)); AMX_FMA64(fma(7, 256 * kx + 192, 128 * ky + 64, S));
  if constexpr (SRC) src_rows(*s, k, 2, bp);
  else { AMX_LDX(xy(b, 0, kQuad)); AMX_LDX(xy(b + 256, 4, kQuad)); }
  MT_K(0, 0, SKIPZ) MT_K(1, 1, false)
  if constexpr (SRC) src_rows(*s, k + 2, 2, bp);
  else { AMX_LDX(xy(b + 512, 0, kQuad)); AMX_LDX(xy(b + 768, 4, kQuad)); }
  MT_K(0, 2, false) MT_K(1, 3, false)
#undef MT_K
}

// A full-width B panel packed by the core while AMX computes: 4 rows per 4-step, zero rows from kb up to kp.
template <class T>
struct NextB {
  const T* src;
  long ldb;
  T* dst;
  int kb, pfrows;
};
constexpr int kPfRowsB = 32;  // B source rows prefetched ahead of the interleaved packing

template <class T>
MT_INL void pack_rows4(const NextB<T>& nb, int r0) {
  constexpr int NR = Ak<T>::NR, nf = NR * int(sizeof(T)) / 4;
  for (int u = 0; u < 4; ++u) {
    const int r = r0 + u;
    float* d = reinterpret_cast<float*>(nb.dst + long(r) * NR);
    if (r < nb.kb) {
      const float* s = reinterpret_cast<const float*>(nb.src + long(r) * nb.ldb);
      if (nb.pfrows && r + nb.pfrows < nb.kb) pf_l2(nb.src + long(r + nb.pfrows) * nb.ldb, NR * int(sizeof(T)));
      for (int c = 0; c < nf; c += 16) vst1q_f32_x4(d + c, vld1q_f32_x4(s + c));
    } else {
      const float32x4x4_t z = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
      for (int c = 0; c < nf; c += 16) vst1q_f32_x4(d + c, z);
    }
  }
}

// Side work of one kernel call. Online B packing: mode 1 packs the next panel with the core (nb), mode 2 has AMX
// read this call's panel from the source and write it to Bp (sb). pfa: next A panel, prefetched into L2 alongside.
template <class T>
struct Onl {
  int mode;
  NextB<T> nb;
  SrcB<T> sb;
  const char* pfa;
};

template <class T, int MODE>
MT_INL void kernel_body(int kp, const T* Ap, const T* Bp, T* C, long ldc, bool load, const Onl<T>* on) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  const char* a = reinterpret_cast<const char*>(Ap);
  const char* b = reinterpret_cast<const char*>(Bp);
  char* bp = const_cast<char*>(b);
  constexpr long sa = 4 * MR * sizeof(T), sb = 4 * NR * sizeof(T);
  auto step = [&](auto skip, int k) __attribute__((always_inline)) {
    if constexpr (sizeof(T) == 4) step4_f32<decltype(skip)::value, MODE == 2>(a, b, MODE == 2 ? &on->sb : nullptr, k, bp);
    else step4_f64<decltype(skip)::value, MODE == 2>(a, b, MODE == 2 ? &on->sb : nullptr, k, bp);
    if constexpr (MODE == 1) pack_rows4(on->nb, k);
    if (on && on->pfa) pf_l2(on->pfa + long(k) * MR * long(sizeof(T)), int(sa));
  };
  int k = 0;
  if (load) {
    z_io<T, false>(C, ldc);
  } else {
    step(std::true_type{}, 0);
    k = 4; a += sa; b += sb;
  }
  for (; k < kp; k += 4, a += sa, b += sb) step(std::false_type{}, k);
  z_io<T, true>(C, ldc);
}

// Micro-kernel: MR x NR block of C += A panel (kp x MR) * B panel (kp x NR), kp a multiple of 4.
// load = false: the first step overwrites Z (skip-Z), so C is not read.
template <class T>
MT_NOINL void kernel(int kp, const T* Ap, const T* Bp, T* C, long ldc, bool load, const Onl<T>* on) {
  if (!on || on->mode == 0) kernel_body<T, 0>(kp, Ap, Bp, C, ldc, load, on);
  else if (on->mode == 1) kernel_body<T, 1>(kp, Ap, Bp, C, ldc, load, on);
  else kernel_body<T, 2>(kp, Ap, Bp, C, ldc, load, on);
}

// Partial C block (rows < MR or cols < NR) through an aligned scratch tile.
template <class T>
MT_NOINL void kernel_edge(int kp, const T* Ap, const T* Bp, T* C, long ldc, int rows, int cols, bool load,
                          const Onl<T>* on) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  alignas(128) T t[MR * NR];
  if (load)
    for (int r = 0; r < rows; ++r) std::memcpy(t + r * NR, C + long(r) * ldc, cols * sizeof(T));
  kernel<T>(kp, Ap, Bp, t, NR, load, on);
  for (int r = 0; r < rows; ++r) std::memcpy(C + long(r) * ldc, t + r * NR, cols * sizeof(T));
}

// A block (mb x kb, row-major, lda) to panels of MR rows: Ap[panel][k][r], depth padded to kp, rows to MR, zeros.
MT_INL void tr4(const float* s, long lda, float* d, int dstride, float32x4_t al, bool sc) {
  float32x4_t r0 = vld1q_f32(s), r1 = vld1q_f32(s + lda), r2 = vld1q_f32(s + 2 * lda), r3 = vld1q_f32(s + 3 * lda);
  if (sc) { r0 = vmulq_f32(r0, al); r1 = vmulq_f32(r1, al); r2 = vmulq_f32(r2, al); r3 = vmulq_f32(r3, al); }
  const float32x4x2_t t01 = vtrnq_f32(r0, r1), t23 = vtrnq_f32(r2, r3);
  vst1q_f32(d, vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0])));
  vst1q_f32(d + dstride, vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1])));
  vst1q_f32(d + 2 * dstride, vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0])));
  vst1q_f32(d + 3 * dstride, vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1])));
}
MT_INL void tr4(const double* s, long lda, double* d, int dstride, float64x2_t al, bool sc) {
  for (int h = 0; h < 2; ++h) {  // 4 rows x 4 depth as 2x2 blocks
    for (int g = 0; g < 2; ++g) {
      float64x2_t r0 = vld1q_f64(s + long(2 * h) * lda + 2 * g), r1 = vld1q_f64(s + long(2 * h + 1) * lda + 2 * g);
      if (sc) { r0 = vmulq_f64(r0, al); r1 = vmulq_f64(r1, al); }
      vst1q_f64(d + (2 * g) * dstride + 2 * h, vzip1q_f64(r0, r1));
      vst1q_f64(d + (2 * g + 1) * dstride + 2 * h, vzip2q_f64(r0, r1));
    }
  }
}

constexpr int kAmxPfChunks = 4;  // source chunks prefetched ahead of the AMX transposition

// 32 depth steps of a full A panel (MR = 2E rows, E = 64 bytes of elements) transposed by AMX through Z: row m,
// depth group g (E steps) goes in with ldz to tile 2g + m/E; each depth step k comes out as a column of the two
// tiles of its group (extrv: y[j] = z[NT j + tile][k % E]) and leaves with one Y pair store.
template <class T>
MT_INL void tr32_amx(const T* A, long lda, T* dst) {
  constexpr int E = 64 / int(sizeof(T)), NT = E == 16 ? 4 : 8, MR = 2 * E;
  constexpr uint64_t kExtrY = (1ull << 63) | (uint64_t(E == 16 ? 8 : 1) << 11) | (1ull << 26) | (1ull << 10);
  for (int m = 0; m < MR; ++m)
    for (int g = 0; g < 32 / E; ++g) AMX_LDZ(zr(A + long(m) * lda + g * E, NT * (m % E) + 2 * g + m / E));
  for (int k = 0; k < 32; ++k) {
    const int t = 2 * (k / E), c = k % E, y = 128 * (k & 3);
    AMX_EXTRY(kExtrY | uint64_t(c * NT + t) << 20 | uint64_t(y));
    AMX_EXTRY(kExtrY | uint64_t(c * NT + t + 1) << 20 | uint64_t(y + 64));
    AMX_STY(xy(dst + long(k) * MR, 2 * (k & 3), kPair));
  }
}

// The panel is written one chunk of KB depth steps at a time (one 128-byte line of each of the MR source rows),
// so each destination line is complete before the next chunk; pf: prefetch the source two chunks ahead.
template <class T, int MR = Ak<T>::MR>
MT_NOINL void pack_a(int mb, int kb, int kp, const T* A, long lda, T alpha, T* Ap, bool pf, bool amx) {
  constexpr int KB = 128 / int(sizeof(T));
  const bool sc = alpha != T(1);
  auto al = [&] { if constexpr (sizeof(T) == 4) return vdupq_n_f32(alpha); else return vdupq_n_f64(alpha); }();
  for (int p0 = 0; p0 < mb; p0 += MR) {
    const int rows = std::min(MR, mb - p0);
    T* dst = Ap + long(p0) * kp;
    const int r4 = rows / 4 * 4, k4 = kb / 4 * 4;
    int kz = 0;
    if (amx && rows == MR && !sc)
      for (; kz + 32 <= kb; kz += 32) {
        if (pf) {  // AMX waits in order on each miss: the whole chunk kAmxPfChunks ahead, into the next panel too
          int kf = kz + 32 * kAmxPfChunks, pf0 = p0;
          if (kf + 32 > kb) { kf -= kb / 32 * 32; pf0 += MR; }
          if (pf0 + MR <= mb && kf >= 0)
            for (int r = 0; r < MR; ++r)
              for (int b = 0; b < 32 * int(sizeof(T)); b += 128) __builtin_prefetch(reinterpret_cast<const char*>(A + long(pf0 + r) * lda + kf) + b, 0, 2);
        }
        tr32_amx<T>(A + long(p0) * lda + kz, lda, dst + long(kz) * MR);
      }
    for (int k0 = kz; k0 < k4; k0 += KB) {
      const int k1 = std::min(k4, k0 + KB);
      for (int r = 0; r < r4; r += 4) {
        const T* src = A + long(p0 + r) * lda;
        if (pf && k0 + 2 * KB < kb)
          for (int u = 0; u < 4; ++u) __builtin_prefetch(src + long(u) * lda + k0 + 2 * KB, 0, 3);
        for (int k = k0; k < k1; k += 4) tr4(src + k, lda, dst + long(k) * MR + r, MR, al, sc);
      }
    }
    for (int r = 0; r < r4; r += 4) {
      const T* src = A + long(p0 + r) * lda;
      for (int k = k4; k < kb; ++k)
        for (int u = 0; u < 4; ++u) dst[long(k) * MR + r + u] = alpha * src[long(u) * lda + k];
    }
    for (int r = r4; r < rows; ++r) {
      const T* src = A + long(p0 + r) * lda;
      for (int k = 0; k < kb; ++k) dst[long(k) * MR + r] = alpha * src[k];
    }
    if (rows < MR)
      for (int k = 0; k < kb; ++k) std::memset(dst + long(k) * MR + rows, 0, (MR - rows) * sizeof(T));
    std::memset(dst + long(kb) * MR, 0, size_t(kp - kb) * MR * sizeof(T));
  }
}

// B block (kb x ncols, row-major, ldb) to one panel of NR columns: Bp[k][c], zero padded to kp x NR.
template <class T>
MT_NOINL void pack_b(int kb, int kp, int ncols, const T* B, long ldb, T* Bp) {
  constexpr int NR = Ak<T>::NR;
  if (ncols == NR) {
    for (int k = 0; k < kb; ++k) {
      const float* s = reinterpret_cast<const float*>(B + long(k) * ldb);
      float* d = reinterpret_cast<float*>(Bp + long(k) * NR);
      constexpr int nf = NR * sizeof(T) / 4;
      for (int c = 0; c < nf; c += 16) vst1q_f32_x4(d + c, vld1q_f32_x4(s + c));
    }
  } else {
    for (int k = 0; k < kb; ++k) {
      std::memcpy(Bp + long(k) * NR, B + long(k) * ldb, ncols * sizeof(T));
      std::memset(Bp + long(k) * NR + ncols, 0, (NR - ncols) * sizeof(T));
    }
  }
  std::memset(Bp + long(kb) * NR, 0, size_t(kp - kb) * NR * sizeof(T));
}

// Whole B block (kb x nb) to its panels in source row order, so that each source row streams contiguously
// (panel order touches one 128-byte piece per row, a page apart when ldb is large).
template <class T>
MT_NOINL void pack_b_rows(int kb, int kp, int nb, const T* B, long ldb, T* Bc, bool pf) {
  constexpr int NR = Ak<T>::NR, nf = NR * int(sizeof(T)) / 4;
  const int nfull = nb / NR * NR;
  for (int k = 0; k < kb; ++k) {
    const T* s = B + long(k) * ldb;
    if (pf && k + 8 < kb)
      for (int c = 0; c < nb; c += 128 / int(sizeof(T))) __builtin_prefetch(s + 8 * ldb + c, 0, 2);
    for (int jj = 0; jj < nfull; jj += NR) {
      const float* sf = reinterpret_cast<const float*>(s + jj);
      float* d = reinterpret_cast<float*>(Bc + long(jj / NR) * bstride<T>(kp) + long(k) * NR);
      for (int c = 0; c < nf; c += 16) vst1q_f32_x4(d + c, vld1q_f32_x4(sf + c));
    }
    if (nfull < nb) {
      T* d = Bc + long(nfull / NR) * bstride<T>(kp) + long(k) * NR;
      std::memcpy(d, s + nfull, (nb - nfull) * sizeof(T));
      std::memset(d + (nb - nfull), 0, (NR - (nb - nfull)) * sizeof(T));
    }
  }
  if (kp > kb)
    for (int jj = 0; jj < nb; jj += NR) std::memset(Bc + long(jj / NR) * bstride<T>(kp) + long(kb) * NR, 0, size_t(kp - kb) * NR * sizeof(T));
}

// AMX blocking: depth of about 4 KB per row (C is reloaded once per depth block), A block up to 8 MB, B block the
// rest of a 6 MB budget but at least 8 panels; each balanced over the problem so that blocks come out even.
inline int balance(int total, int block, int step) {
  if (block >= total) return round_up(total, step);
  const int nblk = (total + block - 1) / block;
  return std::min(round_up(block, step), round_up((total + nblk - 1) / nblk, step));
}
template <class T>
mt_blocking amx_blocking(int M, int N, int K) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR, es = sizeof(T);
  const int kc = balance(K, 4096 / es, kKU);
  const int mc = balance(M, std::max(MR, int((8L << 20) / (long(kc) * es)) / MR * MR), MR);
  const long rest = (6L << 20) - long(mc) * kc * es;
  const int ncap = std::max(8 * NR, int(std::max(rest, 0L) / (long(kc) * es)) / NR * NR);
  return {mc, balance(N, ncap, NR), kc};
}

template <class T>
struct Job {
  int M, N, K;
  T alpha;
  const T* A;
  long lda;
  const T* B;
  long ldb;
  bool load;  // C holds data to accumulate onto (beta = 1 after any pre-scaling)
  T* C;
  long ldc;
  mt_blocking blk;
  int prof, pf, online, pfdist, pack4;
  T* Ac;
  T* Bc;
};

template <class T>
void drive(const Job<T>& jb) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  const int M = jb.M, N = jb.N, K = jb.K;
  const int mc = jb.blk.mc, nc = jb.blk.nc, kc = jb.blk.kc;
  for (int i = 0; i < M; i += mc) {
    const int mb = std::min(mc, M - i);
    for (int k = 0; k < K; k += kc) {
      const int kb = std::min(kc, K - k), kp = round_up(kb, kKU);
      const long bs = bstride<T>(kp);
      if (!(jb.prof & 1)) pack_a<T>(mb, kb, kp, jb.A + long(i) * jb.lda + k, jb.lda, jb.alpha, jb.Ac, jb.pf & 1, jb.pack4);
      const bool load = k > 0 || jb.load;
      for (int j = 0; j < N; j += nc) {
        const int nb = std::min(nc, N - j);
        const T* Bs = jb.B + long(k) * jb.ldb + j;
        // B modes (option online, 1 = auto chosen in run_job): 0 packed up front in source row order; 3 = the core packs each
        // next panel during the kernel before it (first row of kernels, ii = 0); 2 = AMX reads full-width panels
        // from the source in that row and writes the packed copy; 4 = AMX reads full-width panels from the source in
        // every row and nothing is packed.
        const int online = jb.prof & 6 ? 0 : (jb.online == 2 || jb.online == 4 ? jb.online : jb.online == 3 ? 1 : 0);
        const int pfr = (jb.pf & 2) ? jb.pfdist : 0;
        if (jb.prof & 4) {
        } else if (online == 0) {
          pack_b_rows<T>(kb, kp, nb, Bs, jb.ldb, jb.Bc, jb.pf & 2);
        } else {
          for (int jj = 0; jj < nb; jj += NR) {
            const int cols = std::min(NR, nb - jj);
            if ((online == 1 && jj == 0) || cols < NR) pack_b<T>(kb, kp, cols, Bs + jj, jb.ldb, jb.Bc + long(jj / NR) * bs);
          }
        }
        if (jb.prof & 2) continue;
        T* Cb = jb.C + long(i) * jb.ldc + j;
        for (int ii = 0; ii < mb; ii += MR) {
          const int rows = std::min(MR, mb - ii);
          const T* Ap = jb.Ac + long(ii) * kp;
          for (int jj = 0; jj < nb; jj += NR) {
            const int cols = std::min(NR, nb - jj);
            T* Cp = Cb + long(ii) * jb.ldc + jj;
            Onl<T> on{0, {}, {}, nullptr}, *pon = nullptr;
            if (ii == 0 && online == 1 && nb - (jj + NR) >= NR) {
              on.mode = 1;
              on.nb = {Bs + jj + NR, jb.ldb, jb.Bc + long(jj / NR + 1) * bs, kb, pfr};
              pon = &on;
            } else if (((ii == 0 && online == 2) || online == 4) && cols == NR) {
              on.mode = 2;
              on.sb = {Bs + jj, jb.ldb, kb, pfr, online == 2};
              pon = &on;
            }
            if (jj == 0 && (jb.pf & 1) && !(jb.pf & 8)) {  // next A panel of this block (the first again for next j)
              const T* An = ii + MR < mb ? Ap + long(MR) * kp : (j + nc < N ? jb.Ac : nullptr);
              if (An) {
                on.pfa = reinterpret_cast<const char*>(An);
                pon = &on;
              }
            }
            if (load && (jb.pf & 4)) {  // next C tile towards L2: ldz latency on C otherwise stalls the tile start
              const T* Cn = jj + NR < nb ? Cp + NR : Cb + long(ii + MR) * jb.ldc;
              if (jj + NR < nb || ii + MR < mb)
                for (int r = 0; r < MR; ++r) pf_l2(Cn + long(r) * jb.ldc, NR * int(sizeof(T)));
            }
            if (rows == MR && cols == NR) kernel<T>(kp, Ap, jb.Bc + long(jj / NR) * bs, Cp, jb.ldc, load, pon);
            else kernel_edge<T>(kp, Ap, jb.Bc + long(jj / NR) * bs, Cp, jb.ldc, rows, cols, load, pon);
          }
        }
      }
    }
  }
}

// Per-thread packed buffers, reused across calls so that timing does not include page faults.
struct Buf {
  void* p = nullptr;
  size_t n = 0;
  ~Buf() { std::free(p); }
  void* get(size_t bytes) {
    if (bytes > n) {
      std::free(p);
      p = nullptr;
      if (posix_memalign(&p, 16384, bytes)) std::abort();
      n = bytes;
    }
    return p;
  }
};
thread_local Buf tl_buf;

template <class T>
void run_job(Job<T>& jb, const mt_options& o) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  jb.prof = o.prof;
  jb.pf = o.prefetch;
  // online = 1 (default): B is not packed when all of it fits in 2 MB (fp64 always, fp32 only for M <= 128: fp64
  // has twice the fma per load, fp32 pays the extra single-register loads once per row of kernels), else up front.
  jb.online = o.online != 1 ? o.online
              : (long(jb.K) * jb.N * long(sizeof(T)) <= (2L << 20) && (sizeof(T) == 8 || jb.M <= 128) ? 4 : 0);
  jb.pfdist = o.pfdist ? o.pfdist : kPfRowsB;
  jb.pack4 = o.pack4;
  mt_blocking b;
  if (o.mc && o.nc && o.kc) b = {o.mc, o.nc, o.kc};
  else if (o.model == 1) b = amx_blocking<T>(jb.M, jb.N, jb.K);
  else if (o.model == 3) b = mt_model_blocking(jb.M, jb.N, jb.K, sizeof(T), MR, NR);
  else if (o.model == 0) b = {256, 1024, 256};
  else b = {jb.M, jb.N, jb.K};
  b.mc = round_up(std::max(b.mc, MR), MR);
  b.nc = round_up(std::max(b.nc, NR), NR);
  b.kc = std::max(b.kc, 1);
  jb.blk = b;
  const size_t kp = size_t(round_up(b.kc, kKU));
  const size_t na = size_t(b.mc) * kp, nb = size_t(b.nc / NR) * bstride<T>(int(kp));
  char* base = static_cast<char*>(tl_buf.get((na + nb) * sizeof(T) + 512));
  jb.Ac = reinterpret_cast<T*>(base);
  jb.Bc = reinterpret_cast<T*>(base + ((na * sizeof(T) + 255) & ~size_t(255)));
  AMX_SET();
  drive<T>(jb);
  AMX_CLR();
}

template <class T>
struct Half {
  Job<T> jb;
  mt_options o;
};
template <class T>
void run_half(void* p) {
  auto* h = static_cast<Half<T>*>(p);
  if (h->jb.M > 0 && h->jb.N > 0) run_job(h->jb, h->o);
}

template <class T>
void gemm(mt_order order, int M, int N, int K, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C,
          int ldc, const mt_options* opt) {
  const mt_options o = opt ? *opt : mt_options{};
  if (M <= 0 || N <= 0) return;
  if (order == MtColMajor) {
    std::swap(M, N);
    std::swap(A, B);
    std::swap(lda, ldb);
  }
  if (K <= 0 || alpha == T(0)) {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) C[long(i) * ldc + j] = beta == T(0) ? T(0) : beta * C[long(i) * ldc + j];
    return;
  }
  if (beta != T(0) && beta != T(1))
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) C[long(i) * ldc + j] *= beta;
  Job<T> jb{M, N, K, alpha, A, lda, B, ldb, beta != T(0), C, ldc, {}, 0, 0, 0, 0, 0, nullptr, nullptr};
  constexpr int MR = Ak<T>::MR;
  if (o.threads < 2) return run_job(jb, o);  // M2: the cores of the P cluster share one AMX unit (threads=0 is 1)
  // threads = 3: the calling (P-cluster) thread and an efficiency-cluster thread with eshare percent of the work.
  const bool pe = o.threads == 3;
  const double f0 = pe ? 1.0 - o.eshare / 100.0 : 0.5;
  Half<T> h0{jb, o}, h1{jb, o};
  if (M >= N) {
    const int m0 = std::min(M, round_up(int(M * f0 + 0.5), MR));
    h0.jb.M = m0;
    h1.jb.M = M - m0;
    h1.jb.A = A + long(m0) * lda;
    h1.jb.C = C + long(m0) * ldc;
  } else {
    const int n0 = std::min(N, round_up(int(N * f0 + 0.5), Ak<T>::NR));
    h0.jb.N = n0;
    h1.jb.N = N - n0;
    h1.jb.B = B + n0;
    h1.jb.C = C + n0;
  }
  if (pe) run_pair_e(run_half<T>, &h0, &h1);
  else run_pair(run_half<T>, &h0, &h1);
}

}  // namespace
}  // namespace mt

void mt_sgemm(mt_order order, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
              float beta, float* C, int ldc, const mt_options* opt) {
  mt::gemm<float>(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, opt);
}
void mt_dgemm(mt_order order, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb,
              double beta, double* C, int ldc, const mt_options* opt) {
  mt::gemm<double>(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, opt);
}
