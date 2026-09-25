// AMX GEMM (Vision Pro M2): 32x32 fp32 / 16x32 fp64 C blocks in Z, row-major core, column-major as C^T = B^T A^T.
#include "internal.h"
#include "amx.h"
#include <arm_neon.h>
#include <sys/sysctl.h>
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

constexpr int kKU = 4;        // packed depth is padded with zeros to a multiple of this
constexpr int kPfRowsB = 32;  // B source rows prefetched ahead when AMX reads B from the source
constexpr int kAmxPfChunks = 4;  // source chunks prefetched ahead of the AMX transposition of A
enum BMode { kBPacked = 0, kBAuto = 1, kBSource = 2 };  // values of mt_options::online in this backend

// B panel stride: 256 bytes of padding keep the panels of a block out of the same L1 sets.
template <class T>
constexpr long bstride(int kp) { return long(kp) * Ak<T>::NR + 256 / long(sizeof(T)); }

// fp32 C row 16p + j is in Z rows 4j + 2p + {0,1}; fp64 C row 8p + j is in Z rows 8j + 4p + {0..3}.
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

// z_io load with C scaled by beta: each 64-byte piece goes to X, then vector-mode fma (skip-Z) writes x * beta to Z.
template <class T>
MT_INL void z_load_scaled(const T* C, long ldc, const T* bvec) {
  constexpr uint64_t kVec = 1ull << 63;
  constexpr int E = 64 / int(sizeof(T));  // elements per Z row
  AMX_LDY(xy(bvec, 7));
  int u = 0;
  auto row = [&](const T* p, int z) __attribute__((always_inline)) {
    AMX_LDX(xy(p, u));
    const uint64_t op = kVec | fma_op(z, 64 * u, 448, true);
    if constexpr (sizeof(T) == 4) AMX_FMA32(op); else AMX_FMA64(op);
    u = (u + 1) & 7;
  };
  constexpr int RP = sizeof(T) == 4 ? 16 : 8, ZS = sizeof(T) == 4 ? 4 : 8, ZH = sizeof(T) == 4 ? 2 : 4;
  for (int p = 0; p < 2; ++p)
    for (int j = 0; j < RP; ++j)
      for (int q = 0; q < ZH; ++q) row(C + long(RP * p + j) * ldc + q * E, ZS * j + ZH * p + q);
}

// Unpacked B panel: its rows are read from the source with single-register loads; rows from kb on read zeros.
template <class T>
struct SrcB {
  const T* src;
  long ldb;
  int kb;
};
alignas(256) const float kZeros[64] = {};

template <class T>
MT_INL void src_rows(const SrcB<T>& s, int r0, int nrows) {
  constexpr int RB = Ak<T>::NR * int(sizeof(T));  // bytes per B row: 128 (fp32) or 256 (fp64)
  for (int u = 0; u < nrows; ++u) {
    const int r = r0 + u;
    const char* p = r < s.kb ? reinterpret_cast<const char*>(s.src + long(r) * s.ldb) : reinterpret_cast<const char*>(kZeros);
    if (r + kPfRowsB < s.kb) pf_l2(s.src + long(r + kPfRowsB) * s.ldb, RB);
    for (int q = 0; q < RB / 64; ++q) AMX_LDX(xy(p + 64 * q, u * RB / 64 + q));
  }
}

// Four depth steps of fp32: 512-byte slabs of A (Y) and B (X), 2 + 2 quad loads (or B from the source), 16 fma32.
template <bool SKIPZ, bool SRC>
MT_INL void step4_f32(const char* a, const char* b, const SrcB<float>* s, int k) {
  if constexpr (SRC) src_rows(*s, k, 4);
  else { AMX_LDX(xy(b, 0, kQuad)); AMX_LDX(xy(b + 256, 4, kQuad)); }
  AMX_LDY(xy(a, 0, kQuad)); AMX_LDY(xy(a + 256, 4, kQuad));
#define MT_K(k, S) \
  AMX_FMA32(fma_op(0, 128 * k, 128 * k, S)); AMX_FMA32(fma_op(1, 128 * k + 64, 128 * k, S)); \
  AMX_FMA32(fma_op(2, 128 * k, 128 * k + 64, S)); AMX_FMA32(fma_op(3, 128 * k + 64, 128 * k + 64, S));
  MT_K(0, SKIPZ) MT_K(1, false) MT_K(2, false) MT_K(3, false)
#undef MT_K
}

// Four depth steps of fp64: A slab 4 x 16 fills Y, B slab 4 x 32 goes through X two steps at a time, 32 fma64.
template <bool SKIPZ, bool SRC>
MT_INL void step4_f64(const char* a, const char* b, const SrcB<double>* s, int k) {
  AMX_LDY(xy(a, 0, kQuad)); AMX_LDY(xy(a + 256, 4, kQuad));
#define MT_K(kx, ky, S) \
  AMX_FMA64(fma_op(0, 256 * kx, 128 * ky, S)); AMX_FMA64(fma_op(1, 256 * kx + 64, 128 * ky, S)); \
  AMX_FMA64(fma_op(2, 256 * kx + 128, 128 * ky, S)); AMX_FMA64(fma_op(3, 256 * kx + 192, 128 * ky, S)); \
  AMX_FMA64(fma_op(4, 256 * kx, 128 * ky + 64, S)); AMX_FMA64(fma_op(5, 256 * kx + 64, 128 * ky + 64, S)); \
  AMX_FMA64(fma_op(6, 256 * kx + 128, 128 * ky + 64, S)); AMX_FMA64(fma_op(7, 256 * kx + 192, 128 * ky + 64, S));
  if constexpr (SRC) src_rows(*s, k, 2);
  else { AMX_LDX(xy(b, 0, kQuad)); AMX_LDX(xy(b + 256, 4, kQuad)); }
  MT_K(0, 0, SKIPZ) MT_K(1, 1, false)
  if constexpr (SRC) src_rows(*s, k + 2, 2);
  else { AMX_LDX(xy(b + 512, 0, kQuad)); AMX_LDX(xy(b + 768, 4, kQuad)); }
  MT_K(0, 2, false) MT_K(1, 3, false)
#undef MT_K
}

// Per-call extras: B read from the source (src), next A panel to prefetch (pfa), 64 bytes of beta to scale C by.
template <class T>
struct Side {
  const SrcB<T>* src;
  const char* pfa;
  const T* bvec;
};

template <class T, bool SRC>
MT_INL void kernel_body(int kp, const T* Ap, const T* Bp, T* C, long ldc, bool load, const Side<T>& sd) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  constexpr long sa = 4 * MR * sizeof(T), sb = 4 * NR * sizeof(T);
  const char* a = reinterpret_cast<const char*>(Ap);
  const char* b = reinterpret_cast<const char*>(Bp);
  auto step = [&](auto skip, int k) __attribute__((always_inline)) {
    if constexpr (sizeof(T) == 4) step4_f32<decltype(skip)::value, SRC>(a, b, sd.src, k);
    else step4_f64<decltype(skip)::value, SRC>(a, b, sd.src, k);
    if (sd.pfa) pf_l2(sd.pfa + long(k) * MR * long(sizeof(T)), int(sa));
  };
  int k = 0;
  if (load) {
    if (sd.bvec) z_load_scaled<T>(C, ldc, sd.bvec); else z_io<T, false>(C, ldc);
  } else {  // the first step overwrites Z (skip-Z), so C is not read
    step(std::true_type{}, 0);
    k = 4; a += sa; b += sb;
  }
  for (; k < kp; k += 4, a += sa, b += sb) step(std::false_type{}, k);
  z_io<T, true>(C, ldc);
}

// MR x NR block of C (+)= A panel (kp x MR) * B panel (kp x NR), kp a multiple of 4.
template <class T>
MT_NOINL void kernel(int kp, const T* Ap, const T* Bp, T* C, long ldc, bool load, const Side<T>& sd) {
  if (sd.src) kernel_body<T, true>(kp, Ap, Bp, C, ldc, load, sd);
  else kernel_body<T, false>(kp, Ap, Bp, C, ldc, load, sd);
}

// Partial C block (rows < MR or cols < NR) through an aligned scratch tile.
template <class T>
MT_NOINL void kernel_edge(int kp, const T* Ap, const T* Bp, T* C, long ldc, int rows, int cols, bool load,
                          const Side<T>& sd) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  alignas(128) T t[MR * NR];
  if (load)
    for (int r = 0; r < rows; ++r) std::memcpy(t + r * NR, C + long(r) * ldc, cols * sizeof(T));
  kernel<T>(kp, Ap, Bp, t, NR, load, sd);
  for (int r = 0; r < rows; ++r) std::memcpy(C + long(r) * ldc, t + r * NR, cols * sizeof(T));
}

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

// 32 depth steps of a full A panel through Z: rows in with ldz, each depth step out as a Z column (extrv) to Y.
template <class T>
MT_INL void tr32_amx(const T* A, long lda, T* dst) {
  constexpr int E = 64 / int(sizeof(T)), NT = E == 16 ? 4 : 8, MR = 2 * E;
  constexpr uint64_t kExtrY = (1ull << 63) | (uint64_t(E == 16 ? 8 : 1) << 11) | (1ull << 26) | (1ull << 10);
  for (int m = 0; m < MR; ++m)  // row m, depth group g goes to tile 2g + m / E
    for (int g = 0; g < 32 / E; ++g) AMX_LDZ(zr(A + long(m) * lda + g * E, NT * (m % E) + 2 * g + m / E));
  for (int k = 0; k < 32; ++k) {  // y[j] = z[NT j + tile][k % E] for the two tiles of the group of k
    const int t = 2 * (k / E), c = k % E, y = 128 * (k & 3);
    AMX_EXTRY(kExtrY | uint64_t(c * NT + t) << 20 | uint64_t(y));
    AMX_EXTRY(kExtrY | uint64_t(c * NT + t + 1) << 20 | uint64_t(y + 64));
    AMX_STY(xy(dst + long(k) * MR, 2 * (k & 3), kPair));
  }
}

// A block (mb x kb) to panels Ap[panel][k][r] of MR rows, zero padded; full panels through Z (amx, alpha = 1).
template <class T>
MT_NOINL void pack_a(int mb, int kb, int kp, const T* A, long lda, T alpha, T* Ap, bool pf, bool amx) {
  constexpr int MR = Ak<T>::MR, KB = 128 / int(sizeof(T));
  const bool sc = alpha != T(1);
  auto al = [&] { if constexpr (sizeof(T) == 4) return vdupq_n_f32(alpha); else return vdupq_n_f64(alpha); }();
  for (int p0 = 0; p0 < mb; p0 += MR) {
    const int rows = std::min(MR, mb - p0);
    T* dst = Ap + long(p0) * kp;
    const int r4 = rows / 4 * 4, k4 = kb / 4 * 4;
    int kz = 0;
    if (amx && rows == MR && !sc)
      for (; kz + 32 <= kb; kz += 32) {
        if (pf) {  // AMX waits in order on each miss: prefetch whole chunks ahead, into the next panel too
          int kf = kz + 32 * kAmxPfChunks, pf0 = p0;
          if (kf + 32 > kb) { kf -= kb / 32 * 32; pf0 += MR; }
          if (pf0 + MR <= mb && kf >= 0 && kf + 32 <= kb)
            for (int r = 0; r < MR; ++r) pf_l2(A + long(pf0 + r) * lda + kf, 32 * int(sizeof(T)));
        }
        tr32_amx<T>(A + long(p0) * lda + kz, lda, dst + long(kz) * MR);
      }
    for (int k0 = kz; k0 < k4; k0 += KB) {  // one 128-byte line of each source row per chunk
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

// B columns [0, nb) of kb rows to panels of NR columns, in source row order so that each row streams contiguously.
template <class T>
MT_NOINL void pack_b_rows(int kb, int kp, int nb, const T* B, long ldb, T* Bc, bool pf) {
  constexpr int NR = Ak<T>::NR, nf = NR * int(sizeof(T)) / 4;
  const long bs = bstride<T>(kp);
  const int nfull = nb / NR * NR;
  for (int k = 0; k < kb; ++k) {
    const T* s = B + long(k) * ldb;
    if (pf && k + 8 < kb) pf_l2(s + 8 * ldb, nb * int(sizeof(T)));
    for (int jj = 0; jj < nfull; jj += NR) {
      const float* sf = reinterpret_cast<const float*>(s + jj);
      float* d = reinterpret_cast<float*>(Bc + long(jj / NR) * bs + long(k) * NR);
      for (int c = 0; c < nf; c += 16) vst1q_f32_x4(d + c, vld1q_f32_x4(sf + c));
    }
    if (nfull < nb) {
      T* d = Bc + long(nfull / NR) * bs + long(k) * NR;
      std::memcpy(d, s + nfull, (nb - nfull) * sizeof(T));
      std::memset(d + (nb - nfull), 0, (NR - (nb - nfull)) * sizeof(T));
    }
  }
  if (kp > kb)
    for (int jj = 0; jj < nb; jj += NR) std::memset(Bc + long(jj / NR) * bs + long(kb) * NR, 0, size_t(kp - kb) * NR * sizeof(T));
}

// Blocking: depth of about 4 KB per row, A block up to 8 MB, B block the rest of 6 MB but at least 8 panels.
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
  T beta;
  T* C;
  long ldc;
  mt_blocking blk;
  int prof, pf, bmode, pack4;
  T* Ac;
  T* Bc;
};

template <class T>
void drive(const Job<T>& jb) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  const int M = jb.M, N = jb.N, K = jb.K;
  const int mc = jb.blk.mc, nc = jb.blk.nc, kc = jb.blk.kc;
  const bool scale = jb.beta != T(0) && jb.beta != T(1);
  alignas(64) T bvec[64 / sizeof(T)];
  std::fill(bvec, bvec + 64 / sizeof(T), jb.beta);
  for (int i = 0; i < M; i += mc) {
    const int mb = std::min(mc, M - i);
    for (int k = 0; k < K; k += kc) {
      const int kb = std::min(kc, K - k), kp = round_up(kb, kKU);
      const long bs = bstride<T>(kp);
      if (!(jb.prof & 1)) pack_a<T>(mb, kb, kp, jb.A + long(i) * jb.lda + k, jb.lda, jb.alpha, jb.Ac, jb.pf & 1, jb.pack4);
      const bool load = k > 0 || jb.beta != T(0);
      for (int j = 0; j < N; j += nc) {
        const int nb = std::min(nc, N - j);
        const T* Bs = jb.B + long(k) * jb.ldb + j;
        const bool src = jb.bmode == kBSource && !(jb.prof & 4);
        if (jb.prof & 4) {
        } else if (!src) {
          pack_b_rows<T>(kb, kp, nb, Bs, jb.ldb, jb.Bc, jb.pf & 2);
        } else if (nb % NR) {  // the partial panel is packed even when B is read from the source
          pack_b_rows<T>(kb, kp, nb % NR, Bs + nb / NR * NR, jb.ldb, jb.Bc + long(nb / NR) * bs, jb.pf & 2);
        }
        if (jb.prof & 2) continue;
        T* Cb = jb.C + long(i) * jb.ldc + j;
        for (int ii = 0; ii < mb; ii += MR) {
          const int rows = std::min(MR, mb - ii);
          const T* Ap = jb.Ac + long(ii) * kp;
          for (int jj = 0; jj < nb; jj += NR) {
            const int cols = std::min(NR, nb - jj);
            T* Cp = Cb + long(ii) * jb.ldc + jj;
            SrcB<T> sb{Bs + jj, jb.ldb, kb};
            Side<T> sd{src && cols == NR ? &sb : nullptr, nullptr, scale && k == 0 ? bvec : nullptr};
            if (jj == 0 && (jb.pf & 1)) {  // next A panel of this block (the first again for the next j)
              const T* An = ii + MR < mb ? Ap + long(MR) * kp : (j + nc < N ? jb.Ac : nullptr);
              sd.pfa = reinterpret_cast<const char*>(An);
            }
            if (load && (jb.pf & 4)) {  // next C tile towards L2: the tile start otherwise waits for ldz
              const T* Cn = jj + NR < nb ? Cp + NR : Cb + long(ii + MR) * jb.ldc;
              if (jj + NR < nb || ii + MR < mb)
                for (int r = 0; r < MR; ++r) pf_l2(Cn + long(r) * jb.ldc, NR * int(sizeof(T)));
            }
            const T* Bp = jb.Bc + long(jj / NR) * bs;
            if (rows == MR && cols == NR) kernel<T>(kp, Ap, Bp, Cp, jb.ldc, load, sd);
            else kernel_edge<T>(kp, Ap, Bp, Cp, jb.ldc, rows, cols, load, sd);
          }
        }
      }
    }
  }
}

thread_local Buf tl_buf;

template <class T>
void run_job(Job<T>& jb, const mt_options& o) {
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  jb.prof = o.prof;
  jb.pf = o.prefetch;
  jb.pack4 = o.pack4;
  // Auto: B is read from the source when all of it fits in 2 MB (fp64 always, fp32 only for M <= 128).
  const bool small_b = long(jb.K) * jb.N * long(sizeof(T)) <= (2L << 20) && (sizeof(T) == 8 || jb.M <= 128);
  jb.bmode = o.online == kBAuto ? (small_b ? kBSource : kBPacked) : o.online == kBSource ? kBSource : kBPacked;
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

// AMX units: one per performance cluster = P cores / P cores per L2 (1 if the sysctls are missing).
int amx_units() {
  int p = 0, per = 0;
  size_t n = sizeof p, m = sizeof per;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &p, &n, nullptr, 0) || sysctlbyname("hw.perflevel0.cpusperl2", &per, &m, nullptr, 0) || per <= 0)
    return 1;
  return std::max(1, p / per);
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
  Job<T> jb{M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, {}, 0, 0, 0, 0, nullptr, nullptr};
  constexpr int MR = Ak<T>::MR, NR = Ak<T>::NR;
  // threads = 0: one thread per AMX unit (one per P cluster: M2 1, M4 Pro 2) from 2^22 multiply-adds.
  static const int units = amx_units();
  const bool two = o.threads >= 2 || (o.threads == 0 && units >= 2 && double(M) * N * K >= double(1 << 22) &&
                                      std::max(M, N) >= 4 * MR);
  if (!two) return run_job(jb, o);
  Half<T> h0{jb, o}, h1{jb, o};
  if (M >= N) {
    const int m0 = std::min(M, round_up((M + 1) / 2, MR));
    h0.jb.M = m0;
    h1.jb.M = M - m0;
    h1.jb.A = A + long(m0) * lda;
    h1.jb.C = C + long(m0) * ldc;
  } else {
    const int n0 = std::min(N, round_up((N + 1) / 2, NR));
    h0.jb.N = n0;
    h1.jb.N = N - n0;
    h1.jb.B = B + n0;
    h1.jb.C = C + n0;
  }
  run_pair(run_half<T>, &h0, &h1);
}

}  // namespace
}  // namespace mt

void mt::amx_sgemm(mt_order order, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
              float beta, float* C, int ldc, const mt_options* opt) {
  mt::gemm<float>(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, opt);
}
void mt::amx_dgemm(mt_order order, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb,
              double beta, double* C, int ldc, const mt_options* opt) {
  mt::gemm<double>(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, opt);
}
