// Six-level blocked SME GEMM (paper Fig. 5): row-major core, column-major solved as C^T = B^T A^T.
#include "internal.h"
#include <arm_sme.h>
#include <alloca.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>

#define MT_S __arm_streaming
#define MT_ZA __arm_inout("za")
#define MT_INL __attribute__((always_inline)) inline
#define MT_NOINL __attribute__((noinline))

namespace mt {
namespace {

template <int... I, class F>
MT_INL void sf_impl(std::integer_sequence<int, I...>, F&& f) MT_S MT_ZA {
  (f(std::integral_constant<int, I>{}), ...);
}
template <int N, class F>
MT_INL void static_for(F&& f) MT_S MT_ZA {
  sf_impl(std::make_integer_sequence<int, N>{}, f);
}

template <class T> struct Tr;
template <> struct Tr<float> {
  static constexpr int VL = 16, NT = 4;
  using V = svfloat32_t;
  using V4 = svfloat32x4_t;
  static MT_INL svbool_t pt() MT_S { return svptrue_b32(); }
  static MT_INL svbool_t pw(int64_t i, int64_t n) MT_S { return svwhilelt_b32(i, n); }
  static MT_INL svcount_t ct() MT_S { return svptrue_c32(); }
  static MT_INL svcount_t cw(int64_t i, int64_t n) MT_S { return svwhilelt_c32(i, n, 4); }
  template <int t> static MT_INL void mopa(svbool_t p, V a, V b) MT_S MT_ZA { svmopa_za32_f32_m(t, p, p, a, b); }
  template <int t> static MT_INL V rdh(uint32_t s) MT_S MT_ZA { return svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), t, s); }
  template <int t> static MT_INL void wrh(uint32_t s, V v) MT_S MT_ZA { svwrite_hor_za32_f32_m(t, s, svptrue_b32(), v); }
  template <int t> static MT_INL V4 rdv4(uint32_t s) MT_S MT_ZA { return svread_ver_za32_f32_vg4(t, s); }
  template <int t> static MT_INL void wrh4(uint32_t s, V4 v) MT_S MT_ZA { svwrite_hor_za32_f32_vg4(t, s, v); }
  template <int t> static MT_INL void ldh(uint32_t s, svbool_t p, const float* q) MT_S MT_ZA { svld1_hor_za32(t, s, p, q); }
  template <int t> static MT_INL void sth(uint32_t s, svbool_t p, float* q) MT_S MT_ZA { svst1_hor_za32(t, s, p, q); }
  static MT_INL V mul(V v, float a) MT_S { return svmul_n_f32_x(svptrue_b32(), v, a); }
};
template <> struct Tr<double> {
  static constexpr int VL = 8, NT = 8;
  using V = svfloat64_t;
  using V4 = svfloat64x4_t;
  static MT_INL svbool_t pt() MT_S { return svptrue_b64(); }
  static MT_INL svbool_t pw(int64_t i, int64_t n) MT_S { return svwhilelt_b64(i, n); }
  static MT_INL svcount_t ct() MT_S { return svptrue_c64(); }
  static MT_INL svcount_t cw(int64_t i, int64_t n) MT_S { return svwhilelt_c64(i, n, 4); }
  template <int t> static MT_INL void mopa(svbool_t p, V a, V b) MT_S MT_ZA { svmopa_za64_f64_m(t, p, p, a, b); }
  template <int t> static MT_INL V rdh(uint32_t s) MT_S MT_ZA { return svread_hor_za64_f64_m(svundef_f64(), svptrue_b64(), t, s); }
  template <int t> static MT_INL void wrh(uint32_t s, V v) MT_S MT_ZA { svwrite_hor_za64_f64_m(t, s, svptrue_b64(), v); }
  template <int t> static MT_INL V4 rdv4(uint32_t s) MT_S MT_ZA { return svread_ver_za64_f64_vg4(t, s); }
  template <int t> static MT_INL void wrh4(uint32_t s, V4 v) MT_S MT_ZA { svwrite_hor_za64_f64_vg4(t, s, v); }
  template <int t> static MT_INL void ldh(uint32_t s, svbool_t p, const double* q) MT_S MT_ZA { svld1_hor_za64(t, s, p, q); }
  template <int t> static MT_INL void sth(uint32_t s, svbool_t p, double* q) MT_S MT_ZA { svst1_hor_za64(t, s, p, q); }
  static MT_INL V mul(V v, double a) MT_S { return svmul_n_f64_x(svptrue_b64(), v, a); }
};

// Four consecutive vectors: one ld1/st1 x4 instruction, or four single-vector ones (x4 ablation).
template <class T, bool X4>
MT_INL typename Tr<T>::V4 load4(const T* p) MT_S {
  using R = Tr<T>;
  constexpr int VL = R::VL;
  if constexpr (X4) return svld1_x4(R::ct(), p);
  else return svcreate4(svld1(R::pt(), p), svld1(R::pt(), p + VL), svld1(R::pt(), p + 2 * VL), svld1(R::pt(), p + 3 * VL));
}
template <class T, bool X4>
MT_INL typename Tr<T>::V4 load4n(const T* p, int64_t n) MT_S {
  using R = Tr<T>;
  constexpr int VL = R::VL;
  if constexpr (X4) return svld1_x4(R::cw(0, n), p);
  else
    return svcreate4(svld1(R::pw(0, n), p), svld1(R::pw(VL, n), p + VL), svld1(R::pw(2 * VL, n), p + 2 * VL),
                     svld1(R::pw(3 * VL, n), p + 3 * VL));
}
template <class T, bool X4>
MT_INL void store4(T* p, typename Tr<T>::V4 v) MT_S {
  using R = Tr<T>;
  constexpr int VL = R::VL;
  if constexpr (X4) svst1(R::ct(), p, v);
  else {
    svst1(R::pt(), p, svget4(v, 0)); svst1(R::pt(), p + VL, svget4(v, 1));
    svst1(R::pt(), p + 2 * VL, svget4(v, 2)); svst1(R::pt(), p + 3 * VL, svget4(v, 3));
  }
}
template <class T, bool X4>
MT_INL void store4n(T* p, typename Tr<T>::V4 v, int64_t n) MT_S {
  using R = Tr<T>;
  constexpr int VL = R::VL;
  if constexpr (X4) svst1(R::cw(0, n), p, v);
  else {
    svst1(R::pw(0, n), p, svget4(v, 0)); svst1(R::pw(VL, n), p + VL, svget4(v, 1));
    svst1(R::pw(2 * VL, n), p + 2 * VL, svget4(v, 2)); svst1(R::pw(3 * VL, n), p + 3 * VL, svget4(v, 3));
  }
}

inline int64_t clip(int64_t n, int64_t hi) { return n < 0 ? 0 : (n > hi ? hi : n); }

enum CMode { kZero = 0, kLoad = 1, kScale = 2 };

// C tile block <-> ZA. Rows r < mrows, columns c < ncols; tile (p,q) holds rows p*VL.., columns q*VL..
template <class T, int TM, int TN, bool X4, bool CD>
MT_INL void c_load(T* C, long ldc, int mrows, int ncols, int cmode, T beta) MT_S MT_ZA {
  using R = Tr<T>;
  constexpr int VL = R::VL;
  svzero_za();
  if (cmode == kZero) return;
  static_for<TM>([&](auto P) MT_S MT_ZA {
    for (int s = 0; s < VL && P * VL + s < mrows; ++s) {
      T* row = C + long(P * VL + s) * ldc;
      if constexpr (TN % 4 == 0 && !CD) {
        static_for<TN / 4>([&](auto G) MT_S MT_ZA {
          auto v = load4n<T, X4>(row + G * 4 * VL, clip(ncols - G * 4 * VL, 4 * VL));
          static_for<4>([&](auto Q) MT_S MT_ZA {
            auto x = svget4(v, Q);
            if (cmode == kScale) x = R::mul(x, beta);
            R::template wrh<P * TN + G * 4 + Q>(s, x);
          });
        });
      } else {
        static_for<TN>([&](auto Q) MT_S MT_ZA {
          const svbool_t pg = R::pw(Q * VL, ncols);
          if (CD && cmode == kLoad) {
            R::template ldh<P * TN + Q>(s, pg, row + Q * VL);
          } else {
            auto x = svld1(pg, row + Q * VL);
            if (cmode == kScale) x = R::mul(x, beta);
            R::template wrh<P * TN + Q>(s, x);
          }
        });
      }
    }
  });
}

template <class T, int TM, int TN, bool X4, bool CD>
MT_INL void c_store(T* C, long ldc, int mrows, int ncols) MT_S MT_ZA {
  using R = Tr<T>;
  constexpr int VL = R::VL;
  static_for<TM>([&](auto P) MT_S MT_ZA {
    for (int s = 0; s < VL && P * VL + s < mrows; ++s) {
      T* row = C + long(P * VL + s) * ldc;
      if constexpr (TN % 4 == 0 && !CD) {
        static_for<TN / 4>([&](auto G) MT_S MT_ZA {
          constexpr int t = P * TN + G * 4;
          auto v = svcreate4(R::template rdh<t>(s), R::template rdh<t + 1>(s), R::template rdh<t + 2>(s),
                             R::template rdh<t + 3>(s));
          if (ncols >= (G + 1) * 4 * VL) store4<T, X4>(row + G * 4 * VL, v);
          else store4n<T, X4>(row + G * 4 * VL, v, clip(ncols - G * 4 * VL, 4 * VL));
        });
      } else {
        static_for<TN>([&](auto Q) MT_S MT_ZA {
          const svbool_t pg = R::pw(Q * VL, ncols);
          if constexpr (CD) R::template sth<P * TN + Q>(s, pg, row + Q * VL);
          else svst1(pg, row + Q * VL, R::template rdh<P * TN + Q>(s));
        });
      }
    }
  });
}

template <int I, class V4>
MT_INL auto pick(V4 x0, V4 x1) MT_S {
  if constexpr (I < 4) return svget4(x0, I);
  else return svget4(x1, I - 4);
}

// A row of up to 8 vectors of B (TN >= 4): from the packed panel, or from the source and written to the panel.
template <class T, int TN, bool X4, bool ONLINE>
MT_INL void b_row(const T* src, T* dst, int ncols, typename Tr<T>::V4& b0, typename Tr<T>::V4& b1) MT_S {
  constexpr int VL = Tr<T>::VL;
  if constexpr (ONLINE) {
    b0 = ncols >= 4 * VL ? load4<T, X4>(src) : load4n<T, X4>(src, clip(ncols, 4 * VL));
    store4<T, X4>(dst, b0);
    if constexpr (TN == 8) {
      b1 = ncols >= 8 * VL ? load4<T, X4>(src + 4 * VL) : load4n<T, X4>(src + 4 * VL, clip(ncols - 4 * VL, 4 * VL));
      store4<T, X4>(dst + 4 * VL, b1);
    } else {
      b1 = b0;
    }
  } else {
    b0 = load4<T, X4>(dst);
    if constexpr (TN == 8) b1 = load4<T, X4>(dst + 4 * VL);
    else b1 = b0;
  }
}

// Micro-kernel (paper Alg. 1): TM x TN tiles, Ar = TM packed VL-row panels (stride astr), Br = kb x (TN*VL) panel.
// ONLINE: B comes from the source matrix and is written to Br as a side effect (first-round online packing).
template <class T, int TM, int TN, bool X4, bool ONLINE, bool CD>
MT_NOINL void kernel(int kb, const T* Ar, long astr, T* Br, const T* Bs, long ldb, int ncols, T* C, long ldc,
                   int mrows, int cmode, T beta) MT_S MT_ZA {
  using R = Tr<T>;
  using V = typename R::V;
  using V4 = typename R::V4;
  constexpr int VL = R::VL, NR = TN * VL;
  static_assert(TM * TN <= R::NT, "too many tiles");
  static_assert(TN == 1 || TN == 2 || TN == 4 || TN == 8, "unsupported TN");
  c_load<T, TM, TN, X4, CD>(C, ldc, mrows, ncols, cmode, beta);
  const svbool_t pt = R::pt();
  int k = 0;
  if constexpr (TN >= 4) {
    static_assert(TM <= 2, "wide kernels use at most two A panels");
    for (; k + 4 <= kb; k += 4) {
      const V4 a0 = load4<T, X4>(Ar + long(k) * VL);
      const V4 a1 = TM > 1 ? load4<T, X4>(Ar + astr + long(k) * VL) : a0;
      static_for<4>([&](auto U) MT_S MT_ZA {
        V4 b0, b1;
        b_row<T, TN, X4, ONLINE>(Bs + long(k + U) * ldb, Br + long(k + U) * NR, ncols, b0, b1);
        static_for<TM>([&](auto P) MT_S MT_ZA {
          static_for<TN>([&](auto Q) MT_S MT_ZA {
            R::template mopa<P * TN + Q>(pt, svget4(P == 0 ? a0 : a1, U), pick<Q>(b0, b1));
          });
        });
      });
    }
    for (; k < kb; ++k) {
      V4 b0, b1;
      b_row<T, TN, X4, ONLINE>(Bs + long(k) * ldb, Br + long(k) * NR, ncols, b0, b1);
      static_for<TM>([&](auto P) MT_S MT_ZA {
        const V a = svld1(pt, Ar + P * astr + long(k) * VL);
        static_for<TN>([&](auto Q) MT_S MT_ZA { R::template mopa<P * TN + Q>(pt, a, pick<Q>(b0, b1)); });
      });
    }
  } else {
    for (; k + 4 <= kb; k += 4) {
      V4 bb0, bb1;
      T* dst = Br + long(k) * NR;
      if constexpr (ONLINE) {
        const svbool_t p0 = R::pw(0, ncols), p1 = R::pw(VL, ncols);
        const T* s = Bs + long(k) * ldb;
        if constexpr (TN == 1) {
          bb0 = svcreate4(svld1(p0, s), svld1(p0, s + ldb), svld1(p0, s + 2 * ldb), svld1(p0, s + 3 * ldb));
          bb1 = bb0;
        } else {
          bb0 = svcreate4(svld1(p0, s), svld1(p1, s + VL), svld1(p0, s + ldb), svld1(p1, s + ldb + VL));
          bb1 = svcreate4(svld1(p0, s + 2 * ldb), svld1(p1, s + 2 * ldb + VL), svld1(p0, s + 3 * ldb), svld1(p1, s + 3 * ldb + VL));
        }
        store4<T, X4>(dst, bb0);
        if constexpr (TN == 2) store4<T, X4>(dst + 4 * VL, bb1);
      } else {
        bb0 = load4<T, X4>(dst);
        bb1 = TN == 2 ? load4<T, X4>(dst + 4 * VL) : bb0;
      }
      static_for<TM>([&](auto P) MT_S MT_ZA {
        const V4 a = load4<T, X4>(Ar + P * astr + long(k) * VL);
        static_for<4>([&](auto U) MT_S MT_ZA {
          static_for<TN>([&](auto Q) MT_S MT_ZA {
            R::template mopa<P * TN + Q>(pt, svget4(a, U), pick<U * TN + Q>(bb0, bb1));
          });
        });
      });
    }
    for (; k < kb; ++k) {
      T* dst = Br + long(k) * NR;
      V b0, b1;
      if constexpr (ONLINE) {
        const T* s = Bs + long(k) * ldb;
        b0 = svld1(R::pw(0, ncols), s);
        svst1(pt, dst, b0);
        if constexpr (TN == 2) {
          b1 = svld1(R::pw(VL, ncols), s + VL);
          svst1(pt, dst + VL, b1);
        } else {
          b1 = b0;
        }
      } else {
        b0 = svld1(pt, dst);
        b1 = TN == 2 ? svld1(pt, dst + VL) : b0;
      }
      static_for<TM>([&](auto P) MT_S MT_ZA {
        const V a = svld1(pt, Ar + P * astr + long(k) * VL);
        R::template mopa<P * TN>(pt, a, b0);
        if constexpr (TN == 2) R::template mopa<P * TN + 1>(pt, a, b1);
      });
    }
  }
  c_store<T, TM, TN, X4, CD>(C, ldc, mrows, ncols);
}

// Four rows x 64 columns into horizontal slices r..r+3 of every tile: strided-register x4 loads put the four
// rows of one tile into z(4t)..z(4t+3), so one MOVA vg4 per tile suffices (inline asm to pin the registers).
template <class T>
MT_INL void transpose_in4(const T* src, long ld, uint32_t r) MT_S MT_ZA {
  const T* p1 = src + ld;
  const T* p2 = src + 2 * ld;
  const T* p3 = src + 3 * ld;
  if constexpr (sizeof(T) == 4) {
    asm volatile(
        "ptrue pn8.s\n"
        "ld1w {z16.s, z20.s, z24.s, z28.s}, pn8/z, [%[a0]]\n"
        "ld1w {z17.s, z21.s, z25.s, z29.s}, pn8/z, [%[a1]]\n"
        "ld1w {z18.s, z22.s, z26.s, z30.s}, pn8/z, [%[a2]]\n"
        "ld1w {z19.s, z23.s, z27.s, z31.s}, pn8/z, [%[a3]]\n"
        "mova za0h.s[%w[r], 0:3], {z16.s - z19.s}\n"
        "mova za1h.s[%w[r], 0:3], {z20.s - z23.s}\n"
        "mova za2h.s[%w[r], 0:3], {z24.s - z27.s}\n"
        "mova za3h.s[%w[r], 0:3], {z28.s - z31.s}\n"
        :
        : [a0] "r"(src), [a1] "r"(p1), [a2] "r"(p2), [a3] "r"(p3), [r] "Ucj"(r)
        : "p8", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
          "z30", "z31", "memory");
  } else {
    asm volatile(
        "ptrue pn8.d\n"
        "ld1d {z16.d, z20.d, z24.d, z28.d}, pn8/z, [%[a0]]\n"
        "ld1d {z17.d, z21.d, z25.d, z29.d}, pn8/z, [%[a1]]\n"
        "ld1d {z18.d, z22.d, z26.d, z30.d}, pn8/z, [%[a2]]\n"
        "ld1d {z19.d, z23.d, z27.d, z31.d}, pn8/z, [%[a3]]\n"
        "mova za0h.d[%w[r], 0:3], {z16.d - z19.d}\n"
        "mova za1h.d[%w[r], 0:3], {z20.d - z23.d}\n"
        "mova za2h.d[%w[r], 0:3], {z24.d - z27.d}\n"
        "mova za3h.d[%w[r], 0:3], {z28.d - z31.d}\n"
        "ld1d {z16.d, z20.d, z24.d, z28.d}, pn8/z, [%[a0], #4, mul vl]\n"
        "ld1d {z17.d, z21.d, z25.d, z29.d}, pn8/z, [%[a1], #4, mul vl]\n"
        "ld1d {z18.d, z22.d, z26.d, z30.d}, pn8/z, [%[a2], #4, mul vl]\n"
        "ld1d {z19.d, z23.d, z27.d, z31.d}, pn8/z, [%[a3], #4, mul vl]\n"
        "mova za4h.d[%w[r], 0:3], {z16.d - z19.d}\n"
        "mova za5h.d[%w[r], 0:3], {z20.d - z23.d}\n"
        "mova za6h.d[%w[r], 0:3], {z24.d - z27.d}\n"
        "mova za7h.d[%w[r], 0:3], {z28.d - z31.d}\n"
        :
        : [a0] "r"(src), [a1] "r"(p1), [a2] "r"(p2), [a3] "r"(p3), [r] "Ucj"(r)
        : "p8", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
          "z30", "z31", "memory");
  }
}

// On-the-fly transposition of A (paper Fig. 6): VL rows x 64 columns go in through horizontal slices of all
// tiles and leave through vertical slices, giving panels Ac[k*VL + r] of VL rows each.
template <class T, bool X4, bool SCALE, bool PK4>
MT_NOINL void pack_a(int mb, int kb, const T* A, long lda, T alpha, T* Ac) MT_S MT_ZA {
  using R = Tr<T>;
  constexpr int VL = R::VL, NT = R::NT, CH = NT * VL;
  for (int p0 = 0; p0 < mb; p0 += VL) {
    const int rows = std::min(VL, mb - p0);
    T* dst = Ac + long(p0) * kb;
    for (int k0 = 0; k0 < kb; k0 += CH) {
      const int kn = std::min(CH, kb - k0);
      if (rows < VL) svzero_za();
      int r = 0;
      if (PK4 && !SCALE && kn >= CH)
        for (; r + 4 <= rows; r += 4) transpose_in4<T>(A + long(p0 + r) * lda + k0, lda, r);
      for (; r < rows; ++r) {
        const T* src = A + long(p0 + r) * lda + k0;
        static_for<NT / 4>([&](auto G) MT_S MT_ZA {
          auto v = kn >= CH ? load4<T, X4>(src + G * 4 * VL) : load4n<T, X4>(src + G * 4 * VL, clip(kn - G * 4 * VL, 4 * VL));
          static_for<4>([&](auto Q) MT_S MT_ZA {
            auto x = svget4(v, Q);
            if constexpr (SCALE) x = R::mul(x, alpha);
            R::template wrh<G * 4 + Q>(r, x);
          });
        });
      }
      static_for<NT>([&](auto Tt) MT_S MT_ZA {
        const int kt = kn - Tt * VL;
        for (int c = 0; c < VL && c < kt; c += 4) {
          auto v = R::template rdv4<Tt>(c);
          T* d = dst + long(k0 + Tt * VL + c) * VL;
          if (kt - c >= 4) store4<T, X4>(d, v);
          else store4n<T, X4>(d, v, int64_t(kt - c) * VL);
        }
      });
    }
  }
}

// Separate B packing pass (used only when online packing is switched off).
template <class T, int TN, bool X4>
MT_INL void pack_b(int kb, int ncols, const T* Bs, long ldb, T* Bd) MT_S {
  using R = Tr<T>;
  constexpr int VL = R::VL, NR = TN * VL;
  for (int k = 0; k < kb; ++k) {
    const T* s = Bs + long(k) * ldb;
    T* d = Bd + long(k) * NR;
    if constexpr (TN >= 4) {
      for (int g = 0; g < TN / 4; ++g)
        store4<T, X4>(d + g * 4 * VL, ncols >= (g + 1) * 4 * VL ? load4<T, X4>(s + g * 4 * VL)
                                                               : load4n<T, X4>(s + g * 4 * VL, clip(ncols - g * 4 * VL, 4 * VL)));
    } else {
      for (int q = 0; q < TN; ++q) svst1(R::pt(), d + q * VL, svld1(R::pw(q * VL, ncols), s + q * VL));
    }
  }
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
  int prof, pack4;
  T* Ac;
  T* Bc;
};

// Dispatch a runtime tile count 1..MAX to a template argument.
template <int MAX, class F>
MT_INL void with_tm(int tm, F&& f) MT_S MT_ZA {
  static_for<MAX>([&](auto I) MT_S MT_ZA {
    if (tm == I + 1) f(std::integral_constant<int, I + 1>{});
  });
}

template <class T, int TMM, int TNM, bool X4, bool ON, bool CD>
__arm_locally_streaming __arm_new("za") void drive(const Job<T>& jb) {
  using R = Tr<T>;
  constexpr int VL = R::VL, NT = R::NT, MR = TMM * VL, NR = TNM * VL;
  const int M = jb.M, N = jb.N, K = jb.K;
  const int mc = jb.blk.mc, nc = jb.blk.nc, kc = jb.blk.kc;
  for (int i = 0; i < M; i += mc) {
    const int mb = std::min(mc, M - i);
    for (int k = 0; k < K; k += kc) {
      const int kb = std::min(kc, K - k);
      const T* Ap = jb.A + long(i) * jb.lda + k;
      if (jb.prof & 1) {
      } else if (jb.alpha != T(1)) {
        pack_a<T, X4, true, false>(mb, kb, Ap, jb.lda, jb.alpha, jb.Ac);
      } else if (jb.pack4 && X4) {
        pack_a<T, X4, false, true>(mb, kb, Ap, jb.lda, jb.alpha, jb.Ac);
      } else {
        pack_a<T, X4, false, false>(mb, kb, Ap, jb.lda, jb.alpha, jb.Ac);
      }
      const int cmode = k > 0 ? kLoad : (jb.beta == T(0) ? kZero : (jb.beta == T(1) ? kLoad : kScale));
      for (int j = 0; j < N; j += nc) {
        const int nb = std::min(nc, N - j), nmain = nb / NR * NR;
        const T* Bs = jb.B + long(k) * jb.ldb + j;
        T* Cb = jb.C + long(i) * jb.ldc + j;
        if (!ON) {
          for (int jj = 0; jj < nmain; jj += NR) pack_b<T, TNM, X4>(kb, NR, Bs + jj, jb.ldb, jb.Bc + long(jj) * kb);
          for (int jj = nmain; jj < nb; jj += VL) pack_b<T, 1, X4>(kb, std::min(VL, nb - jj), Bs + jj, jb.ldb, jb.Bc + long(jj) * kb);
        }
        if (jb.prof & 2) continue;
        for (int ii = 0; ii < mb; ii += MR) {
          const int rows = std::min(MR, mb - ii);
          with_tm<TMM>((rows + VL - 1) / VL, [&](auto TM) MT_S MT_ZA {
            for (int jj = 0; jj < nmain; jj += NR) {
              T* Cp = Cb + long(ii) * jb.ldc + jj;
              T* Br = jb.Bc + long(jj) * kb;
              if (ON && ii == 0)
                kernel<T, TM, TNM, X4, true, CD>(kb, jb.Ac + long(ii) * kb, long(VL) * kb, Br, Bs + jj, jb.ldb, NR, Cp, jb.ldc, rows, cmode, jb.beta);
              else
                kernel<T, TM, TNM, X4, false, CD>(kb, jb.Ac + long(ii) * kb, long(VL) * kb, Br, nullptr, 0, NR, Cp, jb.ldc, rows, cmode, jb.beta);
            }
          });
        }
        if (nmain < nb) {
          // Edge micro-kernel (paper: 64x16 for fp32): all tiles stacked along M, one VL-wide column panel.
          for (int ii = 0; ii < mb; ii += NT * VL) {
            const int rows = std::min(NT * VL, mb - ii);
            with_tm<NT>((rows + VL - 1) / VL, [&](auto TM) MT_S MT_ZA {
              for (int jj = nmain; jj < nb; jj += VL) {
                const int cols = std::min(VL, nb - jj);
                T* Cp = Cb + long(ii) * jb.ldc + jj;
                T* Br = jb.Bc + long(jj) * kb;
                if (ON && ii == 0)
                  kernel<T, TM, 1, X4, true, CD>(kb, jb.Ac + long(ii) * kb, long(VL) * kb, Br, Bs + jj, jb.ldb, cols, Cp, jb.ldc, rows, cmode, jb.beta);
                else
                  kernel<T, TM, 1, X4, false, CD>(kb, jb.Ac + long(ii) * kb, long(VL) * kb, Br, nullptr, 0, cols, Cp, jb.ldc, rows, cmode, jb.beta);
              }
            });
          }
        }
      }
    }
  }
}

template <class T>
using DriveFn = void (*)(const Job<T>&);

template <class T, int TMM, int TNM>
DriveFn<T> pick3(bool x4, bool on, bool cd) {
  if (x4) {
    if (on) return cd ? drive<T, TMM, TNM, true, true, true> : drive<T, TMM, TNM, true, true, false>;
    return cd ? drive<T, TMM, TNM, true, false, true> : drive<T, TMM, TNM, true, false, false>;
  }
  if (on) return cd ? drive<T, TMM, TNM, false, true, true> : drive<T, TMM, TNM, false, true, false>;
  return cd ? drive<T, TMM, TNM, false, false, true> : drive<T, TMM, TNM, false, false, false>;
}

template <class T>
DriveFn<T> pick(const mt_options& o) {
  constexpr int NT = Tr<T>::NT;
  if (o.shape == 1) return pick3<T, 2, NT / 2>(o.x4, o.online, o.cdirect);
  return pick3<T, 1, NT>(o.x4, o.online, o.cdirect);
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
  constexpr int VL = Tr<T>::VL, NT = Tr<T>::NT;
  const int mr = o.shape == 1 ? 2 * VL : VL, nr = o.shape == 1 ? NT / 2 * VL : NT * VL;
  mt_blocking b;
  if (o.mc && o.nc && o.kc) b = {o.mc, o.nc, o.kc};
  else if (o.model) b = mt_model_blocking(jb.M, jb.N, jb.K, sizeof(T), mr, nr);
  else b = {256, 1024, 256};
  b.mc = round_up(std::max(b.mc, mr), mr);
  b.nc = round_up(std::max(b.nc, nr), nr);
  b.kc = std::max(b.kc, 1);
  jb.blk = b;
  jb.prof = o.prof;
  jb.pack4 = o.pack4;
  const size_t na = size_t(round_up(b.mc, 4 * VL)) * b.kc, nb = size_t(b.nc) * b.kc;
  const size_t bytes = (na + nb) * sizeof(T) + 256;
  char* base;
  if (o.heap) base = static_cast<char*>(tl_buf.get(bytes));
  else base = static_cast<char*>(alloca(bytes + 128));
  base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(base) + 127) & ~uintptr_t(127));
  jb.Ac = reinterpret_cast<T*>(base);
  jb.Bc = reinterpret_cast<T*>(base + ((na * sizeof(T) + 127) & ~size_t(127)));
  pick<T>(o)(jb);
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
  Job<T> jb{M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, {}, 0, 0, nullptr, nullptr};
  if (o.threads < 2) return run_job(jb, o);
  constexpr int VL = Tr<T>::VL;
  Half<T> h0{jb, o}, h1{jb, o};
  h0.o.heap = h1.o.heap = 1;
  if (M >= N) {
    const int m0 = std::min(M, round_up((M + 1) / 2, 4 * VL));
    h0.jb.M = m0;
    h1.jb.M = M - m0;
    h1.jb.A = A + long(m0) * lda;
    h1.jb.C = C + long(m0) * ldc;
  } else {
    const int n0 = std::min(N, round_up((N + 1) / 2, 4 * VL));
    h0.jb.N = n0;
    h1.jb.N = N - n0;
    h1.jb.B = B + n0;
    h1.jb.C = C + n0;
  }
  run_pair(run_half<T>, &h0, &h1);
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
