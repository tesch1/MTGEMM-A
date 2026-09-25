#pragma once
#include "mtgemm.h"
#include <algorithm>
#include <cstdlib>

#if defined(__ARM_FEATURE_SME)
#define MT_STREAMING_COMPAT __arm_streaming_compatible
#else
#define MT_STREAMING_COMPAT
#endif

namespace mt {

constexpr long kL2Bytes = 8L << 20;  // paper Sec. 2: SME bandwidth holds up to an 8 MB working set
constexpr long kPageBytes = 16384;   // macOS arm64 page
constexpr int kTlbEntries = 160;     // assumed L1D TLB entries of a P-core (M1 Firestorm figure)

inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// Block size near `block` (a multiple of step) that splits `total` into equal blocks.
inline int balance(int total, int block, int step) {
  if (block >= total) return round_up(total, step);
  const int nblk = (total + block - 1) / block;
  return std::min(round_up(block, step), round_up((total + nblk - 1) / nblk, step));
}

// Core-side prefetch (prfm pldl2keep) of `bytes` starting at p; static: the attribute differs per backend file.
__attribute__((always_inline)) static inline void pf_l2(const void* p, int bytes) MT_STREAMING_COMPAT {
  for (int l = 0; l < bytes; l += 128) __builtin_prefetch(static_cast<const char*>(p) + l, 0, 2);
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

int model_kc_max(int es, int mr, int nr);
mt_blocking model_blocking(int M, int N, int K, int es, int mr, int nr);

// Runs fn(a1) on the worker thread and fn(a0) on the caller, then waits for both.
void run_pair(void (*fn)(void*), void* a0, void* a1);

// Backends behind mt_sgemm / mt_dgemm (src/dispatch.cpp): same arguments as the public functions.
void sme_sgemm(mt_order, int, int, int, float, const float*, int, const float*, int, float, float*, int, const mt_options*);
void sme_dgemm(mt_order, int, int, int, double, const double*, int, const double*, int, double, double*, int, const mt_options*);
void amx_sgemm(mt_order, int, int, int, float, const float*, int, const float*, int, float, float*, int, const mt_options*);
void amx_dgemm(mt_order, int, int, int, double, const double*, int, const double*, int, double, double*, int, const mt_options*);

}  // namespace mt
