#pragma once
#include "mtgemm.h"

namespace mt {

constexpr long kL2Bytes = 8L << 20;  // paper Sec. 2: SME bandwidth holds up to an 8 MB working set
constexpr long kPageBytes = 16384;   // macOS arm64 page
constexpr int kTlbEntries = 160;     // assumed L1D TLB entries of a P-core (M1 Firestorm figure)

inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

int model_kc_max(int es, int mr, int nr);
mt_blocking model_blocking(int M, int N, int K, int es, int mr, int nr);

// Runs fn(a1) on the worker thread and fn(a0) on the caller, then waits for both.
void run_pair(void (*fn)(void*), void* a0, void* a1);

}  // namespace mt
