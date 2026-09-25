// Analytical blocking model of the paper, eqs. (1)-(3): TLB bound on kc, then max CMR under the L2 bound.
#include "internal.h"
#include <algorithm>
#include <mutex>

namespace mt {

static int tlb_pages(long bytes) { return int((bytes + kPageBytes - 1) / kPageBytes) + 1; }

int model_kc_max(int es, int mr, int nr) {
  int best = 16;
  for (int kc = 16; kc <= 1 << 16; kc += 16) {
    const int ta = tlb_pages(long(mr) * kc * es), tb = tlb_pages(long(nr) * kc * es), tcc = mr;
    if (ta + 2 * tb + tcc < kTlbEntries) best = kc; else break;
  }
  return best;
}

mt_blocking model_blocking(int M, int N, int K, int es, int mr, int nr) {
  const long budget = kL2Bytes / es;
  const int kcap = std::min(model_kc_max(es, mr, nr), round_up(K, 16));
  const int mcap = round_up(M, mr), ncap = round_up(N, nr);
  double best = -1;
  mt_blocking r{mr, nr, 16};
  for (int kc = 16; kc <= kcap; kc += 16)
    for (int mc = mr; mc <= mcap; mc += mr) {
      const long room = budget - long(mc) * kc;
      if (room <= 0) break;
      long nc = room / (2L * kc + 2L * mc);
      nc = std::min<long>(ncap, nc / nr * nr);
      if (nc < nr) break;
      const double cmr = 2.0 * mc * nc * kc / (double(mc) * kc + double(kc) * nc + 2.0 * mc * nc);
      if (cmr > best * (1 + 1e-9)) { best = cmr; r = {mc, int(nc), kc}; }
    }
  r.kc = balance(K, r.kc, 16);
  r.mc = balance(M, r.mc, mr);
  r.nc = balance(N, r.nc, nr);
  return r;
}

}  // namespace mt

mt_blocking mt_model_blocking(int M, int N, int K, int es, int mr, int nr) {
  struct Entry { int M, N, K, es, mr, nr; mt_blocking b; };
  static std::mutex mu;
  static Entry cache[16];
  static int next = 0;
  {
    std::lock_guard<std::mutex> g(mu);
    for (auto& e : cache)
      if (e.M == M && e.N == N && e.K == K && e.es == es && e.mr == mr && e.nr == nr) return e.b;
  }
  const mt_blocking b = mt::model_blocking(M, N, K, es, mr, nr);
  std::lock_guard<std::mutex> g(mu);
  cache[next] = {M, N, K, es, mr, nr, b};
  next = (next + 1) % 16;
  return b;
}
