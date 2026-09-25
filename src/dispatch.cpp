// mt_sgemm / mt_dgemm: SME when the CPU has it, else AMX on known M2-or-later families, else a portable loop.
#include "internal.h"
#include <sys/sysctl.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace mt {
namespace {

int64_t sysctl_value(const char* name) {
  int64_t v = 0;  // the sysctls read here are 4 bytes; the rest stays zero
  size_t n = sizeof v;
  return sysctlbyname(name, &v, &n, nullptr, 0) == 0 ? v : 0;
}

bool cpu_has_sme() { return sysctl_value("hw.optional.arm.FEAT_SME") == 1; }

// macOS traps user reads of MIDR_EL1 and the ID registers, so AMX is known by hw.cpufamily (mach/machine.h).
bool cpu_has_amx_m2() {
#ifdef MT_AMX_EMULATE
  return true;
#else
  static const uint32_t kFamilies[] = {
      0xda33d83d,  // BLIZZARD_AVALANCHE: M2, A15
      0xfa33415e,  // IBIZA: M3
      0x5f4dea93,  // LOBOS: M3 Pro
      0x72015832,  // PALMA: M3 Max
      0x6f5129ac,  // DONAN: M4
      0x17d5b93a,  // BRAVA: M4 Pro, M4 Max
  };
  const uint32_t f = uint32_t(sysctl_value("hw.cpufamily"));
  for (uint32_t x : kFamilies)
    if (f == x) return true;
  return false;
#endif
}

bool available(mt_backend b) {
  [[maybe_unused]] static const bool sme = cpu_has_sme(), amx = cpu_has_amx_m2();
  switch (b) {
#ifdef MT_WITH_SME
    case MtSme: return sme;
#endif
#ifdef MT_WITH_AMX
    case MtAmx: return amx;
#endif
    case MtReference: return true;
    default: return false;
  }
}

mt_backend from_env() {
  const char* e = std::getenv("MTGEMM_BACKEND");
  if (!e) return MtAuto;
  if (!std::strcmp(e, "sme")) return MtSme;
  if (!std::strcmp(e, "amx")) return MtAmx;
  if (!std::strcmp(e, "ref")) return MtReference;
  return MtAuto;
}

// Portable fallback: C (+)= alpha A B, row-major core (column-major as C^T = B^T A^T), rows of C in the inner loop.
template <class T>
void ref_gemm(mt_order order, int M, int N, int K, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C,
              int ldc) {
  if (M <= 0 || N <= 0) return;
  if (order == MtColMajor) {
    std::swap(M, N);
    std::swap(A, B);
    std::swap(lda, ldb);
  }
  for (int i = 0; i < M; ++i) {
    T* c = C + long(i) * ldc;
    for (int j = 0; j < N; ++j) c[j] = beta == T(0) ? T(0) : beta * c[j];
    if (alpha == T(0)) continue;
    for (int k = 0; k < K; ++k) {
      const T a = alpha * A[long(i) * lda + k];
      const T* b = B + long(k) * ldb;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];
    }
  }
}

}  // namespace
}  // namespace mt

mt_backend mt_select_backend(mt_backend requested) {
  using namespace mt;
  if (requested != MtAuto && available(requested)) return requested;
  static const mt_backend env = from_env();
  if (env != MtAuto && available(env)) return env;
  if (available(MtSme)) return MtSme;
  if (available(MtAmx)) return MtAmx;
  return MtReference;
}

const char* mt_backend_name(mt_backend b) {
  switch (b) {
    case MtSme: return "sme";
    case MtAmx: return "amx";
    case MtReference: return "ref";
    default: return "auto";
  }
}

void mt_sgemm(mt_order order, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
              float beta, float* C, int ldc, const mt_options* opt) {
  switch (mt_select_backend(opt ? mt_backend(opt->backend) : MtAuto)) {
#ifdef MT_WITH_SME
    case MtSme: return mt::sme_sgemm(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, opt);
#endif
#ifdef MT_WITH_AMX
    case MtAmx: return mt::amx_sgemm(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, opt);
#endif
    default: return mt::ref_gemm<float>(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}

void mt_dgemm(mt_order order, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb,
              double beta, double* C, int ldc, const mt_options* opt) {
  switch (mt_select_backend(opt ? mt_backend(opt->backend) : MtAuto)) {
#ifdef MT_WITH_SME
    case MtSme: return mt::sme_dgemm(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, opt);
#endif
#ifdef MT_WITH_AMX
    case MtAmx: return mt::amx_dgemm(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, opt);
#endif
    default: return mt::ref_gemm<double>(order, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}
