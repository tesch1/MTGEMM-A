// SME microbenchmarks on M4 Pro: FMOPA peak by tile count, SME load bandwidth (x1/x4) by footprint,
// streaming reads of a large matrix in the A-packing pattern, with and without core prefetch.
#include <arm_sme.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Min over 5 trials of >= 20 ms each; returns seconds per call.
static double timeit(const std::function<void()>& f) {
  f();
  double best = 1e30;
  for (int t = 0; t < 5; ++t) {
    int reps = 0;
    const double t0 = now();
    double s;
    do { f(); ++reps; s = now() - t0; } while (s < 0.02);
    best = std::min(best, s / reps);
  }
  return best;
}

template <int NT>
__arm_locally_streaming __arm_new("za") void mopa_loop(long n) {
  svbool_t p = svptrue_b32();
  svfloat32_t a = svdup_f32(1.0f), b = svdup_f32(0.5f);
  for (long i = 0; i < n; ++i) {
    svmopa_za32_f32_m(0, p, p, a, b);
    if (NT > 1) svmopa_za32_f32_m(1, p, p, a, b);
    if (NT > 2) svmopa_za32_f32_m(2, p, p, a, b);
    if (NT > 3) svmopa_za32_f32_m(3, p, p, a, b);
  }
}
__arm_locally_streaming __arm_new("za") void mopa64_loop(long n) {
  svbool_t p = svptrue_b64();
  svfloat64_t a = svdup_f64(1.0), b = svdup_f64(0.5);
  for (long i = 0; i < n; ++i) {
    svmopa_za64_f64_m(0, p, p, a, b); svmopa_za64_f64_m(1, p, p, a, b);
    svmopa_za64_f64_m(2, p, p, a, b); svmopa_za64_f64_m(3, p, p, a, b);
    svmopa_za64_f64_m(4, p, p, a, b); svmopa_za64_f64_m(5, p, p, a, b);
    svmopa_za64_f64_m(6, p, p, a, b); svmopa_za64_f64_m(7, p, p, a, b);
  }
}

// Sequential read of `bytes`, consumed by one FMOPA per 256 B so that loads cannot be dropped.
template <bool X4>
__arm_locally_streaming __arm_new("za") void read_seq(const float* p, long bytes) {
  svbool_t pt = svptrue_b32();
  svcount_t pn = svptrue_c32();
  const long n = bytes / 4;
  for (long i = 0; i < n; i += 64) {
    svfloat32x4_t v;
    if (X4) v = svld1_x4(pn, p + i);
    else v = svcreate4(svld1(pt, p + i), svld1(pt, p + i + 16), svld1(pt, p + i + 32), svld1(pt, p + i + 48));
    svmopa_za32_f32_m(0, pt, pt, svget4(v, 0), svget4(v, 1));
    svmopa_za32_f32_m(1, pt, pt, svget4(v, 2), svget4(v, 3));
  }
}

// The A-packing access pattern: 16 rows (stride ld floats) x 64 columns per block, blocks along the row.
template <int PF>
__arm_locally_streaming __arm_new("za") void read_rows(const float* p, long rows, long cols, long ld) {
  svbool_t pt = svptrue_b32();
  svcount_t pn = svptrue_c32();
  for (long r0 = 0; r0 < rows; r0 += 16)
    for (long c = 0; c < cols; c += 64)
      for (int r = 0; r < 16; ++r) {
        const float* q = p + (r0 + r) * ld + c;
        if (PF > 0) { __builtin_prefetch(q + PF, 0, 2); __builtin_prefetch(q + PF + 32, 0, 2); }
        svfloat32x4_t v = svld1_x4(pn, q);
        svmopa_za32_f32_m(0, pt, pt, svget4(v, 0), svget4(v, 1));
        svmopa_za32_f32_m(1, pt, pt, svget4(v, 2), svget4(v, 3));
      }
}

// Same pattern but the core (non-streaming) touches the next block first: a helper-free prefetch variant.
static void core_read_rows(const float* p, long rows, long cols, long ld, float* sink) {
  float s = 0;
  for (long r0 = 0; r0 < rows; r0 += 16)
    for (long c = 0; c < cols; c += 64)
      for (int r = 0; r < 16; ++r) {
        const float* q = p + (r0 + r) * ld + c;
        s += q[0] + q[32];
      }
  *sink = s;
}

static void* body(void*) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  {
    const long n = 1 << 22;
    for (int nt : {1, 2, 3, 4}) {
      double t = timeit([&] {
        if (nt == 1) mopa_loop<1>(n); else if (nt == 2) mopa_loop<2>(n); else if (nt == 3) mopa_loop<3>(n); else mopa_loop<4>(n);
      });
      std::printf("fmopa fp32 %d tiles: %.0f GFLOPS\n", nt, 512.0 * nt * n / t / 1e9);
    }
    double t = timeit([&] { mopa64_loop(n / 2); });
    std::printf("fmopa fp64 8 tiles: %.0f GFLOPS\n", 128.0 * 8 * (n / 2) / t / 1e9);
  }
  const long big = 256L << 20;
  float* buf;
  if (posix_memalign(reinterpret_cast<void**>(&buf), 16384, big)) return nullptr;
  for (long i = 0; i < big / 4; ++i) buf[i] = float(i & 1023);
  for (long kb : {64L, 256L, 1024L, 4096L, 8192L, 12288L, 16384L, 32768L, 262144L}) {
    const long bytes = kb << 10;
    const double t4 = timeit([&] { read_seq<true>(buf, bytes); });
    const double t1 = timeit([&] { read_seq<false>(buf, bytes); });
    std::printf("seq read %7ld KB: x4 %5.0f GB/s  x1 %5.0f GB/s\n", kb, bytes / t4 / 1e9, bytes / t1 / 1e9);
  }
  for (long ld : {7168L, 7168L + 16, 4096L, 2048L}) {
    const long rows = 2048, cols = std::min(ld, 7168L);
    const double bytes = double(rows) * cols * 4;
    float sink;
    const double t0 = timeit([&] { read_rows<0>(buf, rows, cols, ld); });
    const double t1 = timeit([&] { read_rows<256>(buf, rows, cols, ld); });
    const double t2 = timeit([&] { read_rows<1024>(buf, rows, cols, ld); });
    const double tc = timeit([&] { core_read_rows(buf, rows, cols, ld, &sink); });
    std::printf("16-row blocks, ld=%5ld (%4.0f MB): SME %4.0f GB/s, +prfm 1KB %4.0f, +prfm 4KB %4.0f, core %4.0f GB/s\n", ld,
                bytes / 1e6, bytes / t0 / 1e9, bytes / t1 / 1e9, bytes / t2 / 1e9, bytes / tc / 1e9);
  }
  std::free(buf);
  return nullptr;
}

int main() {
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_set_qos_class_np(&at, QOS_CLASS_USER_INTERACTIVE, 0);
  pthread_t th;
  pthread_create(&th, &at, body, nullptr);
  pthread_join(th, nullptr);
}
