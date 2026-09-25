// AMX microbenchmarks (M2 target): fma peak by accumulator count and thread placement, X/Y load bandwidth by
// footprint and width, the 32x32 fp32 kernel step (4 quad loads + 16 fma32) by footprint, Z row load/store.
// usage: ubench_amx [fma|threads|load|kern|z|all]
#include "../src/amx.h"
#include <pthread.h>
#include <pthread/qos.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace mt::amx;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Min over 5 trials of >= 30 ms each (after a 100 ms warm-up to raise the clock); returns seconds per call.
static double timeit(const std::function<void()>& f) {
  const double w = now();
  while (now() - w < 0.1) f();
  double best = 1e30;
  for (int t = 0; t < 5; ++t) {
    int reps = 0;
    const double t0 = now();
    double s;
    do { f(); ++reps; s = now() - t0; } while (s < 0.03);
    best = std::min(best, s / reps);
  }
  return best;
}

template <int NA>
__attribute__((noinline)) void fma32_loop(long n) {
  for (long i = 0; i < n; ++i) {
    AMX_FMA32(fma(0, 0, 0));
    if (NA > 1) AMX_FMA32(fma(1, 64, 64));
    if (NA > 2) AMX_FMA32(fma(2, 128, 128));
    if (NA > 3) AMX_FMA32(fma(3, 192, 192));
  }
}
template <int NA>
__attribute__((noinline)) void fma64_loop(long n) {
  for (long i = 0; i < n; ++i) {
    AMX_FMA64(fma(0, 0, 0));
    if (NA > 1) AMX_FMA64(fma(1, 64, 64));
    if (NA > 2) AMX_FMA64(fma(2, 128, 128));
    if (NA > 3) AMX_FMA64(fma(3, 192, 192));
    if (NA > 4) AMX_FMA64(fma(4, 256, 256));
    if (NA > 5) AMX_FMA64(fma(5, 320, 320));
    if (NA > 6) AMX_FMA64(fma(6, 384, 384));
    if (NA > 7) AMX_FMA64(fma(7, 448, 448));
  }
}

static void zero_xy() {
  alignas(256) static float z[128] = {};
  AMX_LDX(xy(z, 0, kQuad)); AMX_LDX(xy(z, 4, kQuad));
  AMX_LDY(xy(z, 0, kQuad)); AMX_LDY(xy(z, 4, kQuad));
}

static void bench_fma() {
  AMX_SET();
  zero_xy();
  const long n = 1 << 16;
  std::printf("# fma peak, one thread: accumulators GFLOPS\n");
  auto r32 = [&](int na, void (*f)(long)) { double s = timeit([&] { f(n); }); std::printf("fma32 %d %7.1f\n", na, n * na * 512.0 / s * 1e-9); };
  r32(1, fma32_loop<1>); r32(2, fma32_loop<2>); r32(3, fma32_loop<3>); r32(4, fma32_loop<4>);
  auto r64 = [&](int na, void (*f)(long)) { double s = timeit([&] { f(n); }); std::printf("fma64 %d %7.1f\n", na, n * na * 128.0 / s * 1e-9); };
  r64(1, fma64_loop<1>); r64(2, fma64_loop<2>); r64(4, fma64_loop<4>); r64(6, fma64_loop<6>); r64(8, fma64_loop<8>);
  AMX_CLR();
}

// T threads each run fma32_loop<4> (or fma64_loop<8>) for a fixed wall time; per-thread QoS picks P or E cores.
struct ThrArg { qos_class_t qos; bool f64; std::atomic<int>* go; std::atomic<int>* stop; long iters; };
static void* thr_body(void* p) {
  auto* a = static_cast<ThrArg*>(p);
  pthread_set_qos_class_self_np(a->qos, 0);
  AMX_SET();
  zero_xy();
  while (!a->go->load()) {}
  long it = 0;
  while (!a->stop->load(std::memory_order_relaxed)) {
    if (a->f64) fma64_loop<8>(4096); else fma32_loop<4>(4096);
    it += 4096;
  }
  a->iters = it;
  AMX_CLR();
  return nullptr;
}
static void run_threads(int np, int ne, bool f64) {
  std::atomic<int> go{0}, stop{0};
  std::vector<ThrArg> args(np + ne);
  std::vector<pthread_t> th(np + ne);
  for (int i = 0; i < np + ne; ++i) {
    args[i] = {i < np ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND, f64, &go, &stop, 0};
    pthread_create(&th[i], nullptr, thr_body, &args[i]);
  }
  const double t0 = now();
  go = 1;
  while (now() - t0 < 0.6) {}
  stop = 1;
  const double s = now() - t0;
  double gp = 0, ge = 0;
  for (int i = 0; i < np + ne; ++i) {
    pthread_join(th[i], nullptr);
    const double g = args[i].iters * (f64 ? 8 * 128.0 : 4 * 512.0) / s * 1e-9;
    (i < np ? gp : ge) += g;
  }
  std::printf("%s P=%d E=%d  P-threads %7.1f  E-threads %7.1f  total %7.1f GFLOPS\n", f64 ? "fma64" : "fma32", np, ne, gp, ge, gp + ge);
}
static void bench_threads() {
  std::printf("# fma peak by threads (QoS user-interactive = P, background = E)\n");
  for (bool f64 : {false, true}) {
    for (int np = 1; np <= 4; ++np) run_threads(np, 0, f64);
    for (int ne = 1; ne <= 4; ne *= 2) run_threads(0, ne, f64);
    run_threads(1, 1, f64); run_threads(1, 4, f64); run_threads(2, 4, f64);
  }
}

static void* alloc(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 16384, bytes)) std::abort();
  std::memset(p, 0, bytes);
  return p;
}

// X/Y loads over a footprint: width 1 (64 B), 2 (pair, 128 B), 4 (quad, 256 B).
__attribute__((noinline)) void load_loop(const char* p, long bytes, int width) {
  if (width == 4)
    for (long o = 0; o < bytes; o += 512) { AMX_LDX(xy(p + o, 0, kQuad)); AMX_LDY(xy(p + o + 256, 0, kQuad)); }
  else if (width == 2)
    for (long o = 0; o < bytes; o += 256) { AMX_LDX(xy(p + o, 0, kPair)); AMX_LDY(xy(p + o + 128, 0, kPair)); }
  else
    for (long o = 0; o < bytes; o += 128) { AMX_LDX(xy(p + o, 0)); AMX_LDY(xy(p + o + 64, 0)); }
}
static const long kFoot[] = {16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 8 << 20, 12 << 20, 32 << 20, 128 << 20};
static void bench_load() {
  std::printf("# X/Y load bandwidth, one thread: footprint width GB/s\n");
  char* p = static_cast<char*>(alloc(128 << 20));
  AMX_SET();
  for (long f : kFoot)
    for (int w : {1, 2, 4}) {
      const double s = timeit([&] { load_loop(p, f, w); });
      std::printf("load %6ld KB x%d %7.1f\n", f >> 10, w, f / s * 1e-9);
    }
  AMX_CLR();
  std::free(p);
}

// One 32x32 fp32 kernel step per 512 bytes: X <- 4 k-rows of 32 B values, Y <- 4 k-rows of 32 A values, 16 fma32.
template <bool LOADS>
__attribute__((noinline)) void kern_loop(const char* a, const char* b, long bytes) {
  for (long o = 0; o < bytes; o += 512) {
    if (LOADS) {
      AMX_LDX(xy(b + o, 0, kQuad)); AMX_LDX(xy(b + o + 256, 4, kQuad));
      AMX_LDY(xy(a + o, 0, kQuad)); AMX_LDY(xy(a + o + 256, 4, kQuad));
    }
#define MT_K(k) \
    AMX_FMA32(fma(0, 128 * k, 128 * k)); AMX_FMA32(fma(1, 128 * k + 64, 128 * k)); \
    AMX_FMA32(fma(2, 128 * k, 128 * k + 64)); AMX_FMA32(fma(3, 128 * k + 64, 128 * k + 64));
    MT_K(0) MT_K(1) MT_K(2) MT_K(3)
#undef MT_K
  }
}
static void bench_kern() {
  std::printf("# 32x32 fp32 kernel step (4 quad loads + 16 fma32), A and B streams of equal size: footprint GFLOPS\n");
  char* a = static_cast<char*>(alloc(64 << 20));
  char* b = static_cast<char*>(alloc(64 << 20));
  AMX_SET();
  const long n = 1 << 20;
  const double s0 = timeit([&] { kern_loop<false>(a, b, n); });
  std::printf("kern no-loads %7.1f\n", n / 512 * 16 * 512.0 / s0 * 1e-9);
  for (long f : kFoot) {
    if (f > (64 << 20)) break;
    const double s = timeit([&] { kern_loop<true>(a, b, f); });
    std::printf("kern %6ld KB x2 %7.1f\n", f >> 10, f / 512 * 16 * 512.0 / s * 1e-9);
  }
  AMX_CLR();
  std::free(a);
  std::free(b);
}

// C tile I/O: 64 Z rows <-> a 32x32 fp32 block of a row-major C (ld = 4096 floats) as 32 pair loads / stores.
static void bench_z() {
  std::printf("# Z rows: 32 ldz/stz pairs per 32x32 fp32 tile, C tile resident in L1: ns per tile\n");
  const long ld = 4096;
  float* c = static_cast<float*>(alloc(ld * 32 * sizeof(float)));
  AMX_SET();
  const int n = 1000;
  auto st = [&] { for (int r = 0; r < n; ++r) for (int m = 0; m < 16; ++m) { AMX_STZ(zr(c + m * ld, 4 * m, true)); AMX_STZ(zr(c + (16 + m) * ld, 4 * m + 2, true)); } };
  auto ldz = [&] { for (int r = 0; r < n; ++r) for (int m = 0; m < 16; ++m) { AMX_LDZ(zr(c + m * ld, 4 * m, true)); AMX_LDZ(zr(c + (16 + m) * ld, 4 * m + 2, true)); } };
  auto st1 = [&] { for (int r = 0; r < n; ++r) for (int m = 0; m < 16; ++m) {
    AMX_STZ(zr(c + m * ld, 4 * m)); AMX_STZ(zr(c + m * ld + 16, 4 * m + 1));
    AMX_STZ(zr(c + (16 + m) * ld, 4 * m + 2)); AMX_STZ(zr(c + (16 + m) * ld + 16, 4 * m + 3)); } };
  std::printf("stz pair %6.1f\n", timeit(st) / n * 1e9);
  std::printf("ldz pair %6.1f\n", timeit(ldz) / n * 1e9);
  std::printf("stz single %6.1f\n", timeit(st1) / n * 1e9);
  AMX_CLR();
  std::free(c);
}

// Do Z stores / loads of tiles 2-3 overlap with fma32 into tiles 0-1? 16 fma32 + S Z ops per iteration.
template <int S, bool LD>
__attribute__((noinline)) void overlap_loop(float* c, long n) {
  for (long i = 0; i < n; ++i) {
    for (int u = 0; u < 8; ++u) { AMX_FMA32(fma(0, 64 * u, 0)); AMX_FMA32(fma(1, 64 * u, 64)); }
    for (int s = 0; s < S; ++s) {
      if (LD) AMX_LDZ(zr(c + ((i * S + s) & 16383) * 16, 4 * (s & 15) + 2 + (s >> 4 & 1)));
      else AMX_STZ(zr(c + ((i * S + s) & 16383) * 16, 4 * (s & 15) + 2 + (s >> 4 & 1)));
    }
  }
}
template <int S, bool LD>
__attribute__((noinline)) void zonly_loop(float* c, long n) {
  for (long i = 0; i < n; ++i)
    for (int s = 0; s < S; ++s) {
      if (LD) AMX_LDZ(zr(c + ((i * S + s) & 16383) * 16, 4 * (s & 15) + 2 + (s >> 4 & 1)));
      else AMX_STZ(zr(c + ((i * S + s) & 16383) * 16, 4 * (s & 15) + 2 + (s >> 4 & 1)));
    }
}
static void bench_overlap() {
  std::printf("# overlap: ns per iteration of 16 fma32 (tiles 0-1) and S Z-row ops (tiles 2-3)\n");
  static float* c = static_cast<float*>(alloc(16384 * 64));
  AMX_SET();
  zero_xy();
  const long n = 4096;
  const double f = timeit([&] { overlap_loop<0, false>(c, n); }) / n * 1e9;
  std::printf("fma only %6.2f\n", f);
  auto row = [&](int s, void (*both)(float*, long), void (*zo)(float*, long), const char* w) {
    const double b = timeit([&] { both(c, n); }) / n * 1e9, z = timeit([&] { zo(c, n); }) / n * 1e9;
    std::printf("%s S=%2d  z only %6.2f  both %6.2f  (sum %6.2f)\n", w, s, z, b, f + z);
  };
  row(4, overlap_loop<4, false>, zonly_loop<4, false>, "stz");
  row(8, overlap_loop<8, false>, zonly_loop<8, false>, "stz");
  row(16, overlap_loop<16, false>, zonly_loop<16, false>, "stz");
  row(4, overlap_loop<4, true>, zonly_loop<4, true>, "ldz");
  row(8, overlap_loop<8, true>, zonly_loop<8, true>, "ldz");
  row(16, overlap_loop<16, true>, zonly_loop<16, true>, "ldz");
  AMX_CLR();
}

int main(int argc, char** argv) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  const std::string w = argc > 1 ? argv[1] : "all";
  if (w == "fma" || w == "all") bench_fma();
  if (w == "threads" || w == "all") bench_threads();
  if (w == "load" || w == "all") bench_load();
  if (w == "kern" || w == "all") bench_kern();
  if (w == "z" || w == "all") bench_z();
  if (w == "overlap" || w == "all") bench_overlap();
  return 0;
}
