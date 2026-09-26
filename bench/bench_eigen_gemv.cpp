// Eigen column-major GEMV, y += alpha A x, over a grid of shapes: nanoseconds and GFLOPS per shape, best of trials.
// usage: bench_eigen_gemv <grid|MxN> [f64] [ms=20] [trials=5]
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifndef BENCH_EIGEN_NAME
#define BENCH_EIGEN_NAME "eigen"
#endif

template <class T>
static int run(const std::vector<std::pair<int, int>>& shapes, double ms, int trials) {
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  std::printf("# gemv %s %s\n", BENCH_EIGEN_NAME, sizeof(T) == 8 ? "f64" : "f32");
  int failures = 0;
  for (const auto& s : shapes) {
    const Mat a = Mat::Random(s.first, s.second);
    const Vec x = Vec::Random(s.second);
    Vec y = Vec::Random(s.first);
    const Vec ref = y + T(0.5) * a.lazyProduct(x);
    y.noalias() += T(0.5) * a * x;
    const double err = double((y - ref).cwiseAbs().maxCoeff());
    const bool ok = err <= (sizeof(T) == 8 ? 1e-12 : 1e-4) * s.second;
    failures += !ok;
    double best = 1e30;
    for (int t = 0; t < trials; ++t) {
      int reps = 0;
      double sec = 0;
      const auto t0 = std::chrono::steady_clock::now();
      do {
        y.noalias() += T(0.5) * a * x;
        ++reps;
        sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      } while (sec < ms * 1e-3);
      best = std::min(best, sec / reps);
    }
    std::printf("%6d %6d %10.1f %7.1f %s\n", s.first, s.second, best * 1e9, 2.0 * s.first * s.second / best / 1e9,
                ok ? "" : "FAIL");
    std::fflush(stdout);
  }
  return failures ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_eigen_gemv <grid|MxN> [f64] [ms=20] [trials=5]\n");
    return 2;
  }
  std::vector<std::pair<int, int>> shapes;
  int m, n;
  if (!std::strcmp(argv[1], "grid")) {
    for (int r : {16, 32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192})
      for (int c : {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096})
        if (double(r) * c <= double(1 << 25)) shapes.push_back({r, c});
  } else if (std::sscanf(argv[1], "%dx%d", &m, &n) == 2) {
    shapes.push_back({m, n});
  }
  bool f64 = false;
  double ms = 20;
  int trials = 5;
  for (int i = 2; i < argc; ++i) {
    double x;
    int v;
    if (!std::strcmp(argv[i], "f64")) f64 = true;
    else if (std::sscanf(argv[i], "ms=%lf", &x) == 1) ms = x;
    else if (std::sscanf(argv[i], "trials=%d", &v) == 1) trials = v;
  }
  return f64 ? run<double>(shapes, ms, trials) : run<float>(shapes, ms, trials);
}
