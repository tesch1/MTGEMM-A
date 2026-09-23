// Shape sets shared by the benchmark binaries.
#pragma once
#include <cstdio>
#include <string>
#include <vector>

struct Shape { int id, m, n, k; };
static const int W[24][3] = {{64, 2112, 7168},   {64, 24576, 1536},  {64, 32768, 512},   {64, 7168, 16384},
                             {64, 4096, 7168},   {64, 7168, 2048},   {128, 2112, 7168},  {128, 24576, 1536},
                             {128, 32768, 512},  {128, 7168, 16384}, {128, 4096, 7168},  {128, 7168, 2048},
                             {4096, 2112, 7168}, {4096, 24576, 1536}, {4096, 32768, 512}, {4096, 7168, 16384},
                             {4096, 4096, 7168}, {4096, 7168, 2048}, {4096, 256, 4096},  {11008, 256, 4096},
                             {4096, 256, 11008}, {5120, 256, 5120},  {13824, 256, 5120}, {5120, 256, 13824}};

// squares | paper | all (squares + paper) | irr | small (squares 4-384) | thin (M or N from 1 to 64, rest 4096)
// | gridK (M, N = 4..4096 in powers of two, depth K) | MxNxK
inline std::vector<Shape> make_shapes(const std::string& set, int id_lo = 1, int id_hi = 1000) {
  std::vector<Shape> v;
  if (set == "squares" || set == "all")
    for (int s : {512, 1000, 1024, 2048, 3000, 4096}) v.push_back({0, s, s, s});
  if (set == "paper" || set == "all")
    for (int i = 0; i < 24; ++i)
      if (i + 1 >= id_lo && i + 1 <= id_hi) v.push_back({i + 1, W[i][0], W[i][1], W[i][2]});
  if (set == "irr")
    for (int s = 80; s <= 200; s += 30) v.push_back({0, s, s, 25600});
  if (set == "small")
    for (int s : {4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384}) v.push_back({0, s, s, s});
  if (set == "thin") {
    for (int t : {1, 4, 8, 16, 32, 64}) v.push_back({0, t, 4096, 4096});
    for (int t : {1, 4, 8, 16, 32, 64}) v.push_back({0, 4096, t, 4096});
    for (int t : {8, 16, 32}) v.push_back({0, t, t, 4096});
  }
  int kg;
  if (std::sscanf(set.c_str(), "grid%d", &kg) == 1)  // M, N over powers of two 4-4096 at depth kg
    for (int m = 4; m <= 4096; m *= 2)
      for (int n = 4; n <= 4096; n *= 2) v.push_back({0, m, n, kg});
  int m, n, k;
  if (std::sscanf(set.c_str(), "%dx%dx%d", &m, &n, &k) == 3) v.push_back({0, m, n, k});
  return v;
}
