#!/bin/bash
# Fetches LIBXSMM, KleidiAI and Eigen (header-only) at pinned commits into third_party/src and builds them into third_party/install.
set -euo pipefail
cd "$(dirname "$0")"
LIBXSMM_URL=https://github.com/libxsmm/libxsmm.git
LIBXSMM_REV=55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1
KLEIDIAI_URL=https://github.com/ARM-software/kleidiai.git
KLEIDIAI_REV=64270e8f8926aa47f764ffbfe6f2785e00e93c2a
EIGEN_URL=https://gitlab.com/libeigen/eigen.git
EIGEN_REV=ec8593a7dbbf45d370b8e4feda5de106706b01bc
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
mkdir -p src install

fetch() {  # name url rev
  if [ ! -d "src/$1/.git" ]; then git clone -q "$2" "src/$1"; fi
  git -C "src/$1" fetch -q --depth 1 origin "$3" 2>/dev/null || git -C "src/$1" fetch -q origin
  git -C "src/$1" -c advice.detachedHead=false checkout -q "$3"
  echo "$1 at $(git -C "src/$1" rev-parse --short HEAD)"
}

fetch libxsmm "$LIBXSMM_URL" "$LIBXSMM_REV"
fetch kleidiai "$KLEIDIAI_URL" "$KLEIDIAI_REV"
fetch eigen "$EIGEN_URL" "$EIGEN_REV"

if [ ! -f install/libxsmm/lib/libxsmm.a ]; then
  make -C src/libxsmm -j"$JOBS" CC=clang CXX=clang++ STATIC=1 BLAS=0 FORTRAN=0 \
    PREFIX="$PWD/install/libxsmm" install-minimal > install/libxsmm.log 2>&1 || { tail -30 install/libxsmm.log; exit 1; }
fi
echo "libxsmm built"

if [ ! -f install/kleidiai/lib/libkleidiai.a ]; then
  cmake -S src/kleidiai -B src/kleidiai/build -DCMAKE_BUILD_TYPE=Release -DKLEIDIAI_BUILD_TESTS=OFF \
    -DKLEIDIAI_BUILD_BENCHMARK=OFF -DCMAKE_INSTALL_PREFIX="$PWD/install/kleidiai" > install/kleidiai.log 2>&1
  cmake --build src/kleidiai/build -j"$JOBS" >> install/kleidiai.log 2>&1 || { tail -30 install/kleidiai.log; exit 1; }
  cmake --install src/kleidiai/build >> install/kleidiai.log 2>&1
fi
echo "kleidiai built"
