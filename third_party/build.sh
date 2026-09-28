#!/bin/bash
# Fetches LIBXSMM, KleidiAI, OpenBLAS (also patched with openblas-sme2.patch), Eigen master and the Eigen SME branch (header-only) at pinned commits into third_party/src and builds them into third_party/install.
set -euo pipefail
cd "$(dirname "$0")"
LIBXSMM_URL=https://github.com/libxsmm/libxsmm.git
LIBXSMM_REV=55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1
KLEIDIAI_URL=https://github.com/ARM-software/kleidiai.git
KLEIDIAI_REV=64270e8f8926aa47f764ffbfe6f2785e00e93c2a
EIGEN_URL=https://gitlab.com/libeigen/eigen.git
EIGEN_REV=ec8593a7dbbf45d370b8e4feda5de106706b01bc
# Eigen with the SME backend work of merge requests !3164 and follow-ups (branch sme-phase4 on the fork).
EIGEN_BRANCH_URL=https://gitlab.com/tesch1/eigen.git
EIGEN_BRANCH_REV=35683b6d7d8f1173a6064847947fa942c9a7a1ea
OPENBLAS_URL=https://github.com/OpenMathLib/OpenBLAS.git
OPENBLAS_REV=63d7f22e42577e413c2775d84daaa1da24c8cc46
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
fetch openblas "$OPENBLAS_URL" "$OPENBLAS_REV"
fetch eigen "$EIGEN_URL" "$EIGEN_REV"
fetch eigen-branch "$EIGEN_BRANCH_URL" "$EIGEN_BRANCH_REV"

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

# TARGET=VORTEXM4 selects the SME kernels; autodetection does not know every M4 variant (hw.cpufamily 399882554).
OB_OPTS="CC=clang TARGET=VORTEXM4 NOFORTRAN=1 NO_LAPACK=1 NO_SHARED=1 USE_THREAD=1 USE_OPENMP=0 NUM_THREADS=12"
if [ ! -f install/openblas/lib/libopenblas.a ]; then
  { make -C src/openblas -j"$JOBS" $OB_OPTS libs && make -C src/openblas $OB_OPTS PREFIX="$PWD/install/openblas" install; } \
    > install/openblas.log 2>&1 || { tail -30 install/openblas.log; exit 1; }
fi
echo "openblas built"

# The same OpenBLAS with the SME2 port of this design (openblas-sme2.patch), in a second worktree.
if [ ! -f install/openblas-sme2/lib/libopenblas.a ]; then
  [ -d src/openblas-sme2 ] || git -C src/openblas worktree add -q --detach ../openblas-sme2 "$OPENBLAS_REV"
  git -C src/openblas-sme2 -c advice.detachedHead=false checkout -q "$OPENBLAS_REV"
  git -C src/openblas-sme2 apply --check ../../openblas-sme2.patch 2>/dev/null && git -C src/openblas-sme2 apply ../../openblas-sme2.patch
  { make -C src/openblas-sme2 -j"$JOBS" $OB_OPTS libs && make -C src/openblas-sme2 $OB_OPTS PREFIX="$PWD/install/openblas-sme2" install; } \
    > install/openblas-sme2.log 2>&1 || { tail -30 install/openblas-sme2.log; exit 1; }
fi
echo "openblas-sme2 built"
