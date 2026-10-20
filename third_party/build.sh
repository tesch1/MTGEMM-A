#!/bin/bash
# Fetches LIBXSMM, KleidiAI, OpenBLAS develop and Eigen master (header-only) at pinned commits into third_party/src and builds them into third_party/install.
set -euo pipefail
cd "$(dirname "$0")"
LIBXSMM_URL=https://github.com/libxsmm/libxsmm.git
LIBXSMM_REV=55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1
KLEIDIAI_URL=https://github.com/ARM-software/kleidiai.git
KLEIDIAI_REV=64270e8f8926aa47f764ffbfe6f2785e00e93c2a
EIGEN_URL=https://gitlab.com/libeigen/eigen.git
# Eigen master with the SME backend (merge request !3164 and later SME work).
EIGEN_REV=3c3bf4532bf9e059e3999827b00b7452c2bcde7d
# OpenBLAS develop with the SME2 kernels after this design (pull request #6074, merged 2026-10-16).
OPENBLAS_URL=${OPENBLAS_URL:-https://github.com/OpenMathLib/OpenBLAS.git}
OPENBLAS_REV=5137bb62b4ca24fa88873809b845ffc18e601256
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
mkdir -p src install

fetch() {  # name url rev
  if [ ! -d "src/$1/.git" ]; then git clone -q "$2" "src/$1"; fi
  git -C "src/$1" remote set-url origin "$2"
  git -C "src/$1" fetch -q --depth 1 origin "$3" 2>/dev/null || git -C "src/$1" fetch -q origin
  git -C "src/$1" -c advice.detachedHead=false checkout -q "$3"
  echo "$1 at $(git -C "src/$1" rev-parse --short HEAD)"
}

fetch libxsmm "$LIBXSMM_URL" "$LIBXSMM_REV"
fetch kleidiai "$KLEIDIAI_URL" "$KLEIDIAI_REV"
fetch openblas "$OPENBLAS_URL" "$OPENBLAS_REV"
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

# TARGET=VORTEXM4 selects the SME kernels; autodetection does not know every M4 variant (hw.cpufamily 399882554).
# Rebuilt when the pin moves.
OB_OPTS="CC=clang TARGET=VORTEXM4 NOFORTRAN=1 NO_LAPACK=1 NO_SHARED=1 USE_THREAD=1 USE_OPENMP=0 NUM_THREADS=12"
if [ "$(cat install/openblas/REV 2>/dev/null)" != "$OPENBLAS_REV" ]; then
  rm -rf install/openblas
  { make -C src/openblas clean && make -C src/openblas -j"$JOBS" $OB_OPTS libs &&
    make -C src/openblas $OB_OPTS PREFIX="$PWD/install/openblas" install; } \
    > install/openblas.log 2>&1 || { tail -30 install/openblas.log; exit 1; }
  echo "$OPENBLAS_REV" > install/openblas/REV
fi
echo "openblas built"
