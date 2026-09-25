CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -march=armv8.6-a+sme2+sme-f64f64 -Wall -Wextra -Wno-unused-parameter
CPPFLAGS += -Iinclude
BUILD    := build
LIB      := $(BUILD)/libmtgemm.a
OBJS     := $(BUILD)/mtgemm.o $(BUILD)/model.o $(BUILD)/threads.o
ACCEL    := -framework Accelerate

all: $(LIB) $(BUILD)/test_gemm $(BUILD)/bench $(BUILD)/ubench

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp include/mtgemm.h src/internal.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	ar rcs $@ $^

$(BUILD)/test_gemm: tests/test_gemm.cpp $(LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB) -o $@

$(BUILD)/bench: bench/bench.cpp bench/shapes.h $(LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK $< $(LIB) $(ACCEL) -o $@

$(BUILD)/ubench: bench/ubench.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

# Google Benchmark cross-check (fetched into build/, not part of 'all').
GB := $(BUILD)/_deps/benchmark
$(GB)/build/src/libbenchmark.a:
	git clone -q --depth 1 --branch v1.9.1 https://github.com/google/benchmark $(GB)
	cmake -S $(GB) -B $(GB)/build -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=OFF -DBENCHMARK_ENABLE_GTEST_TESTS=OFF >/dev/null
	cmake --build $(GB)/build -j4 >/dev/null

$(BUILD)/gbench: bench/gbench.cpp $(LIB) $(GB)/build/src/libbenchmark.a
	$(CXX) $(CPPFLAGS) -I$(GB)/include $(CXXFLAGS) -DACCELERATE_NEW_LAPACK $< $(LIB) $(GB)/build/src/libbenchmark.a $(ACCEL) -o $@

gbench: $(BUILD)/gbench

# LIBXSMM and KleidiAI comparison (third_party/build.sh fetches and builds them; not part of 'all').
TP := third_party/install
$(TP)/libxsmm/lib/libxsmm.a $(TP)/kleidiai/lib/libkleidiai.a:
	./third_party/build.sh

$(BUILD)/bench_ext: bench/bench_ext.cpp bench/shapes.h $(TP)/libxsmm/lib/libxsmm.a $(TP)/kleidiai/lib/libkleidiai.a | $(BUILD)
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -I$(TP)/libxsmm/include/libxsmm -I$(TP)/kleidiai/include $< \
	  $(TP)/libxsmm/lib/libxsmm.a $(TP)/libxsmm/lib/libxsmmgen.a $(TP)/kleidiai/lib/libkleidiai.a $(ACCEL) -o $@

bench_ext: $(BUILD)/bench_ext

# Eigen master and the Eigen SME branch, one thread and on a thread pool; header-only, fetched by third_party/build.sh.
EIGEN_INC := third_party/src/eigen
EIGEN_BR_INC := third_party/src/eigen-branch
EIGEN_BENCH := bench/bench_eigen.cpp bench/shapes.h
$(EIGEN_INC)/Eigen/Core $(EIGEN_BR_INC)/Eigen/Core:
	./third_party/build.sh

$(BUILD)/bench_eigen: $(EIGEN_BENCH) $(EIGEN_INC)/Eigen/Core | $(BUILD)
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -I$(EIGEN_INC) $< $(ACCEL) -o $@

$(BUILD)/bench_eigen_mt: $(EIGEN_BENCH) $(EIGEN_INC)/Eigen/Core | $(BUILD)
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -DEIGEN_GEMM_THREADPOOL -DBENCH_EIGEN_NAME='"eigen-mt"' -I$(EIGEN_INC) $< $(ACCEL) -o $@

$(BUILD)/bench_eigen_br: $(EIGEN_BENCH) $(EIGEN_BR_INC)/Eigen/Core | $(BUILD)
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -DBENCH_EIGEN_HAS_SME_UNITS -DBENCH_EIGEN_NAME='"eigen-branch"' -I$(EIGEN_BR_INC) $< $(ACCEL) -o $@

$(BUILD)/bench_eigen_br_mt: $(EIGEN_BENCH) $(EIGEN_BR_INC)/Eigen/Core | $(BUILD)
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -DEIGEN_GEMM_THREADPOOL -DBENCH_EIGEN_HAS_SME_UNITS -DBENCH_EIGEN_NAME='"eigen-branch-mt"' -I$(EIGEN_BR_INC) $< $(ACCEL) -o $@

bench_eigen: $(BUILD)/bench_eigen $(BUILD)/bench_eigen_mt $(BUILD)/bench_eigen_br $(BUILD)/bench_eigen_br_mt

test_ext: $(BUILD)/bench_ext
	./$(BUILD)/bench_ext all libxsmm check
	./$(BUILD)/bench_ext irr libxsmm check
	./$(BUILD)/bench_ext all kleidiai check
	./$(BUILD)/bench_ext irr kleidiai check
	./$(BUILD)/bench_ext small libxsmm check
	./$(BUILD)/bench_ext thin libxsmm check
	./$(BUILD)/bench_ext small kleidiai check
	./$(BUILD)/bench_ext thin kleidiai check

test_eigen: bench_eigen
	for b in bench_eigen bench_eigen_mt bench_eigen_br bench_eigen_br_mt; do for o in row col; do for set in all small thin; do \
	  ./$(BUILD)/$$b $$set $$o check || exit 1; done; done; done

test: $(BUILD)/test_gemm
	./$(BUILD)/test_gemm

# AMX backend (Vision Pro M2, also runs on M4); _emu builds run every AMX instruction in corsix's M2 emulator.
CORSIX_REV := 483714bb051da088d08a66724b22dd08a5db3c99
AMXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wno-unused-parameter
AMX_OBJS := $(BUILD)/amx/mtgemm_amx.o $(BUILD)/amx/model.o $(BUILD)/amx/threads.o
AMX_LIB  := $(BUILD)/libmtgemm_amx.a
CORSIX   := $(BUILD)/_deps/amx
EMU_SRC  := ldst extr fma fms genlut mac16 matfp matint vecfp vecint
EMU_OBJS := $(EMU_SRC:%=$(BUILD)/emu/%.o) $(BUILD)/emu/amx_ver.o
EMU_LOBJS := $(BUILD)/emu/mtgemm_amx.o $(BUILD)/emu/model.o $(BUILD)/emu/threads.o

$(BUILD)/amx $(BUILD)/emu:
	mkdir -p $@

$(BUILD)/amx/%.o: src/%.cpp include/mtgemm.h src/internal.h src/amx.h | $(BUILD)/amx
	$(CXX) $(CPPFLAGS) $(AMXFLAGS) -c $< -o $@

$(AMX_LIB): $(AMX_OBJS)
	ar rcs $@ $^

$(BUILD)/test_gemm_amx: tests/test_gemm.cpp $(AMX_LIB)
	$(CXX) $(CPPFLAGS) $(AMXFLAGS) $< $(AMX_LIB) -o $@

$(BUILD)/bench_amx: bench/bench.cpp bench/shapes.h $(AMX_LIB)
	$(CXX) $(CPPFLAGS) $(AMXFLAGS) -DACCELERATE_NEW_LAPACK -DMT_BENCH_AMX $< $(AMX_LIB) $(ACCEL) -o $@

$(BUILD)/ubench_amx: bench/ubench_amx.cpp src/amx.h | $(BUILD)
	$(CXX) $(CPPFLAGS) $(AMXFLAGS) $< -o $@

$(CORSIX)/emulate.h:
	git clone -q https://github.com/corsix/amx $(CORSIX) && git -C $(CORSIX) checkout -q $(CORSIX_REV)

$(BUILD)/emu/%.o: $(CORSIX)/%.c $(CORSIX)/emulate.h | $(BUILD)/emu
	$(CC) -O2 -w -c $< -o $@

$(BUILD)/emu/amx_ver.o: tests/amx_ver.c $(CORSIX)/emulate.h | $(BUILD)/emu
	$(CC) -O2 -I$(CORSIX) -c $< -o $@

$(BUILD)/emu/%.o: src/%.cpp include/mtgemm.h src/internal.h src/amx.h $(CORSIX)/emulate.h | $(BUILD)/emu
	$(CXX) $(CPPFLAGS) -I$(CORSIX) -DMT_AMX_EMULATE $(AMXFLAGS) -c $< -o $@

$(BUILD)/test_gemm_amx_emu: tests/test_gemm.cpp $(EMU_LOBJS) $(EMU_OBJS)
	$(CXX) $(CPPFLAGS) $(AMXFLAGS) $< $(EMU_LOBJS) $(EMU_OBJS) -o $@

amx: $(AMX_LIB) $(BUILD)/test_gemm_amx $(BUILD)/bench_amx $(BUILD)/ubench_amx

test_amx: $(BUILD)/test_gemm_amx $(BUILD)/test_gemm_amx_emu
	./$(BUILD)/test_gemm_amx
	./$(BUILD)/test_gemm_amx_emu 60

clean:
	rm -rf $(BUILD)

.PHONY: all test clean gbench bench_ext test_ext bench_eigen test_eigen amx test_amx
