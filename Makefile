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

$(BUILD)/bench: bench/bench.cpp $(LIB)
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

test: $(BUILD)/test_gemm
	./$(BUILD)/test_gemm

clean:
	rm -rf $(BUILD)

.PHONY: all test clean gbench
