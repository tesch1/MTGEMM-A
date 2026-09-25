// Apple AMX (M1-M4) instruction encodings after corsix/amx; only M2-level features are used.
// MT_AMX_EMULATE routes every instruction to corsix's emulator (per-thread state) for checks against M2 semantics.
#pragma once
#include <cstdint>

#ifdef MT_AMX_EMULATE
struct alignas(128) amx_state { uint8_t x[8][64], y[8][64], z[64][64]; };  // layout of corsix's amx_state
extern "C" {
void emulate_AMX_LDX(amx_state*, uint64_t);
void emulate_AMX_LDY(amx_state*, uint64_t);
void emulate_AMX_STX(amx_state*, uint64_t);
void emulate_AMX_STY(amx_state*, uint64_t);
void emulate_AMX_LDZ(amx_state*, uint64_t);
void emulate_AMX_STZ(amx_state*, uint64_t);
void emulate_AMX_LDZI(amx_state*, uint64_t);
void emulate_AMX_STZI(amx_state*, uint64_t);
void emulate_AMX_EXTRX(amx_state*, uint64_t);
void emulate_AMX_EXTRY(amx_state*, uint64_t);
void emulate_AMX_FMA64(amx_state*, uint64_t);
void emulate_AMX_FMA32(amx_state*, uint64_t);
void emulate_AMX_VECFP(amx_state*, uint64_t);
void emulate_AMX_MATFP(amx_state*, uint64_t);
}
namespace mt::amx { inline thread_local amx_state emu_state; }
#define MT_AMX_OP(name, g) emulate_AMX_##name(&mt::amx::emu_state, uint64_t(g))
#define AMX_SET() ((void)0)
#define AMX_CLR() ((void)0)
#else
#define MT_AMX_NOP_OP_IMM5(op, imm5) \
  __asm volatile("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")
#define MT_AMX_OP_GPR(op, gpr) \
  __asm volatile(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"(uint64_t(gpr)) : "memory")
#define MT_AMX_CODE_LDX 0
#define MT_AMX_CODE_LDY 1
#define MT_AMX_CODE_STX 2
#define MT_AMX_CODE_STY 3
#define MT_AMX_CODE_LDZ 4
#define MT_AMX_CODE_STZ 5
#define MT_AMX_CODE_LDZI 6
#define MT_AMX_CODE_STZI 7
#define MT_AMX_CODE_EXTRX 8
#define MT_AMX_CODE_EXTRY 9
#define MT_AMX_CODE_FMA64 10
#define MT_AMX_CODE_FMA32 12
#define MT_AMX_CODE_VECFP 19
#define MT_AMX_CODE_MATFP 21
#define MT_AMX_OP(name, g) MT_AMX_OP_GPR(MT_AMX_CODE_##name, g)
#define AMX_SET() MT_AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR() MT_AMX_NOP_OP_IMM5(17, 1)
#endif

#define AMX_LDX(g) MT_AMX_OP(LDX, g)
#define AMX_LDY(g) MT_AMX_OP(LDY, g)
#define AMX_STX(g) MT_AMX_OP(STX, g)
#define AMX_STY(g) MT_AMX_OP(STY, g)
#define AMX_LDZ(g) MT_AMX_OP(LDZ, g)
#define AMX_STZ(g) MT_AMX_OP(STZ, g)
#define AMX_LDZI(g) MT_AMX_OP(LDZI, g)
#define AMX_STZI(g) MT_AMX_OP(STZI, g)
#define AMX_EXTRX(g) MT_AMX_OP(EXTRX, g)
#define AMX_EXTRY(g) MT_AMX_OP(EXTRY, g)
#define AMX_FMA64(g) MT_AMX_OP(FMA64, g)
#define AMX_FMA32(g) MT_AMX_OP(FMA32, g)
#define AMX_VECFP(g) MT_AMX_OP(VECFP, g)
#define AMX_MATFP(g) MT_AMX_OP(MATFP, g)

namespace mt::amx {

// Operand builders. Loads and stores: pointer in bits 0-55, register in 56+, pair (62), quad on M2+ (62 + 60).
constexpr uint64_t kPair = 1ull << 62, kQuad = (1ull << 62) | (1ull << 60);
inline uint64_t ptr(const void* p) { return reinterpret_cast<uint64_t>(p) & ((1ull << 56) - 1); }
inline uint64_t xy(const void* p, int reg, uint64_t mult = 0) { return ptr(p) | uint64_t(reg) << 56 | mult; }
inline uint64_t zr(const void* p, int row, bool pair = false) { return ptr(p) | uint64_t(row) << 56 | (pair ? kPair : 0); }

// fma32 / fma64 matrix mode: z[row + stride*j][i] += x[i] * y[j]; offsets in bytes into the 512-byte X / Y files.
constexpr uint64_t kSkipZ = 1ull << 27;
constexpr uint64_t fma(int zrow, int xoff, int yoff, bool skipz = false) {
  return uint64_t(zrow) << 20 | uint64_t(xoff & 0x1FF) << 10 | uint64_t(yoff & 0x1FF) | (skipz ? kSkipZ : 0);
}
// Writemask on X lanes (bits 41-47) and Y lanes (32-38): mode 2 = first n lanes (n > 0).
constexpr uint64_t xfirst(int n) { return (2ull << 46) | uint64_t(n & 31) << 41; }
constexpr uint64_t yfirst(int n) { return (2ull << 37) | uint64_t(n & 31) << 32; }

}  // namespace mt::amx
