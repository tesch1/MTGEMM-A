// AMX probe: checks that AMX instructions execute, checks one fma32 result, times fma32 throughput.
#include <mach/mach_time.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>

#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm volatile("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")
#define AMX_OP_GPR(op, gpr) \
    __asm volatile(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")
#define AMX_LDX(g) AMX_OP_GPR(0, g)
#define AMX_LDY(g) AMX_OP_GPR(1, g)
#define AMX_STZ(g) AMX_OP_GPR(5, g)
#define AMX_FMA32(g) AMX_OP_GPR(12, g)
#define AMX_SET() AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR() AMX_NOP_OP_IMM5(17, 1)

static sigjmp_buf jb;
static void on_sigill(int s) { siglongjmp(jb, 1); }

static void sysstr(FILE *o, const char *k) {
    char b[256]; size_t n = sizeof b;
    if (sysctlbyname(k, b, &n, 0, 0) == 0) fprintf(o, "%s: %s\n", k, b);
}
static void sysint(FILE *o, const char *k) {
    int64_t v = 0; size_t n = sizeof v;
    if (sysctlbyname(k, &v, &n, 0, 0) == 0) fprintf(o, "%s: %lld\n", k, (long long)v);
}

static double now_s(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

int amx_probe_main(FILE *o) {
    sysstr(o, "machdep.cpu.brand_string"); sysstr(o, "hw.machine"); sysstr(o, "hw.model");
    sysstr(o, "kern.osversion"); sysint(o, "hw.ncpu"); sysint(o, "hw.perflevel0.physicalcpu");
    sysint(o, "hw.perflevel1.physicalcpu"); sysint(o, "hw.perflevel0.cpusperl2"); sysint(o, "hw.cpufamily");
    sysint(o, "hw.perflevel0.l2cachesize");
    sysint(o, "hw.perflevel1.l2cachesize"); sysint(o, "hw.l1dcachesize"); sysint(o, "hw.memsize");
    sysint(o, "hw.optional.arm.FEAT_SME"); sysint(o, "hw.optional.arm.FEAT_BF16"); sysint(o, "hw.optional.arm.FEAT_I8MM");
    fflush(o);

    struct sigaction sa = {0}, old;
    sa.sa_handler = on_sigill;
    sigaction(SIGILL, &sa, &old);
    if (sigsetjmp(jb, 1)) { fprintf(o, "AMX: SIGILL (not available)\n"); sigaction(SIGILL, &old, 0); return 1; }
    AMX_SET();
    sigaction(SIGILL, &old, 0);

    float __attribute__((aligned(128))) x[16], y[16], z[16];
    for (int i = 0; i < 16; i++) { x[i] = (float)(i + 1); y[i] = 2.0f; }
    AMX_LDX((uint64_t)x); AMX_LDY((uint64_t)y);
    AMX_FMA32(1ull << 27);               // z[4j][i] = x[i]*y[j], Z input skipped
    AMX_STZ((uint64_t)z | (4ull << 56)); // row 4 is j=1
    int ok = 1;
    for (int i = 0; i < 16; i++) ok &= z[i] == 2.0f * (i + 1);
    fprintf(o, "AMX: executes; fma32 check %s (z[4][15]=%g)\n", ok ? "PASS" : "FAIL", z[15]);

    const long N = 20000000;
    for (int rep = 0; rep < 3; rep++) {
        double t = now_s();
        for (long k = 0; k < N; k++) {
            AMX_FMA32(0ull << 20); AMX_FMA32(1ull << 20); AMX_FMA32(2ull << 20); AMX_FMA32(3ull << 20);
        }
        t = now_s() - t;
        fprintf(o, "fma32 4 accumulators: %.1f GFLOPS (%.3f s)\n", N * 4.0 * 512 / t * 1e-9, t);
    }
    AMX_CLR();
    fflush(o);
    return ok ? 0 : 2;
}
