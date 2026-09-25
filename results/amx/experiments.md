# AMX backend on the Vision Pro: design experiments

Summary of the intermediate device batches that the README cites (the raw files were removed; the final
results are in `final1` and `final2`, the check after the review in `review1`). GFLOPS, fp32 row-major one
thread unless noted; Accelerate with its own threading. Each comparison is within one batch, because the device
varies up to 10% between batches. Option values are those of the final code (`online` 0 = B packed up front,
2 = B read from the source).

## Where the time goes (2048^3, batches b1, b2 and b4)

| run | GFLOPS | note |
|---|---|---|
| kernel only (`prof=5`), kc=2048, mc=64, nc=128 | 1587 | all in L2: 100% of the fma32 peak |
| kernel only, kc=2048, mc=256, nc=512 | 1475 | 6 MB footprint |
| kernel only, kc=512, mc=256, nc=512, C prefetch on / off | 1378 / 1263 | C reload at each depth block |
| kernel only, mc=2048, nc=128, kc=1024 | 1392 | A block of 8 MB streams from DRAM |
| full, mc=2048, nc=128, kc=1024: skip A packing / skip B packing | 1272 / 1289 | full run 1160 |
| full, model of the SME version (mc=512, nc=416, kc=688) | 925 | B packed 4 times |

## Blocking (2048^3, batch b7, two rounds in one batch)

| mc, nc, kc | round 1 | round 2 |
|---|---|---|
| 2048, 128, 1024 | 1230 | 1089 |
| 1024, 128, 1024 | 1166 | 1098 |
| 1024, 256, 1024 | 1161 | 1014 |
| 2048, 128, 512 | 1109 | 986 |
| 512, 128, 1024 | 1024 | 1038 |
| 512, 256, 512 | 977 | 969 |
| Accelerate | 1028 | 1023 |

The AMX peak (`peak=`) stayed at 1607-1640 GFLOPS in both rounds: the drift is in the memory system.

## B packing (paper shapes 1-12, M = 64 and 128, geomean)

| variant | GFLOPS | batch |
|---|---|---|
| B packed panel by panel, unpadded panel stride | 274 | b8 |
| B packed up front in row order, padded panel stride | 610 | b9 |
| 64x16 kernel with B read straight from the source (removed) | 282 | b9 |
| Accelerate | 377 | b9 |

Squares (b10): the core packing the next B panel inside the kernel (removed) 823 against 1106 with B packed up front.

## Reading B without packing (batch b13)

| set | Accelerate | B packed | B from the source |
|---|---|---|---|
| small 4-384 | 143 | 83 | 94 (faster than packed from 32 to 256) |
| squares 512-4096 | 1169 | 1179 | 814 |
| paper's 24 | 603 | 738 | 407 |
| small 4-384, fp64 | 73 | 42 | 49 (faster than packed at every size from 32 up) |

The default (`online=1`) reads B from the source when it fits in 2 MB (fp64, or fp32 with M <= 128).

## A transposition through Z (batches b11 and b12)

| shape | NEON | AMX through Z |
|---|---|---|
| 32x32x4096 (b12) | 238 | 307 |
| 4096x64x4096, prefetch 2 chunks ahead (b11) | 426 | 328 |
| 4096x64x4096, prefetch 4 chunks ahead (b12) | 408 | 394 |
| fp64 squares geomean (b11) | 253 | 283 |

## Efficiency-cluster share (batch b14, removed)

| shape | P only | 5% on E | 8% on E | 12% on E |
|---|---|---|---|---|
| 2048^3 | 1254 | 1225 | 1270 | 1091 |
| 4096^3 | 985 | 1010 | 686 | 512 |
| 4096x2112x7168 | 995 | 1020 | 872 | 514 |
