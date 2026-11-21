# Batched modular arithmetic on Triton

Atlas ran 256 entries per batch on a Ryzen 7 6800H and a Radeon 680M (gfx1035) using Triton 3.5.1. The comparison checked every value in every trial against the native GMP calculation and Singular 4.4.1. The modular reconstruction inputs represent small signed fractions. For Hensel lifting the two roots of $T^2-4$ modulo 65521 are lifted modulo $65521^2$. The Chinese remainder inputs have coprime moduli 65521 and 65519. [Benchmark script](triton_modular.py) · [Raw samples](triton-modular-680m.json)

The medians below are milliseconds after one warm-up and five timed trials per mode. Atlas timings cover structured dispatch, input conversion, device transfers, kernel execution, result conversion and creation of the replay record. Singular timings cover its warm arithmetic loop on prepared inputs and exclude its answer checks. CPU and GPU trial order alternates. All GPU trials required a recorded physical Triton stage.

| Operation | Atlas CPU | Atlas Triton | Singular |
|---|---:|---:|---:|
| reduce | 0.341 | 0.605 | 0.656 |
| add | 0.465 | 0.723 | 0.761 |
| subtract | 0.462 | 0.750 | 0.767 |
| multiply | 0.472 | 0.680 | 0.834 |
| divide | 0.493 | 0.773 | 1.080 |
| inverse | 0.392 | 0.598 | 0.969 |
| reconstruct | 0.485 | 0.662 | 0.630 |
| hensel | 0.429 | 0.627 | 1.492 |
| crt | 0.515 | 0.789 | 1.362 |

Packing all coefficient columns into one upload and all result columns into one download reduced the GPU medians on every case in the original comparison. The inverse batch went from 0.878 to 0.598 ms and rational reconstruction from 0.878 to 0.662 ms. CRT also reuses the inverse of the first modulus across the batch. The native CPU solver remained faster on every measured modular input. Singular was faster than Triton on rational reconstruction. The default automatic policy uses the CPU here.

Run `python benchmarks/triton_modular.py --count 256 --trials 5 --singular` in the Triton environment to reproduce the comparison. A missing GPU, a CPU fallback or an incorrect answer fails the benchmark.
