# Triton polynomial arithmetic

Run `python benchmarks/gpu_polynomial.py --singular --trials 5` in the Atlas Triton environment. The benchmark requires a successful physical Triton launch for every input. Each solver runs one warmup per case and five timed repetitions. All timed polynomial results agreed exactly.

These medians are milliseconds on a Ryzen 7 6800H and its Radeon 680M (`gfx1035`) using Triton 3.5.1. The two Atlas columns measure the same sparse-row entry, including parsing, packing, transfers, launch, reconstruction and result serialization. Singular 4 uses the same rational polynomials and operations in a lexicographic ring with one thread. Its timer covers the operation on parsed elements. Equality is checked against the whole Atlas result outside its timer. [Individual trials](gpu-polynomial-680m.json)

| Case | FLINT CPU | Triton GPU | Singular |
|---|---:|---:|---:|
| add | 0.523 | 0.791 | 0.008 |
| subtract | 0.542 | 0.858 | 0.009 |
| sparse-product | 0.284 | 1.801 | 0.060 |
| dense-product | 0.811 | 1.515 | 0.075 |
| wide-product | 0.048 | 0.347 | 0.002 |
| divide | 0.091 | 0.364 | 0.008 |
| power | 0.079 | 1.610 | 0.067 |
| differentiate | 0.164 | 0.370 | 0.005 |
| evaluate | 0.064 | 0.456 | 0.008 |
| compose | 0.069 | 2.162 | 0.002 |

The dense univariate product uses one lane per output coefficient when the native bound proves its signed accumulator fits. This reduces its output transfer from 262,144 pair products to 1,023 coefficients for the 512-by-512 control. Wider products use base-32768 limb convolution and GMP carry propagation. Simultaneous monomial substitution reuses repeated scalar powers and applies each coordinate to all source terms in a batch. On the 32-term evaluation control, this reduced the median Triton time from 35.385 ms to 0.456 ms. The CPU and Singular were faster than Triton on every measured arithmetic input. Atlas keeps automatic mode on the CPU for these operations.
