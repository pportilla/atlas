# Rational matrices and paired prime lifting on Triton

Atlas returned identical CPU and GPU results on five matrix problems and one modular Gröbner basis. Singular independently verified the matrix ranks, square determinants, row spaces, images, kernels, and solvability. Its particular solutions and Atlas's particular solutions both satisfied the original systems. Reducing each basis modulo the other checked equality of the two ideals in the modular example. [Reproducible comparison](matrix_triton.py)

## Whole-operation timings

Median wall time in milliseconds, one warm-up and nine measured trials, rotating solver order for each input.

| Input | Atlas CPU | Atlas Triton | Singular |
|---|---:|---:|---:|
| Rational 2 by 2 | 0.170 | 0.801 | 0.241 |
| Dependent 2 by 3, inconsistent right-hand side | 0.180 | 0.490 | 0.261 |
| Dense 6 by 6 | 0.290 | 1.024 | 0.411 |
| Tall 80 by 6 | 1.179 | 2.565 | 1.912 |
| Rectangular 12 by 14 | 0.591 | 1.626 | 1.155 |
| Lex basis of `(x² + (2^301 + 123)y − 1, xy − 2)` | 1.351 | 4.854 | 0.455 |

The host was a Ryzen 7 6800H with an integrated Radeon 680M (gfx1035), Triton 3.5.1, HIP-enabled PyTorch 2.9.1 and Singular 4.4.1. Atlas ran through the public session function for the matrices and the shared algebra dispatcher for the complete basis. Wall times include native computation, required transfers, exact host certificates, result construction and replay preparation. Comparison checks ran after timing. Singular held the inputs in a persistent process, returning a completion marker over a pipe after computing its result. Its field was the rationals in a one-variable polynomial ring with constant matrix entries. It computed standard bases of the image and row space, syzygies, determinant for a square matrix, and a lift for a consistent system. The two programs can choose different vector-space bases.

The CPU completed all six requests faster than the GPU on this device. Atlas CPU completed the five matrix cases faster than Singular in this run. Singular completed the modular Gröbner basis fastest. The workload is small and coefficient lifting accounts for only part of total basis time. Whole-operation timings do not isolate the CRT kernel.

## Rank kernel optimization

For matrices with up to eight columns and 256 rows, a single Triton program keeps the columns in registers and eliminates by modular cross products. It writes the original indices of independent rows. The tiled implementation issues successive pivot and elimination launches. In a separate comparison, the table gives 30-trial medians in milliseconds for the same input and device, including host-to-device transfer and witness readback.

| Rows by columns | Fused GPU | Tiled GPU |
|---|---:|---:|
| 2 by 3 | 0.134 | 0.240 |
| 6 by 7 | 0.146 | 0.338 |
| 16 by 7 | 0.154 | 0.336 |
| 32 by 7 | 0.164 | 0.393 |
| 64 by 7 | 0.152 | 0.341 |
| 80 by 7 | 0.171 | 0.408 |
| 128 by 7 | 0.175 | 0.407 |
| 256 by 7 | 0.201 | 0.397 |

After removing a redundant upload, duplicate input validation and eager Python rational-vector conversion, the tall matrix's complete CPU and GPU times changed from 2.081 and 3.657 ms to 1.179 and 2.565 ms. The initial run timed an additional comparison of two result records inside the Atlas samples. The direct fused/tiled comparison uses identical transfer, launch and readback timing boundaries.

For the matrix calculation, a failed modular row-space witness triggers rational row reduction. Square matrices through order six with bounded primitive row products can also dispatch their determinant to the GPU. The host independently certifies the returned integer with sufficient modular determinants and a proven bound. Other full-rank determinants use the native CPU algorithm. The modular Gröbner route packs pairs of 31-bit prime images into one Triton launch and checks both residues of each reconstructed coefficient. Arbitrary-precision combination of the 62-bit results, rational reconstruction, the independent witness prime, and final ideal certification run on the CPU.

On a configured HIP or CUDA system, build Atlas and run the following in its Python environment.

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/matrix_triton.py --trials 9 --rank-trials 30
python3 -m unittest tests.test_matrix_analysis tests.test_gpu_crt tests.test_gpu_rank
```
