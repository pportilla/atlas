# Device timings for chart transformations and rank

## CPU versus OpenCL on Intel Iris Xe

The CPU had the lower median total solve time on the two workloads below. Both backends produced identical chart maps, strict transforms, exceptional component IDs and dual complexes. GPU samples were accepted only when the on-device path was actually used. This is a comparison of this implementation on one laptop, not a comparison against other resolution packages or GPU vendors.

**Hardware:** 13th Gen Intel(R) Core(TM) i7-1355U; Intel(R) Iris(R) Xe Graphics; OpenCL driver 26.35.39758. Every published timing is the median of seven runs per backend. Each backend was warmed once per case, and order alternated between repetitions. The CLI column times a fresh CLI and worker process, including JSON output. OS/driver file caches were not flushed. Times are wall-clock milliseconds, not pure GPU event timings.

## Independent chart transformations

The intrinsic resolution of **x² + y¹⁹ + z³⁷ + x³(x+y+z)²⁷** has a diagonal Newton face with weights (703, 74, 38). Its regular fan has **191 charts**. The equation expands to 409 monomials. After the fan and ray orders are known, all **78,119 chart–monomial pairs** can be transformed independently in one OpenCL launch. The GPU computes the three new exponents and the CPU retains the rational coefficients and assembles the resulting chart equations. The resulting geometric exceptional complex has 40 vertices and 39 edges.

| Time in ms | C++ CPU | OpenCL path |
|---|---:|---:|
| Native exponent transforms, including device transfers when used | 0.309 | 1.401 |
| Whole transform batch, including host packing and materializing tuples | 9.98 | 12.05 |
| Complete warm resolution | 96.82 | 102.28 |
| Fresh CLI invocation with full JSON output | 231.1 | 256.6 |

Parallelism was available; each work item was small. The CPU transformed every exponent in under a millisecond. Most of the solve was spent outside that native transform. The relatively small end-to-end difference should be interpreted in light of machine load and run-to-run variation. [Per-run samples and atlas checksum](iris-xe-charts.json); [benchmark script](charts.py).

## Parallel modular rank

The input is a dense homogeneous ternary form of degree d with deterministic small integer coefficients, plus x^(d+1). Its projective tangent curve is proved smooth by the full-rank Jacobian test. For d=11 this gives a smooth plane curve of genus 45 as the exceptional curve in the intrinsic resolution. One blowup has three affine charts. The computational challenge is the smoothness proof: 570 rows, 435 columns, arithmetic modulo 1,000,000,007. Each pivot permits independent updates of the trailing matrix; subsequent pivots depend on the previous ones.

| Degree | Matrix | Rank: CPU / OpenCL | Warm resolution: CPU / OpenCL | Fresh CLI: CPU / OpenCL |
|---:|---:|---:|---:|---:|
| 4 | 45 × 36 | 0.017 / 3.487 | 1.08 / 5.28 | 115.7 / 134.9 |
| 8 | 273 × 210 | 1.609 / 25.326 | 5.60 / 28.40 | 109.4 / 150.3 |
| 11 | 570 × 435 | 12.631 / 62.751 | 18.96 / 68.56 | 127.7 / 198.4 |

The OpenCL rank time includes device work, transfers and CPU verification of the returned nonzero minor. The benchmark never compares an unverified result to a verified one. [Exact polynomials, per-run samples and atlas checksums](iris-xe.json); [benchmark script](gpu.py).

## Reproduce on another device

~~~sh
make gpu
./atlas devices
python3 benchmarks/charts.py --repeats 7 --output /tmp/charts.json
python3 benchmarks/gpu.py --degrees 4 8 11 --repeats 7 --output /tmp/rank.json
~~~

Set RESOLUTION_GPU_INDEX to compare a different enumerated device. A missing or unused GPU causes the benchmark to stop rather than counting a fallback as a GPU timing. The raw samples allow a different summary statistic or comparison across machines.

The chart, rank and published-input drivers also accept `--backend triton` with a working Triton installation. To exercise the exact chart, modular rank and Gröbner replay tests on the selected physical Triton device, run `make test-triton`.
