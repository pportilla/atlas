# CPU versus OpenCL on Intel Iris Xe

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

## Published inputs and the four-variable germ

The [comparison driver](published_comparison.py) invokes only the Atlas CLI. It checks two published reference cases and records a four-variable mixed-term germ in both modes. Times below are medians of **5 fresh CLI invocations** on the i7-1355U laptop. Each invocation includes process startup, the calculation, and writing the full JSON output. Each case/backend received one untimed warm-up. The driver alternates case order and checks the geometry checksum on every resolved run. [Per-run samples and checksums](atlas-reference-cases.json)

| Input / mode | Atlas outcome | Fresh CPU CLI | Fresh OpenCL CLI |
|---|---|---:|---:|
| `x^7+y^2-z^2`, intrinsic | 7 charts, 6 rational (−2)-curves in a path | 106.2 ms | — |
| `x*y^2*z^3-t^6`, embedded | 12 charts, 8 vertices, 15 edges, 8 triangles | 126.6 ms | — |
| Same binomial, intrinsic | 12 charts, 7 vertices, 12 edges, 6 triangles | 120.0 ms | — |
| `x^5+z*y^3+z^15+t^4*x+t^8`, intrinsic | 598 charts, 99 vertices, 251 edges, 153 two-cells | 190.8 ms | 227.2 ms |
| Same polynomial, embedded | 598 charts, 169 vertices, 593 edges, 720 two-cells, 295 three-cells | 190.5 ms | 220.4 ms |

For the 598-chart germ the GPU processed all 12,558 chart–monomial pairs. CPU and GPU produced identical chart maps, strict transforms, geometric components and cell attachments. In intrinsic mode the native chart batch took a median of 0.088 ms on CPU and 12.020 ms on OpenCL including device setup and transfers.

Atlas reproduces the intersection matrix and genera for the A₆ polynomial in the Singular tutorial. [Singular tutorial](https://www.singular.uni-kl.de/ftp/pub/Math/Singular/doc/tutor.pdf)

For the four-variable binomial, Atlas used eight smooth codimension-two blowups. Blanco and Frühbis-Krüger reported 9 seconds and 240 final charts for Singular’s `resbinomial` on this input. Their general `resolve` routine exceeded 7,200 seconds. The published run used a 2.13 GHz Core i3-330M with 4 GB of RAM, and its timing covered a different implementation and output. A same-machine speed ratio remains unknown. [Study, §5, Table 1](https://arxiv.org/pdf/1109.1655#page=10)

To reproduce after `make gpu`, run `python3 benchmarks/published_comparison.py --gpu-target --output /tmp/atlas-reference-cases.json`. Omit `--gpu-target` for a CPU-only machine.

## Gröbner kernel

All times are on the i7-1355U and count the computation only: no process start, parsing or printing. Singular 4.4.1 ran in one session and was timed around the call. The kernel never uses Singular; it is the yardstick.

### The verifier's bases

`groebner-corpus.json` holds 796 distinct bases that the ideal-blowup verifier requested while checking candidate centres for seven germs and running the blowup tests: up to 378 generators in at most 7 variables, in the graded reverse lexicographic and the lexicographic order. Each comes with SymPy's reduced basis and SymPy's time. The kernel reproduces all 796 term by term.

| Problems | Kernel | Singular `std` | SymPy 1.14 |
|---|---:|---:|---:|
| All 796 | 38.5 ms | 181.2 ms | 115.7 s |
| 699 grevlex | 34.7 ms | 59.9 ms | 92.8 s |
| 97 lex | 3.8 ms | 121.3 ms | 22.9 s |
| 75 with at least 50 generators | 30.9 ms | 47.0 ms | 89.9 s |

Sums over the problems of the best of five runs (three for Singular). The kernel is not slower than Singular on any single problem.

### Ideals whose coefficients swell

The default entry point starts with exact Buchberger strategies. When they swell it switches, for a lexicographic order, to a Hilbert-driven computation on the homogenized ideal, and otherwise to modular images with a certificate over the rationals ([native/KERNEL.md](../native/KERNEL.md), sections 7 and 8). The route it ended up using is in brackets. Singular's `modStd` is its own modular algorithm.

| Ideal | Order | Kernel | Singular `std` | Singular `modStd` |
|---|---|---:|---:|---:|
| Katsura-5 | grevlex | 5.2 ms (exact) | 5.7 ms | 123 ms |
| Katsura-6 | grevlex | 47.0 ms (modular) | 46.7 ms | 263 ms |
| Katsura-7 | grevlex | 314 ms (modular) | 429 ms | 1,029 ms |
| Cyclic-6 | grevlex | 29.7 ms (modular) | 37.0 ms | 162 ms |
| Cyclic-5 | lex | 1.5 ms (exact) | 2.6 ms | 91 ms |
| Katsura-4 | lex | 2.0 ms (Hilbert) | 2.0 ms | 254 ms |
| Katsura-5 | lex | 31.1 ms (Hilbert) | 22.3 ms | over 120 s |

Best of five runs. All routes of the kernel return identical bases. Before the Hilbert-driven route the last two rows took 10.2 ms and 668 ms.

On the last row Singular is still 1.4 times faster. Its `std` also homogenizes and uses a Hilbert function, but takes it from a computation modulo 32003. The kernel takes it from an exact graded basis, which costs 8 of its 31 ms, because a Hilbert function modulo an unlucky prime is too large and makes the criterion drop pairs it should keep; the kernel's result is then certified by comparing Hilbert series. Whether Singular re-checks its own result was not examined.

### Replays on the GPU

The modular route records, for one prime, the row operations that produce the basis, and replays them for further primes. A replay has no data-dependent branches, so primes can run side by side: on the host's 12 threads, or as one OpenCL work item per prime. Both produce the same residues, bit for bit.

| Ideal | Replays | Host threads | Iris Xe (OpenCL) |
|---|---:|---:|---:|
| Katsura-6 | 2,048 primes | 24.0 ms | 235 ms |
| Katsura-7 | 2,048 primes | 123 ms | 1,897 ms |
| Cyclic-6 | 2,048 primes | 42.4 ms | 553 ms |

On this laptop the integrated GPU is about ten times slower than the CPU, so `ATLAS_GROEBNER_GPU` is off by default. A replay costs about 12 µs per prime for Katsura-6 against 20 ms for the trace and 10 ms for the certificate, and these ideals need 7 to 127 primes, so replays are a small part of the total here; a device with many more parallel lanes changes the replay column only.

### One verification, end to end

Rejecting the centre `(w,x,z)` for `w^5+x^2*y^2+z^4` took 41.5 s with SymPy alone, 1.3 s once the Gröbner bases came from the kernel, and takes 0.4 ms now that the whole test runs in the kernel (20 ms through the Python binding). Rejecting the origin for the same germ takes 0.3 ms. In an exhaustive search, the five priority candidates of this germ are decided in 0.3 s; before, the search had not decided them after 60 s.

~~~sh
python3 benchmarks/groebner.py --singular --repeats 5
python3 benchmarks/groebner_hard.py --singular --gpu
~~~
