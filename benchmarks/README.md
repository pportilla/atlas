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

The chart, rank and published-input drivers also accept `--backend triton` with a working Triton installation. To exercise the exact chart, modular rank and Gröbner replay tests on the selected physical Triton device, run `make test-triton`.

## Shared torus verification

Restricting each chart to its coordinate hyperplanes once reduced the median time for a complete warm intrinsic resolution. Seven timed runs per version and case alternated between the original and updated verifier. Every run checked the same owned atlas and labelled incidence data. Values are milliseconds on the Ryzen 7 6800H host. The OpenCL runs used its Radeon 680M. [Raw samples, input polynomials and version details](torus-section-timings.json)

| Workload | CPU before → after | OpenCL before → after |
|---|---:|---:|
| 191 charts, 409 monomials | 1298.3 → 1104.8 | 1182.8 → 1014.3 |
| 598 charts, four variables | 453.2 → 395.0 | 381.0 → 338.9 |

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

### Full Katsura-7 solve on CPU and GPU

On the Ryzen 7 6800H with Radeon 680M integrated graphics (Mesa 26.0.8, OpenCL 3.0), the updated GPU mode took **100.18 ms** for a warm solve, **5.84× faster** than the previous GPU mode. The updated CPU mode finished in **82.49 ms**, the fastest on this machine. Times below are wall-clock medians in milliseconds from seven rounds with rotating case order. The CPU could use all 16 hardware threads. [Samples, execution receipts and build fingerprints](katsura7-parallel.json)

| Mode | Full warm solve | Full fresh solve | Trace | Replay | Lift | Exact verification |
|---|---:|---:|---:|---:|---:|---:|
| Previous CPU | 286.30 | 345.35 | 153.47 | 2.90 | 17.55 | 96.58 |
| Updated CPU | 82.49 | 152.73 | 22.49 | 2.52 | 17.21 | 23.98 |
| Previous GPU | 585.09 | 728.68 | 151.77 | 301.00 | 19.55 | 94.75 |
| Updated GPU | 100.18 | 260.68 | 38.18 | 2.82 | 17.27 | 23.91 |

In each updated GPU solve, OpenCL reduced 367 critical-pair rows in six trace batches. The three replay batches contained 2, 4 and 8 new primes and ran on CPU threads. Previously those replay batches ran on the GPU. Exact rational verification ran on the host in all four configurations. Every solve produced the same 74 basis polynomials (352,915 serialized bytes).

The driver starts each solve with the eight original polynomials in eight variables over the rationals, in graded reverse lexicographic order. It calls the public `atlas_groebner` entry point, checks that the solver took the certified modular route, and compares the entire reduced basis byte for byte with the baseline CPU result. Every round measures both builds on the CPU and with the selected GPU backend. Build both checkouts with the same compiler options and native dependencies.

~~~sh
python3 benchmarks/groebner_parallel.py --baseline /path/to/built/baseline \
    --candidate /path/to/built/candidate --backend opencl --index 0 \
    --require-gpu-trace --repeats 7 --output /tmp/katsura7.json
~~~

`--backend triton` selects a working Triton device. `--require-gpu-trace` checks that the candidate actually reduced trace pairs on the selected GPU. Every GPU timing requires a device execution receipt. A warm solve includes input generation, route selection, exact certification and full basis serialization. A fresh solve also includes interpreter startup, returning all the basis text to the caller and process exit. Filesystem and driver compiler caches can be warm in both cases. Separate diagnostic runs time trace construction, modular replay, rational lifting and exact verification directly in the modular solver. [Benchmark driver](groebner_parallel.py)

### One verification, end to end

Rejecting the centre `(w,x,z)` for `w^5+x^2*y^2+z^4` took 41.5 s with SymPy alone, 1.3 s once the Gröbner bases came from the kernel, and takes 0.4 ms now that the whole test runs in the kernel (20 ms through the Python binding). Rejecting the origin for the same germ takes 0.3 ms. In an exhaustive search, the five priority candidates of this germ are decided in 0.3 s; before, the search had not decided them after 60 s.

~~~sh
python3 benchmarks/groebner.py --singular --repeats 5
python3 benchmarks/groebner_hard.py --singular --gpu
~~~

## Same-host resolution and zeta functions

[The headless comparison](resolution_zeta.py) runs the public Atlas resolver and the installed Singular libraries on three isolated surface germs over the rationals. Each solver starts from the polynomial and returns the local topological zeta function and the local monodromy invariant at the origin. The inputs cover an ordinary double point, the cone over a smooth plane cubic, and the A₆ surface from the Singular tutorial. The driver checks both exact rational functions on every run.

Atlas requests embedded resolution with both zeta functions, CPU computation, and quick search. This includes its Python-to-native calls, verification, atlas construction and computation of the embedded dual complex. Singular loads `reszeta.lib`, calls `resolve(ideal(f),"A")` to retain all charts, then calls `zetaDL(re,1,"local","A")`. Singular builds its own resolution and computes the characteristic polynomial on middle cohomology. If this polynomial is χ(s) of degree µ, Atlas's A'Campo convention gives 1 / ((1 − T) T^µ χ(1/T)) for these isolated surfaces. The driver checks that conversion outside the timed interval. The full chart records have different formats and neither program prints them during the benchmark.

Fresh wall time includes starting the interpreter, loading libraries, reading the input, the complete calculation, printing both expressions and exiting. Warm wall time covers sending a polynomial to an initialized interpreter through reading its expressions back. Each interpreter performs one untimed solve of each case before timing. Solvers run sequentially, the order reverses on alternating repetitions, and repeated calls can reuse their normal in-process caches. Filesystem caches are left warm. The JSON records the raw expressions, all wall times, machine and software versions, and a digest of each checkout's engine sources and installed solver binaries.

~~~sh
python3 benchmarks/resolution_zeta.py --repeats 5 --output /tmp/zeta-times.json
~~~

Use `--atlas-root /path/to/built/candidate --baseline /path/to/built/baseline` to compare two Atlas builds with the same interpreter and Singular installation. The external reference is optional and used only by this benchmark. A missing result, Singular error or algebraic disagreement fails the run.

### Measurements on the headless host

Median wall times in milliseconds from five runs per case and scope on 2026-11-14. The AMD Ryzen 7 6800H ran Linux x86-64, Python 3.14.4 and Singular 4.4.1. The driver and its children were pinned to logical CPU 5 with `taskset -c 5`. Other host activity continued. The Atlas source-and-binary fingerprint prefixes were `a67f9d3d2d08` for the candidate and `694c612747fc` for the pre-migration baseline.

| Input | Candidate fresh | Baseline fresh | Singular fresh | Candidate warm | Baseline warm | Singular warm |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary double point, $x^2+y^2+z^2$ | 38.15 | 302.52 | 170.33 | 1.99 | 7.23 | 136.94 |
| Cubic cone, $x^3+y^3+z^3$ | 37.28 | 286.67 | 172.38 | 1.48 | 3.76 | 135.81 |
| A₆ surface, $x^7+y^2-z^2$ | 39.99 | 321.79 | 1351.45 | 4.34 | 22.04 | 1307.31 |

The ordinary double point and the cubic cone use the sparse point-chart verifier without importing SymPy. Their fresh baseline-to-candidate speedups were 7.93× and 7.69×. For the longer A₆ chain, fresh wall time improved 8.05× and warm wall time improved 5.07×. Across all three inputs the candidate was 2.55–5.07× faster than the warm baseline and 4.46–33.80× faster than fresh Singular. All 90 timed requests passed both exact invariant checks.

### General blowups and intrinsic chains

The following measurements compare the previous native release with the sparse chart migration on the same headless CPU. Each sample used the public CPU resolver with both zeta functions enabled. Each pair ran in a fresh interpreter, then repeated the same request in that interpreter. The five pairs per input alternated the release order on CPU 4. The zeta expressions, number of cells and completion status agreed for every timed request. Cold timings include interpreter startup and worker initialization. These figures compare Atlas versions.

| Input and mode | Previous cold | Native cold | Previous warm | Native warm |
|---|---:|---:|---:|---:|
| $(x+y)^2+z^3+x^3$, embedded | 734 ms | 244 ms | 454 ms | 180 ms |
| $(xy-z^2)^2+x^5+y^5$, embedded | 826 ms | 316 ms | 577 ms | 255 ms |
| $x^7+y^2-z^2$, intrinsic with embedded companion | 274 ms | 43 ms | 17.6 ms | 8.5 ms |

The default-mode audit traces plane, chain, graph and canonical-centre examples. The current driver uses native inputs by default. The `--symbolic-inputs` option exercises the compatibility adapter for existing Python callers. [Production path inventory](symbolic-paths.md)
