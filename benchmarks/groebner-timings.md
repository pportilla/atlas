# Gröbner basis and normal form timings

## Comparing fully reduced monic bases

Run this comparison on Linux with Singular on the path. Both engines receive the same ordered generators over `QQ`, scaled to primitive integers before timing. The 796 verifier cases include both lex and grevlex. Seven Cyclic and Katsura cases exercise the two orders on classical ideals.

```sh
make -C native -f groebner.mk build/bench_groebner_basis
python3 benchmarks/groebner_comparison.py --cpu single --output build/benchmarks/groebner-single.json
python3 benchmarks/groebner_comparison.py --cpu host --output build/benchmarks/groebner-host.json
```

The timer starts with parsed polynomials and stops with a fully reduced basis in memory, including its monic rational coefficients. Atlas calls its production C++ solver. Singular runs `groebner`, `std`, `slimgb`, or `std` with `option(pure_gb)`, with `option(redSB)` and leading-coefficient normalization in each case. For the classical problems the driver also tests `stdfglm` on zero-dimensional lex ideals and `modStd(I,1)` on grevlex ideals. The benchmark checks the exact polynomial answers after timing. Parsing, printing and process startup are outside the measurement.

Each method warms up on each ideal. Five measured rounds rotate method order. For the verifier corpus each round averages eight solves per ideal and sums those averages. The reported total is the median of the five sums. A classical trial times one solve. The same 30-second process limit applies to every classical method, including warm-up and checking. A timeout is reported separately from measured solve times. `--cpu single` restricts every process and its threads to one logical CPU. `--cpu host` gives every method the same set of available CPUs. Singular's measurements include its in-process command invocation and assignment. Atlas calls the production solver directly from C++.

For direct C++ timings on both sides, install the development headers for the same version of libSingular as your Singular executable. These commands run six paired rounds, alternating which kernel runs first. Both drivers read identical requests, materialize the monic basis inside the timer, and return every rational coefficient and exponent for comparison.

```sh
make -C native -f groebner.mk build/bench_groebner_basis build/bench_singular_basis
python3 benchmarks/groebner_native_comparison.py --cpu single --output build/benchmarks/groebner-native-single.json
python3 benchmarks/groebner_native_comparison.py --cpu host --output build/benchmarks/groebner-native-host.json
```

On the Ryzen 7 6800H with Singular 4.4.1, `std` with `pure_gb` was the fastest of the four fixed Singular settings over the full verifier corpus. The C++ driver calls that method directly in libSingular. The following totals are milliseconds for all the ideals in each row. Each CPU column gives both processes the indicated affinity. [Individual timings, options, hashes and timeouts](groebner-reduced-ryzen-6800h.json.gz)

| Verifier ideals | Atlas, 1 logical CPU | libSingular, 1 | Atlas, 16 available | libSingular, 16 |
|---|---:|---:|---:|---:|
| All 796 | 37.35 | 53.35 | 37.71 | 54.50 |
| 699 grevlex | 33.75 | 48.62 | 34.01 | 49.64 |
| 97 lex | 3.60 | 4.74 | 3.70 | 4.85 |

For the classical ideals, the fixed `std`/`pure_gb` setting was often slower than another Singular method. With one logical CPU, Singular's best completed Katsura-5 lex time was 37.31 ms using `stdfglm`, against 22.15 ms for Atlas in the same five-round comparison. `slimgb` and `std` with `pure_gb` hit the process limit on that ideal at both CPU settings. The verifier inputs were collected from Atlas resolution workloads. The two monomial-order subsets contain different ideals.

## Comparing Atlas with OSCAR

This benchmark uses the same 796 verifier ideals and seven classical ideals as the Singular comparison. It compares the Atlas C++ kernel with the public Julia API of [OSCAR](https://www.oscar-system.org/). OSCAR's default Buchberger method calls Singular internally. Its F4 method uses msolve and accepts grevlex here; its FGLM method computes a starting basis and changes order for the three zero-dimensional classical lex inputs. These methods are reported separately.

With Julia installed, instantiate the [pinned OSCAR project](oscar/Manifest.toml), build the Atlas benchmark executable and run the comparison. Use `--julia /path/to/julia` if Julia is not on your path. See also [OSCAR's Linux installation instructions](https://www.oscar-system.org/install/linux/).

```sh
julia --project=benchmarks/oscar -e 'using Pkg; Pkg.instantiate()'
make -C native -f groebner.mk build/bench_groebner_basis
python3 benchmarks/groebner_oscar.py --cpu single --output build/benchmarks/oscar-single.json
python3 benchmarks/groebner_oscar.py --cpu host --output build/benchmarks/oscar-host.json
```

Both engines start with the same parsed rational input and finish with a reduced monic basis in memory. OSCAR starts each solve on a fresh ideal; timing repeated calls on the same ideal could return a cached basis. Its public call includes any conversion into the backend and back into Julia polynomials. Startup, Julia compilation during warm-up, parsing, printing and the independent comparison of rational coefficients and exponents are outside the solve timer. F4 uses multi-modular lifting over `QQ` without a final rational verification inside msolve; its returned basis faces the same exact comparison outside the timer. See [OSCAR's Gröbner basis documentation](https://docs.oscar-system.org/stable/CommutativeAlgebra/GroebnerBases/groebner_bases/).

On the Ryzen 7 6800H, Julia 1.10.13 and OSCAR 1.8.2 used Singular.jl 0.28.19 for the default method and AlgebraicSolving 0.10.6 for F4. Six paired rounds rotated method order. Each verifier round timed eight fresh solves per ideal, averaged those eight, then summed over the same ideals for each method. Each classical trial timed one solve. Values below are medians in milliseconds with all methods pinned to one logical CPU.

| Input and order | Atlas | OSCAR default | OSCAR F4 | OSCAR FGLM |
|---|---:|---:|---:|---:|
| All 796 verifier ideals, mixed orders | 37.905 | 332.271 | — | — |
| Same 696 nonempty grevlex verifier inputs | 34.201 | 218.350 | 624.729 | — |
| Katsura-5 grevlex | 3.456 | 12.873 | 3.913 | — |
| Katsura-6 grevlex | 30.522 | 172.007 | 13.747 | — |
| Katsura-7 grevlex | 168.079 | 963.504 | 51.293 | — |
| Cyclic-6 grevlex | 20.795 | 78.461 | 8.892 | — |
| Cyclic-5 lex | 1.210 | 3.981 | — | 3.196 |
| Katsura-4 lex | 1.341 | 4.596 | — | 4.515 |
| Katsura-5 lex | 21.662 | 132.669 | — | 51.485 |

OSCAR's default completed all 803 inputs. F4 does not accept lex; on the grevlex corpus, its wrapper also rejected the three empty generator lists (verifier-63, -137 and -383) with a `MethodError`. The 696-input row excludes those three ideals from every column. The other 97 verifier inputs use lex. FGLM is measured only on the three known zero-dimensional classical lex cases. Every returned basis agreed exactly.

OSCAR defaults to one msolve thread. The runner keeps that setting for the verifier ideals at either CPU affinity. For classical F4 it explicitly requests the full CPU allocation (one or 16 here). To try full threading on the verifier corpus too, use `--cpu host --msolve-corpus-threads available`. An exploratory run with this setting took over four minutes of wall time for its first complete corpus round across all methods; it was stopped before six rounds. This is not an F4 solve time. The runner allows 240 seconds of wall time per method per corpus round, separately from startup; `--corpus-time-limit` changes that budget.

With the same 16 logical CPUs available to both systems, the six-round medians in milliseconds were:

| Input and order | Atlas | OSCAR default | OSCAR F4 | OSCAR FGLM |
|---|---:|---:|---:|---:|
| All 796 verifier ideals, mixed orders | 37.662 | 350.411 | — | — |
| Same 696 nonempty grevlex verifier inputs | 33.900 | 194.649 | 692.334 | — |
| Katsura-5 grevlex | 3.432 | 13.253 | 12.287 | — |
| Katsura-6 grevlex | 17.268 | 150.906 | 1681.019 | — |
| Katsura-7 grevlex | 52.958 | 1130.605 | 1952.206 | — |
| Cyclic-6 grevlex | 15.131 | 85.805 | 26.319 | — |
| Cyclic-5 lex | 1.225 | 4.236 | — | 3.558 |
| Katsura-4 lex | 1.354 | 4.869 | — | 4.864 |
| Katsura-5 lex | 22.136 | 182.030 | — | 53.585 |

The one-thread F4 setting was faster on all four classical grevlex ideals than requesting 16 threads on this host. The verifier corpus and these classical ideals have different winners. [Raw trials, exclusions, thread settings and input hashes](groebner-oscar-ryzen-6800h.json.gz).

## Earlier kernel timings

The following i7-1355U measurements use Atlas's internal solver timer and Singular 4.4.1's default `std`. The timer stops before Atlas materializes monic coefficients, and default `std` can return a partially reduced basis. Use the comparison above for like-for-like timings.

## The verifier's bases

`groebner-corpus.json` holds 796 distinct bases that the ideal-blowup verifier requested while checking candidate centres for seven germs and running the blowup tests: up to 378 generators in at most 7 variables, in the graded reverse lexicographic and the lexicographic order. Each comes with SymPy's reduced basis and SymPy's time. The kernel reproduces all 796 term by term.

| Problems | Kernel | Singular `std` | SymPy 1.14 |
|---|---:|---:|---:|
| All 796 | 38.5 ms | 181.2 ms | 115.7 s |
| 699 grevlex | 34.7 ms | 59.9 ms | 92.8 s |
| 97 lex | 3.8 ms | 121.3 ms | 22.9 s |
| 75 with at least 50 generators | 30.9 ms | 47.0 ms | 89.9 s |

Sums over the problems of the best of five runs (three for Singular). The kernel is not slower than Singular on any single problem.

## Ideals whose coefficients swell

The default entry point starts with exact Buchberger strategies. When they swell it switches, for a lexicographic order, to a Hilbert-driven computation on the homogenized ideal, and otherwise to modular images with a certificate over the rationals. [Modular certification](../native/KERNEL.md#certification) · [Hilbert-driven basis](../native/KERNEL.md#the-hilbert-driven-route) The route it ended up using is in brackets. Singular's `modStd` is its own modular algorithm.

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

## Device execution for rational ideal arithmetic

The public `./atlas algebra groebner` command returned the same ring, grevlex order and 41 reduced QQ basis polynomials in all nine runs below. The test substitutes `2^192*u6` for the last variable of Katsura-6 to require a device-sized batch of replay primes as well as trace work. The host reconstructs the rational coefficients and certifies the lifted basis. Times are wall-clock medians over three fresh CLI processes per case, with rotating case order, through verification, serialization and decoding the full JSON result. [Raw samples and exact result digest](algebra-radeon-680m.json)

| Requested execution | Trace pairs on GPU / CPU | Replay primes on GPU / CPU | End to end |
|---|---:|---:|---:|
| CPU | 0 / 150 | 0 / 126 | 195.543 ms |
| Radeon 680M, OpenCL 3.0, index 0 | 150 / 0 | 96 / 30 | 377.469 ms |
| GPU, unavailable OpenCL index 1 | 0 / 150 | 0 / 126 | 287.151 ms |

The Ryzen 7 6800H host used the Radeon 680M with Mesa driver 26.0.8-1ubuntu0.3. Each GPU run submitted four trace batches and two replay batches. Four smaller replay batches stayed on the CPU, which certified one basis in one attempt. The comparison also ran with no OpenCL device visible and recovered the same basis.

For `(1+x+y)^120/7` modulo `(x^8,y^9)`, the CPU reducer now skips each divisible term in place. In two builds with identical compiler settings, this cut the fresh CLI median from **808.029 to 153.169 ms**. With the basis already loaded, the native call fell from **810.344 to 1.658 ms**, including input parsing and output serialization. Each median uses seven samples after warm-up.

On the optimized build, the Radeon 680M took **252.827 ms** end to end. Automatic selection used the CPU (**170.848 ms**), and requesting an unavailable OpenCL index fell back to the CPU (**239.447 ms**). The Radeon processed all 7,381 terms and the host verified all 7,309 deletions. Every run matched all 72 coefficients given by `binom(120,i) * binom(120-i,j) / 7` for `0 <= i < 8` and `0 <= j < 9`. [Matched samples, execution records and exact remainder](algebra-normal-form-radeon-680m.json)

Run these commands with a working OpenCL installation, or select `--backend triton` for a physical Triton device. [Benchmark driver](algebra_gpu.py)

~~~sh
make gpu
python3 benchmarks/algebra_gpu.py --backend opencl --index 0
python3 benchmarks/algebra_gpu.py --backend opencl --workload normal-form
python3 benchmarks/algebra_gpu.py --backend opencl --fallback-only
~~~

## Membership with a prepared basis

In the lexicographic ring `QQ[x,y,z]` with `x > y > z`, the basis `(y-z-1)` leaves the leading term `x` of `x+y^300` irreducible. Returning false there cut the median CPU call from **243.706 ms to 3.776 µs**. Queries that require the full reduction took comparable time on both builds. For `y^300-(1+z)^300+1` the median went from 577.538 to 591.069 ms, and for the member `y^300-(1+z)^300` it went from 578.135 to 586.052 ms. Each figure is the median of seven paired rounds on the Ryzen 7 6800H using one saved basis per process. [Input polynomials, build hashes and raw samples](algebra-membership-ryzen-6800h.json)

## Replays on the GPU

The modular route records, for one prime, the row operations that produce the basis, and replays them for further primes. A replay has no data-dependent branches, so primes can run side by side: on the host's 12 threads, or as one OpenCL work item per prime. Both produce the same residues, bit for bit.

| Ideal | Replays | Host threads | Iris Xe (OpenCL) |
|---|---:|---:|---:|
| Katsura-6 | 2,048 primes | 24.0 ms | 235 ms |
| Katsura-7 | 2,048 primes | 123 ms | 1,897 ms |
| Cyclic-6 | 2,048 primes | 42.4 ms | 553 ms |

On this laptop the integrated GPU is about ten times slower than the CPU, so `ATLAS_GROEBNER_GPU` is off by default. A replay costs about 12 µs per prime for Katsura-6 against 20 ms for the trace and 10 ms for the certificate, and these ideals need 7 to 127 primes, so replays are a small part of the total here; a device with many more parallel lanes changes the replay column only.

## Full Katsura-7 solve on CPU and GPU

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

## Profiling Katsura-6, -7 and -8

The starting point for this comparison is the optimized Katsura-7 implementation measured above. Each cell gives the median full solve time in milliseconds, warm / fresh, over seven rounds with rotating mode order on the Ryzen 7 6800H and Radeon 680M. All 315 timed warm, fresh and diagnostic solves returned byte-identical certified bases within each workload, containing 41, 74 and 143 polynomials respectively. [Full solve samples, execution receipts, phase timings and output digests](katsura-dispatch.json)

| Workload | Previous CPU | Previous GPU | Updated CPU | Updated GPU | Automatic |
|---|---:|---:|---:|---:|---:|
| Katsura-6 | 30.93 / 92.93 | 35.04 / 196.16 | 25.96 / 89.89 | 29.01 / 198.44 | 40.85 / 98.21 |
| Katsura-7 | 96.37 / 162.33 | 108.64 / 274.68 | 68.52 / 141.27 | 79.71 / 245.87 | 68.99 / 144.65 |
| Katsura-8 | 726.72 / 813.91 | 1372.64 / 1547.62 | 434.23 / 536.60 | 438.65 / 625.81 | 423.93 / 529.64 |

The larger Katsura-6 automatic median varied across workers under concurrent host load. In a separate check alternating 30 CPU and automatic solves in the same warm interpreter, the medians were 24.18 ms and 24.46 ms. Both modes ran all 150 trace rows on native threads in every solve. [Paired CPU and auto samples](katsura6-auto-check.json)

For Katsura-8, the updated GPU mode finished **3.13× faster warm** and **2.47× faster fresh** than the previous GPU mode by ratio of medians. It processed all 870 trace rows on the GPU and replayed all 30 primes on CPU threads. Previously, 16 of those primes ran on the GPU. Replay took 641.48 ms before and 18.13 ms after, trace construction 214.18 and 117.10 ms, lifting 205.12 and 24.26 ms, and exact verification 258.13 and 229.54 ms. Automatic dispatch chose CPU for every trace and replay batch in all three workloads.

Other computations used the host during this comparison. The smaller Katsura-6 samples varied from 23.8 to 41.2 ms warm in automatic mode and from 24.2 to 41.2 ms on the updated CPU. Both used the same native trace and replay backends. Warm Katsura-8 timings ranged from 379 to 462 ms on the updated CPU and from 393 to 504 ms in updated GPU mode. The updated GPU beat the previous GPU in all seven matched warm rounds, with a median paired speedup of 3.09× (interquartile range 3.06–3.34×). The updated CPU finished ahead of the updated GPU in five of those rounds.

On the Radeon 680M, the GPU kernel spent 15.84 ms executing the seven Katsura-8 trace packets, down from 39.90 ms. Once host preparation and transfer costs were included, all 17 measured packet shapes across the three problems still favored native CPU reduction. The first physical GPU dispatch cost an additional 82.01 ms relative to the warmed first packet. [Before samples](katsura-trace-before.json), [after samples](katsura-trace-after.json), [measured calibration](katsura-trace-calibration.json)

| Workload | Shared symbolic setup | GPU row dispatch with transfers | Device kernel execution | Host echelon |
|---|---:|---:|---:|---:|
| Katsura-6, four packets | 1.03 → 0.58 | 3.15 → 2.44 | 1.47 → 0.55 | 0.11 → 0.10 |
| Katsura-7, six packets | 4.60 → 2.81 | 8.58 → 5.61 | 4.81 → 2.22 | 0.76 → 0.70 |
| Katsura-8, seven packets | 34.60 → 12.26 | 45.61 → 25.03 | 39.90 → 15.84 | 6.73 → 5.98 |

Values are milliseconds before and after, summing the seven-run median for each degree packet. Device kernel execution uses OpenCL event timestamps. The other columns use host wall clocks. Separate runs measured each column. The raw data includes each packet's degree, prime, rank, reducer dimensions, CPU time and upload, kernel and download timings.

The profiler captures every screened degree from an actual certified solve. It reuses the packed inputs to compare the native CPU row reducer with a complete GPU dispatch, including data transfers. It checks every returned remainder against the CPU coefficients. Separate CPU solves measure shared symbolic setup, row reduction and echelon work. An optional instrumented OpenCL plugin measures buffer setup and upload, kernel execution and readback. Device event durations are part of the corresponding wall times. [Trace profiler](groebner_trace_profile.py)

The following commands build a separate profiling plugin on Linux, calibrate the machine from seven rounds, then run both checkouts through the public solver. The calibration measures the first GPU call in three fresh processes and records timings by prime, row count, width and number of reducer entries. The profile identifies the installed native solver and backend binaries so that rebuilding the solver invalidates stale timing evidence when their contents change. Set `ATLAS_TRACE_PROFILE` to the generated profile when using automatic dispatch outside the benchmark. For Triton, choose `--backend triton` and omit `--opencl-profile`.

~~~sh
make gpu
make -C native -f groebner.mk gpu GROEBNER_GPU_LIB=build/libgroebner_profile.so \
    CXXFLAGS='-O3 -std=c++17 -DATLAS_OPENCL_PROFILE'
python3 benchmarks/groebner_trace_profile.py --sizes 6 7 8 --repeats 7 \
    --opencl-profile native/build/libgroebner_profile.so --output /tmp/trace.json \
    --write-auto-profile /tmp/trace-policy.json
python3 benchmarks/groebner_parallel.py --baseline /path/to/built/baseline \
    --sizes 6 7 8 --repeats 7 --require-gpu-trace --calibration /tmp/trace-policy.json \
    --output /tmp/katsura-solves.json
~~~

## One verification, end to end

Rejecting the centre `(w,x,z)` for `w^5+x^2*y^2+z^4` took 41.5 s with SymPy alone, 1.3 s once the Gröbner bases came from the kernel, and takes 0.4 ms now that the whole test runs in the kernel (20 ms through the Python binding). Rejecting the origin for the same germ takes 0.3 ms. In an exhaustive search, the five priority candidates of this germ are decided in 0.3 s; before, the search had not decided them after 60 s.

~~~sh
python3 benchmarks/groebner.py --singular --repeats 5
python3 benchmarks/groebner_hard.py --singular --gpu
~~~
