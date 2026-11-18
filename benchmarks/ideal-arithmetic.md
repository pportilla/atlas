# Ideal and quotient arithmetic timings

## Primary decomposition over QQ

The benchmark times an irredundant primary decomposition together with its associated primes. In each trial it compares Atlas against both `primdecGTZ` and `primdecSY` from Singular's `primdec.lib`. For every input, Singular independently checks the intersection, tests that each component is primary with the reported radical, and verifies that dropping any component changes the intersection. This permits different valid choices for an embedded primary component.

Run the main suite and the additional batch of 32 coordinate-mixed product ideals with the following commands. The optional `--cpu` argument pins both applications to a chosen core on Linux.

```sh
python3 benchmarks/primary_decomposition.py --repeats 7 --cold-repeats 7 --timeout 120
python3 benchmarks/primary_decomposition.py --stress-probe --repeats 3 --timeout 15
```

Warm timings exclude process startup. Atlas decodes the sparse input and serializes the returned generator lists within the timed call. Singular starts with its ring and input ideal loaded. Each trial discards one warm-up and runs a fresh decomposition. Fresh-command timings include startup, parsing and full result output for both systems. The run order rotates and Singular's random seed changes each trial. [Benchmark program and inputs](primary_decomposition.py)

On a Ryzen 7 6800H with both programs pinned to CPU 5, Atlas beat both Singular 4.4.1 procedures in all 91 warm trials and all 91 fresh CLI trials. The table gives seven-trial medians in milliseconds. Each Singular column uses the lower of the two procedure medians for that case and timing mode. [Full input ideals, individual trials, versions and binary fingerprints](primary-decomposition-ryzen-6800h.json)

| Ideal | Atlas warm | Singular warm | Atlas CLI | Singular CLI |
| --- | ---: | ---: | ---: | ---: |
| Factorable principal ideal | 0.151 | 0.282 | 6.47 | 22.30 |
| Monomial ideal with embedded components | 0.064 | 18.873 | 6.28 | 42.03 |
| Monomial primary ideal | 0.053 | 0.732 | 6.89 | 23.68 |
| Algebraic graph | 0.185 | 7.216 | 7.21 | 31.95 |
| Finite nonreduced quadratic points | 0.753 | 2.590 | 7.92 | 26.00 |
| Mixed line and point | 0.163 | 5.876 | 7.43 | 30.68 |
| Embedded curve | 0.422 | 7.741 | 8.00 | 35.76 |
| Exceptional fiber | 0.766 | 9.313 | 15.62 | 73.64 |
| Katsura-3 | 4.928 | 6.619 | 20.55 | 56.68 |
| Coordinate-mixed common factor | 1.246 | 26.371 | 8.24 | 41.25 |
| Coordinate-mixed comaximal product | 6.780 | 25.088 | 9.96 | 34.95 |
| Two-parameter tensor product | 2.273 | 17.588 | 8.89 | 38.52 |
| Nonreduced number-field fiber | 1.186 | 6.629 | 7.39 | 28.73 |

The first nine ideals were chosen before measuring. The initial Python-launched CLI lost every fresh trial, so the CLI now calls the same C++ computation directly. Changes of coordinates then exposed two expensive product ideals. Extracting a common factor or splitting a comaximal product reduced one 30-second timeout and one 2-second computation to millisecond computations. Two further ideals exposed costs in contraction and in representing nilpotents. Using a smaller saturation polynomial, discarding lower-dimensional branches of certified complete intersections, and lifting powers of graph equations improved those cases. All original inputs were retained and retimed. [Baseline and intermediate measurements](primary-decomposition-baseline-ryzen-6800h.json)

The separate 32-ideal stress suite used the same binaries. Atlas beat both Singular procedures on all 96 fresh commands. Across those ideals, median CLI speedups ranged from 2.63× to 6.75×. Singular verified all 64 returned components. [Stress inputs and timings](primary-decomposition-stress-ryzen-6800h.json)

## Normalization of affine algebras over QQ

The benchmark computes the entire normalization over the rationals, including the defining equations and the map from the input algebra. It times both Singular's default `normal(I)` and `normal(I,"prim")` from `normal.lib`. The second form provides a fractional module on each minimal prime. Outside the timed region, the checker uses that module to compare both inclusions, the source maps, and the defining relations supplied by Atlas. [Benchmark program and exact inputs](normalization.py)

```sh
python3 benchmarks/normalization.py --repeats 7 --cold-repeats 7 --timeout 120
```

Each warm trial discards one warm-up and excludes process startup. Atlas reads the sparse input and serializes its complete result in the timed call. Singular starts with the input ideal and library loaded. Fresh trials measure each CLI from launch through full output, including construction of relations and coordinate maps. The programs rotate their run order and Singular's random seed changes each trial. On Linux, `--cpu` pins both programs to a selected core.

On a Ryzen 7 6800H pinned to CPU 5, Atlas beat both procedures in Singular 4.4.1 in all 161 warm trials and all 161 fresh CLI trials. Across the 23 inputs, the median speedup over the faster Singular procedure ranged from 8.11× to 335.32× warm and from 5.20× to 17.18× through the CLI. Singular independently verified 27 normalized components. The table gives seven-trial median wall times in milliseconds on representative inputs. [All 23 inputs, both Singular procedures and individual timings](normalization-ryzen-6800h.json)

| Input | Atlas warm | Faster Singular warm | Atlas CLI | Faster Singular CLI |
| --- | ---: | ---: | ---: | ---: |
| Normal quadric cone | 0.141 | 6.016 | 7.948 | 48.103 |
| Cusp, $y^2=x^3$ | 0.072 | 13.910 | 8.157 | 54.726 |
| Monomial space curve | 0.374 | 19.336 | 7.914 | 62.170 |
| Cyclic surface with two branch divisors | 0.325 | 54.638 | 6.230 | 90.927 |
| Noncyclic plane quartic | 3.713 | 47.999 | 10.740 | 82.421 |
| Plane and translated space curve | 5.062 | 83.643 | 18.562 | 156.745 |
| Nonmonic surface | 7.851 | 63.699 | 14.076 | 93.908 |

Thirteen inputs were fixed before measuring. The initial general kernel won all 39 warm comparisons, while the Python-launched CLI lost all 39 fresh comparisons. The native CLI removed that startup cost. Ten further inputs exposed expensive coordinate changes, repeated integral extensions, and nonmonic equations. All 23 inputs were rerun after the optimizations. [Baseline, timeouts and intermediate measurements](normalization-baseline-ryzen-6800h.json)

## Localization at a prime over QQ

Each input specifies a rational polynomial ring, an ideal $I$, a candidate prime $P$, and optionally a fraction $f/s$. Each system verifies primality, computes $J=IR_P\cap R$, and checks fraction membership in $IR_P$ and invertibility in $R_P$. Singular independently checks both ideal containments. It also checks that each returned generator $g$ has $(I:g)\not\subseteq P$, that $I\subseteq J$, and that every associated prime of $J$ lies in $P$. These conditions certify the contraction, including embedded components. [Benchmark and exact inputs](localization.py) · [Tests in both monomial orders](../tests/test_localization_singular.py)

The three Singular procedures use direct calculations for zero, unit, monomial, principal, height-one and finite-dimensional cases. For the remaining ideals, two filter primary components using `primdecGTZ` or `primdecSY` and the third uses Ishihara's `locPrime`. Singular loads libraries on demand. The warm test starts with parsed inputs in Singular. Atlas reads sparse polynomial rows and serializes the complete result in its timed native call. Each warm sample averages five full computations after one discarded warm-up. The fresh-command test times one process from launch through its complete output. Both tests rotate the execution order. [Pinned locPrime source](https://github.com/IshiharaYuki/localprime/blob/5624175785c7982358d9d31f89d29345bad1680b/locprime.lib)

Run from the repository root in the native build environment. The optional `--locprime` argument adds the third procedure. The driver checks its source hash and adapts the saturation call to Singular 4.4.

```sh
make -C native -f groebner.mk build/localization
curl -fsSL https://raw.githubusercontent.com/IshiharaYuki/localprime/5624175785c7982358d9d31f89d29345bad1680b/locprime.lib -o /tmp/atlas-locprime.lib
python3 benchmarks/localization.py --cpu 5 --repeats 7 --batch 5 --cold-repeats 7 \
  --locprime /tmp/atlas-locprime.lib --output /tmp/atlas-localization.json
```

On the Ryzen 7 6800H, with Atlas and Singular 4.4.1 pinned to CPU 5, Atlas had the lower median on all 22 cases in both modes. Compared with the fastest of the three Singular procedures on each case, the median per-case speedup was 4.70× warm (range 1.48–63.34×) and 3.51× from a fresh CLI (range 1.28–6.69×). Atlas won 152 of 154 paired warm samples and all 154 fresh samples. One warm sample each on the unit-ideal and height-one inputs ran less than 1% slower than the fastest Singular procedure. All three procedures completed and verified every case. [Individual timings, inputs and build hashes](localization-ryzen-6800h.json)

Seven-sample medians in milliseconds follow.

| Input | Atlas warm | Fastest Singular warm | Atlas CLI | Fastest Singular CLI |
| --- | ---: | ---: | ---: | ---: |
| Principal cusp | 0.064 | 0.206 | 5.583 | 7.929 |
| Hidden common factor | 0.944 | 16.330 | 6.690 | 44.743 |
| Nonreduced conjugate points | 0.893 | 4.599 | 5.880 | 23.494 |
| Height one, no fraction query | 0.185 | 0.274 | 6.083 | 7.792 |
| Dense change of generators | 1.172 | 2.551 | 6.821 | 30.152 |
| Mixed curve in four variables | 1.754 | 15.900 | 7.745 | 42.073 |
| Translated nonreduced point | 0.212 | 1.119 | 5.604 | 19.434 |

The first implementation used general primary decomposition and the Python CLI launcher. Against the final Singular procedures it won 15 of 21 completed warm case medians and none of the fresh-command medians. One further height-one case exceeded the 120-second native limit. All 22 inputs were retained, including the six that exposed expensive common factors, changed generators, and nonreduced fibers during development. [Baseline timings and timeout](localization-baseline-ryzen-6800h.json)

## Ordinary and symbolic powers over QQ

The benchmark compares the complete reduced Gröbner basis in the requested order. It covers ideals whose two powers agree, embedded components removed by a symbolic power, a singular non-complete-intersection prime, mixed-dimensional unions, changes of generators, nonreduced ordinary inputs and the zero and unit ideals. It verifies both ideal containments and matches every monic generator against Singular. Separate tests run the corpus in both lex and grevlex orders. [Inputs and benchmark driver](ideal_powers.py) · [Singular agreement and edge cases](../tests/test_ideal_powers_singular.py)

For ordinary powers, Singular uses its built-in ideal power and `std` with reduced bases enabled. The benchmark applies the convention $I^0=(1)$ in both systems. The symbolic procedures check radicality and use direct calculations for squarefree monomials, complete intersections and finite reduced algebras. General inputs compare primary-component filtering with Jacobian saturation on equidimensional inputs or a second primary-decomposition algorithm on mixed-dimensional inputs. The procedures split off suitable common factors. Atlas is compared with the fastest applicable procedure on each case.

The **warm computation** starts with parsed polynomials and returns the entire native ideal in each system. The separate **warm JSON** boundary includes formatting the complete result, including all the monic polynomials. Atlas's JSON time also includes decoding sparse input rows. **Fresh CLI** runs include process startup, parsing, arithmetic and output. A warm-up is discarded before each warm sample. Both systems run on the same CPU and their execution order rotates.

```sh
make -C native -f groebner.mk build/ideal_power build/symbolic_power build/bench_ideal_powers
python3 benchmarks/ideal_powers.py --cpu 5 --batch 5 --repeats 7 --cold-repeats 7 \
  --warm-json --output /tmp/atlas-ideal-powers.json
```

On the Ryzen 7 6800H with both systems pinned to logical CPU 5, Atlas had the lower seven-trial median on all 28 inputs at all three timing boundaries. The suite contained 12 ordinary and 16 symbolic powers. The comparisons use Singular 4.4.1. [All inputs, timings, method names and binary hashes](ideal-powers-ryzen-6800h.json)

| Timing boundary | Atlas wins in paired trials | Median per-case speedup | Range of per-case speedups |
| --- | ---: | ---: | ---: |
| Warm ideal computation | 188/196 | 8.73× | 1.15–84.30× |
| Warm result including JSON | 195/196 | 3.28× | 1.16–21.54× |
| Fresh CLI | 192/196 | 1.64× | 1.14–4.30× |

For example, the symbolic cube of the union of a plane and a space curve took 2.199 ms warm in Atlas and 5.420 ms with the faster Singular procedure. The baseline Atlas implementation took about 767 ms on that input. As a separate measure of transport cost, the complete Atlas sparse request with JSON output beat the unformatted Singular ideal computation on 19 of 28 case medians.

The baseline completed 27 of the 28 cases. The mixed cusp and plane at exponent four exceeded the 10-second native cutoff. Among the completed cases Atlas had the lower median on 21 warm computations and 26 fresh commands. The later implementations kept the products in their native polynomial representations, used fraction-free elimination on small certified systems, and factored out common hypersurface components before computing symbolic powers. All original cases and all five harder inputs were retained. [Baseline measurements](ideal-powers-baseline-ryzen-6800h.json)

## Arithmetic in declared quotient rings over QQ

The 27 cases cover all 11 operations, including negative powers, equality and unit tests. The inputs include a nonreduced algebra, a reduced grid, a number field, positive-dimensional quotients and the zero ring. Each result is checked against an actual Singular `qring`. The check compares the full monic reduced basis and the canonical representative as ambient polynomials. Inverse and division cases also check the multiplicative identity. The accompanying tests repeat all cases in lex and grevlex and check wide input and nonunit errors. [Benchmark driver and inputs](quotient_ring.py) · [Agreement tests](../tests/test_quotient_ring_singular.py)

| Timing boundary | Work included |
| --- | --- |
| Prepared quotient | An Atlas C++ `Ring` and a Singular `qring` each hold the prepared basis. Operands are parsed. Both return a typed, canonical result. Ring declaration has no separate arithmetic to time. |
| Construct and compute | Both systems start with parsed ambient polynomials, build a new quotient and compute the same typed result for every iteration. |
| Fresh CLI | Each sample launches the public Atlas CLI or Singular, parses the expressions, constructs the quotient and prints JSON with the ring, reduced basis and result. Atlas also produces its execution and replay record. |

Singular computes the reduced defining basis with `std`, the representative with `reduce` and inverses with `division`, including its remainder check. For nonzero powers the driver measures both Singular's built-in power and modular repeated squaring and selects the faster median per case and timing boundary. Each warm sample discards one warm-up and times 80 evaluations. Singular uses a microsecond timer and Atlas uses a monotonic nanosecond timer. The comparison rotates the two systems and power methods between trials on the same logical CPU.

```sh
make -C native -f groebner.mk build/quotient_ring build/bench_quotient_ring
python3 benchmarks/quotient_ring.py --cpu 1 --batch 80 --trials 7 --cold-trials 7 \
  --output /tmp/atlas-quotient-ring.json
```

On the Ryzen 7 6800H, pinned to logical CPU 1, Atlas had the lower median for every case at each applicable boundary. It also completed all 553 paired trials faster than Singular 4.4.1. The speedup for a case divides its Singular median by its Atlas median. [Full timings, inputs, results and binary hashes](quotient-ring-ryzen-6800h.json)

| Timing boundary | Faster case medians | Faster paired trials | Median per-case speedup | Per-case range |
| --- | ---: | ---: | ---: | ---: |
| Prepared quotient | 25/25 | 175/175 | 4.61× | 1.31–122.77× |
| Construct and compute | 27/27 | 189/189 | 8.14× | 1.41–51.88× |
| Fresh CLI | 27/27 | 189/189 | 1.36× | 1.27–1.42× |

The first implementation won 17 of 25 prepared case medians, 24 of 27 with construction included and all 27 fresh command medians. The final run kept the same cases. Packed monomial reduction and multiplication improved the dense reductions and the twisted-cubic product. Arithmetic on the prepared twisted cubic fell from 21.14 to 5.25 µs, compared with 7.64 µs in the final Singular run. Exact multiplication-matrix inversion, Bézout certificates and univariate modular powering improved the unit and power cases. [Five-trial baseline on the initial binaries](quotient-ring-baseline-ryzen-6800h.json)

## Ideal operations on the Radeon 680M

For the Katsura-6 ideal $K$ and an independent $z$, the following run used OpenCL on an AMD Radeon 680M. Each row replayed 14 primes in three CPU batches and certified one basis over $\mathbf Q$ on the CPU. The four ideal results each had 41 generators and agreed with Singular 4.4.1 by mutual reduction.

| Request | OpenCL trace pairs | OpenCL batches |
| --- | ---: | ---: |
| Eliminate $z$ from $K+(z)$ | 150 | 4 |
| $K\cap(z)$ | 482 | 6 |
| $(zK):(z)$ | 150 | 4 |
| $(zK):z^\infty$ | 482 | 6 |
| Dimension of the Katsura-6 quotient ring ($0$) | 150 | 4 |

For $(zK):(z)$, dividing first reduced the warm median request time from 29.12 ms to 18.61 ms on this device. Eight alternating pairs of requests returned the same sequence of 41 generators. For Katsura-5 the corresponding times were 11.87 ms and 7.14 ms, with 22 generators.

## Multivariate polynomial GCD over QQ

Seven trials on a Ryzen 7 6800H compared the public `gcd(f,g)` call to Singular 4.4.1 on nine Liao–Fateman examples and one case with a 401-bit coefficient. Each system computed the expected monic polynomial in grevlex order. The largest dense case had 6,561 terms per input in eight variables and a 256-term GCD. On one logical CPU, the median Atlas call fell from 159.66 ms to 8.02 ms. First use on fresh expanded operands took 10.29 ms. Singular took 24.78 ms per warmed call. [Benchmark and reproduction commands](polynomial_gcd.py) · [Original samples](polynomial-gcd-ryzen-6800h.json) · [Updated samples](polynomial-gcd-prepared-ryzen-6800h.json)

Atlas had the lower median on all ten cases when both systems timed repeated calls in batches. The original comparison times one Atlas call against Singular's per-call mean in a warmed batch. In that comparison the two pinned coprime cases still favored Singular. At three variables the medians were 10.91 µs for Atlas and 6.54 µs for Singular. At six variables they were 15.37 µs and 12.01 µs. With default scheduling, the three-variable fully dense case also favored Singular (53.73 µs and 33.83 µs). [Default-scheduling samples](polynomial-gcd-prepared-unpinned-ryzen-6800h.json)

## Polynomial GCD on the Radeon 680M

The [physical-device test](../tests/test_polynomial_gcd.py) uses $hA$ and $h(A+1)$, with $h=3x^2+4yz-5z+1$ and a 401-bit coefficient in $A$. Each input has 25 terms. Since $A$ and $A+1$ are coprime, the expected grevlex result is $h/3$. On the Radeon 680M, OpenCL replayed 62 primes in five batches. The CPU certified one basis and performed both exact divisions. After one warm-up per mode, median dispatch time over three runs was 1.15 ms for direct FLINT and 10.24 ms for the GPU-assisted calculation. Automatic mode uses the native CPU kernel for this operation.

## Ring homomorphisms

The session benchmark applies maps from $\mathbb Q[x,y]$ to $\mathbb Q[u,v,w]$. For each input $f+ix$, it checks that the image equals $h(f)+i h(x)$. Five alternating trials used 64 distinct inputs per map per trial after warm-up, on logical CPU 3 of the Ryzen 7 6800H. Medians are microseconds per image, including its construction. Both implementations start from a prepared map and operands. [Cases and driver](ring_maps.py) · [Samples and timing conditions](ring-maps-ryzen-6800h.json) · [Session agreement checks](../tests/test_session_maps.py)

| Map | Images of $x,y$ | Atlas | Singular 4.4.1 |
| --- | --- | ---: | ---: |
| inclusion | `u, v` | 6.71 | 2.97 |
| nonlinear | `u+v^2, u` | 85.52 | 4.39 |
| collision | `2*u/3, u` | 58.77 | 4.61 |

The recorded Atlas medians are slower for all three maps. To reproduce the comparison, run `python3 benchmarks/ring_maps.py --trials 5 --count 64` on one logical CPU.
