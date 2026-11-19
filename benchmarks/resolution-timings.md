# Resolution and zeta timings

## Same-host resolution and zeta functions

For singular-line examples, the reports include the chosen centres, the chart
counts, and separate timings for solving and checking the saved charts.
[Sheared centres](sheared-centre-resolution.md) ·
[Inclined singular line](inclined-centre-resolution.md) ·
[Exhaustive search and Singular's two-chart resolution](exhaustive-search-resolution.md) ·
[Discovering smooth linear and curved centres](reduced-centre-resolution.md)

[The headless comparison](resolution_zeta.py) runs the public Atlas resolver and the installed Singular libraries on three isolated surface germs over the rationals. Each solver starts from the polynomial and returns the local topological zeta function and the local monodromy invariant at the origin. The inputs cover an ordinary double point, the cone over a smooth plane cubic, and the A₆ surface. The driver checks both exact rational functions on every run.

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

### Local monodromy on nearby strata

The table gives median wall times in milliseconds on an AMD Ryzen 7 6800H
with Python 3.14.4 and Singular 4.4.1. Run
`taskset -c 5 python3 benchmarks/nearby_monodromy.py --trials 9` to repeat
the comparison. Each case has one warm-up and nine measured pairs on one
logical CPU, with alternating solver order. [Benchmark driver](nearby_monodromy.py)

Both systems use the same rational polynomial. Atlas starts after the origin
resolution and times the additional call that identifies the nearby loci and
returns their factored zetas. Singular receives the indicated normal coordinates,
sets the remaining coordinates to one, and runs `gmssing.lib`'s `monodromy` on
the resulting isolated transverse germ. Its timing includes the request to a
warm Singular process and delivery of the Jordan data. Outside the timed
interval the driver converts that data to the unreduced Lefschetz convention
and checks the multiplicity of every root against Atlas on the indicated locus.

| Polynomial | Normal coordinates | Atlas (ms) | Singular (ms) |
|---|---|---:|---:|
| $x^4y^3$ | $x$ | 0.124 | 2.831 |
| $x^3+y^4z$ | $x,y$ | 0.155 | 3.593 |
| $x^2+y^2z+y^3$ | $x,y$ | 0.162 | 2.634 |
| $xy+z^5t$ | $x,y,z$ | 0.155 | 3.521 |
| $x^2+y^2+tz^2+z^3$ | $x,y,z$ | 0.151 | 2.997 |
| $x^2+y^3+y^4$ in $x,y,z$ | $x,y$ | 0.159 | 2.951 |

### Topological zeta reduction from certified strata

The native reducer builds each stratum's denominator from the components that meet there. It adds the resulting rational functions in a balanced tree, cancelling common factors at each addition. The measurements below start with the same certified strata and include producing the factored expression, expanded numerator and ordered pole records. Each of the seven inputs reproduced its complete topological zeta record. [Raw samples and record hashes](topological-reduction-timings.json)

| Charts | Components | Strata | Previous median | Balanced median | Speedup |
|---:|---:|---:|---:|---:|---:|
| 598 | 170 | 604 | 37.20 ms | 7.72 ms | 4.82× |
| 3,915 | 566 | 3,860 | 2,210.52 ms | 77.17 ms | 28.64× |

The 598-chart case used seven alternating pairs and the 3,915-chart case used three. For a smooth germ, a node and a cusp the absolute change ranged from a 0.1 μs saving to a 1.7 μs increase. These CPU measurements cover reduction after resolution.

After installation, the full 3,915-chart CLI resolve with both zeta functions took **6.703 s → 4.763 s** across three alternating fresh-process pairs. The 598-chart warm API resolve took **0.398 s → 0.358 s**. Both versions returned identical charts, atlases, labelled complexes and zeta functions. [Post-install samples and checksums](topological-reduction-timings.json)

### Verifying the 3,915-chart fan

The five-variable embedded solve with both zetas took **5.535 s → 2.348 s** (2.36×) across three alternating fresh CLI pairs on one Ryzen 7 6800H core. Both builds used the balanced zeta reducer. The timed command includes startup and writing the full 19.27 MB JSON result. Repeated calls to the public solver in a warm interpreter took **4.413 s → 1.837 s**. [Command, samples and output hashes](fan-verification-3915.json)

| Phase in a separate full CLI profile | Before | After |
|---|---:|---:|
| Euler strata on the owned charts | 1.222 s | 0.083 s |
| Normal crossings | 0.977 s | 0.090 s |
| Remaining fan work, including pullbacks, chart records and incidence | 2.051 s | 1.381 s |
| Newton chart generation | 0.275 s | 0.323 s |
| Zeta reduction | 0.083 s | 0.081 s |
| Other solver work | 0.027 s | 0.028 s |
| JSON encoding, decoding and formatting | 0.284 s | 0.257 s |
| Startup, transfers and remaining CLI work | 0.253 s | 0.254 s |
| **Total** | **5.173 s** | **2.497 s** |

Chart verification accounted for 82% of the original profile. The kernel now certifies transverse coordinate divisors and one hypersurface with a single ideal test. It computes eligible Euler strata on disjoint coordinate tori. The verifier also recognizes canonical projection text and uses insertion ranks to order faces. Replaying the saved equations reproduced every attachment among the 565 vertices and 9,972 higher-dimensional cells, the 3,860 nonzero strata and both complete zeta records. [Proofs for the native chart operations](../native/resolution-algorithms.md#9-operations-for-the-blowups)

#### Pullbacks and Jacobian orders on the regular fan

The 1.381 s of remaining fan work in the preceding profile includes verifying chart maps, constructing strict equations, and assembling the labelled complex. In a fresh profile of that build, general native chart calls took 538 ms of 1.241 s. Python spent another 236 ms decoding and formatting the strict equations. Timers inside the native operation attributed 330 of 509 ms to computing polynomial Jacobian determinants.

The kernel now applies the common linear change to the original polynomial once, then processes the 3,915 verified cones in 62 blocks. Each ray gives the orders of the pulled-back equation and volume form. The kernel emits the strict polynomial in both sparse and display form. All 3,915 crossings and Euler calls still use the verified equations. [Derivation](../native/resolution-algorithms.md#95-affine-chart-maps)

| Remaining fan work in a separate matched profile | Before | After |
|---|---:|---:|
| Native chart calculation and transfer, including strict-equation text in the new build | 731 ms | 104 ms |
| Python strict-equation decoding and formatting, or reading native result headers | 319 ms | 10 ms |
| Projection formatting and verification | 134 ms | 124 ms |
| Collecting faces and assembling the labelled complex | 131 ms | 77 ms |
| Certifying the fan cover | 84 ms | 95 ms |
| Preparing owned fibres and divisor ideals | 71 ms | 78 ms |
| Other chart setup and records | 251 ms | 210 ms |
| **Total excluding crossings and Euler strata** | **1.721 s** | **0.698 s** |

Five alternating pairs with timers only at the fan, crossings and Euler boundaries gave median residual times of **1.880 s → 0.789 s** (2.38×). The final fresh CLI median was **2.889 s → 2.042 s** over five further alternating pairs on the same shared host. Each pair included the full JSON transfer and both zetas. Warm solver calls in separate paired runs fell from 2.192 s to 1.154 s. Reading the saved equations reproduced the labelled complex, every codimension-one attachment, all 3,860 nonzero strata and the complete monodromy and topological zeta records. [Raw samples, disjoint profiles and record hashes](fan-chart-pullback-3915.json)

#### Projection formatting and verification on the regular fan

On this input, 15,660 of the 19,575 projection coordinates are single monomials. The previous formatter sorted all 97,875 possible terms, including 74,385 with zero coefficients. Replaying the chart data attributed 38.2 ms to rendering powers and joining terms, 20.9 ms to sorting, and 15.7 ms to filtering and formatting coefficients. Only 81 coordinates required comparison of alternate polynomial spellings.

The native chart batch now constructs the projections from the nonzero entries of the common linear map and the ray matrix. Six alternating full solves on the same core measured **106.5 ms → 49.5 ms** for projection formatting and verification. The new estimate includes 24.1 ms of validation and setup and 25.4 ms for the additional native computation and transfer. The latter is the median difference from replaying all 62 real chart batches through both operations in 19 alternating pairs. The historical 124 ms lies within the 100–162 ms range of the six baseline solves.

The **full CLI median fell from 1.571 s to 1.494 s** across 36 alternating fresh-process pairs, including both zetas and writing the full 19.27 MB JSON result. The median paired saving was 66.8 ms. The 31 pairs that met the predeclared low-contention criterion saved 64.3 ms. An earlier 12-pair run on a saturated host had a paired interval spanning zero. Saved-chart replay recovered all 565 vertices, 9,972 higher-dimensional cells, 31,574 codimension-one attachments, 3,860 nonzero strata and both complete zeta records. [Paired measurements, command and checksums](fan-projection-3915.json)

### Newton coordinate choices

The native kernel returns the axial intercepts, their volume product, and the expansion budget estimates for every donor coordinate in one call. Atlas keeps this result with each coordinate trial. A shear that changes the polynomial gets fresh statistics before the trial is ranked.

| Intrinsic resolution | Terms | Native calls | Time in axial statistics | Full CLI median | Median paired saving |
|---|---:|---:|---:|---:|---:|
| Five variables, 13 charts | 3,065 | 10 → 1 | 32.6 → 3.4 ms | 472 → 448 ms | 28 ms |
| Seven variables, 13 charts | 3,010 | 14 → 1 | 56.9 → 4.0 ms | 523 → 468 ms | 53 ms |

The CLI timings include startup, both zetas and JSON output, with 15 alternating fresh-process pairs per input on the same CPU. Both labelled complexes and both zetas matched in every pair with SymPy imports blocked. A separate comparison reproduced all 651 ordered coordinate choices on 154 inputs, including trials that exhausted the eight-candidate limit and shears whose prospective axial term cancelled. [Inputs, commands, raw timings and record hashes](newton-axial-reuse.json)

#### Native chart equations and intrinsic orbit restrictions

On the two 13-chart inputs above, the largest cost after axial statistics was decoding each full strict transform and sending its exponents back to compute the orbit sections. The verifier now returns a sparse equation containing every term that survives on at least one exceptional coordinate hyperplane. The existing orbit test then restricts those terms to each individual face. Five variables require 57 of 39,845 chart terms. Seven variables require 85 of 39,130. The complete strict equations still supply the published charts and the zeta computation. The producer formats those equations directly from native chart exponents in graded lexicographic order. [Boundary restriction proof](../native/resolution-algorithms.md#95-affine-chart-maps)

| Median time | Five variables | Seven variables |
|---|---:|---:|
| Decode strict terms and request orbit restrictions | 228.7 → 1.0 ms | 266.6 → 2.1 ms |
| Format producer chart equations and adapted input | 167.1 → 28.7 ms | 163.6 → 28.8 ms |
| Full CLI, 29 paired runs per input | 863 → 478 ms | 935 → 524 ms |
| Median paired full CLI speedup | 1.81× | 1.78× |

The stage timings use worker CPU clocks across 15 separate pairs. The full CLI timings include startup, both zetas, formatting JSON and transferring the entire result. Every timed run matched the previous mathematical record. The two CLI series used the same native libraries on both sides and alternated old and new code. The first series shared a busy core. The second measured 716 → 401 ms in five variables and 591 → 355 ms in seven. The largest remaining measured worker operation is native fan verification and transfer, taking 106 and 115 ms respectively in the stage profile. Replaying the saved charts reproduced all six labelled incidences across both inputs, all ten nonzero strata and both complete zeta records per input. [Commands, individual samples, stage timings and exact output hashes](newton-chart-equations.json)

#### Formatting the independently verified fan equations

In the previous profile, fan verification and transfer took 106 ms in five variables and 115 ms in seven. Separate timings inside the kernel showed that formatting the full strict equations accounted for 54% of this stage in both cases. The verifier now sorts and formats its own verified exponents with the producer’s native formatter. It computes the exact rational coefficient strings once per block of cones. [Derivation](../native/resolution-algorithms.md#95-affine-chart-maps)

| Median time | Five variables | Seven variables |
|---|---:|---:|
| Strict equation formatting, instrumented native replay | 33.43 → 10.60 ms | 33.48 → 9.60 ms |
| Complete fan stage, measured in the resolver | 100.10 → 58.59 ms | 104.68 → 62.88 ms |
| Full CLI wall time, original → final build, 15 pairs | 210.5 → 191.4 ms | 236.2 → 217.3 ms |
| Full CLI wall time saved, median within those pairs | 19.1 ms (1.10×) | 19.8 ms (1.09×) |

The native replay alternated 17 pairs on the three actual request packets per input. Before the change, the general polynomial printer took 26.6 and 25.2 ms, and converting to graded order took 7.0 and 8.3 ms. On the optimized build, writing the complete sparse equations took 10.3 and 10.7 ms, while parsing the input took 8.2 and 7.3 ms. The resolver profile used 14 separate pairs and included packing, copying, decoding, release and the same response hash on each build. Copying and decoding the reply took about 2 ms.

The first full CLI comparison used 44 alternating fresh-process pairs per input on a shared host, from startup through both zetas and complete JSON output. The median paired savings were 36.5 ms (1.12×) and 30.8 ms (1.10×). After a change to release temporary FLINT buffers on allocation failure, 15 fresh pairs per case compared the final build directly with the original. The final build was faster in 14 of 15 pairs on each input. Every timed run matched the reference mathematical record. Independently replaying the saved equations reproduced all 13 charts per case, both labelled intrinsic complexes, every incidence and both full zeta records. All six native replies also matched byte for byte. [Commands, phase timings, raw pairs and verification records](newton-fan-verification-format.json)

### General blowups and intrinsic chains

The following measurements compare the previous native release with the sparse chart migration on the same headless CPU. Each sample used the public CPU resolver with both zeta functions enabled. Each pair ran in a fresh interpreter, then repeated the same request in that interpreter. The five pairs per input alternated the release order on CPU 4. The zeta expressions, number of cells and completion status agreed for every timed request. Cold timings include interpreter startup and worker initialization. These figures compare Atlas versions.

| Input and mode | Previous cold | Native cold | Previous warm | Native warm |
|---|---:|---:|---:|---:|
| $(x+y)^2+z^3+x^3$, embedded | 623 ms | 215 ms | 390 ms | 160 ms |
| $(xy-z^2)^2+x^5+y^5$, embedded | 718 ms | 296 ms | 479 ms | 229 ms |
| $x^7+y^2-z^2$, intrinsic with embedded companion | 244 ms | 39 ms | 15.7 ms | 7.6 ms |

The default-mode audit traces plane, chain, graph and canonical-centre examples. The current driver uses native inputs by default. The `--symbolic-inputs` option exercises the compatibility adapter for existing Python callers. [Production path inventory](symbolic-paths.md)

## Affine Euler characteristics on the double-quadric resolution

On the Ryzen 7 6800H, Atlas computed the compactly supported Euler characteristic of the affine chart $(zt-u^2)^5+1+z^5+t^5+u^5=0$ as 145 in three trials. Median calculation CPU time was 2.68 ms, and median fresh-process wall time was 61 ms. Singular 4.4.1 returned 145 in one trial with `chern.lib::EulerAff`. Its other two trials reached the 15-second wall limit.

Both systems also found $-99$ on $x^{11}+y^{11}=1$ and $16$ on $x^2y^2+x^2z^2+y^2z^2=1$. For those two controls Atlas used less calculation CPU time (1.53 and 1.76 ms versus 9 and 31 ms) and had higher fresh-process wall times (62 and 63 ms versus 29 and 51 ms). [Comparison script](euler_lattice.py) and [all trials and conditions](euler-lattice-ryzen-6800h.json).

For the contact fourfold the strict equation restricts on its most expensive chart to $A h^2-(q^3+u^3+1)h+1=0$, with $q=zt-u^2$ and $A=2q^6+3z^6+5t^6+7u^6+1$. Atlas obtained Euler characteristic $-622$ in all three trials, with median 0.463 s calculation CPU and 0.524 s fresh-process wall. Singular reached its 15-second cutoff in all three trials. For the control $xy^2+y+1=0$, both returned zero. The median process wall times were 62 ms for Atlas and 28 ms for Singular. [All contact and control trials](euler-contact-ryzen-6800h.json).

## Shared torus verification

Restricting each chart to its coordinate hyperplanes once reduced the median time for a complete warm intrinsic resolution. Seven timed runs per version and case alternated between the original and updated verifier. Every run checked the same owned atlas and labelled incidence data. Values are milliseconds on the Ryzen 7 6800H host. The OpenCL runs used its Radeon 680M. [Raw samples, input polynomials and version details](torus-section-timings.json)

| Workload | CPU before → after | OpenCL before → after |
|---|---:|---:|
| 191 charts, 409 monomials | 1298.3 → 1104.8 | 1182.8 → 1014.3 |
| 598 charts, four variables | 453.2 → 395.0 | 381.0 → 338.9 |

For the 191-chart intrinsic CPU solve, the native kernel now filters the strict transform to its value and first derivatives on the owned fibre before checking crossings. Seven alternating before-and-after pairs gave median solve times of **1.355 s → 0.568 s** in fresh processes and **0.601 s → 0.276 s** after one warm-up. Each solve returned identical chart equations and all 39 labelled edges. These solve timings exclude imports and output serialization. An independent full CLI comparison, including both, measured **1.670 s → 0.806 s** across seven pairs. After installation, seven more full CLI pairs gave **1.454 s → 0.647 s** with the same labelled complex. The live verifier checked 325 strict-transform terms across 191 charts. All timings in this comparison used the intrinsic resolution with optional zeta calculations disabled. [Samples and output hashes](fan-crossing-timings.json)

The installed verifier also reduced the median warm CPU time for the 598-chart reference from **392.10 ms to 355.45 ms** across five alternating pairs. All five runs improved and reproduced the same chart equations and labelled intrinsic complex. [Samples and geometry hash](fan-crossing-timings.json)

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

The intrinsic resolution of the A₆ surface produces six genus-zero curves of self-intersection −2 in a chain. The driver checks the vertices, their degrees and the connected path.

For the four-variable binomial, Atlas used eight smooth codimension-two blowups. Blanco and Frühbis-Krüger reported 9 seconds and 240 final charts for Singular’s `resbinomial` on this input. Their general `resolve` routine exceeded 7,200 seconds. The published run used a 2.13 GHz Core i3-330M with 4 GB of RAM, and its timing covered a different implementation and output. A same-machine speed ratio remains unknown. [Study, §5, Table 1](https://arxiv.org/pdf/1109.1655#page=10)

To reproduce after `make gpu`, run `python3 benchmarks/published_comparison.py --gpu-target --output /tmp/atlas-reference-cases.json`. Omit `--gpu-target` for a CPU-only machine.

## Skipping saturation during resolution

Atlas can return the original ideal when it proves the multiplier avoids every associated prime, including embedded primes. The bounded certificate uses a verified decomposition and falls back to saturation when the proof is inconclusive. On the three B9 chart inputs, nine isolated trials gave median Atlas certificate times of 12.4, 23.8 and 23.2 ms. Singular's component checks took 19.6, 77.6 and 239.4 ms with the inputs already loaded. Direct Singular saturation and ideal equality were faster than the Atlas certificate on all five small controls. [Inputs and driver](component_regularity.py) · [Individual trials and conditions](component-regularity-ryzen-6800h.json)

In the full intrinsic resolution of the benchmark surface, those three checks took 35.8, 45.7 and 73.4 ms. The previous build spent 368.6 seconds saturating chart 1 and was still saturating chart 22 after 511 seconds. All 36 chart records after B8 matched. At the 90-minute comparison limit, Atlas was reducing derivatives for maximal contact on chart 23 during B9 centre selection. The worker peaked at 1.31 GiB. [Surface equation, public command and measured stages](component-regularity-surface-ryzen-6800h.json)

## Residual order at smooth rational points

During centre selection, Atlas can compute the residual order on a smooth frame when it proves the working set is a finite set of rational points. It can also exclude a direction for maximal contact when the directional Taylor expansion vanishes through the required degree at every eligible point. Atlas still constructs the global equations for the chosen centre, which can be a curve through one of those points. [Regression tests](../tests/test_resolution_point_order.py)

The comparison covers orders on translated and implicit frames, rational point support with multiplicities and open conditions, and derivatives for maximal contact. Over nine alternating trials, Atlas had the lower median on 14 of 16 inputs. On B9 chart 23, Atlas certified order six in 11.75 ms versus 35.79 ms in Singular. Computing six successive derivatives on that chart took 13.81 s versus 22.45 s, and all seven polynomials agreed coefficient by coefficient. Singular was faster on two small derivative chains, taking 0.320 ms versus 0.361 ms on the translated frame and 0.451 ms versus 0.589 ms on the coupled frame. [Benchmark driver](resolution_point_order.py) · [Inputs, conditions and individual trials](resolution-point-order-ryzen-6800h.json)

The public CLI comparison used a frozen build that also contained concurrent resolution changes. It reproduced all 36 chart records after B8 on the hard surface and all 16 chart centres available for comparison during B9 selection. A separate comparison isolated the point-order change on five smaller canonical examples. The certificate did not fire on those examples, and Atlas with the change won 17 of 30 timing pairs, with a median speedup of 1.014×. [Full surface run and matched CLI timings](resolution-point-order-surface-ryzen-6800h.json)

On chart 23 the full run certified order six in 11.22 ms and reached coordinate division after 28 min 49 s. At the 90-minute limit, B9 selection was still open. The worker peaked at 3.81 GiB and spent its final 48 min 13 s building an elimination basis to divide by the B8 coordinate. In a separate replay, Atlas set that coordinate to zero and certified a point on the open section where the residual polynomial was nonzero. This proved nondivisibility in a median of 69.95 ms across nine trials. Singular confirmed the result with a nonzero five-term remainder. [Chart operations and section comparison](resolution-point-order-surface-ryzen-6800h.json)

## Arithmetic on chart coordinates

These runs isolate a session operation on a prepared chart. In the graph ring with relation $tx=y$, multiplication by $t$ sends $x+k$ to $y+kt$. The coefficient-division case works over $\mathbb Q(a)$ with $a^2=2$. The comparison excludes the resolution, ring declarations, operand preparation and verification of the answers. It includes the returned polynomial. Five alternating trials each measured 256 operations after an untimed warm-up on logical CPU 11 of a Ryzen 7 6800H. [Driver and exact checks](session_chart_arithmetic.py) · [Per-trial samples and conditions](session-chart-arithmetic-ryzen-6800h.json)

| Chart operation | Atlas (µs) | Singular 4.4.1 (µs) |
| --- | ---: | ---: |
| chart product | 98.30 | 3.32 |
| coefficient division | 207.59 | 9.46 |

The recorded Atlas medians are slower on both operations. To repeat with the same batch size, run `python3 benchmarks/session_chart_arithmetic.py --trials 5 --batch 256` on one logical CPU.
