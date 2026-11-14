# Shifted sparse local dimension and smoothness

These measurements exercise the one-million-term translation guard reached by the former local dimension path. For example, over `QQ[x,y,z]` at `(1,1,0)`, set `U=x^1001-1` and `V=y^1001-1`. The ideal `(UV,Uz,Vz)` defines three branches locally. Its dimension is 1 and its embedding dimension is 3. Expanding the monomial `x^1001*y^1001` about this point produces 1,004,004 terms. The previous Atlas CLI returned `Large coordinate translation.` for both operations on this input.

## Results on the Ryzen 7 6800H

Atlas returned the verified answer or the expected off-support rejection on all 30 requests. The previous executable reached the translation guard on 20, exceeded three seconds on two, and completed eight. Direct Singular completed 26 requests. Atlas had the faster fresh-process median on all 26, with a median paired speed ratio of 1.42. Across all 30 requests, the median Atlas time was 7.18 ms and the range was 5.50–17.22 ms. [All samples, exact inputs and executable hashes](large-coupled-local-ryzen-6800h.json)

The pairs below give dimension / smoothness timing in milliseconds. The invariant column gives local dimension / embedding dimension.

| Input | Invariants | Atlas CPU | Best direct Singular |
| --- | --- | ---: | ---: |
| Axis union, N=7 | 1 / 3 | 5.5 / 6.5 | 7.3 / 7.4 |
| Axis union, N=1,001 | 1 / 3 | 6.5 / 6.7 | 373.6 / 611.7 |
| Parameter squares, N=17 | 0 / 3 | 6.8 / 6.8 | 9.6 / 9.3 |
| Parameter squares, N=1,000,001 | 0 / 3 | 6.8 / 7.3 | 104.5 / 103.3 |
| Parameter squares, N=2⁶⁴+7 | 0 / 3 | 7.1 / 7.4 | Exponent unsupported |
| Cancellation, isolated point | 0 / 2 | 6.9 / 6.9 | 136.5 / 80.1 |
| Cancellation, curve | 1 / 3 | 10.0 / 16.9 | 11.0 / 21.2 |
| Independent higher terms | 0 / 3 | 13.9 / 17.0 | 14.8 / 19.0 |
| Nonreduced line | 1 / 3 | 16.1 / 17.2 | 35.1 / 31.4 |
| Oblique union, N=7 | 1 / 3 | 10.5 / 10.2 | 10.9 / 10.6 |
| Oblique union, N=1,001 | 1 / 3 | 6.9 / 6.7 | Both methods >3,000 |
| Hyperplane and line, intersection | 3 / 4 | 7.3 / 7.6 | 23.5 / 24.9 |
| Hyperplane and line, point on line only | 1 / 1 | 5.9 / 5.5 | 8.4 / 7.0 |
| Hyperplane and line, point on hyperplane only | 3 / 3 | 6.0 / 11.8 | 32.2 / 30.9 |
| Hyperplane and line, outside | Off support | 12.1 / 14.1 | 14.5 / 14.6 |

The sparse substitution for the isolated cancellation example reduced the measured Atlas CLI medians from 34.4 / 35.6 ms to 6.9 / 6.9 ms. For the oblique degree-1,001 union, both direct Singular methods exceeded the time budget. For the exponent above 2⁶⁴, both reported that polynomial exponentiation rejected the bigint exponent. The independent chart runs verified all four answers. The JSON preserves the separate direct and chart outcomes for every request.

## Reproduce

Build the native CPU executable using the repository’s build environment, then run the following on Linux with Singular on the path. Choose a logical CPU allowed by the current host. Pass `--baseline-root /path/to/previous/build` to include a previous Atlas CLI and its matching library. The JSON includes their hashes when supplied.

```sh
make -C native -f groebner.mk build/local_algebra
python3 benchmarks/large_coupled_local.py --cpu 5 --seconds 3 --memory-mib 768 --repeats 3 --output /tmp/large-coupled-local.json
```

Each trial starts a fresh CLI, parses the original expressions and point, performs the calculation and reads its result. All trials use one logical CPU. Atlas requests CPU explicitly and verifies the backend on successful responses. Singular 4.4.1 uses one Singular and one FLINT thread. Solver order rotates across three repetitions. Reported times are medians in milliseconds. Each child has three seconds of wall time and 768 MiB of virtual address space. A timeout records only a lower bound. Allocation failures, arithmetic overflow and unrepresentable exponents are reported separately. After any failure, further repetitions of that case and method are skipped.

Two direct Singular formulations receive the same original generators and point as Atlas. The first translates the ideal into a local ring and computes `dim(std(J))`. The second computes a global standard basis and tests the dimension bounds. If the bounds differ, it uses `equidim` and takes the greatest dimension among components through the point. Both test support first. Smoothness uses the rank of the original Jacobian at the point. A rank attaining the number of equations or ambient dimension certifies a complete intersection or an isolated smooth point. We report the faster successful direct median where both are available.

Singular’s ring-construction list supports a larger `maxExp` for the million-degree inputs. We use local `ds` when widening the direct reference ring. On this installation, constructing a widened `Ds` ring gives dimension 1 for `((x+y)^2,(x+z)^2,(y+z)^2)`, even though these three independent linear forms cut out a zero-dimensional thickening. Default `Ds`, default `ds` and widened `ds` give dimension 0. Each case records the Atlas order. [Singular manual on ring attributes](https://www.singular.uni-kl.de/ftp/pub/Math/Singular/src/4-1-1/singular.pdf)

## Independent local oracle

The source data gives coordinates and a nonzero Jacobian determinant for each supplied chart. All charts have rational residue field, so these changes induce isomorphisms on completed local rings. For the parameter-square family, `u=x^N-1` has derivative `N` at 1, including for `N=2^64+7`. For the oblique union, the chart determinant at the point is `-N`. Singular verifies the dimension in these charts by a local standard basis and independently computes the embedding dimension as `vdim(std(J+maxideal(2)))-1`. These chart runs receive simpler equations and are excluded from direct speed comparisons.

The two graph-cancellation examples admit a further exact simplification in the chart. The first input equation fixes `x=1`, which eliminates every term carrying `x^N-1` or `x-1`. The remaining equations are `y²+z³` and `yz`. Their ideal contains `z⁴=z(y²+z³)-y(yz)`. This gives local dimension 0 in three variables and dimension 1 with a fourth free coordinate. The mixed family consists of a hyperplane and a line in the chart. The same four-dimensional source scheme has local dimensions 3 at the intersection, 1 on the line away from the intersection, and 3 on the hyperplane away from the line. Public errors at an off-support point correspond to Singular reporting a local unit ideal.

## Ordinary corpus on the same build

The 12-example suite measured local dimension and smoothness on logical CPU 3. Each warm median covers seven batches of 96 recomputations. Each cold median covers five fresh processes. For each request, the comparison uses the faster median of the two Singular formulations. [Raw timings, inputs and build hashes](large-coupled-ordinary-recheck-ryzen-6800h.json)

| Atlas measurement | Faster than Singular | Median paired speed ratio (Singular / Atlas) |
| --- | ---: | ---: |
| Warm public CPU handler | 23 of 24 | 5.12× |
| Warm typed C++ kernel | 24 of 24 | 31.02× |
| Fresh CLI process | 24 of 24 | 1.70× |

The warm handler's slower request was the dimension of the regular plane with zero ideal, at 4.30 µs for Atlas and 3.64 µs for Singular. Five further runs of this example gave Atlas medians of 2.36–4.14 µs and Singular medians of 6.25–7.34 µs. Atlas was faster in all five. For the dimension of the plane-and-line intersection, Atlas took 8.45 µs and Singular took 13.90 µs. In three separate runs of that example with the same batch settings, Atlas medians ranged from 6.79 to 7.41 µs and Singular medians from 9.22 to 13.11 µs.

The focused repeats used the same benchmark and measured source files as the full run. Other work relinked the native binaries between runs. The raw report above records the full run.
