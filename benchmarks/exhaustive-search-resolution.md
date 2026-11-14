# Why Singular is faster on the inclined pinch point

Singular blows up the singular line of `(x+y+z)^2-z*(x-y)^2` once and finishes in two charts. Atlas's canonical procedure starts at the origin and takes four point blowups and five curve blowups, producing 36 charts. Giving the canonical worker priority and accelerating the producer's affine-locus tests reduced fresh Atlas wall time from **991.8 to 512.7 ms** on this example. Singular's default resolution took **87.3 ms**. [Measurements and complete charts](exhaustive-search-resolution-ryzen-6800h.json.gz)

The line can be read from the equation. In the invertible linear coordinates `a=x+y+z`, `b=x-y`, `t=z`, the polynomial is `a²-t*b²`. Its singular locus is `a=b=0`. In the chart `a=b*r` the strict transform is `r²-t`, which is smooth and transverse to `b=0`. In the chart `b=a*r` the strict transform is `1-t*r²`, which misses the fibre over the origin. Singular actually reported `(2*y+z, x+y+z)` as its first centre. An independent ideal-containment check identifies it with the singular line. Its chart tree contains the root and the two final charts.

## Fresh embedded resolutions

These are median milliseconds from the second complete series of seven alternating fresh trials per solver and input on logical CPU 13 of a Ryzen 7 6800H. The host was running integration tests. It was 63% busy at the final sample before this series. The baseline is the build immediately after the chart-overlap optimization. Atlas wrote the full ordered geometry as JSON. Its zeta invocation also returned the monodromy and topological zeta functions. Singular 4.4.1 returned its standard resolution and printed the chart count. Its checked run additionally used the `resolve.lib` correctness checks. CPU measurements sum child process user and system time. [Driver and input definitions](exhaustive_search_resolution.py)

| Input | Atlas wall before → after | Atlas CPU before → after | Atlas wall with both zetas | Singular default / checked wall |
| --- | ---: | ---: | ---: | ---: |
| `(x+y+z)^2-z*(x-y)^2` | 991.8 → 512.7 | 981.0 → 510.9 | 1022.4 → 534.4 | 87.3 / 134.4 |
| `(x/2+2*y/3-z/5)^2-z*(3*x/4-y/5)^2` | 1002.6 → 499.2 | 999.4 → 497.6 | 1044.5 → 489.8 | 87.5 / 111.3 |
| `(x+y+z)^2-x*(y-z)^2`, variables `y,z,x` | 928.9 → 416.5 | 921.0 → 415.0 | 868.1 → 419.6 | 75.7 / 94.7 |
| `x^2-z*y^2` | 51.8 → 50.9 | 50.4 → 49.5 | 51.8 → 50.8 | 67.9 / 85.3 |

The first complete series ran during concurrent builds and browser tests, with up to 99.8% total CPU use. Its main-result wall medians were 1313.7 → 902.9 ms and its Singular default median was 148.7 ms. Both complete series are archived. An independent seven-trial comparison on two physical cores allowed the search workers to run concurrently. On the main input, Atlas used 766.2 → 570.3 ms wall and 1073.8 → 550.5 ms aggregate CPU. Singular took 106.3 ms wall and 104.6 ms CPU in that series.

Each timed Atlas result matched its entire baseline record except for recorded durations. The inclined results retained all 36 charts, their maps, multiplicities and discrepancies, and the embedded dual complex with 9 vertices, 18 edges and 9 triangles. Both zeta functions replayed from the saved charts. The monodromy zeta is `1/(1-T²)` and the topological zeta is `(s+2)/(2*(s+1)²)`. On the final build the three inclined variants also recovered all 26 strata and 10 numerical rows. Singular independently reproduced all 112 saturated overlap ideals used for the main dual complex. The axis-aligned control retained the two-chart binomial resolution.

Seven additional stress inputs also recovered their original public results. They include `(x+y+z)^2+z^3`, `(x+y+z)^2+x^3+z^3`, and the square of the main polynomial. Two inputs reached the 75-second limit in both builds, `(1+x)*((x+y+z)^2-z*(x-y)^2)` and `(x+y+z)^2+x^4+y^4+z^4`. Their runs establish only the timeout.

## Where the time went

In a separate seven-trial instrumented run on the main input, Atlas's exhaustive search took **529.5 → 167.7 ms**. Producing the chart outputs, including checking the owned strata and computing the dual complex, took **102.4 → 98.8 ms**. The new search recognizes an exact obstruction to resolving this tangent by the first Newton modification. It starts the canonical origin worker first and allows the general certificate worker to join after a short deadline if the origin worker is still running. Two process traces showed one origin worker on both runs and a certificate worker on one. Neither launched a Newton worker. These are phase timings from separate invocations and should be compared within each series.

The search-only build was measured separately before the producer change landed. On the main input it took 744.4 → 395.9 ms against its own paired baseline. To measure the producer increment directly, eleven further alternating pairs compared that exact search-only binary with the final binary on the main input with both zetas. They gave **381.3 → 370.3 ms wall** and **380.9 → 369.9 ms CPU** on logical CPU 13. The producer answers eligible affine nonemptiness tests directly in the native polynomial representation. It preserves the same centres and chart equations.

Singular's internal timer measured **47 ms CPU** inside the default `resolve` call and **15 ms CPU** for loading `resolve.lib`. The following controls isolate the effect of choosing the centre and the cost of checking a supplied witness.

- **Same line supplied to both systems.** Atlas's internal `verify_candidate` constructed and certified the two graph charts in **3.9 ms CPU** with the input already parsed. Singular's `blowUp` constructed its two affine charts in **9 ms CPU** with the same polynomial and centre already parsed. These calls have different output and certification duties. An untimed independent comparison verified both projections, both graph relations, and equality of both strict-transform and exceptional ideals in the corresponding free affine coordinates.
- **Same 36-chart Atlas witness.** Singular took **50.7 ms wall** in a fresh process to verify pullbacks, Jacobian orders, the origin fibre and simple normal crossings on every owned region, using the removed ideals. This check excludes global gluing, dual-complex reconstruction and zeta evaluation. With the same charts already parsed, Atlas took **5.4 ms** for its narrower native-backed local normal-crossings and component-consistency check. The archive includes the exact Singular verification program and separate timing boundaries for each measurement.

To reproduce, point `BEFORE` and `AFTER` at the built Atlas checkouts and set `CPU` to an available logical CPU. The driver uses the installed Singular with `resolve.lib`.

```sh
python3 benchmarks/exhaustive_search_resolution.py --baseline "$BEFORE" \
  --atlas-root "$AFTER" --cpu "$CPU" --trials 7 --output /tmp/exhaustive-search.json.gz
```
