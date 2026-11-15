# Canonical centres and exceptional intersections

Atlas uses a single native call per eligible chart to descend through coefficient presentations, choose a curve on a smooth surface, or count exceptional intersections. These operations use the chart equations, divisor history and ownership conditions. A failed certificate returns control to the general procedure. [Resolution algorithm map](../docs/resolution-pipeline.md)

## Full CLI timings

The comparisons use fresh processes on a Ryzen 7 6800H. Atlas runs `./atlas resolve` in exhaustive mode over the rationals, including chart verification and JSON output. The zeta runs add `--monodromy`, which produces both zeta functions and pole witnesses. Singular 4.4.1 constructs a resolution with `resolve(I,"L","A")` and prints its final chart count. A timeout is a wall-time lower bound. The raw measurements also include Singular's checked invocation and child CPU times. [Inputs, samples and full records](canonical-chart-batches-ryzen-6800h.json.gz)

Both builds include the native residual-order change in `392decc`. The candidate is `e3499ba`. Each row gives median wall times in milliseconds. The three canonical cases used 11 trials per build on logical CPUs 13, 4 and 8 respectively. The remaining controls used seven trials on CPU 2. Each row used the same core for both builds and Singular, with other jobs active on the host.

| Equation | Atlas charts | Before | After | Before with zetas | After with zetas | Singular |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(x+y)^2+z^3+x^3` | 15 | 518.45 | 385.15 | 520.77 | 384.43 | >60,000 |
| `(x*y-z^2)^2+x^5+y^5` | 18 | 1,309.32 | 1,203.30 | 1,395.71 | 1,249.31 | >60,000 |
| `x^2+(y+z)^3+y^4` | 19 | 567.30 | 467.29 | 566.50 | 465.64 | 642.38 |
| `(x+y+z)^2-z*(x-y)^2` | 2 | 72.35 | 70.97 | 71.89 | 72.05 | 67.44 |
| `x^2-y^2*(z+t)` | 3 | 70.65 | 70.49 | 73.26 | 72.48 | 68.80 |
| `x*y*z+x^4+y^4+z^4` | 9 | 113.48 | 110.32 | 118.48 | 117.82 | 504.80 |
| `x^2+y^19+z^37+x^3*(x+y+z)^27` | 191 | 601.96 | 582.93 | — | — | >60,000 |

With default CLI settings and plain text output, the 15-chart case fell from 547.92 to 422.98 ms. Default JSON output fell from 540.46 to 407.07 ms. All 11 alternating pairs per format were faster on CPU 14. Host load particularly affected the 18-chart case. An additional 11-pair resolution-only comparison on CPU 13 measured 592.36 to 517.01 ms with the same full result. Singular remained faster on the two- and three-chart controls. It produced 31 charts for the 19-chart case.

## Where time was saved

The following are median process CPU times on CPU 10 over 15 trials. Centre selection covers every chosen blowup. Incidence starts with decoded charts and includes geometric and rational component counts.

| Canonical charts | Centre selection before | Centre selection after | Incidence before | Incidence after |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 29.61 ms | 4.35 ms | 17.56 ms | 3.82 ms |
| 18 | 43.75 ms | 45.52 ms | 24.18 ms | 6.64 ms |
| 19 | 54.57 ms | 39.21 ms | 19.49 ms | 4.13 ms |

The general centre procedure still handles the 18-chart case. Its overall gain came from faster incidence and from avoiding an unused verifier in the competing certificate search. In a process trace that spare consumed 48.9 ms of CPU and handled zero requests. With both builds using the native chart changes, deferring the spare alone reduced the median full CLI time from 548.25 to 500.48 ms across 15 alternating pairs. Rejected candidates continue on the active verifier. An undecided result or timeout enables the spare for subsequent attempts.

## Agreement

For each fresh CLI run the driver compares every JSON field apart from the two elapsed-time fields. It strips the calculated outputs, recomputes the dual complex and both zetas through the saved-record interfaces, and checks the pole witnesses. Singular independently checks the owned pullback, exceptional and volume orders, and normal crossings on all 52 charts of the three canonical cases. [CLI and Singular checks](native_resolution_boundary.py)

For the first two blowups of the 15-chart cubic, the native descent returns the same invariant and ordered ideal as the general frame procedure, including the conjugate points on the second blowup. In the last two blowups the native surface procedure certifies six chart choices. The incidence pass processes all 41 nonempty exceptional intersections in its 15 charts. The tests also cover partially owned loci, nontrivial coefficient fields and exponents beyond the packed representation. [Affine centres](../tests/test_centre_affine_maximum.py) · [Surface centres](../tests/test_surface_centre_choice.py) · [Owned incidence](../tests/test_owned_chart_incidence.py)

The two intrinsic controls also matched under default, quick and exhaustive search. Removing the cached numerical data and strata from five records reproduced 29 numerical rows and 76 strata, both zetas and all pole checks. The archived 36-chart inclined surface again yielded 26 cells with the duplicated B1–B9 edge removed. A fresh solve of that surface uses two charts. [Historical edge verification](historic-inclined-edge.json)

## Reproduce

Build both revisions, then run the fresh CLI comparison with a logical CPU available on the host. Each build gets a warm-up and the driver alternates their order. The stage benchmark uses fresh interpreters and measures process CPU time around the complete centre choice, the canonical construction, and the first count of geometric and rational components on decoded charts.

```sh
python3 benchmarks/native_resolution_boundary.py \
    --before /path/to/baseline --after /path/to/candidate --cpu 13 \
    --trials 11 --singular-timeout 60 --cases centre tangent double_cone \
    --output /tmp/canonical-cli.json.gz
python3 benchmarks/canonical_chart_stages.py \
    --before /path/to/baseline --after /path/to/candidate --cpu 10 \
    --trials 15 --output /tmp/canonical-stages.json
```
