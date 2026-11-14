# Native resolution and saved-chart replay

The regressions exercise four paths through `./atlas resolve` on a Ryzen 7 6800H. Each command starts a fresh process, resolves over the rational input field and writes its complete JSON result. The fifth case checks a 191-chart regular fan. [Resolution algorithm map](../docs/resolution-pipeline.md)

## Comparison

The table reports median wall time in milliseconds over seven runs after one warm-up on logical CPU 13. Atlas before and after ran in alternating order while other validation jobs shared the host. The Singular column times `resolve(I,"L","A")` in Singular 4.4.1. Atlas writes chart and incidence data to standard output. Singular constructs its resolution and prints the chart count. The zeta columns include both zeta functions and the pole witnesses requested by `--monodromy`. [Samples, complete four-case records and checksums](native-resolution-boundary-ryzen-6800h.json.gz)

| Input | Atlas charts | Before | After | Before with zetas | After with zetas | Singular |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `x^2-y^2*(z+t)` | 3 | 66.32 | 63.47 | 108.32 | 66.15 | 73.25 |
| `(x+y+z)^2-z*(x-y)^2` | 2 | 90.85 | 88.21 | 117.41 | 92.04 | 78.14 |
| `(x+y)^2+z^3+x^3` | 15 | 591.27 | 610.91 | 595.59 | 611.82 | >60,000 |
| `x*y*z+x^4+y^4+z^4` | 9 | 214.51 | 226.70 | 282.90 | 211.63 | 714.23 |
| `x^2+y^19+z^37+x^3*(x+y+z)^27` | 191 | 985.11 | 961.13 | — | — | >60,000 |

On the inclined graph, a resident worker produced the complete record with both zetas in 8.02 ms. A separate cold-process trace attributed 49.43 ms to module imports. Deferring two imports until their optional workers start saved a median 2.18 ms per paired full CLI run over 49 pairs on two logical CPUs. [Profile and matched samples](inclined-cli-startup-ryzen-6800h.json)

## Exact comparisons

- For every fresh run the driver compares every JSON field except `elapsed_ms` and `native_elapsed_ms`. It also strips the computed complex and zetas, recomputes them through the saved-record interfaces, and compares the complete replay results across versions. Fresh Newton records carry more ray and root annotations than the historical replay format. Their labelled vertices, cells and faces agree.
- Singular independently checks the three four-variable Whitney charts and their fibre, the inclined graph centre and its four overlaps, and the pullback, divisor orders and normal crossings of all 15 canonical and all 9 Newton charts. It also checks both local Newton zeta functions. Singular constructs 15 charts for that input and Atlas constructs 9. [Benchmark driver](native_resolution_boundary.py)
- The earlier 36-chart resolution of the inclined surface stores 27 cells. Both Atlas versions reconstruct 26. Singular verifies that the four chart copies of the B1–B9 intersection give one irreducible curve. Its extra archived edge is a duplicate. The original chart equations and both zetas survive replay. [Archived edge certificate](historic-inclined-edge.json)
- Related inclined surfaces with a rational change of coordinates and a coordinate permutation retain both graph charts. Larger archived Newton examples also exercise fallback gluing. [Inclined controls](native-resolution-inclined-controls.json) · [Gluing results](chart-gluing-ryzen-6800h.json)

## Reproduce

Build both revisions and pass their directories to the comparison script. The compressed result includes the full Atlas records for the first four cases, the per-run samples and the 191-chart checksum. Frozen installations with a `source-hashes.json` file also record the individual engine and binary hashes.

```sh
python3 benchmarks/native_resolution_boundary.py \
    --before /path/to/baseline --after /path/to/candidate --cpu 13 \
    --trials 7 --singular-timeout 60 --toric-control --output /tmp/resolution.json.gz
```
