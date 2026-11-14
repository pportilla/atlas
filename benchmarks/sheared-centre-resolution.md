# Embedded resolution along a sheared singular line

Atlas's default search returned the same two-chart blowup about six times faster on four surface germs. Fresh processes on the Ryzen 7 6800H still took 2–6 ms longer than Singular 4.4.1's default resolution on those inputs. Atlas printed the complete ordered resolution as JSON. Singular ran `resolve.lib` and printed the chart count. The checked Singular column includes its resolution check. [Timings, full Atlas records and validation](sheared-centre-resolution-ryzen-6800h.json)

All entries are median wall milliseconds from seven fresh runs per command on logical CPU 3. Each command had one discarded warm-up. Trial order alternated. The last column requests both monodromy and topological zeta functions in the Atlas invocation.

| Polynomial | Atlas before → after | Singular default | Singular checked | Atlas with zetas before → after |
| --- | ---: | ---: | ---: | ---: |
| `(x+y)^2-z*(x-y)^2` | 456.5 → 75.1 | 69.2 | 85.0 | 467.3 → 73.7 |
| `(x/2+2*y/3)^2-z*(3*x/4-y/5)^2` | 381.4 → 64.5 | 62.0 | 77.2 | 387.9 → 64.8 |
| `(x+y)^3-z*(x-y)^3` | 376.5 → 63.4 | 61.0 | 76.6 | 384.1 → 63.0 |
| `(3*x-5*y)^2+z*(x^2+3*x*y+2*y^2)` | 399.0 → 68.0 | 62.6 | 78.8 | 413.9 → 68.0 |
| `x^2-z*y^2` | 46.2 → 45.9 | 61.0 | 76.2 | 46.4 → 45.9 |

An execution trace outside the timer confirmed one call to the installed native graph verifier on each of the four sheared inputs. The unsheared case used the existing two-chart binomial computation. Three separate pairs per input with `--quick` kept the previous records. The first, second and fourth inputs produced 30 charts and took 338, 332 and 343 ms after the change. The cubic took 3.16 s for two charts. The unsheared control took 43 ms.

The independent checks substitute every chart map into the original polynomial and compare the exceptional factors and strict transform over the rationals. They certify the Jacobian orders and simple normal crossings over the origin. In Singular's two standard charts the exceptional fibre is a projective line. Its strict-transform intersection is supported at the single projective zero of the tangent linear form, with multiplicity `d`. The open exceptional stratum and its intersection with the strict transform each have Euler characteristic one. This gives monodromy zeta `1/(1-T^d)`, Milnor fibre Euler characteristic `d` and topological zeta `(s+2)/((s+1)*(d*s+2))`. Both saved zeta records were recomputed from the archived charts. Nine deliberate changes to chart data were rejected. [Benchmark and independent chart checks](sheared_centre_resolution.py)

For each search policy and zeta setting, every pinned before/after record matched after excluding elapsed-time fields. An unrestricted-affinity control exposed a different valid presentation on the original binary. In 16 launches including warm-up, 15 returned the two-chart graph and one returned 30 canonical charts. The staged recognizer returned two charts in all 16. Independent checks of both saved presentations recovered the same zeta functions.

The chart oracle now retains the entire ideal returned by `sat` on Singular 4.4.1 when excluding a removed locus. Rechecking all ten archived default and quick records reproduced the published chart checks. This adjustment was made after the timings and driver digest recorded in the archive.

An additional nine-triplet default CLI comparison measured the quartic `(x+y)^4-z*(2*x-y)^4` at 417.30 → 68.75 ms against 64.52 ms in Singular, and the permuted cubic `(-7/19)*(x-z)^3+y*(2*z^3+x^3)` at 403.47 → 67.69 ms against 63.37 ms. The inclined germ `(x+y+z)^2-z*(x-y)^2` still produced 36 charts and took 710.84 → 718.76 ms. Singular produced two charts in 62.32 ms. The archived supplemental control includes the nine timings per program and before/after complete-record checks with and without zetas.

To reproduce, point `BEFORE` and `AFTER` at built Atlas checkouts and set `CPU` to a logical CPU available on the host. Install Singular with `resolve.lib`.

```sh
python3 benchmarks/sheared_centre_resolution.py --baseline "$BEFORE" \
  --atlas-root "$AFTER" --cpu "$CPU" --trials 7 --output /tmp/sheared-default.json
python3 benchmarks/sheared_centre_resolution.py --baseline "$BEFORE" \
  --atlas-root "$AFTER" --cpu "$CPU" --search quick --trials 3 --output /tmp/sheared-quick.json
```
