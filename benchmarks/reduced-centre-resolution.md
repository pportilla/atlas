# Blowing up the reduced singular locus

Atlas resolved the inclined pinch point `(x+y+z)^2-z*(x-y)^2` in **61.4 ms**, down from **293.5 ms**. It found the singular line directly and certified one blowup with two charts. Singular independently chose the same line and took **59.8 ms** by default or **74.9 ms** with its correctness checks enabled. The old Atlas resolution used nine blowups and 36 charts. [Complete measurements, charts and checks](reduced-centre-resolution-ryzen-6800h.json.gz)

Medians are from seven interleaved fresh processes per command and input on logical CPU 13 of an AMD Ryzen 7 6800H, after one untimed warm-up per command. Both Atlas builds ran the default embedded CPU resolution and printed full JSON. The zeta commands additionally calculated both the monodromy and topological zeta functions. Singular 4.4.1 ran its default or checked `resolve.lib` resolution and printed the final chart count. The archive includes the wall and aggregate child CPU samples for all 210 launches and hashes for both frozen builds.

| Polynomial | Atlas charts | Atlas wall before → after, ms | Atlas CPU before → after, ms | Atlas wall with both zetas, ms | Singular default / checked wall, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(x+y+z)^2-z*(x-y)^2` | 36 → 2 | 293.5 → 61.4 | 291.9 → 58.9 | 296.9 → 61.8 | 59.8 / 74.9 |
| `(x/2+2*y/3-z/5)^2-z*(3*x/4-y/5)^2` | 36 → 2 | 302.8 → 62.0 | 301.3 → 59.3 | 306.0 → 62.4 | 60.0 / 75.5 |
| `(x+y+z)^2-x*(y-z)^2`, variables `y,z,x` | 36 → 2 | 292.7 → 61.2 | 290.9 → 58.6 | 296.5 → 61.0 | 59.7 / 75.1 |
| `(y-x^2)^2-x*z^2` | 30 → 2 | 483.0 → 59.5 | 472.0 → 56.9 | 485.9 → 60.3 | 58.3 / 72.6 |
| `x^2-z*y^2` | 2 → 2 | 42.3 → 41.9 | 40.7 → 40.3 | 42.5 → 42.9 | 58.1 / 72.9 |

Default Singular finished 1.3–2.1 ms earlier on the four inputs that changed resolution. The axis-aligned control kept its original binomial computation, including its entire normalized output with both zetas. An earlier complete 210-launch series overlapped with regression tests and two long-running resolution controls on other cores. In that series the main case took 373.8 → 76.1 ms versus 68.5 / 86.7 ms for Singular. All of its samples are retained in the archive. The primary series ran after those tasks ended.

## Which geometry changed

Set `a=x+y+z`, `b=x-y`, `t=z`. The main equation becomes `a²-t*b²`, with reduced singular locus `a=b=0`. Atlas found generators `x+z/2, y+z/2`. Singular's checked run reported `2*y+z, x+y+z`. Independent computation of the radical Jacobian ideal and mutual ideal-containment checks established equality over the rationals. For `(y-x²)²-x*z²`, Singular reported the smooth parabola `z=x²-y=0`, and Atlas certified this nonlinear graph centre. Its chart count fell from 30 to two.

On each of the four intersections of an Atlas graph chart with a Singular chart, the oracle expresses the Atlas ratio in the Singular coordinates and inverts the relevant denominator. It checks the projected coordinates, both ambient graph equations, strict-transform ideals, exceptional ideals, and scheme-theoretic fibre. It also checks the disjoint ownership convention in the saved atlas. This comparison allows the solvers to choose different pivot generators. Eight deliberately changed chart records were rejected. [Benchmark driver and independent Singular checks](reduced_centre_resolution.py)

For the three inclined cases the old exceptional complex had 9 vertices, 18 edges and 9 triangles. The old curved example had 9 vertices, 17 edges and 9 triangles. In each new presentation the exceptional divisor is irreducible, giving one vertex and no higher-dimensional cells. Its multiplicity and log discrepancy are both 2. The exceptional fibre over the origin is a projective line whose strict-transform intersection is a doubled point. This leaves two disjoint strata, each with Euler characteristic one. Both presentations gave monodromy zeta `1/(1-T²)`, Milnor fibre Euler characteristic 2, and topological zeta `(s+2)/(2*(s+1)²)`. The driver deleted the stored answers and replayed each presentation's own exceptional complex and both full zeta records. The final build also replayed the previous build’s stripped chart records for all five inputs. The archive retains the changed chart maps, certificates and divisor labels and the list of changed fields.

A separate control multiplied the inclined equation by the local unit `1+x`. Both builds reached the 120-second limit on this input, so its performance and final geometry remain unmeasured.

## Centre discovery and graph verification

In seven fresh diagnostic processes with the main polynomial already parsed, Atlas's combined native operation recovered the centre, constructed its charts and certified the graph in **0.70 ms CPU**. Running the installed standalone graph helper, including process startup, took **5.06 ms wall** and returned the same packed proof. A separate diagnostic measured **0.48 ms CPU** for centre discovery alone and **3.32 ms CPU** for the arbitrary-candidate verifier to construct and certify the same public graph. The latter includes constructing the structured record. These three measurements have distinct timing boundaries. Singular's internal timers measured **6 ms CPU** for its radical Jacobian calculation and **7 ms CPU** for constructing two charts using the supplied centre.

To reproduce, set `BEFORE` and `AFTER` to the built Atlas checkouts, select an available logical CPU, and install Singular with `resolve.lib` and `primdec.lib`.

```sh
python3 benchmarks/reduced_centre_resolution.py --baseline "$BEFORE" \
  --atlas-root "$AFTER" --cpu "$CPU" --trials 7 --output /tmp/reduced-centre.json.gz
```
