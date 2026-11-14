# Embedded resolution along an inclined singular line

Native chart-overlap and open-locus checks reduced the median time for the complete resolution of `(x+y+z)^2-z*(x-y)^2` from **697.3 to 598.2 ms**. Atlas returned the same 36-chart resolution, including its certificates and execution record. Singular 4.4.1 produced its two-chart resolution in **61.3 ms**. Rebuilding the Atlas dual complex from the saved charts fell from **262.9 to 186.6 ms** in nine independent fresh CLI pairs. [Raw timings, records, hashes and checks](inclined-centre-resolution-ryzen-6800h.json.gz)

All table entries are median milliseconds on an AMD Ryzen 7 6800H. Each command received one warm-up and seven measured fresh launches on logical CPU 13 in alternating order. Atlas wrote its complete JSON result. Singular's `resolve.lib` returned the embedded resolution and printed its chart count. Its checked timing also runs its own resolution validation. Atlas CPU time sums user and system time for the child processes.

| Polynomial | Atlas wall before → after | Atlas CPU before → after | Atlas wall with both zetas | Singular default / checked wall |
| --- | ---: | ---: | ---: | ---: |
| `(x+y+z)^2-z*(x-y)^2` | 697.3 → 598.2 | 694.5 → 595.0 | 722.0 → 599.3 | 61.3 / 77.0 |
| `(x/2+2*y/3-z/5)^2-z*(3*x/4-y/5)^2` | 701.2 → 600.8 | 698.2 → 598.0 | 708.9 → 613.8 | 61.0 / 76.7 |
| `(x+y+z)^2-x*(y-z)^2`, variables `y,z,x` | 696.1 → 604.7 | 693.0 → 601.7 | 697.8 → 604.4 | 61.2 / 76.4 |
| `x^2-z*y^2` | 43.6 → 44.3 | 42.1 → 43.0 | 44.2 → 43.9 | 59.9 / 73.8 |

**Independent geometry checks.** For each inclined input, Singular checked every Atlas chart map, its total transform, Jacobian divisor and simple normal crossings on the owned origin fibre. Twenty of the 36 charts carry removed loci, giving 24 removed ideals in total. The oracle also rejected five deliberate changes to chart 9, including deletion of a removed locus on which the represented divisors actually fail transversality. The axis-aligned control uses two charts. [Benchmark and checks](inclined_centre_resolution.py)

The Singular oracle verified the two standard blowup charts in independent linear coordinates adapted to the centre. For the main input the centre is `(2*x+z, 2*y+z)`. The reciprocal chart ratios identify the fibre with a projective line, and the strict transform cuts it in a doubled point. The exceptional divisor has multiplicity 2 and Jacobian order 1, and the open exceptional stratum and its intersection with the strict transform each have Euler characteristic one. Both methods give monodromy zeta `1/(1-T^2)` and topological zeta `(s+2)/(2*(s+1)^2)`. Replaying each saved Atlas record with the zeta data removed recovered both functions.

**Overlap and replay checks.** A complete replay used 112 native chart-overlap calls and 326 open-locus calls. A separate Singular computation reconstructed those 112 saturated ideals and verified each inclusion in both directions. Across all 1,296 ordered pairs of charts, before and after produced byte-identical sparse rows for the ideals, coordinate substitutions and removed loci. Their SHA-256 starts with `6980ca72d65375d2`. The archive includes the complete digest, the ordered-pair encoding, the Singular program and its output digest.

The benchmark compared every field of every timed Atlas result except elapsed-time fields. Sixteen additional public comparisons included quick search, intrinsic mode, affine variants and both zeta settings. The embedded inclined result had 9 vertices, 18 edges and 9 triangles. The intrinsic control retained its partial complex with 5 vertices. On the unsheared binomial control, the general saved-chart dual-complex replay reconstructs the single vertex and its dimension. The fresh binomial result also carries toric ray metadata. This distinction was present in both builds.

To reproduce, set `BEFORE` and `AFTER` to built Atlas checkouts and choose an available logical CPU. The driver requires Singular with `resolve.lib`. A plain `.json` output path writes the archive without compression.

```sh
python3 benchmarks/inclined_centre_resolution.py --baseline "$BEFORE" \
  --atlas-root "$AFTER" --cpu "$CPU" --trials 7 --output /tmp/inclined.json.gz
```
