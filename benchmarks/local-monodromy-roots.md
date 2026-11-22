# Local monodromy eigenvalues and full Bernstein–Sato polynomials

On the four Zoo inputs, an exact check of the previously missing eigenvalue took **3.9–39.6 ms** in Atlas. Singular computed each entire Bernstein–Sato polynomial in **145–2,781 ms**. These measurements start with the polynomial and a supplied pole. The cost of finding the poles by embedded resolution is separate. The raw data includes the full root sets, multiplicities, equations and individual trials. [Measurements](local-monodromy-roots-ryzen-6800h.json) · [Driver](local_monodromy_roots.py)

| Input | Atlas local test (ms) | Singular local test (ms) | Singular full polynomial (ms) |
|---|---:|---:|---:|
| Two tangent conics | 4.58 | 7.47 | 286.56 |
| E7 cylinder | 3.87 | 6.28 | 145.20 |
| Cubic parameters | 15.79 | 26.27 | 855.36 |
| Mixed parameters | 39.58 | 53.06 | 2,781.23 |
| Plane cusp, three residue classes | 0.42 | 1.91 | 8.42 |
| Five squares | 1.54 | 3.82 | 8.80 |
| Cusp with a distant double line | 1.67 | 2.02 | — |

The local tests build the annihilator of the formal power of the input, specialize its parameter at each candidate root, and decide support at the origin by eliminating the derivatives. The plane cusp checks two conjugate eigenvalues and an absent eigenvalue. For five squares the first two candidates fail and the third candidate, −5/2, succeeds. The distant double line contributes a global root −1/2 whose support misses the origin. All three trials of both local tests agreed on each residue class.

For comparison, a fresh Atlas CLI process computed the embedded resolution, both zeta functions and the geometric witnesses in 2,107 ms for the conics, 93 ms for the E7 cylinder, 101 ms for the cubic parameters and 137 ms for the mixed parameters. These are medians of three wall-clock runs including startup, using `atlas resolve` with `--monodromy --compute cpu --json`. Each run used the Zoo record's equation, variables and search policy. The conics used exhaustive search and the other three used quick search. The full Singular polynomials also show that every actual pole in these four records is itself a root, with multiplicity at least its pole order.

Singular's full calculation uses `bfct` from `bfun.lib`. For the six positively weighted homogeneous equations, the scaling action moves every point toward the origin and preserves its local Bernstein–Sato polynomial along the orbit. The local polynomial at the origin is a multiple of the local polynomial at all sufficiently nearby points. It therefore equals the global polynomial for each of these inputs. The comparison uses local elimination for the distant double line. [Local Bernstein–Sato ideals and localization](https://arxiv.org/abs/0806.4869)

The table gives median milliseconds on logical CPU 13 of an AMD Ryzen 7 6800H, using one discarded warmup and three measured trials per system. Atlas includes JSON transport and recomputes its native result for every trial. Singular 4.4.1 uses one thread, a fresh interpreter per trial and `rtimer` around the operation. The timings exclude process startup, library loading and polynomial declarations. On the cubic-parameter example, putting the exponent of the central parameter ahead of total degree in the first elimination reduced the Atlas median from 4,675 ms to 15.8 ms. The same term order improved all seven measured Atlas medians. Atlas had a lower targeted-test median than Singular on every input in this suite.

Run from the repository root after building the native kernel, selecting a logical CPU available on the host.

```sh
taskset -c 13 python3 benchmarks/local_monodromy_roots.py --trials 3 --output /tmp/local-monodromy-roots.json
```
