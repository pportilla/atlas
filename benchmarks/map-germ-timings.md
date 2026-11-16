# Map-germ codimension timings

## Fresh CLI comparisons

The benchmark compares the extended tangent-space codimension over QQ at the origin. Atlas runs the public `ae-codimension` command and prints its complete JSON record. Singular 4.4.1 loads `classifyMapGerms.lib` in a local ring ordered by `(c,ds)` and returns the second entry of `coDimMap(f)`. Each timed run checks the value and finite or infinite status. Both programs start a fresh process for every trial. One discarded run per implementation warms the file cache, then the benchmark alternates execution order across seven pairs. The driver requested logical CPU 13. A later check found that the installed single-threaded Singular can reset its affinity to CPU 0 after launch. [Inputs and benchmark driver](ae_codimension.py)

```sh
python3 benchmarks/ae_codimension.py --cpu 13 --trials 7 --output /tmp/atlas-ae.json
```

On a Ryzen 7 6800H, Atlas had the lower fresh-process time in 84 of 84 pairs. The median of the 12 per-case speedups was 11.08×, with a range of 4.96–208.35×. The dense plane case changes source coordinates in the germ $(u,uv+v^7)$ using $u=x+y+xy+y^2$ and $v=y+y^2$. Both implementations computed codimension 10. [All CLI samples, results, binary hashes and native profiles](ae-codimension-ryzen-6800h.json)

| Map germ | Codimension | Atlas median (ms) | Singular median (ms) |
| --- | ---: | ---: | ---: |
| Immersion | 0 | 4.639 | 154.921 |
| Univariate scalar | 10 | 5.016 | 24.885 |
| Separated scalar | 11 | 4.510 | 23.519 |
| Mixed scalar | 10 | 4.982 | 28.741 |
| Monomial curve | 7 | 4.804 | 29.733 |
| Crosscap | 0 | 4.454 | 148.441 |
| Higher-order double cover | 11 | 4.641 | 292.062 |
| Graded plane map | 3 | 4.850 | 44.132 |
| Plane map after change of coordinates | 3 | 4.981 | 60.199 |
| Graded map in three variables | 1 | 5.229 | 546.448 |
| Positive-dimensional contact locus | ∞ | 4.697 | 47.322 |
| Dense plane map after change of coordinates | 10 | 15.226 | 3,172.467 |

Native timings start with parsed polynomials. On the dense plane map, reducing representatives modulo the Jacobian before multiplying by target coordinates cut target-space closure from 6.596 to 3.293 ms, and the complete jet computation from 11.985 to 8.616 ms. Two controls that forced the jet solver past the normal shortcuts took longer after the change, from 0.066 to 0.082 ms and from 5.814 to 5.851 ms. On four graded maps, packed Gröbner elimination reduced that stage by 43–48%. On the scalar $x^5+y^5+x^3y^3$, the Jacobian product criterion, a nilpotence bound and Euler cancellation reduced the complete native call from 25.871 to 2.756 µs. The double-cover hypersurface calculation reduced the complete native computation for $(x,y^2,y^3+x^{12}y)$ from 117.99 to 1.77 µs.

To time the current kernel on another map, pass the source variables on the first input line and one target component on each following line. Choose `parse`, `solve` or `graded` to time input parsing, the public native computation or graded elimination. The harness checks the answer against the public kernel. [Native timing harness](ae_codimension_kernel.cpp)

```sh
make -C native -f groebner.mk build/bench_ae_codimension
printf 'x,y\nx\nx*y+y^7\n' | native/build/bench_ae_codimension 35 graded
```

## Pairwise-coprime powers of three lines

For $(x^a,y^b,(x+y)^c)$ with pairwise-coprime positive degrees, the native
solver computes the extended tangent quotient over the rational numbers. The benchmark times the public Atlas CLI, Singular's
`coDimMap(f)[2]`, and a separate Singular computation of exactly the same
extended normal quotient on a free basis over $\mathbb Q[x^a,y^b]$. The
second Singular computation provides an independent dimension even on cases
where the general procedure times out. All three use fresh processes on
logical CPU 0. Each completed median uses three trials after one warmup.
[Native tangent-space computation](../native/map-germ-algorithms.md) · [Singular module computation](ae_coprime_module.sing)

Run `python3 benchmarks/ae_coprime.py --output ae-coprime.json` to repeat the
comparison. The JSON records each trial and the cutoff for any timeout.
[Measurements on the Ryzen 7 6800H](ae-coprime-ryzen-6800h.json)

| $(a,b,c)$ | Extended codimension | Atlas CLI (ms) | Singular `coDimMap` (ms) | Singular module (ms) |
| --- | ---: | ---: | ---: | ---: |
| (1, 2, 3) | 1 | 3.94 | 259.04 | 6.51 |
| (1, 3, 5) | 20 | 3.68 | 198.18 | 7.65 |
| (2, 3, 5) | 115 | 4.02 | 5,092.31 | 9.93 |
| (2, 3, 7) | 242 | 5.92 | 33,373.54 | 11.23 |
| (3, 4, 5) | 529 | 4.92 | timeout at 15 s | 21.33 |
| (2, 5, 7) | 729 | 5.80 | timeout at 15 s | 27.47 |
| (3, 5, 7) | 1710 | 9.16 | timeout at 15 s | 77.14 |

## Atlas kernel timing for graded surface maps

The native `solve` harness times the complete computation on parsed polynomials.
These are medians of 12 trials on one logical CPU of a Ryzen 7 6800H. Each build
was warmed up on each germ, and the order of inputs and builds rotated between
trials. The three revisions, all sample times, the exact input polynomials and
the stopping weights are in the [kernel measurements](ae-graded-rank-kernel-ryzen-6800h.json).

| $(a,b,c)$ | Codimension | Before (ms) | Derivative reuse (ms) | With modular rank changes (ms) |
| --- | ---: | ---: | ---: | ---: |
| (1,2,3) | 1 | 0.00180 | 0.00178 | 0.00178 |
| (1,3,5) | 20 | 0.0408 | 0.0397 | 0.0360 |
| (2,3,5) | 115 | 0.0942 | 0.0847 | 0.0511 |
| (2,3,7) | 242 | 0.228 | 0.190 | 0.0839 |
| (3,4,5) | 529 | 0.734 | 0.607 | 0.239 |
| (2,5,7) | 729 | 1.307 | 1.035 | 0.378 |
| (3,5,7) | 1710 | 5.242 | 4.270 | 1.173 |

For $(3,5,7)$ a separate profile measured 3.853 ms in modular rank before the
changes and 0.819 ms afterwards. For $(x^2,y^3,x^5+x^4y-xy^4+y^5)$ the full
solve fell from 0.0832 to 0.0452 ms. The exact-rank controls used
$(x^2,y^3,(65521x+y)^5)$ and $(x^3,y^7,x^2-3x(65521y)-(65521y)^2)$.
Their timings fell from 1.852 to 1.782 ms and from 23.365 to 20.970 ms,
respectively, with codimensions 115 and 242.

## Published map germs

Run `python3 -m unittest tests.test_ae_literature` to check the native solver
against 29 literature presentations, including plane branches, corank-two double
cusps and four-variable augmentations. The [regression cases](../tests/test_ae_literature.py)
give the input polynomials, expected dimensions, sources and parameter choices.

Sharland's corank-two maps from three-space to four-space start with the
component $x$. The last three components are
listed below in the original target coordinates. [Sharland, equations (6)–(7) and Table 1](https://arxiv.org/pdf/1202.4992#page=6)

| Germ | Last three components | Published $\mathcal A_e$-codimension |
| --- | --- | ---: |
| $\widehat A_2$ | $y^2+xz+x^2y,\ yz,\ z^2+y^3$ | 18 |
| $\widehat B_3$ | $y^2+xz,\ z^2-xy,\ y^3+y^2z+yz^2-z^3$ | 33 |
| $\widehat A_3$ | $y^3+xz+x^4y,\ yz,\ z^2+y^5$ | 186 |
| $\widehat B_5$ | $y^2+xz,\ z^2+\alpha xy,\ y^5+y^4z+yz^4-z^5$, $\alpha=\pm1$ | 252 |
| $\widehat A_4$ | $y^4+xz+x^6 y,\ yz,\ z^2+y^7$ | 844 |
| $\widehat B_7$ | $y^2+xz,\ z^2+\alpha xy,\ y^7+y^6z+yz^6-z^7$, $\alpha=\pm1$ | 837 |

The complete Atlas kernel was measured on the parsed original maps with a
120-second timeout and 2 GiB of address space per process on a Ryzen 7 6800H.
The old kernel ran three times per map. The new kernel ran 15 times per map,
each in a fresh process on the same pinned CPU used for that map's baseline.

| Germ | Before (revision `66a47b6`) | After, single call (median) | After, warmed (median) |
| --- | --- | ---: | ---: |
| $\widehat A_2$ | 120 s timeout in all 3 trials | 0.792 ms | 0.519 ms |
| $\widehat B_3$ | memory limit after 32.05, 45.03 and 61.30 s | 3.301 ms | 2.203 ms |

The original graph calculation used 11 variables. For $\widehat A_2$ its
472-element Gröbner basis stalled in rational verification. For $\widehat B_3$
the reconstruction was still incomplete after 127 primes and the next replay
ran out of memory. Weighted tangent ranks now prove the dimensions at weights
10 and 7, respectively. The regression suite also adds the square of the
first target coordinate to the fourth and independently recovers both
dimensions using ordinary filtered jets. [Samples and reproduction instructions](ae-sharland-kernel-ryzen-6800h.json)

For $\widehat A_3$ and both signs of $\widehat B_5$, revision `3c5ca16`
timed out at 120 seconds in all nine trials under the same 2 GiB limit.
The following medians time the complete native solve after parsing.
First-call wall times use 15 fresh processes per map. The two CPU-time
columns compare nine fresh processes per build on the same logical core.
The dense control uses the previous algorithm with larger work limits so
that both builds finish and prove the same answer.

| Original map | New first-call wall time | Dense control CPU | New kernel CPU |
| --- | ---: | ---: | ---: |
| $\widehat A_3$ | 2.171 ms | 30.441 ms | 2.090 ms |
| $\widehat B_5$, $\alpha=+1$ | 52.605 ms | 241.631 ms | 33.852 ms |
| $\widehat B_5$, $\alpha=-1$ | 33.746 ms | 237.188 ms | 40.568 ms |

The dense solver reached 186 at weight 22 on $\widehat A_3$ but ran out of
matrix cells just before proving the last zero at weight 32. On $\widehat B_5$
the first oversized matrix occurred at weight 11. The completed dense control
spent 118 ms of its 206 ms weighted stage on exact integer rank for
$\alpha=+1$. The new solver carries the preceding weight's tangent quotient
forward using a monomial target and certifies both signs at weight 17.
[Raw measurements, profiles and reproduction instructions](ae-sharland-next-kernel-ryzen-6800h.json)

The following are complete kernel times for the next three presentations on
the same Ryzen 7 6800H. Each new wall time is the median of nine fresh processes,
with parsing and startup excluded. The CPU times were measured separately on
one pinned core. The run at revision `33cfd19` used the same 120 s and 2 GiB
limits and the same per-map pins as the new wall-time measurements.

| Original map | Previous kernel, wall | New kernel, wall | New kernel, thread CPU | Last certified weight |
| --- | ---: | ---: | ---: | ---: |
| $\widehat A_4$ | 40.33 ms | 41.13 ms | 23.85 ms | 62 |
| $\widehat B_7$, $\alpha=+1$ | 120 s timeout twice | 1.100 s | 0.831 s | 27 |
| $\widehat B_7$, $\alpha=-1$ | 120 s timeout twice | 0.880 s | 0.849 s | 27 |

At weight 16 for $\widehat B_7$ the projected matrix has 86 rows and 113
columns. The old work estimate charged each pivot the largest coefficient
height in the whole matrix. The new estimate bounds minors by the individual
column heights. Composition of the exact dual bases now skips zero entries of
the change-of-basis matrix and charges the products it performs.
[Samples, certificates and reproduction instructions](ae-sharland-47-kernel-ryzen-6800h.json)
