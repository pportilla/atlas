# Performance log: Jung's method for surfaces

`jung_resolution` / `atlas jung` ([guide](../thematic/jung-method.md)) on the test surfaces,
next to `atlas resolve` on the same inputs. Reproduce with
`python3 benchmarks/jung_timings.py --trials 5 --cap 40`.

## How each result is verified

Every result is checked against data computed independently of the construction:

- **Laufer's formula.** For a hypersurface (smoothable Gorenstein) surface
  singularity $\mu=12p_g+K^2+\chi_{\mathrm{top}}(E)-1$. The record computes the
  right-hand side from its own graph and genus (column *Laufer*). The Milnor
  number $\mu$ comes from Atlas's local algebra
  (`atlas algebra milnor-number`, `milnor_number(f)`), a separate Gröbner
  computation in the local ring. The two agree exactly on every isolated surface below.
- **Known graphs.** `tests/test_jung.py` checks the Dynkin graphs of the
  rational double points (arm lengths of E8, E6, D4; the A20 chain), the cycle
  of three $(-3)$-curves of the cusp $T_{4,4,4}$, and the hand resolution of
  the key surface.
- **Internal certificates.** Every record has an integral canonical cycle
  (Gorenstein) and a negative definite intersection matrix. Each crossing's
  branch orbits match the index of their lattice. The local covering degrees
  over each exceptional line match its generic sheets. The cover curves get
  nonnegative integral genera.

## Test surfaces

12 CPUs, load average 17 from other sessions during the run. Five trials of
`atlas jung` and at most three of `atlas resolve --vars x,y,z` (stopped at
40 s); medians. *native* is the record's `elapsed_ms`; the CLI time adds
about 0.3 s of Python start-up.

| Surface | Equation | μ (local algebra) | p_g | K² | χ | curves (min. good / Jung) | Laufer | `jung` native | `atlas jung` CLI | `atlas resolve` CLI |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| E8 | `x^2+y^3+z^5` | 8 | 0 | 0 | 9 | 8 / 8 | ✓ 8 | 4 ms | 331 ms | 216 ms |
| A20 | `x*y-z^21` | 20 | 0 | 0 | 21 | 20 / 22 | ✓ 20 | 20 ms | 429 ms | 394 ms |
| | `x^3+y^3+z^4` | 12 | 1 | -4 | 5 | 4 / 7 | ✓ 12 | 22 ms | 427 ms | 400 ms |
| cusp $T_{4,4,4}$ | `x*y*z+x^4+y^4+z^4` | 11 | 1 | -3 | 3 | 3 / 15 | ✓ 11 | 310 ms | 888 ms | 434 ms |
| D4 | `(x+y)^2+z^3+x^3` | 4 | 0 | 0 | 5 | 4 / 4 | ✓ 4 | 21 ms | 486 ms | 1133 ms |
| order 4 | `(x*y-z^2)^2+x^5+y^5` | 43 | 4 | -16 | 12 | 11 / 19 | ✓ 43 | 4890 ms (1.7 s unloaded) | 5259 ms | 1408 ms |
| E6 | `x^2+(y+z)^3+y^4` | 6 | 0 | 0 | 7 | 6 / 6 | ✓ 6 | 7 ms | 408 ms | 1351 ms |
| | `x^2*y+y^4+z^3` | 10 | 1 | -6 | 5 | 4 / 5 | ✓ 10 | 27 ms | 444 ms | 418 ms |
| E6 | `(x+2*y)^2+(y+z)^3+z^4` | 6 | 0 | 0 | 7 | 6 / 6 | ✓ 6 | 6 ms | 342 ms | 1404 ms |
| key | `z^2 - (x^2*(x+2*y)^2 + 4*y^3*(x+y)^2)*(x^2*(x+2*y)^2 + 4*y*(x+y)^2)` | 39 | 3 | -8 | 12 | 11 / 11 | ✓ 39 | 47 ms | 427 ms | no result within 40 s |
| non-isolated | `(x+y+z)^2-z*(x-y)^2` | ∞ | | | | | declined | 13 ms | | 451 ms |

The key surface matches the hand reference exactly: 11 curves, a central
$(-4)$-curve with one $(-2)$-leaf and three $D_4$-forks of $(-2)$-curves,
$K^2=-8$, $\chi_{\mathrm{top}}=12$, $p_g=3$, $\mu=39$. Jung's method needs 7
point blowups of the discriminant and no Gröbner basis. The canonical route of
`atlas resolve` does not finish within 40 s.

`atlas resolve` is faster through its CLI on several rational double points:
the Jung CLI spends most of its time starting Python, and `atlas resolve`
starts natively through `resolve_supervisor`. The Jung engine itself takes
4–50 ms on eight of the ten surfaces. The two exceptions are the cusp
(0.3 s), where the global sheet test runs four times, and the order-4
surface. Its discriminant $(x^2y^2+x^5+y^5)(x^5+y^5)^2$ has its tangent
directions at the fifth roots of $-1$, so the local series there have
coefficients of degree 4 over $\mathbf Q$.

## Changes that made it fast

Each change is generic; the answers did not change.

1. **Rational fast path.** FLINT's `qqbar` computes an enclosure for every
   operation, even on rationals. `native/jung_coef.hpp` keeps rationals in
   `fmpq` and switches to `qqbar` only for irrational numbers. Unit series
   are inverted by Newton's iteration instead of a geometric series. The cusp
   went from more than 200 s to 0.1 s; the time had been in inverting series
   inside Newton's method for roots.
2. **Target precision.** Roots are lifted only to the precision a test
   needs, and the box of the tree comes from the tree's own differences:
   $B\ge\sum n_\nu\delta_\nu$ and $B\ge$ target $+\sum(n_\nu-1)\delta_\nu$
   along each path.
3. **Certified small boxes.** The first box is $D(\alpha+2)$ instead of
   $D(m\,n_{\max}\alpha/2+2)$; every node certifies its own precision
   ($B\ge n\delta$ and $\delta$ attained) and the box doubles otherwise. The
   order-4 surface went from 23 s to 1.7 s; its local boxes had been 6–8 times
   larger than the trees needed.
4. **Cyclic covers.** For $t^m+g(p,q)$ the group of $m$-th roots of unity
   labels branches and curves. This replaces the generic test, which on
   $z^4+x^5+y^6$ did not finish in 60 s.
5. **Cheap rejection.** A candidate set of sheets is first tested with four
   Taylor degrees: a nonzero coefficient already proves it is not a union of
   curves. Only survivors get the full certified precision. Sets of at most
   half the remaining sheets are tested, since a complement of an invariant
   set is invariant.

## Further surfaces (stress corpus)

38 more surfaces: Brieskorn–Pham, $T_{p,q,r}$, non-quasihomogeneous and
higher-order equations. All 33 isolated surfaces that finished agree with
Laufer's formula; the two non-isolated ones (`x^3+y^2*z^2+z^5`,
`(x^2-y^3)^2+z^3`) are declined. Slower cases, run alone:

Measured alone at load average 19–25 (other sessions), single trials unless a
range is given; the earlier numbers in parentheses were taken at lower load.

| Equation | μ | p_g | Laufer | `jung` native | Bottleneck |
| --- | ---: | ---: | --- | ---: | --- |
| `(x*y-z^2)^2+x^5+y^5` | 43 | 4 | ✓ | 4.6–6.8 s (1.7 s) | coefficients in $\mathbf Q(\zeta_{10})$ |
| `x*y*z+x^4+y^4+z^4` | 11 | 1 | ✓ | 0.33–0.42 s | four global sheet tests |
| `z^4+x^5+y^6` | 60 | 6 | ✓ | 13.9–14.2 s (8.9 s) | not profiled yet |
| `x^3+y^3+z^3+x*y*z*(x+y+z)` | 8 | 1 | ✓ | 22.7 s at load 25 (1.0 s) | global sheet test |
| `x^4+y^4+z^4+x^2*y*z` | 27 | | | > 60 s | one local analysis with algebraic coefficients |
| `(x^2-y^3)^2+z^5+x*y^4` | 52 | | | > 60 s | sheets that separate at $u^8$ |

## Remaining slow cases

- Coefficients in number fields of degree $\ge 4$ make each `qqbar`
  operation expensive. A number-field representation (FLINT `nf_elem`, one
  primitive element per chart and tree path) is the next step.
- When the sheets over a curve separate late (high separation exponent $T$),
  the certified Taylor test needs $\binom{f}{s}T(1+N_E)$ terms with growing
  rational heights. Subtracting the common part of the sheets first would
  lower $T$.
