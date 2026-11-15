# Reading a resolution record

```sh
./atlas resolve 'y^2-x^3' --vars x,y --mode embedded --monodromy --json > cusp.json
./atlas recompute-dual-complex cusp.json
./atlas monodromy cusp.json --json
```

The first command saves the resolution and both zetas. The second reconstructs
the labelled complex from the saved record. The third computes the monodromy
zeta and prints its factors and A’Campo contributions. Both replay commands
accept `-` for standard input. The complex command also reads `.json.gz` files.

At the Atlas prompt, `R.dual_complex` reads the complex already stored by
`R = resolve(f)`. Use `recompute_dual_complex(R)` to rebuild incidence from the
charts. Its vertices carry the data recovered during that computation. The
recorded vertices retain their original annotations, such as self-intersections
and multiplicities. Use `R.dual_complex.record` for the full recorded complex.

Use distinct variable names beginning with a letter, explicit multiplication,
and nonnegative integer powers written with `^` or `**`. Coefficients may be
rational. The input polynomial must vanish at the origin. To study a germ at
another rational point, first translate that point to the origin.

## Resolution modes

Embedded mode certifies normal crossings of the total transform and retains
multiplicities, including those of repeated input components. Its complex is
that of the ambient exceptional divisor. Intrinsic mode takes a reduced germ,
resolves the strict transform, and describes its reduced exceptional divisor.
For the cusp the strict curve becomes smooth after one blowup. Embedded
resolution continues through the tangency and produces three exceptional curves.

The CLI uses exhaustive search by default. Use `--quick` for a bounded attempt.
Quick parsing allows eight variables, degree 128, 4,096 expanded terms, and
8,192 input characters. Integer literals may have up to 256 decimal digits,
and numerators and denominators may reach 4,096 bits during expansion. The
quick worker has a 55-second deadline. Exit code 3 signals an inconclusive
resolution or missing requested monodromy. [Search and verification](resolution-pipeline.md)

## Chart and complex conventions

Read each chart’s map as an expression for each original coordinate in the
chart coordinates. In a toric chart, substituting the map into the input
polynomial gives the pullback factor times the strict transform. A graph chart
has an `ambient_ideal` of relations. There its `strict_transform` gives the
additional equations in that coordinate ring.

Chart domains matter when following a later blowup. For plane charts a child
starts on the parent’s `base_domain` before the parent’s `exclude_points`
are applied. The parent still covers the complement of the replaced points.
The `algebraic_numbers` entries specify centres over extensions of the rationals.

A vertex identifies a geometric irreducible component of the exceptional divisor. Each higher cell
identifies a geometric component of an intersection. Two cells can therefore
have the same vertex set. Use cell IDs and the supplied codimension-one faces
to follow their attachments. The record reports the status of the complex.
When the data do not certify all attachments, inspect the diagnostic before
using the listed cells as a full complex. [How Atlas certifies and replays incidence](resolution-pipeline.md#what-the-record-certifies)

In an embedded plane-curve resolution, the strict transform of a branch maps
onto the original branch. The exceptional curves contract to points. In a
decorated resolution graph, an arrowhead marks where the strict transform
meets an exceptional curve. Atlas stores this attachment in the branch record.
For the cusp `y^2-x^3`, `K.dual_complex` gives the chain `E1 — E3 — E2`,
and `K.branches[0].meets` gives `[E3]`. To see that intersection in local
coordinates, open `C = K.branches[0].chart`, then read `C.strict_transform`
and `K.divisor("E3").equation_in(C)`. [Inspect a cusp at the prompt](sessions.md#inspect-a-cusp)

## Local monodromy zeta function

Atlas uses the Lefschetz convention for **unreduced** cohomology of the
complex Milnor fibre. For the total transform components with multiplicities
$N_i$ and open strata $E_i^\circ$, A’Campo’s formula gives

$$
\zeta_{f,0}(T)=\prod_i(1-T^{N_i})^{-\chi_c(E_i^\circ\cap\pi^{-1}(0))}.
$$

Each open stratum omits every other component, including the strict transform.
In the cusp record the exceptional multiplicities are 2, 3, and 6 and the open
Euler characteristics are 1, 1, and −1. This gives
$(1-T^6)/((1-T^2)(1-T^3))$. [A’Campo’s formula, theorem 3](https://www.e-periodica.ch/cntmng?pid=com-001%3A1975%3A50%3A%3A27)

## Topological zeta function

The local topological zeta uses all the disjoint strata $E_I^\circ$ of the
total transform. If $\nu_i-1$ is the order of the pulled-back volume form on
component $i$, then

$$
Z_{\mathrm{top},0}(s)=\sum_I \chi_c(E_I^\circ\cap\pi^{-1}(0))
\prod_{i\in I}\frac{1}{N_i s+\nu_i}.
$$

At the Atlas prompt, `M = monodromy_check(f)` checks the actual poles for
monodromy witnesses. Inspect `M.status` for the outcome and `M.poles` for the
individual poles and their witnesses.

For the cusp this gives $(4s+5)/((s+1)(6s+5))$. The `topological_zeta`
record includes the numerical data, strata, and poles. For an intrinsic
resolution that needs further blowups to compute the zetas, a new solve can
supply an embedded companion. Saved-record replay reports when the available
data are insufficient. [Strata and verification](../native/KERNEL.md#local-topological-zeta-reduction)
