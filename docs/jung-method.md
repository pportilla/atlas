# Jung's method for surface singularities

`jung_resolution` resolves an isolated surface singularity $f(x,y,z)=0$ at the
origin over $\mathbf Q$ and returns its resolution graph, canonical cycle,
$K^2$, the Euler characteristic of the exceptional set and the geometric genus
$p_g$. All of the mathematics runs in the native program
[`native/jung.cpp`](../native/jung.cpp); Python only parses the input and
presents the record. The method does not use Gröbner bases or a centre search:
it is a projection, one resultant, a plane curve resolution and toric
calculations with integer matrices.

## Resolve a surface with Jung's method

At the prompt or in a script:

```text
x, y, z = coordinates("x", "y", "z")
J = jung_resolution(x^2 + y^3 + z^5)
J.curves
J.edges
print(J.K2, J.euler_characteristic, J.geometric_genus, J.milnor_number_from_laufer)
```

From the shell, `atlas jung "x^2+y^3+z^5"` prints the graph and
`atlas jung "x^2+y^3+z^5" --json` the complete record. `--vars` changes the
three variable names (default `x,y,z`).

The value describes the minimal good resolution: Jung's resolution after
contracting smooth rational $(-1)$-curves for as long as the exceptional set
stays a normal crossing divisor of smooth curves. `J.jung_graph` keeps the
graph before the contractions.

## Resolve with method jung

`resolve` and `atlas resolve` accept a method. The default, `auto`, keeps
Atlas's usual choice among its engines. `method="jung"` asks for Jung's
method: Atlas first checks that the method applies to this request and this
surface, and then uses it in the same call.

```text
x, y, z = coordinates("x", "y", "z")
R = resolve(x^2 + y^3 + z^5, method="jung")
R.dual_complex.vertices
R.invariants
R.certificate
```

From the shell, `atlas resolve "x^2+y^3+z^5" --vars x,y,z --method jung`. The
HTTP API takes `"method": "jung"` in the request object, for `/api/resolve`
and for background searches.

The request is checked first. Jung's method resolves the surface itself, so
its mode is `intrinsic` (the default when the method is `jung`; asking for
`embedded` is refused). The equation must have three variables. The method
gives no embedded resolution, so it cannot supply zeta functions, and a
request for `monodromy` or `zetas` is refused.

The native resolver then verifies, on this surface, each hypothesis it
uses, before using it:

| Hypothesis | Checked by |
| --- | --- |
| `origin_on_surface` | $f(0)=0$ |
| `reduced_irreducible_germ` | exactly one rational factor of $f$ vanishes at the origin, with multiplicity 1 |
| `regular_projection` | a direction with $f_m(d)\ne0$ and simple remaining roots of $f(td)$ |
| `reduced_discriminant` | $\operatorname{Res}_t(f,f_t)\ne0$ |
| `isolated_singularity` | over every branch of the discriminant the roots that agree form one orbit with exponent $1/n$, so the surface is smooth there |

The verified hypotheses are listed in `R.certificate["hypotheses"]`. When
one fails, the call stops with an error that names it, for example
`Jung's method does not apply here (isolated_singularity): ...`; the CLI
exits with status 2.

The result is an ordinary resolution record (`engine` is `jung`). Its dual
complex lists the curves of the minimal good resolution with `genus` and
`self_intersection`, and its cells carry the number of intersection points.
`R.invariants` gives $K^2$, $\chi_{\mathrm{top}}(E)$, $p_g$, the canonical
cycle and $12p_g+K^2+\chi-1$. `R.record["jung"]` is the complete record of
the method, the same one `jung_resolution` returns. The record has no
charts.

## How the method works

1. **Projection.** A fixed search over directions $d$ with entries in
   $\{-2,\dots,2\}$ (coordinate axes first) picks the fibre direction. It
   needs $f_m(d)\ne 0$ for $m=\operatorname{ord} f$, so that $f$ is regular of
   order $m$ in the fibre coordinate $t$, and the roots of $f(td)/t^m$ must be
   simple. Then the discriminant is the branch locus of the Weierstrass
   polynomial times a unit. Among valid directions it prefers the smallest
   degree in $t$, then the fewest terms.
2. **Discriminant.** $\Delta=\operatorname{Res}_t(f,\partial_t f)$, computed by
   FLINT. Its rational factors through the origin form the reduced germ.
3. **Resolution of the discriminant.** The plane engine's `Resolver`
   ([`native/plane_resolver.hpp`](../native/plane_resolver.hpp), shared with
   `atlas resolve --vars x,y`) blows up points until the total transform has
   normal crossings. It keeps every chart map, the exceptional axes and the
   points blown up later.
4. **Quasi-ordinary normalization.** Over each crossing of the total
   transform, $\Delta$ is $u^aw^b$ times a unit, so the Weierstrass polynomial
   is quasi-ordinary: its roots are series in $u^{1/D},w^{1/D}$ with
   $D=\operatorname{lcm}(1,\dots,m)$ whose differences are monomials times
   units. The engine splits off the Weierstrass factor by Hensel lifting in a
   truncated ring of power series, removes the mean of the roots, divides by
   the smallest difference monomial $u^{\delta}$ (the componentwise minimum of
   $e/k$ over the terms $u^e$ of the coefficient of $t^{n-k}$; it is attained)
   and splits again by Hensel's lemma. The result is a tree of roots. Every
   step checks that the truncation determines the next monomial and doubles
   the precision otherwise.
5. **Cyclic quotients.** The loops around $u=0$ and $w=0$ act on the roots
   (a constant at depth $k$ is multiplied by $\omega^{\delta_1+\dots+\delta_k}$).
   An orbit is a normalized branch, the toric surface of the positive quadrant
   in the lattice $N$ of loops fixing one of its roots. $N$ has index the
   number of roots in the orbit; its rays $(k_1,0)$ and $(0,k_2)$ give the
   ramification along both curves and the type $\frac1n(1,q)$, resolved by the
   Hirzebruch–Jung string of $n/q=[b_1,\dots,b_r]$.
6. **Covering curves.** Over each exceptional line $E$ of the plane
   resolution, the sheets at a generic point form a cover $C\to E$, branched
   only at the crossings. Its components get their genus from
   Riemann–Hurwitz and their self-intersection from
   $e\,C^2+\sum_P \mu_E(H_P)=f\,E^2$, where $e$ is the ramification index
   along $E$, $f$ the degree of $C\to E$, and $H_P$ the string curve next to
   $C$ at a point $P$ of $C$.
7. **Canonical cycle and invariants.** $K=\sum k_iE_i$ solves
   $K\cdot E_j=2g_j-2-E_j^2$; then $K^2=\sum k_i(2g_i-2-E_i^2)$ and
   $\chi_{\mathrm{top}}(E)=\sum(2-2g_i)-\#\{\text{intersection points}\}$.

### Which sheets form one curve

Local data often decide: a single sheet; a crossing with a single point of
the class over it; sections through one crossing; the two sheets of a double
cover, exchanged by its involution. A cyclic cover $t^m+g(p,q)$ is decided by
its group: the $m$-th roots of unity act on the roots compatibly with every
local monodromy, so the branches at a crossing $p$ form one orbit
$\mathbf Z/l_p$, and the curves over $E$ form $\mathbf Z/c_E$ with
$c_E=\gcd_p l_p$. Each branch lies on the curve given by its label mod $c_E$.
The labels start at an arbitrary branch at each crossing. Every choice gives
the same graph, because the dual graph of the plane resolution is a tree.

Otherwise the engine decides from the sheets at a generic point $(0,v^*)$ of
the creation chart of $E$. A set $S$ of sheets is a union of curves exactly
when the product $P_S$ of its root factors is single valued near $E$. Then
its coefficient of $u^j$ is a polynomial of degree at most $j$ in $v$. The
engine checks the Taylor coefficients at $v^*$ up to degree
$\binom fs\,j\,(1+N_E)$, where $N_E$ counts the blowups centred on $E$. Any
other $S$ has a nonzero coefficient below that bound, a bound that also
counts the poles a conjugate coefficient can have at blown-up points of
$E$. Only the coefficients up to the separation exponent of $S$ matter. A
first pass with four extra degrees rejects most sets cheaply. Sets of at most
half of the remaining sheets suffice, because the complement of an invariant
set is invariant. The exact factor, pulled back to every crossing, vanishes
at the local roots of exactly the points of that curve.
`record["global_monodromy_tests"]` counts how often this test ran, and
`record["cyclic_cover"]` whether the group labels were used.

### Geometric genus

$p_g=\dim\mathcal O_X/I$ for the adjoint ideal
$I=\{g:\ v_D(g)\ge -k_D \text{ for all } D\}$, because the Gorenstein form
pulls back with divisor $K$. A string curve is a toric divisor, so its
condition excludes the terms below a line at one local point. For a covering
curve $C$ with ray $n$ and adjacent ray $w'$, the terms of level
$\langle n,\alpha\rangle=c$ restricted to $C$ form a section of
$\mathcal O(-cC)|_C$ of degree $-cC^2$; vanishing at the corner to a higher
order forces the level to vanish. With $N_0=\max(-k_D)$ the monomials
$p^iq^jt^c$, $c<m$, $i+j+c<N_0$ span $\mathcal O_X/\mathfrak m^{N_0}$, and
$p_g$ is the rank of the conditions, computed exactly.

## Fields of a Jung resolution

| Field | Meaning |
| --- | --- |
| `J.curves` | curves of the minimal good resolution: `id`, `genus`, `self_intersection`, `kind` (`cover` over an exceptional line or `string` at a crossing), `over` |
| `J.edges` | `(curve, curve, number of intersection points)` |
| `J.K2`, `J.euler_characteristic` | $K^2$ and $\chi_{\mathrm{top}}(E)$ of the minimal good resolution |
| `J.canonical_cycle` | the coefficients $k_i$ of $K$ |
| `J.geometric_genus` | $p_g$ |
| `J.milnor_number_from_laufer` | $12p_g+K^2+\chi_{\mathrm{top}}(E)-1$ |
| `J.jung_graph` | the graph before contractions |
| `J.projection`, `J.discriminant` | the coordinate change, and the factors of $\Delta$ through the origin |
| `J.record` | the complete record, including every crossing with its chart, lattice, quotient type and string |

For a hypersurface singularity, Laufer's formula
$\mu=12p_g+K^2+\chi_{\mathrm{top}}(E)-1$ holds. The tests compare
`J.milnor_number_from_laufer` with `milnor_number(f)`, which Atlas computes
from the local algebra, independently of this construction.

## What is declined

- germs that are reducible or nonreduced at the origin, and surfaces singular
  along a curve (detected over the branches of the discriminant: there the
  roots that agree must form one orbit with exponent $1/n$);
- inputs for which no direction in the search box gives a valid projection;
- more than 20 sheets over one curve in the global test.

Timings and checks on the test surfaces are in the
[performance log](perf-log-jung.md).
