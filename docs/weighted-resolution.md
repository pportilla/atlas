# Embedded resolution by weighted blowups

The weighted engine resolves a hypersurface germ $f = 0$ at the origin of
$\mathbf A^n$ over $\mathbf Q$ by the weighted blowups of Abramovich, Temkin
and Włodarczyk, in their logarithmic form (Abramovich and Quek). Every centre,
blowup, search for bad points and certificate runs in the native engine
[`native/weighted_resolution.cpp`](../native/weighted_resolution.cpp); Python
only sends the polynomial and presents the record
([`engine/weighted_blowups.py`](../engine/weighted_blowups.py)).

## Resolve with method weighted

`resolve` and `atlas resolve` accept a method. The default, `auto`, keeps
Atlas's usual choice among its engines. `method="weighted"` resolves by
weighted blowups.

```text
x, y, z = coordinates("x", "y", "z")
R = resolve((x*y - z^2)^2 + x^5 + y^5, method="weighted")
R.steps
R.divisors
R.certificate
```

From the shell, `atlas resolve "(x*y-z^2)^2+x^5+y^5" --vars x,y,z --method weighted`
prints the charts, and `--json` the complete record. The HTTP API takes
`"method": "weighted"` in the request object, for `/api/resolve` and for
background searches.

The resolution is embedded, so the mode must be `embedded` (the default). The
engine computes no zeta functions yet, so a request for `monodromy` or `zetas`
is refused; use `method="auto"` for them. The origin must lie on the
hypersurface. A germ that is smooth at the origin gets the identity chart.

The record has `engine: "weighted"`:

- `steps` lists the weighted blowups in order. Each gives the point or locus it
  was made at, the invariant $(a_1, \dots, a_k)$ of its centre
  $J = (x_1^{a_1}, \dots, x_k^{a_k}) + M$, the monomial part $M$ as exponent
  vectors on the boundary coordinates, and whether the centre is a point or a
  locus of positive dimension.
- `charts` gives each chart's coordinates `t1, t2, …`, the original
  coordinates as polynomials in them (`map`), the strict transform and the
  exceptional divisors through the chart. A chart made at an orbit of points
  defined over a number field carries that field as `coefficient_field`; one
  chart then stands for the whole Galois orbit. A chart that carries equations
  in `ambient_ideal` is the subvariety they define (see below). `over` holds
  equations of the locus whose preimage the chart owns, and `owned_faces` the
  sets of its coordinates whose common zeros, with the other coordinates of its
  cone nonzero, it owns there.
- `divisors` (the vertices of `dual_complex`) are the exceptional divisors,
  each with its order `total_multiplicity` $N$ of $f$ and $\nu$, its log
  discrepancy plus one. The engine does not compute their intersections, so the
  dual complex has no cells and the status `unavailable`.
- `certificate` says that the strict transform and the exceptional divisors
  cross normally over the origin on every face that each chart owns. The
  kernel's normal-crossings test checks every chart; points that a chart hands
  to a later blowup are compared with the bad loci it found.

## How a centre is chosen

At a point, the engine computes the invariant from coefficient collections in
Tschirnhausen coordinates on a jet of $f$. The jet grows until the centre is
admissible for $f$ itself. The blowup is the toric modification given by the
normal fan of the Newton polyhedron of $J$; each chart is the monomial map of
one cone. The points over the centre where the strict transform still has
logarithmic order at least two are found as the prime components of a finite
set over $\mathbf Q$, one Galois orbit at a time.

A bad locus over the centre can have positive dimension: after one blowup the
strict transform of $(xy - z^2)^2 + x^5 + y^5$ is singular along a conic in the
exceptional plane. The engine decomposes such a locus with the native primary
decomposition and computes the centre at the generic point of each component
$Z$. On its boundary stratum, $Z$ is cut out by a Gröbner basis
$X_1, \dots, X_c$. An $X_i$ that is linear in a variable of its own becomes
that variable by a triangular change of coordinates. Any other $X_i$ becomes a
new coordinate $w_i$ with the ambient equation $w_i = X_i$, and $f$ is written
in powers of the $w_i$. Its coefficients are then units at the generic point of
$Z$, where the centre is read off in the coordinates $X$. When a Tschirnhausen
change could alter the result, the engine checks the invariant against the jet
centre at a random point of $Z$. An admissible centre with that invariant is
the centre of Abramovich, Temkin and Włodarczyk along $Z$, the unique
admissible centre of maximal invariant there.

Blowing up a centre with a new coordinate $w$ gives charts that are
subvarieties of affine space, given by `ambient_ideal`. On them the engine
searches bad points with the logarithmic Jacobian minors of the ambient
equations and $f$. At a point it puts the ambient equations into the
coefficient collection, where they have order one, so the centres and blowups
there stay toric in a full set of coordinates.

A point of a chart's face that lies on a bad locus whose general points belong
to another chart gets that locus's centre, blown up for the points over it
only. A locus inside a larger bad locus is blown up with the larger one, and
where two loci meet, the one with the larger invariant owns the points.

## Charts of weighted blowups

A cone of the fan that is not regular gives a chart that covers the blowup
with a finite cyclic group of order `orbifold_order`, its determinant; the
blowup has a cyclic quotient singularity there. The certificate concerns these
charts. Every other chart is a chart of the blowup itself.

## Limits

- A bad locus of positive dimension on a chart that already carries ambient
  equations is declined with its reason.
- Each resolution and its certificate have 40 seconds; the engine runs in a
  forked helper that Atlas stops at the deadline.
- No zeta functions yet: they need the strata of the weighted charts.

The timings against `method="auto"` on the same inputs, with their conditions,
are in [the weighted resolution benchmark](../benchmarks/weighted-resolution.md).
