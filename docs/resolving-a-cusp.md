# Resolve a plane cusp

`resolve(f)` uses embedded resolution by default. For a plane curve, the
vertices of its dual complex represent the exceptional curves above the
origin. An edge records an intersection point of two exceptional curves.
The strict transform is the closure of the original curve away from the
origin, lifted to the resolved surface. Its branches have their own attachments.

## Follow the branch to the exceptional divisor

```text
atlas> x, y = coordinates("x", "y")
atlas> f = y^2 - x^3
atlas> K = resolve(f)
atlas> K.polynomial == f
True
atlas> K.dual_complex
Dual complex (embedded, complete): 3 vertices, 2 edges
  E1 — E3, E2 — E3
atlas> K.branches
[P1]
atlas> b = K.branches[0]
atlas> b.meets
[E3]
atlas> b.chart
B3.u
atlas> b.point
['0', '1']
```

The cusp has one branch. In a decorated resolution graph you would draw its
arrow at `E3`. Atlas records that arrow as `b.meets`. In intrinsic mode the
complex instead records the exceptional locus **on the resolved curve**.
For the cusp this is one point. Use `help K.dual_complex` or `help K.branches`
at the prompt.

`resolve(f, method="weighted")` resolves the same germ by weighted blowups
instead. Its record lists the blowups in `K.steps` and the exceptional divisors
with their numerical data, and certifies normal crossings over the origin; see
[Embedded resolution by weighted blowups](weighted-resolution.md#resolve-with-method-weighted).

## Look inside a chart

`K.charts` returns chart objects. The prompt displays each by its ID.
For the cusp, `B3.u` and `B3.v` are the two coordinate charts of the third
blow-up. Select an entry by position, look it up by ID, or follow a branch
directly to its chart.

```text
atlas> K.charts
[B1.u, B1.v, B2.u, B2.v, B3.u, B3.v]
atlas> K.charts[0].id
B1.u
atlas> C = K.chart("B3.u")
atlas> C == b.chart
True
atlas> C.coordinates
(u, v)
atlas> C.map
{'x': u^2*v, 'y': u^3*v^2}
atlas> C.divisors
[E3, E2]
atlas> C.strict_transform
v-1
```

`C.map` gives the original coordinates as polynomials in this chart.
`C.pullback_map` is the corresponding ring homomorphism, with the chart's
function ring `C.parent` as its codomain. In a chart with relations, read the
defining polynomials through `C.ambient_ideal` in `C.ambient`. Computations
with `C.coordinates` respect these relations.
`C.pullback(f)` substitutes those polynomials into `f`. The strict transform
has equation `v-1 = 0`. The exceptional divisor `E3` has equation `u = 0`,
so the branch meets it at `(u, v) = (0, 1)`.

## Check the multiplicity in the pullback

```text
atlas> u, v = C.coordinates
atlas> E = b.meets[0]
atlas> E.equation_in(C)
u
atlas> total = C.pullback(f)
atlas> total == u^6 * v^3 * C.strict_transform
True
atlas> total.multiplicity(u)
6
```

Type `help K.charts` for chart access, or type `C.` and press Tab for the
fields and methods on this chart. `E.charts` lists the charts containing
a local equation for `E`. The same divisor has equation `v = 0` in `B3.v`.
For a divisor defined by several equations, `E.ideal_in(C)` gives its generators.
To read a polynomial from text in the chart, use
`polynomial("v-1", variables=C.coordinates)`.

`K.polynomial` belongs to the original coordinate ring. Its serialized text
is available as `K.record.polynomial`. To rebuild the incidence from the charts,
use `D = recompute_dual_complex(K)` and inspect `D.cells`. The result has its
own vertex records. The old `dual_complex(K)` spelling returns the raw replay
dictionary for existing scripts.

The [cusp script](../examples/cusp.atlas) follows the branch through both charts containing `E3`. The [prompt and script checks](../tests/test_cli_session.py) verify the factorization and multiplicity shown above. The [plane-curve kernel](../native/plane.cpp) constructs the blowups. To inspect a saved calculation, continue to [resolution records](resolution-records.md).

To compute the intersection in this chart, run `./atlas run examples/chart-elimination.atlas`. The script imposes $u=0$ and $v-1=0$, eliminates $u$, then reduces $v^2+v+1$ to $3$. [Chart elimination script](../examples/chart-elimination.atlas)
