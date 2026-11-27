# Embedded resolution by weighted blowups

The surface $x^2+y^3+z^5=0$ has three terms of different degrees, but all
three have **weighted degree 30** if we give $x,y,z$ weights $15,10,6$.
A single weighted blowup with these weights resolves the germ on quotient
charts. We can see the weights in the chart maps and read the exceptional
multiplicity directly from the pulled-back equation.

## Resolve with method weighted

```text
atlas> x, y, z = coordinates("x", "y", "z")
atlas> R = resolve(x^2 + y^3 + z^5, method="weighted")
atlas> R.steps[0].invariant
['2', '3', '5']
atlas> R.divisors
[E1]
atlas> R.charts
[W1.c0, W1.c1, W1.c2]
atlas> C = R.charts[0]
atlas> C.map
{'x': t1^15, 'y': t1^10*t2, 'z': t1^6*t3}
atlas> C.strict_transform
t3^5+t2^3+1
```

In this chart $E_1$ has equation $t_1=0$. Substituting the chart map into
the input gives $t_1^{30}(1+t_2^3+t_3^5)$. The exponent of $t_1$ is the
order $N$ of the input along $E_1$. The pulled-back volume form
$dx\wedge dy\wedge dz$ has order $15+10+6-1=30$, so the numerical datum $\nu$ is $30+1=31$.

```text
atlas> E = R.divisor("E1")
atlas> E.total_multiplicity
30
atlas> E.nu
31
atlas> R.certificate.certified
True
```

The input to `resolve` is a rational hypersurface equation vanishing at the
origin. `method="weighted"` takes up to 16 variables and requires
`mode="embedded"`, the default. A smooth germ gets the identity chart.
With `monodromy=True`, Atlas retains the weighted charts and computes both
local zeta functions on an ordinary embedded companion for the same equation.
Its evidence is saved in `R.record["embedded_companion"]`. The input above
has topological zeta $(22s+31)/((s+1)(30s+31))$. Use `method="auto"` for the
ordinary dual complex of intersections between divisors. [Resolution modes](resolution-records.md#resolution-modes)

The same example from the repository directory is `./atlas resolve 'x^2+y^3+z^5' --vars x,y,z --method weighted`.
Add `--json` to save the full record. For `/api/resolve` or an HTTP background
search, include `"method": "weighted"` in the request object. [HTTP requests](../reference/http-and-compute.md)

## How a centre is chosen

For $x^2+y^3+z^5$, the invariant $(2,3,5)$ in `R.steps[0].invariant` records
the exponents in the centre $(x^2,y^3,z^5)$. Its Newton polyhedron has a face whose primitive
inward normal is $(15,10,6)$. In general the engine chooses local
coordinates adapted to the centre. It computes successive coefficient
collections in Tschirnhausen coordinates on a jet of the equation and
increases the jet order until the original equation satisfies the
admissibility test. The normal fan of the centre's Newton polyhedron gives
the chart maps. The record of each step includes the invariant
$(a_1,\ldots,a_k)$ and the monomial part $M$ of the centre
$J=(x_1^{a_1},\ldots,x_k^{a_k})+M$ along the existing boundary.

For the theory of the centre and invariant, see Abramovich, Temkin and
Włodarczyk. Abramovich and Quek describe the logarithmic construction in the
presence of a boundary. [*Functorial embedded resolution via weighted blowings up* (2024)](https://arxiv.org/abs/1906.07106) · [*Logarithmic resolution via multi-weighted blow-ups* (2024)](https://epiga.episciences.org/14648)

### A centre of positive dimension

For $(xy-z^2)^2+x^5+y^5$ the projectivized tangent cone is a double conic.
In the $x$-chart of the first blowup, put $y=xv$ and $z=xw$. The strict
transform has equation $(v-w^2)^2+x(1+v^5)=0$. Along a general point of
the conic $x=v-w^2=0$ the surface is smooth and tangent to the exceptional
plane. The next step therefore needs a centre of positive dimension.

```text
atlas> x, y, z = coordinates("x", "y", "z")
atlas> S = resolve((x*y - z^2)^2 + x^5 + y^5, method="weighted")
atlas> S.steps[0].locus
point
atlas> S.steps[1].locus
positive dimension
atlas> len(S.steps)
4
atlas> S.certificate.certified
True
```

To find the next centre Atlas decomposes the locus where the strict transform
still has logarithmic order at least two. An isolated point defined over a
number field stands for its whole Galois orbit. For a positive-dimensional
component $Z$, the calculation takes place at its generic point. A Gröbner
basis supplies equations $X_1,\ldots,X_c$ in the boundary stratum. A basis
element linear in a variable of its own can replace that variable. For any
other element the engine introduces a chart coordinate $w_i$ and the relation
$w_i=X_i$. Expanding the equation in these coordinates exposes coefficients
that are units at the generic point. When a Tschirnhausen change could alter
the invariant, the engine also compares it with the jet calculation at a
random point of $Z$ and tests admissibility.

In a chart with these extra relations, logarithmic Jacobian minors of the
ambient equations together with the strict equation locate the remaining
bad points. At a point the ambient equations have order one and enter the
coefficient collection. A chart that meets a centre along a face uses the
centre already chosen at its generic point. A locus contained in a larger
one is blown up with it. Where two loci meet, the one with the larger
invariant governs their common points.

## Charts of weighted blowups

Each maximal cone gives a chart. For a regular cone the map is a chart of
the blowup itself. For a nonregular cone Atlas writes the monomial map on a
finite cover. `orbifold_order` gives its degree, the absolute determinant
of the ray matrix. For example, the first chart of $x^2+y^3+z^5$ has degree 15.

```text
atlas> x, y, z = coordinates("x", "y", "z")
atlas> R = resolve(x^2 + y^3 + z^5, method="weighted")
atlas> C = R.chart("W1.c0")
atlas> C.map
{'x': t1^15, 'y': t1^10*t2, 'z': t1^6*t3}
atlas> C.orbifold_order
15
```

The certificate checks that the strict transform and exceptional divisors
cross normally **on these chart covers over the origin**. The following
fields explain the extent of a chart and its role in the certificate.

| Field | How to read it |
| --- | --- |
| `ambient_ideal` | Equations satisfied by the chart coordinates, including any added to describe a centre. |
| `coefficient_field` | Field of definition for a chart at a Galois orbit of points. |
| `over` and `owned_faces` | Equations for the locus covered and the coordinate faces assigned to this chart. On each face the listed coordinates vanish and the other cone coordinates are nonzero. |
| `divisors` on the resolution | Exceptional components with $N$ and $\nu$. The weighted engine currently lists the components only, so the status of their dual complex is `unavailable`. |

When a later blowup replaces part of an earlier chart, the certificate
compares the removed points with the locus where normal crossings failed.
[Weighted chart construction and verification](../../native/weighted_resolution.cpp)

## Limits

Exhaustive weighted resolution runs without a wall-clock limit. The weighted
worker represents orders by signed native words. It can also encounter a
chart whose admissible centre or normal-crossings certificate it cannot
construct. In these cases Atlas continues with ordinary embedded resolution
of the original polynomial and computes the selected zetas. The record's
`engine` identifies the algorithm that produced the certified geometry.
`requested_method` and `weighted_attempt.reason` identify the original
method and the reason for switching.

An explicit quick search gives the weighted construction and its
certificate 25% of its time budget each, and at most 40 seconds.
After a decline it continues with the ordinary embedded resolution, as an
exhaustive search does, and is `inconclusive` only when the budget ends. [Weighted resolution benchmark](../../benchmarks/weighted-resolution.md)
