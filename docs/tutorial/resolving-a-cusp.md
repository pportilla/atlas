# Resolving a plane cusp

Where does the cusp $y^2=x^3$ meet the exceptional divisor after resolution?
We can read the answer from the resolution graph, then check it in local coordinates.

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

The edges join `E1` to `E3` and `E2` to `E3`. Each vertex represents an
exceptional curve above the origin. The cusp contributes one branch, `P1`.
In a drawing of the graph, its arrowhead would attach to `E3`. The branch
record even tells us where to look for that point, in the chart `B3.u`.

Why did the calculation take three blowups? Look at the chart through the
branch immediately after the first one.

```text
atlas> first = K.chart("B1.u")
atlas> first.map
{'x': u, 'y': u*v}
atlas> first.strict_transform
v^2-u
```

Substituting $x=u$, $y=uv$ gives $u^2(v^2-u)$. The strict transform
$u=v^2$ is already smooth, but at the origin it is tangent to the exceptional
curve $u=0$. The default, embedded resolution continues until the whole divisor
has normal crossings. To resolve the curve intrinsically, ask for
`resolve(f, mode="intrinsic")`. Its exceptional locus on the smooth curve
consists of one point. [Resolution modes](../thematic/resolution-records.md#resolution-modes)

## Look inside a chart

The names `B3.u` and `B3.v` denote the two charts of the third blowup.
We can look up the first by name or reach it directly from the branch.

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

The map expresses the original $x,y$ in the new coordinates $u,v$.
Here `E3` has equation $u=0$, and the strict transform has equation $v-1=0$.
They meet transversely at $(0,1)$, precisely the point in the branch record.

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

Thus `E3` occurs with multiplicity six in the total transform. The factor
$v^3$ belongs to `E2`, which misses the branch point because $v=1$ there.
The same `E3` has a different local equation in the other chart of this
blowup. Use `E.charts` to find its charts and `E.equation_in(C)` to ask for
its equation in a particular one. For a divisor defined by several equations,
`E.ideal_in(C)` gives the local ideal.

The ring of the chart is `C.parent`. For example, use `C.pullback_map` to
transport a polynomial from the original ring, or read an expression directly
in the chart with `polynomial("v-1", variables=C.coordinates)`. In a chart
with relations, `C.ambient_ideal` gives those relations in the ambient ring
`C.ambient`. [Rings and ideals](rings.md#choose-the-ring-of-a-polynomial)

To reconstruct the incidence from the charts, use `recompute_dual_complex(K)`.
The cusp script also follows `E3` through both charts. The chart elimination
example computes the ideal of the branch point and reduces $v^2+v+1$ there
to $3$. [Reading a resolution record](../thematic/resolution-records.md) · [Cusp script](../../examples/cusp.atlas) · [Chart elimination script](../../examples/chart-elimination.atlas)
