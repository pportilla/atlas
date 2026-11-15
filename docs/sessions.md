# Prompt and scripts

Run `./atlas` at a terminal. Assignments bind names quietly. Enter a name or
expression to inspect it. Tab completes names, fields such as `R.divisors`,
and methods such as `R.chart(`. Use `:workspace` to inspect the rings and
variables in use. Type `help functions` to list operations or `help resolve`
for its arguments and an example.

For a polynomial map germ, type `help ae_codimension`. The returned object's
`.finite` field tells you how to interpret `.value`.
[Map-germ inputs, results, and an example](../examples/algebra-api.md#extended-right-left-codimension-of-a-map-germ)

## Choose the ring of a polynomial

`coordinates("x", "y")` gives you the generators of `QQ[x, y]`. Repeating
the declaration in the same session gives you those same generators.

```text
atlas> x, y = coordinates("x", "y")
atlas> f = y^2 + x
atlas> x, y = coordinates("x", "y")
atlas> x.parent == f.parent
True
atlas> g = f + x
atlas> g.parent == f.parent
True
```

The `Rings` row now shows `QQ[x, y] → x, y, f, g`. The names to the right
of the arrow belong to that ring. Type `:workspace` for the full list.
Type `help f` or `help f.parent` to inspect a saved polynomial or its ring.
The same help command accepts a saved ideal, quotient ring, or ring map.

Atlas uses the coefficient field and the generator names in the order you
give them. Omitting `coeffs=QQ` selects the rationals. If a direct coordinate
declaration would replace a saved value, Atlas rejects the whole assignment
and suggests an available name. This also applies when selecting one of the
new generators with an index. To also work
in `QQ[x, z]`, use a different name on the left for its first generator.

```text
atlas> x2, z2 = coordinates("x", "z")
atlas> x2.parent
QQ[x, z]
atlas> x.parent == f.parent
True
```

Here `x2` and `z2` name the generators spelled `x` and `z` in the second ring. Similarly,
`x3, y3, z3 = coordinates("x", "y", "z")` selects `QQ[x, y, z]` while `x`,
`y`, and `f` still belong to `QQ[x, y]`. Asking for
`coordinates("y", "x")` also selects a different parent because the order changed.

To work with two independent copies of `QQ[x, y]`, request a fresh one
explicitly. You can still read a formula in either ring by passing its parent.

```text
atlas> u, v = coordinates("x", "y", fresh=True)
atlas> u.parent == f.parent
False
atlas> g = polynomial("x^2+y", ring=f.parent)
atlas> g.parent == f.parent
True
atlas> h = u^2 + v
atlas> h.parent == u.parent
True
```

Another ordinary call to `coordinates("x", "y")` returns the generators of
`f.parent`. A declaration with `fresh=True` explicitly requests a new ring
and may reuse the names on the left. Ordinary assignment and unpacking a
saved tuple also rebind names. To restore the original generators,
assign `x, y = f.parent.coordinates`.

## Move a polynomial between rings

To add `z` to a polynomial in `QQ[x, y]`, declare the larger ring under
available names and specify where each generator of the original ring goes.

```text
atlas> x, y = coordinates("x", "y")
atlas> f = y^2 + x^3
atlas> X, Y, z = coordinates("x", "y", "z")
atlas> include = hom(f.parent, z.parent, images=(X, Y))
atlas> g = include(f) + z
atlas> g.parent
QQ[x, y, z]
atlas> g == Y^2 + X^3 + z
True
```

`include.domain`, `include.codomain`, and `include.images` describe the map.
The images follow the generator order of the domain and are substituted
simultaneously. Between `QQ[x, y]` and `QQ[x, z]`, choosing images
`(x2, z2)` sends `y` to the generator spelled `z`. This choice is explicit.

Use `I.parent` for the live polynomial ring of an ideal. For example,
`I = ideal(f)`, then `polynomial("x+y", ring=I.parent)` uses the same
generators as `f`. The field `I.ring` contains the serializable description
and monomial order. Use `ideal(ring=f.parent)` for the zero ideal and
`polynomial("1/2", ring=QQ)` for a scalar. Both typed and quoted inputs to
`resolve` can use an existing ring: `resolve(f)` and
`resolve("y^2+x^3", ring=f.parent)` have the same original parent.

For `A = quotient_ring(I)`, `A.ambient` is the original polynomial ring,
`A.gens` are its residue classes, and `A.projection` sends ambient polynomials
to their classes. Compare inside the quotient with `A(f) == A(0)`. For
`J = eliminate(I, x)`, `J.inclusion` sends a polynomial in the surviving
coordinates back into the original ring.

## Inspect a cusp

`resolve(f)` uses embedded resolution by default. For a plane curve, the
vertices of its dual complex represent the exceptional curves above the
origin. An edge records an intersection point of two exceptional curves.
The strict transform is the closure of the original curve away from the
origin, lifted to the resolved surface. Its branches have their own attachments.

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

### Look inside a chart

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
defining polynomials through `C.ambient_ideal` in `C.ambient`; computations
with `C.coordinates` respect these relations.
`C.pullback(f)` substitutes those polynomials into `f`. The strict transform
has equation `v-1 = 0`. The exceptional divisor `E3` has equation `u = 0`,
so the branch meets it at `(u, v) = (0, 1)`.

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

## Run or save a calculation

```sh
./atlas run examples/cusp.atlas
./atlas run examples/chart-elimination.atlas
./atlas run examples/groebner.atlas
```

Scripts use the same functions, assignments, and field access as the prompt.
In a script, use `print(...)` to write output. The chart-elimination example
intersects the strict transform with the last exceptional curve, eliminates one
coordinate, and computes a normal form in the resulting ring.
[Chart elimination script](../examples/chart-elimination.atlas) ·
[Algebra operations and examples](../examples/algebra-api.md)

At the prompt a blank line finishes an indented block. Use
`:save "my session.atlas"` to save successful mathematical statements in
order, including explicit prints. Saving again replaces the file. A failed or
interrupted block restores the previous variable bindings. Replay the file with
`./atlas run "my session.atlas"`, or use `./atlas run -` to read standard input.
