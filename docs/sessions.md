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

Atlas uses the coefficient field and the generator names in the order you
give them. Omitting `coeffs=QQ` selects the rationals. Declaring
`x, y, z = coordinates("x", "y", "z")` binds those names to generators of
`QQ[x, y, z]`. The saved polynomial `f` still belongs to `QQ[x, y]`.
Arithmetic requires both polynomials to have the same parent. Asking for
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
`f.parent`.

## Inspect a cusp

```text
atlas> x, y = coordinates("x", "y")
atlas> f = y^2 - x^3
atlas> R = resolve(f)
atlas> R.polynomial == f
True
atlas> R.dual_complex
Dual complex (embedded, complete): 3 vertices, 2 edges
  E1 — E3, E2 — E3
atlas> R.dual_complex.vertices
[E1, E2, E3]
atlas> E = R.branches[0].meets[0]
atlas> E.charts
[B3.u, B3.v]
atlas> C = R.chart("B3.u")
atlas> u, v = C.coordinates
atlas> E.equation_in(C)
u
atlas> total = C.pullback(f)
atlas> total == u^6 * v^3 * C.strict_transform
True
atlas> total.multiplicity(u)
6
```

The same divisor has local equation `v` in chart `B3.v`. Use
`C.coordinates` to construct expressions in a chart and `R.coordinates` for
the original variables. To read a polynomial from text in the chart, use
`polynomial("v-1", variables=C.coordinates)`. For a divisor defined by several
equations, `E.ideal_in(C)` gives its generators.

`R.polynomial` is a polynomial in the original coordinate ring. The serialized
equation text is available as `R.record.polynomial`. Each recorded vertex in
`R.dual_complex.vertices` is one of `R.divisors`. The strict transform meets
`E3`, as shown by `R.branches[0].meets`. Type `help R.dual_complex` for the
geometric convention. To rebuild the incidence from the charts, use
`D = recompute_dual_complex(R)` and inspect `D.cells`. The result has its own
vertex records. The old `dual_complex(R)` spelling returns the raw replay
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
