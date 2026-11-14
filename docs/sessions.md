# Prompt and scripts

Run `./atlas` at a terminal. Assignments bind names quietly. Enter a name or
expression to inspect it. Tab completes names, fields such as `R.divisors`,
and methods such as `R.chart(`. Use `:workspace` to inspect the rings and
variables in use. Type `help functions` to list operations or `help resolve`
for its arguments and an example.

## Inspect a cusp

```text
atlas> x, y = coordinates("x", "y")
atlas> f = y^2 - x^3
atlas> R = resolve(f)
atlas> R.divisors
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

Each call to `coordinates` creates a coordinate space over the rationals by
default. The same divisor has local equation `v` in chart `B3.v`. Use
`C.coordinates` to construct expressions in a chart and `R.coordinates` for
the original variables. To read a polynomial from text in the chart, use
`polynomial("v-1", variables=C.coordinates)`. For a divisor defined by several
equations, `E.ideal_in(C)` gives its generators.

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
