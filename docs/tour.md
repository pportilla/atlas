# A tour of Atlas

Atlas computes exactly with polynomials over the rationals, studies a variety at a rational point, and resolves hypersurface singularities. This page shows a few short calculations at the Atlas prompt. To try them, [install Atlas](installation.md), type `./atlas` in the project directory and enter the lines that follow `atlas>`.

## Polynomials and ideals

In $\mathbb Q[x,y]$, suppose we impose the relation $x^2=y$. What does $x^3$ become? Name the ring, put the relation in an ideal, and ask for the normal form.

```text
atlas> R = ring("x", "y")
atlas> I = ideal(x^2 - y)
atlas> normal_form(x^3, I)
x*y
atlas> membership(x^3 - x*y, I).value
True
```

The answer says that $x^3$ and $xy$ represent the same class modulo $I$, and the membership test confirms that their difference lies in the ideal. Factorization also takes place over $\mathbb Q$.

```text
atlas> factor(x^4 - y^2)
1*(x^2-y)*(x^2+y)
```

## A singular point

The curve $y^2=x^3$ has a cusp at the origin. Its Milnor number is two, and its tangent cone is the double line $y^2=0$.

```text
atlas> f = y^2 - x^3
atlas> milnor_number(f).value
2
atlas> tangent_cone(ideal(f)).generators
(y^2,)
```

## Resolve the cusp

Three point blowups resolve the cusp. The exceptional curves `E1` and `E2` each meet `E3`, and the branch of the curve passes through `E3`. The local topological and monodromy zeta functions of the cusp are read from the same resolution.

```text
atlas> K = resolve(f)
atlas> K.dual_complex
Dual complex (embedded, complete): 3 vertices, 2 edges
  E1 — E3, E2 — E3
atlas> K.branches[0].meets
[E3]
atlas> K.topological_zeta.expression
(4*s + 5)/((s + 1)*(6*s + 5))
atlas> K.monodromy_zeta.expression
(1-T^6)/((1-T^2)*(1-T^3))
```

## Keep the calculation

Type `:save "tour.atlas"` to write the statements you have entered to a file. The command `./atlas run tour.atlas` repeats the calculation from the shell. At the prompt, `help milnor_number` describes a function and `help K` describes the resolution you saved.

## Where to go next

The [tutorial](tutorial/index.md) explains each of these calculations and many more, chapter by chapter. [Constructions](constructions.md) answer short questions of the form “How do I compute … in Atlas?” The [reference manual](reference/index.md) documents every function and every field of its results, and the [thematic tutorials](thematic/index.md) treat zeta functions and the monodromy conjecture in depth.
