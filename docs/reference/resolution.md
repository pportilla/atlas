# Resolution, zeta functions and monodromy

Atlas resolves a hypersurface germ $f=0$ at the origin of affine space over $\mathbb Q$. Every calculation starts with `resolve(f)`, which returns a resolution object. The charts, the exceptional divisors, the dual complex, and the local topological and monodromy zeta functions are all fields of that object. `monodromy_check(f)` tests the monodromy conjecture for the germ, and `zoo_search` looks up resolved examples in the Resolution Zoo.

This page describes the functions. Every field of the values they return is listed in [Resolution objects and zeta functions](resolution-values.md). For a first calculation, follow [Resolving a plane cusp](../tutorial/resolving-a-cusp.md). The mathematics of the zeta functions is in [Zeta functions and the monodromy conjecture](../thematic/zeta-functions.md).

## From a polynomial to its zeta functions

Declare the coordinates, write the equation, resolve it and read what you need from the result.

```atlas
R = ring("x", "y")                     # the ring Q[x, y] and its coordinates x, y
f = y^2 - x^3                          # the plane cusp
Z = resolve(f)                         # an embedded resolution of f = 0 at the origin
print(Z.divisors)                      # [E1, E2, E3]
print(Z.topological_zeta.expression)   # (4*s + 5)/((s + 1)*(6*s + 5))
print(Z.monodromy_zeta.expression)     # (1-T^6)/((1-T^2)*(1-T^3))
print(monodromy_check(f).status)       # verified
```

The table lists the most common questions. `Z` is a resolution and `f` its equation.

| To obtain | Write | Details |
| --- | --- | --- |
| An embedded resolution | `Z = resolve(f)` | [`resolve`](#resolve) |
| A resolution of the hypersurface itself | `resolve(f, mode="intrinsic")` | [Resolution modes](../thematic/resolution-records.md#resolution-modes) |
| A resolution by weighted blowups | `resolve(f, method="weighted")` | [Weighted resolution](../thematic/weighted-resolution.md) |
| The charts and their maps | `Z.charts`, `Z.chart("B3.u").map` | [Charts](resolution-values.md#charts) |
| The exceptional divisors and their multiplicities $N$ | `Z.divisors`, `Z.divisor("E3").total_multiplicity` | [Exceptional divisors](resolution-values.md#exceptional-divisors) |
| The numerical data $(N_i,\nu_i)$ of all components | `Z.topological_zeta.numerical_data` | [Topological zeta function](resolution-values.md#topological-zeta-function) |
| The dual complex of the exceptional divisor | `Z.dual_complex` | [Dual complex](resolution-values.md#dual-complex) |
| The divisor met by a branch of a plane curve | `Z.branches[0].meets` | [Branches](resolution-values.md#branches-of-a-plane-curve) |
| The local topological zeta function | `Z.topological_zeta.expression` | [Topological zeta function](resolution-values.md#topological-zeta-function) |
| Its poles | `Z.topological_zeta.poles` | [Topological zeta function](resolution-values.md#topological-zeta-function) |
| The local monodromy zeta function | `Z.monodromy_zeta.expression` | [Monodromy zeta function](resolution-values.md#monodromy-zeta-function) |
| The Euler characteristic of the Milnor fibre | `Z.monodromy_zeta.milnor_fiber_euler_characteristic` | [Monodromy zeta function](resolution-values.md#monodromy-zeta-function) |
| Monodromy zeta functions at points near the origin | `Z.monodromy_zeta.nearby_strata` | [Monodromy near the origin](../thematic/zeta-functions.md#monodromy-near-the-origin) |
| A witness for each pole (the monodromy conjecture) | `monodromy_check(f).poles` | [`monodromy_check`](#monodromy_check), [Pole check](resolution-values.md#pole-check) |
| The complete record as JSON | `print_json(record=Z.record)` | [The resolution](resolution-values.md#the-resolution) |
| Catalogue records containing given monomials | `zoo_search("x^2, y^3").ids` | [`zoo_search`](#zoo_search) |

## Give the equation

The equation is a polynomial over $\mathbb Q$ that vanishes at the origin. A polynomial already knows its ring. A formula given as a string needs its coordinates, either with `variables=` or `ring=`, or from the ring selected with `use_ring`. To study a germ at another rational point, move the point to the origin with a ring map first.

```atlas
Z = resolve("y^2 - x^3", variables=["x", "y"])
print(Z.coordinates)                   # (x, y)
R = ring("x", "y")
f = (y - 1)^2 - x^3                    # a cusp at the point (0, 1)
move = hom(R, R, images=(x, y + 1))    # substitute y + 1 for y
print(move(f))                         # -x^3+y^2
```

Atlas reads `^` and `**` as powers and needs an explicit `*` for products. The command line and the HTTP API accept the same formulas with an ordered list of coordinates. [Command line](command-line.md#resolve-a-germ) · [HTTP requests](http-and-compute.md#request-a-resolution)

## Choose what to compute

The options of `resolve` choose the kind of resolution and how long Atlas may search for it.

| Option | Values |
| --- | --- |
| `mode` | `"embedded"` (default) makes the total transform a divisor with simple normal crossings and records the multiplicity of each component. `"intrinsic"` resolves the singularities of the reduced hypersurface. |
| `method` | `"auto"` (default) chooses the construction. `"weighted"` uses weighted blowups in embedded mode. |
| `search` | `"exhaustive"` (default) continues until the resolution is certified or you interrupt it. `"quick"` stops at a bound and then returns status `"inconclusive"`. A quick search has a 55-second deadline and reads at most eight variables, degree 128 and 4,096 expanded terms. |
| `monodromy` | `True` computes both zeta functions during the call. With `False` (default) they are computed the first time you read one of them. |
| `compute` | `"cpu"`, `"auto"` or `"gpu"`. Omit it to use the configured device, normally the CPU. [Select a device](http-and-compute.md#select-a-device) |

For the cusp, the embedded resolution needs three blowups and has three exceptional curves. The intrinsic resolution stops after one blowup, when the curve is smooth. Its exceptional locus on the curve is the single point `P1`.

```atlas
R = ring("x", "y")
f = y^2 - x^3
print(resolve(f).divisors)                     # [E1, E2, E3]
print(resolve(f, mode="intrinsic").divisors)   # [P1]
print(resolve(f, search="quick").status)       # resolved
```

The zeta functions are always computed on an embedded resolution. For an intrinsic or weighted resolution, Atlas resolves the same equation in embedded mode when you first read a zeta function. `monodromy_check(f)` needs only the zeta functions, so it can also use a certificate that avoids charts altogether. [Zeta functions and the monodromy conjecture](../thematic/zeta-functions.md)

## Search the Resolution Zoo

The Resolution Zoo is a catalogue of germs that Atlas has resolved, with both resolutions, both zeta functions and the pole check. Install the catalogue beside Atlas and open it in a browser with `./atlas serve`. `zoo_search` finds the records whose equation contains all the monomials you give, up to renaming the variables and changing the overall sign. [Install the Resolution Zoo](../installation.md#install-the-resolution-zoo)

```atlas
hits = zoo_search("x^2, -y^2*z")               # records containing x^2 and -y^2*z
print(len(hits.ids) > 0)                       # True
R = ring("x", "y", "z")
print(zoo_search([x^2, -y^2*z]).ids == hits.ids)   # True
```

Separate the monomials with commas, write them as a signed sum such as `x*y^2 - y^3`, or give a list. Coefficients must match exactly. All monomials share one renaming of the variables, and different query variables stand for different coordinates. Thus `zoo_search("1/2*x*y^3, -x^2")` matches an equation containing $-\tfrac12vu^3$ and $v^2$, even with further terms. A name like `xy` is a single variable. Text queries need no declared coordinates. Polynomial queries must come from one ring. A blank query or an empty list returns every record.

The result lists the IDs in catalogue order. The catalogue writes each equation in the coordinates `x, y, z, w, u, v, x7, x8, …` with its terms in descending lexicographic order, compares these term sequences, and orders identical equations by ID. The browser's search box accepts the same text. Selecting a record shows its name, labels and resolution engine, and **Copy LaTeX** and **Copy Python** copy its equation. For the record format, see [The Resolution Zoo catalogue](../thematic/resolution-records.md#resolution-zoo-catalogue).
