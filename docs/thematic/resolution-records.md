# Resolution records

Every resolution that Atlas computes is a record, a JSON document with the charts, the exceptional divisors, their dual complex and, when requested, the zeta functions. At the prompt, a resolution `Z` is a view of this record, and `Z.record` returns it as a dictionary. The command line prints the same record with `--json`, and the HTTP API returns it in its responses.

This chapter shows how to save a record and compute from it later, how the embedded and intrinsic modes differ, how to read charts and dual complexes, and how the Resolution Zoo stores its catalogue. The [cusp tutorial](../tutorial/resolving-a-cusp.md) works through a resolution at the prompt, and [Zeta functions and the monodromy conjecture](zeta-functions.md) explains the zeta functions. Every field is listed in [Resolution objects and zeta functions](../reference/resolution-values.md).

## Save and replay the cusp

From the repository directory, resolve the cusp $y^2=x^3$ in embedded mode with both zeta functions and save the record.

```sh
./atlas resolve 'y^2-x^3' --vars x,y --monodromy --json > cusp.json
```

Without `--json`, the command prints a summary of the charts instead. Later commands read the saved file without resolving again.

```sh
./atlas monodromy cusp.json
```

```text
Local monodromy zeta(T) = (1-T^6)/((1-T^2)*(1-T^3))
Euler characteristic of the complex Milnor fiber: -1
```

Add `--json` to `./atlas monodromy` to see the factors and A’Campo's contribution of each divisor. `./atlas recompute-dual-complex cusp.json` rebuilds the intersections of the exceptional curves from the saved charts. Its `cells` list two edges, joining `E1` to `E3` and `E2` to `E3`. Both commands accept `-` for standard input, and the complex command also reads `.json.gz` files. [Command line](../reference/command-line.md)

At the prompt, the same information comes from the resolution object.

```text
atlas> x, y = coordinates("x", "y")
atlas> Z = resolve(y^2 - x^3, monodromy=True)
atlas> Z.mode
embedded
atlas> Z.dual_complex
Dual complex (embedded, complete): 3 vertices, 2 edges
  E1 — E3, E2 — E3
atlas> recompute_dual_complex(Z).vertices
[E1, E2, E3]
atlas> Z.monodromy_zeta.expression
(1-T^6)/((1-T^2)*(1-T^3))
```

`Z.dual_complex` keeps everything recorded during the resolution, including self-intersections and multiplicities when they are known. `recompute_dual_complex(Z)` rebuilds the incidences from the charts alone and returns new vertices with the data that this replay recovers. In a script, `print_json(record=Z.record)` writes the same JSON as `./atlas resolve --json`.

Reading a zeta function from a saved record replays its certificate against the equation. If the record lacks the embedded data, for example an intrinsic or inconclusive record, Atlas starts a fresh embedded calculation from the recorded equation and coordinates. Over HTTP, `POST /api/zetas` does the same for a record you send. [Replay zeta functions over HTTP](../reference/http-and-compute.md#replay-zeta-functions)

## Resolution modes

An **embedded** resolution, the default, follows the hypersurface inside its ambient space. The total transform must become a divisor with simple normal crossings, and the record keeps the multiplicity of each component, repeated factors of the input included. Its dual complex describes the ambient exceptional divisor. An **intrinsic** resolution resolves the singularities of the hypersurface itself and describes the exceptional locus on the resolved hypersurface. It requires a reduced germ.

For the cusp, one blowup already makes the curve smooth. The intrinsic resolution stops there, and the exceptional locus on the curve is a single point.

```text
atlas> x, y = coordinates("x", "y")
atlas> I = resolve(y^2 - x^3, mode="intrinsic")
atlas> I.charts
[B1.u, B1.v]
atlas> I.dual_complex
Dual complex (intrinsic, complete): 1 vertex, 0 higher cells
atlas> I.divisors
[P1]
```

The embedded resolution needs two more blowups, because after the first one the smooth curve is tangent to the exceptional line. Embedded and intrinsic resolutions also differ in higher dimension. For the cusp cylinder $y^2=x^3$ in three-space, one blowup along the singular line makes the surface smooth, while embedded resolution takes three. Zeta functions always come from an embedded resolution. Reading them from an intrinsic resolution computes one. [Choose what to compute](../reference/resolution.md#choose-what-to-compute)

## Chart and complex conventions

A chart's `map` expresses the original coordinates in the chart coordinates. Substituting these expressions into the equation gives its total transform. In a toric chart, removing the exceptional monomial leaves the strict transform. In a chart with relations, the chart coordinates also satisfy the equations in `ambient_ideal`, and `strict_transform` gives further equations in that quotient ring. [Look inside a chart of the cusp](../tutorial/resolving-a-cusp.md#look-inside-a-chart)

Later blowups replace parts of earlier charts. For plane charts, a child chart starts on the parent's `base_domain`, before the parent's `exclude_points` are removed. The parent still covers the complement of the replaced points. Centres defined over extensions of the rationals have entries in `algebraic_numbers`.

A vertex of the dual complex is a geometric irreducible component of the exceptional divisor. A cell of higher dimension is a geometric irreducible component of an intersection of such components. Two components can meet in several points and so give several edges with the same endpoints. Use the cell IDs and their codimension-one `faces` to follow the attachments. The `status` of the complex says whether all of these have been certified. A branch of a plane curve has its own attachment. For example, `Z.branches[0].meets` gives the exceptional curves that the strict transform meets. [How Atlas certifies incidence](resolution-pipeline.md#what-the-record-certifies)

```text
atlas> x, y = coordinates("x", "y")
atlas> Z = resolve(y^2 - x^3)
atlas> Z.dual_complex.cells[0]["faces"]
['E3', 'E1']
atlas> Z.branches[0].meets
[E3]
```

## Resolution Zoo catalogue

The Resolution Zoo repository stores one hypersurface germ per stable identifier. The
index at `index.json` carries the name, dimension, labels and
equation of each germ. The ordered `variables` give the coordinate order
for the exponent vectors in `terms`. For example, the term
`{"coefficient":"-1/2","exponents":["1","3","0"]}` in coordinates
`["x","y","z"]` means $-xy^3/2$. Coefficients and exponents are decimal
strings so clients can preserve exact values of arbitrary size. The rational
coefficient is in lowest terms with a positive denominator. [Resolution Zoo repository](https://github.com/pportilla/resolution-zoo)

Open `records/<id>.json` for a standalone catalogue record.
The versioned contract for both JSON documents is in
`schema.json`. Version 1 records describe germs at the
coordinate origin over the rationals. Each carries the geometric field
returned by Atlas. A partial or unavailable zeta carries a status and a
reason. The index lists the current catalogue entries and their dimensions.

| Field | Interpretation |
| --- | --- |
| `invariants` | Total degree of the displayed polynomial, multiplicity at the origin, and the dimensions of the local Milnor and Tjurina algebras. A positive-dimensional algebra has `finite: false` and `value: null`. |
| `intrinsic` | A resolution of the reduced hypersurface germ. Its complex describes the exceptional components on the resolved hypersurface. |
| `embedded` | A resolution of the hypersurface inside its ambient space. Its complex describes the ambient exceptional divisor. |
| `zeta.monodromy` | The monodromy zeta expression at the origin, its factors, A’Campo contributions with multiplicities and Euler characteristics, and the nearby strata where Atlas computed a transverse zeta. |
| `zeta.topological` | The topological zeta at the origin, numerator, denominator, numerical data $(N_i,\nu_i)$, contributing open intersection strata and surviving poles. |
| `zeta.pole_check` | Actual and candidate poles, with the eigenvalue checks. Each nearby match refers to a zero-based entry of `zeta.monodromy.nearby_strata`. |
| `source` | The Atlas resolution record version, method, compute backend and geometry search policy. An exhaustive pole check following quick geometry is recorded in `pole_check_search`. |

Both resolutions retain the ordered modifications in `steps`. A toric
modification can occupy one step. Its `fan.rays` gives vectors and each
maximal cone in `fan.cones` lists zero-based indices into these rays.
Other steps may instead give the centre's ideal on each relevant chart
and the chart's history. Each complex owns its vertices, cells and
codimension-one face references. Genus and self-intersection appear when
Atlas records them. Repeated edges with the same endpoints can represent
different intersection points.

The local Milnor algebra supplies the isolated or non-isolated label.
Family and equation labels are curated. Homogeneous examples also carry the
quasi-homogeneous label. The nearby strata are those certified by the
solver and an empty list makes no claim about isolation.

From a built Atlas checkout, regenerate the installed catalogue with the following command. [Install the Resolution Zoo](../installation.md#install-the-resolution-zoo)

```sh
python3 tools/build_resolution_zoo.py --jobs 6
```

Add an equation, name, stable ID and labels to the appropriate dimension
and coordinate group in `seeds.json` in the catalogue checkout. The generator
requests both resolutions, both zetas and the local algebra dimensions.
It requires a complete complex in each mode and both computed zetas before
replacing the catalogue index. `--jobs` sets how many records are computed
at once; the index keeps the order of the seed file. To build a separate
collection, pass `--seeds my-seeds.json --out my-zoo`. Keep partial collections in a separate output directory. Commit the updated seeds, records and index together in the Resolution Zoo repository.

For the full Atlas record, including its affine charts and replayable
certificate, run the solver on the catalogue equation with the recorded
variables and search policy. These commands reproduce the Whitney umbrella.

```sh
./atlas resolve 'x^2-y^2*z' --vars x,y,z --mode embedded --compute cpu --quick --monodromy --json
./atlas resolve 'x^2-y^2*z' --vars x,y,z --mode intrinsic --compute cpu --quick --json
```
