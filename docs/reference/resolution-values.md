# Resolution objects and zeta functions

`resolve(f)` returns a **resolution** of the hypersurface germ $f=0$ at the origin. This page lists everything you can read from it: the resolution itself, its charts, exceptional divisors, branches and dual complex, its local topological and monodromy zeta functions, and the pole check returned by `monodromy_check`. The functions that create these values are described in [Resolution, zeta functions and monodromy](resolution.md).

Each section starts from an empty session. Paste its script at the prompt or save it in a file and run it with `./atlas run`. The comments show the printed output. At the prompt, `help Z` describes a saved resolution `Z` and `help Z.charts` describes one of its fields.

All these values are read-only views of one saved record. A field is read as an attribute, such as `Z.mode`. Entries of a dictionary, such as a pole, are read the same way, so `T.poles[1].value` and `T.poles[1]["value"]` agree. Rational numbers in the record, such as pole values and point coordinates, are exact decimal strings like `"-5/6"`.

## The resolution

The cusp $y^2=x^3$ is resolved by three point blowups. The resolution remembers the equation, its ring, and how it was obtained.

```atlas
R = ring("x", "y")
f = y^2 - x^3
Z = resolve(f)
print(Z)                       # Resolution(-x^3+y^2, mode=embedded, status=resolved, charts=6, divisors=3)
print(Z.polynomial == f)       # True
print(Z.coordinates)           # (x, y)
print(Z.status, Z.mode)        # resolved embedded
print(Z.summary)               # 3 point blowups; 1 geometric branch above the origin.
print(Z.charts)                # [B1.u, B1.v, B2.u, B2.v, B3.u, B3.v]
print(Z.divisors)              # [E1, E2, E3]
print(Z.topological_zeta.expression)  # (4*s + 5)/((s + 1)*(6*s + 5))
```

| Field or method | Value |
| --- | --- |
| `Z.polynomial` | The input equation as a polynomial in the ring of `Z.coordinates`. |
| `Z.coordinates` | The ordered generators of the input ring. Chart maps refer to these names. |
| `Z.mode` | `"embedded"` or `"intrinsic"`, as passed to `resolve`. [Resolution modes](../thematic/resolution-records.md#resolution-modes) |
| `Z.status` | `"resolved"` when the search certified the resolution, `"inconclusive"` when a quick search stopped first. A zeta calculation by `monodromy_check` that needed no charts has `"zetas_computed"`. |
| `Z.summary` | One sentence describing the modification. An inconclusive record explains its stop in `Z.diagnostic`. |
| `Z.charts`, `Z.chart(id)` | The affine charts, as a list or by ID. [Charts](#charts) |
| `Z.divisors`, `Z.divisor(id)` | The exceptional divisors, as a list or by ID. [Exceptional divisors](#exceptional-divisors) |
| `Z.branches` | The branches of the strict transform of a plane curve over the origin. An error says when no branch data were recorded. [Branches](#branches-of-a-plane-curve) |
| `Z.dual_complex` | The intersection complex of the exceptional divisors. [Dual complex](#dual-complex) |
| `Z.steps` | The modifications in order. Each step has an `id`, a `type` such as `"point blowup"`, its `center`, the `exceptional_id` it creates and the multiplicities of the strict and total transforms there. |
| `Z.topological_zeta` | The local topological zeta function. [Topological zeta function](#topological-zeta-function) |
| `Z.monodromy_zeta` | The local monodromy zeta function. [Monodromy zeta function](#monodromy-zeta-function) |
| `Z.monodromy_check` | The pole check of the monodromy conjecture for this germ. [Pole check](#pole-check) |
| `Z.engine` | The algorithm that produced the geometry, such as `"C++ / FLINT plane point blowups"`. |
| `Z.base_field`, `Z.geometric_field`, `Z.algebraic_numbers` | The input is over $\mathbb Q$. Divisors and points are geometric objects over $\overline{\mathbb Q}$. Centres over number fields list their algebraic numbers. |
| `Z.certificate` | What the verifier checked before accepting the resolution. |
| `Z.execution`, `Z.elapsed_ms` | The device and stages used for this run, and its duration in milliseconds. |
| `Z.record` | A copy of the complete record as a dictionary. Every field above is read from it. It is the JSON printed by `./atlas resolve --json`. |

The fields `Z.topological_zeta`, `Z.monodromy_zeta` and `Z.monodromy_check` are computed the first time you read one of them and then kept. Pass `monodromy=True` to `resolve` to compute them during the call. For an intrinsic or weighted resolution, Atlas computes the zeta functions on an embedded resolution of the same equation.

To keep a resolution, write its record with `print_json(record=Z.record)` in a script, or run `./atlas resolve 'y^2-x^3' --vars x,y --json > cusp.json` from the shell. [Saving and replaying a resolution](../thematic/resolution-records.md#save-and-replay-the-cusp)

## Charts

A chart is an affine piece of the resolved space. Its map expresses the original coordinates in the chart coordinates. Substituting the map into the equation gives the total transform.

```atlas
R = ring("x", "y")
f = y^2 - x^3
Z = resolve(f)
C = Z.chart("B3.u")
print(C.id, C.step, C.parent_chart)   # B3.u B3 B2.v
print(C.coordinates)                  # (u, v)
print(C.map)                          # {'x': u^2*v, 'y': u^3*v^2}
print(C.pullback(f))                  # u^6*v^4-u^6*v^3
print(C.strict_transform)             # v-1
print(C.divisors)                     # [E3, E2]
print(C.pullback_map(x + y))          # u^3*v^2+u^2*v
```

The ID `B3.u` names the chart of the third blowup `B3` in which the new exceptional curve is $u=0$. In this chart the total transform $u^6v^3(v-1)$ splits into `E3` with multiplicity six, `E2` with multiplicity three and the strict transform $v=1$.

| Field or method | Value |
| --- | --- |
| `C.id` | The chart's ID in this resolution. |
| `C.coordinates` | The ordered generators of the chart ring. |
| `C.parent` | The chart ring. For a chart with relations, this is a quotient ring. |
| `C.ambient` | The polynomial ring in which the chart's relations are written. |
| `C.ambient_ideal` | The relations among the chart coordinates, for a chart with relations. |
| `C.map` | A dictionary sending each original coordinate name to a polynomial on the chart. |
| `C.pullback(g)` | The pullback of a polynomial `g` of the original ring. For `g = Z.polynomial` it is the total transform. |
| `C.pullback_map` | The same substitution as a ring map from the original ring to `C.parent`. [Ring homomorphisms](saved-values.md#ring-homomorphisms) |
| `C.strict_transform` | The equation of the strict transform on this chart, or a list of equations. |
| `C.divisors` | The exceptional divisors with a local equation on this chart. |
| `C.exceptional` | For each of those divisors, its `id`, local `equation` and `total_multiplicity`. |
| `C.step`, `C.parent_chart`, `C.map_to_parent` | The blowup that created the chart, the chart it was taken from, and the map to the coordinates of that chart. |
| `C.base_domain`, `C.exclude_points` | The part of the parent chart that this chart covers, and the points that later blowups replace. [Chart conventions](../thematic/resolution-records.md#chart-and-complex-conventions) |
| `C.orbifold_order` | For a weighted blowup, the degree of the finite cover on which the chart is written. [Weighted charts](../thematic/weighted-resolution.md#charts-of-weighted-blowups) |

Some fields exist only for some construction methods. Reading a field that a chart lacks raises an error that names the field.

## Exceptional divisors

A divisor is a geometric irreducible component of the exceptional locus over the origin. Its total multiplicity $N$ is the order of vanishing of the pulled-back equation along it.

```atlas
R = ring("x", "y")
f = y^2 - x^3
Z = resolve(f)
E = Z.divisor("E3")
print(E.total_multiplicity, E.self_intersection, E.dimension)   # 6 -1 1
print(E.charts)                                                 # [B3.u, B3.v]
C = Z.chart("B3.u")
print(E.equation_in(C))                                         # u
print(C.pullback(f).multiplicity(E.equation_in(C)))             # 6
```

The last line checks $N=6$ directly. The factor $u$ divides the total transform exactly six times.

| Field or method | Value |
| --- | --- |
| `E.id` | The divisor's ID, such as `"E3"`. |
| `E.dimension` | The dimension of the component. |
| `E.total_multiplicity` | The multiplicity $N$ of the divisor in the total transform. |
| `E.self_intersection` | The self-intersection of an exceptional curve on a surface, when Atlas records it. |
| `E.nu` | The numerical datum $\nu$, on divisors of a weighted resolution. For every method, both $N$ and $\nu$ appear in `Z.topological_zeta.numerical_data`. |
| `E.charts` | The charts on which the divisor has a local equation. |
| `E.equation_in(C)` | The local equation on chart `C`, when it is a single polynomial. |
| `E.ideal_in(C)` | The local equations on chart `C` as a list. Use it when a divisor needs several equations. |

## Branches of a plane curve

For a plane curve, the strict transform meets the exceptional divisor in one point per branch. Each branch records that point and the chart that contains it.

```atlas
R = ring("x", "y")
Z = resolve(y^2 - x^3)
b = Z.branches[0]
print(Z.branches)             # [P1]
print(b.meets)                # [E3]
print(b.chart, b.point)       # B3.u ['0', '1']
print(b.chart.strict_transform)   # v-1
```

| Field | Value |
| --- | --- |
| `b.id` | The branch's ID, such as `"P1"`. |
| `b.meets` | The exceptional divisors that the strict transform meets along this branch. |
| `b.chart` | The chart containing the branch point. |
| `b.point` | The coordinates of the branch point in that chart, as exact strings. |

## Dual complex

The dual complex has a vertex for each exceptional divisor and a cell for each irreducible component of an intersection of divisors. For a curve, the cells are the edges of the resolution graph.

```atlas
R = ring("x", "y")
Z = resolve(y^2 - x^3)
D = Z.dual_complex
print(D.status)                    # complete
print(D.vertices)                  # [E1, E2, E3]
print(len(D.cells))                # 2
print(D.cells[0]["vertices"])      # ['E1', 'E3']
print(recompute_dual_complex(Z).vertices)   # [E1, E2, E3]
```

At the prompt, `Z.dual_complex` on a line by itself displays the summary and the edges, here `E1 — E3, E2 — E3`.

| Field | Value |
| --- | --- |
| `D.vertices` | The exceptional divisors, as the same objects as `Z.divisors`. |
| `D.cells` | One dictionary per cell, with its `id`, `dimension`, `vertices` and the IDs of its codimension-one `faces`. Two curves meeting in several points give several cells with the same vertices. |
| `D.status` | `"complete"` when every vertex and cell is certified. Otherwise `D.record["reason"]` explains what is missing. |
| `D.record` | A copy of the complete complex, including its `convention`. |

`Z.dual_complex` is the complex recorded during resolution. `recompute_dual_complex(Z)` rebuilds the incidences from the saved charts alone. [Chart and complex conventions](../thematic/resolution-records.md#chart-and-complex-conventions)

## Topological zeta function

Let $\pi$ be an embedded resolution of $f$ with components $E_i$ of the total transform, including the strict transform. Write $N_i$ for the multiplicity of $f\circ\pi$ along $E_i$ and $\nu_i-1$ for the order of the pulled-back volume form. For a set $I$ of components let $E_I^\circ$ be the points that lie on exactly the components in $I$. The local topological zeta function of Denef and Loeser is

$$Z_{\mathrm{top},0}(s)=\sum_{I}\chi\big(E_I^\circ\cap\pi^{-1}(0)\big)\prod_{i\in I}\frac{1}{N_is+\nu_i}.$$

Read it from any resolution of the germ.

```atlas
R = ring("x", "y")
Z = resolve(y^2 - x^3)
T = Z.topological_zeta
print(T.status)                  # computed
print(T.expression)              # (4*s + 5)/((s + 1)*(6*s + 5))
print(T.numerator)               # 4*s + 5
print(T.denominator)             # (s + 1)*(6*s + 5)
print(T.numerical_data[2])       # {'component': 'E3', 'multiplicity': 6, 'nu': 5}
print(T.poles[1].value, T.poles[1].order, T.poles[1].components)   # -5/6 1 ['E3']
print(T.strata[2])               # {'components': ['E3'], 'euler_characteristic': -1}
```

The pole $-5/6=-\nu/N$ comes from `E3`, and the pole $-1$ from several components.

| Field | Value |
| --- | --- |
| `status` | `"computed"`, or `"inconclusive"` or `"unavailable"` with a `reason`. |
| `variable` | `"s"`. |
| `expression` | The zeta function as an exact rational function in lowest terms. |
| `numerator`, `denominator` | Its numerator and factored denominator. |
| `numerical_data` | One entry per component, with its `component` ID, its multiplicity $N$ as `multiplicity` and its `nu`. The strict transform has the ID `"strict"`. Entries marked `auxiliary` belong to a contact calculation rather than to a divisor. |
| `strata` | One entry per nonempty stratum $E_I^\circ\cap\pi^{-1}(0)$, with the set $I$ as `components` and its `euler_characteristic`. |
| `poles` | The poles that survive cancellation. Each has an exact `value`, its `order`, the `components` with $-\nu_i/N_i$ equal to the value and `in_monodromy_zeta_at_origin`, which says whether the predicted eigenvalue already appears in the monodromy zeta at the origin. |
| `formula`, `method` | The formula and the way the Euler characteristics were counted. |

[Zeta functions and the monodromy conjecture](../thematic/zeta-functions.md)

## Monodromy zeta function

Atlas uses A’Campo's formula for the zeta function of the monodromy on the complex Milnor fibre $F_0$ at the origin. Here $E_i^\circ$ is $E_i$ with its intersections with all other components removed, including the strict transform.

$$\zeta_{f,0}(T)=\prod_q\det\big(1-T\,h^*\mid H^q(F_0;\mathbb Q)\big)^{(-1)^{q+1}}=\prod_i\big(1-T^{N_i}\big)^{-\chi(E_i^\circ\cap\pi^{-1}(0))}.$$

The cohomology is unreduced, so the trivial eigenvalue of $H^0$ is included.

```atlas
R = ring("x", "y")
Z = resolve(y^2 - x^3)
M = Z.monodromy_zeta
print(M.expression)                          # (1-T^6)/((1-T^2)*(1-T^3))
print(M.factors[2])                          # {'multiplicity': 6, 'exponent': 1}
print(M.contributions[2])                    # {'component': 'E3', 'multiplicity': 6, 'euler_characteristic': -1}
print(M.milnor_fiber_euler_characteristic)   # -1
print(M.nearby_strata)                       # []
```

| Field | Value |
| --- | --- |
| `status` | `"computed"`, or `"inconclusive"` or `"unavailable"` with a `reason`. |
| `variable` | `"T"`. |
| `expression` | The zeta function as a product of factors $(1-T^m)^{\pm1}$. |
| `factors` | One entry per factor $(1-T^m)^e$, with $m$ as `multiplicity` and $e$ as `exponent`. |
| `contributions` | A’Campo's terms. Each has the `component`, its multiplicity $N_i$ and the Euler characteristic of $E_i^\circ\cap\pi^{-1}(0)$. |
| `milnor_fiber_euler_characteristic` | The Euler characteristic $\chi(F_0)=\sum_i N_i\,\chi(E_i^\circ\cap\pi^{-1}(0))$ of the Milnor fibre, which is minus the degree of the zeta function. |
| `nearby_strata` | Zeta functions that Atlas certified at points arbitrarily close to the origin. Each entry gives a `model`, its `expression` and `factors`, and the data of the model. [Monodromy on nearby strata](../thematic/zeta-functions.md#monodromy-near-the-origin) |
| `convention`, `formula`, `method`, `resolution_engine` | The cohomological convention, the formula, how it was evaluated and which resolution supplied it. |

An empty `nearby_strata` list makes no claim that the singularity is isolated.

## Pole check

The monodromy conjecture predicts that for every pole $s_0$ of the topological zeta function, $e^{2\pi i s_0}$ is an eigenvalue of the monodromy at some point of $f=0$ near the origin. `monodromy_check(f)` resolves $f$, computes both zeta functions and looks for such an eigenvalue for each pole.

```atlas
R = ring("x", "y")
f = y^2 - x^3
P = monodromy_check(f)
print(P.status)                              # verified
print(P.reason)                              # Every actual pole has a local monodromy eigenvalue witness.
p = P.poles[1]
print(p.value, p.eigenvalue_order)           # -5/6 6
print(p.cyclotomic_exponent, p.witness)      # 1 origin zeta
print(P.monodromy_zeta.expression)           # (1-T^6)/((1-T^2)*(1-T^3))
print(resolve(f).monodromy_check.status)     # verified
```

The pole $-5/6$ predicts a primitive sixth root of unity. The cyclotomic polynomial $\Phi_6$ divides the numerator of the monodromy zeta at the origin once, so `cyclotomic_exponent` is $1$ and the origin itself is the witness.

| Field | Value |
| --- | --- |
| `status` | One of the statuses below. |
| `reason` | A sentence explaining the status. |
| `poles` | One entry per actual pole, described below. |
| `candidate_poles` | Every candidate $-\nu_i/N_i$ with its `components` and its `pole_order` after cancellation. A candidate with order zero is not a pole. Candidates from a contact calculation also carry `eigenvalue_order` and `cyclotomic_exponent`. |
| `topological_zeta`, `monodromy_zeta` | The two zeta functions, as on a resolution. |
| `resolution` | The resolution used, or `None` when a quick search stopped before one was found. |
| `local_bernstein_sato` | The exact eigenvalue test of an exhaustive search. Its `by_pole` maps each tested pole to its `tested_roots` and the `root` that was found, or `None`. |
| `milnor_number` | With `check_isolation=True`, the local Milnor number of the germ. Otherwise `None`. |

`Z.monodromy_check` on a resolution `Z` returns the same `status`, `reason`, `poles`, `candidate_poles` and, when it ran, `local_bernstein_sato`.

| Status | Meaning |
| --- | --- |
| `verified` | Every actual pole has a witness. |
| `refuted` | An exhaustive check found a pole whose eigenvalue is absent from every Milnor fibre near the origin. |
| `unresolved` | Both zetas were computed, but a quick check left a pole without a witness. |
| `inconclusive` | A quick search stopped before both zetas were computed. |
| `isolated_candidate` | A quick check with `check_isolation=True` found an isolated singularity whose zeta lacks a predicted eigenvalue. Check the resolution independently. |

Each entry of `poles` has the fields of a pole of the topological zeta function together with the following.

| Field | Value |
| --- | --- |
| `eigenvalue_order` | The order $q$ of the root of unity $e^{2\pi i s_0}$, the denominator of the pole. |
| `cyclotomic_exponent` | The exponent of the cyclotomic polynomial $\Phi_q$ in the monodromy zeta at the origin. It is positive for a zero and negative for a pole. |
| `witness` | Where the eigenvalue was found, from the table below, or `None`. |
| `witness_component`, `witness_multiplicity` | For a nearby smooth component, its ID and multiplicity. |
| `nearby_matches` | Each nearby stratum whose zeta contains $\Phi_q$, as a zero-based `stratum_index` into `monodromy_zeta.nearby_strata` and a signed `cyclotomic_exponent`. |
| `transverse` | For a nearby transverse model, its data. For example a transverse composition $f=G(h_1,\dots,h_k)$ records the $h_i$ as `coordinates`, their names as `transverse_coordinates` and $G$ as `transverse_germ`. |
| `bernstein_root` | For the witness `local Bernstein–Sato root`, the root found. |

| Witness | Where the eigenvalue was found |
| --- | --- |
| `origin zeta` | In the monodromy zeta at the origin. |
| `origin H^0` | In $H^0$ of the Milnor fibre at the origin, for the eigenvalue $1$. |
| `nearby smooth component` | On the Milnor fibre at a smooth point of a component of $f=0$ of multiplicity divisible by $q$. |
| `nearby transverse binomial`, `nearby transverse Morse`, `nearby constant transverse type`, `nearby tangent branches`, `nearby ordinary multiple point`, `nearby transverse composition` | In the zeta of a transverse model along a stratum through the origin. |
| `local Bernstein–Sato root` | In the exact test with roots of the local Bernstein–Sato polynomial, which decides the remaining poles in an exhaustive check. |

[Zeta functions and the monodromy conjecture](../thematic/zeta-functions.md#check-the-monodromy-conjecture)
