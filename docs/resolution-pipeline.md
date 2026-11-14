# How Atlas resolves a hypersurface germ

A request specifies a polynomial over the rationals, its ordered coordinates, and a resolution mode. Atlas resolves the germ at the origin. The `charts` field describes the modification. The `atlas` field records the pieces of those charts used to verify the answer and count strata over the origin. [Request and dispatch](../engine/resolve.py) · [Atlas serialization](../engine/atlas.py)

The CLI, prompt, scripts and HTTP workers call the same solver. An invalid variable list, a zero equation or a hypersurface missing the origin is an input error. Exceeding a quick parsing limit gives an inconclusive result. [Session entry point](../engine/session.py) · [Worker](../engine/worker.py) · [Input parser](../engine/polynomial.py)

```mermaid
flowchart TD
    A["Parse the equation and validate the request"] --> B["Try eligible structured methods in order"]
    B -->|success| V["Verify the owned charts or reuse their native proof"]
    B -->|no result| X["Try eligible exhaustive embedded fan and coordinate-centre proofs"]
    X -->|no result or ineligible| G["Try blowing up the smooth reduced singular locus"]
    G -->|all graph charts certified| V
    G -->|declined, timed out or ineligible| P{"Search policy and dimension"}
    P -->|quick| Q["Bounded Newton, origin blowups and priority ideals where eligible"]
    P -->|exhaustive, singular in 3 to 8 variables| N{"Does the native tangent test rule out Newton?"}
    N -->|yes| O["Start origin blowups, joined by ideal search after 250 ms or on decline"]
    N -->|no| W["Start Newton, joined by origin and ideal search after two seconds or on decline"]
    P -->|other exhaustive cases| I["Eligible serial Newton, then exhaustive ideal search"]
    Q -->|success| V
    Q -->|budget exhausted| U["Inconclusive with diagnostic"]
    X -->|success| V
    O -->|success| V
    W -->|success| V
    I -->|success| V
    V --> R["Resolution record with owned atlas and exceptional dual complex"]
    R -->|zetas requested| S["Use embedded strata from this atlas or an embedded companion"]
    S --> Z["Monodromy and topological zeta functions"]
    H["Saved resolution JSON"] -->|incidence| D["Recompute the dual complex in the saved mode"]
    H -->|integrals| T["Recover embedded strata from the record"]
    T --> Z
```

## How Atlas chooses a modification

1. For an embedded singular germ in three to eight variables, Atlas first checks for a divisor with normal crossings at the origin. Coordinate powers or smooth factors through the origin with independent differentials certify the identity map. Singular plane curves go to the plane blowup algorithm. A smooth germ outside the size range of the ordinary engine gets the identity from its nonzero linear part. [Dispatcher](../engine/resolve.py) · [Normal crossings test](../engine/monodromy_special.py) · [Plane resolver](../native/plane.cpp)
2. Within the ordinary engine's range (two to eight variables, at most 4,096 terms, total degree at most 128), singular three-variable inputs first try the A-series surface chain. The ordinary engine handles smooth germs and tests for a smooth projective tangent cone. In three variables, a diagonal Newton surface is the next candidate. If still unresolved, two-term inputs in two to eight variables can try coordinate blowups, except after an exhaustive plane-worker limit. [Surface chains](../engine/chain_backend.py) · [Ordinary points](../engine/native_backend.py) · [Diagonal surfaces](../engine/toric_backend.py) · [Binomials](../engine/binomial_backend.py)
3. **Exhaustive search**, the CLI and session default, first tries two bounded embedded proofs where eligible. A small fan can certify the first monomial ideal candidate. For a ternary pencil `H(x,y) + z*K(x,y)`, the next test requires binary homogeneous forms of equal degree 2–15, with `H` a scalar times a power of a noncoordinate rational linear form and `K` nonzero at its projective root. After certifying that the preceding Newton candidates cannot resolve the germ, an isolated graph worker tests the identity, origin and coordinate-line centres in their scheduled order. The third must pass and both predecessors must fail. [Bounded trials](../engine/newton_trials.py) · [Pencil test and size limits](../native/resolution_candidate.cpp) · [Graph worker](../native/graph_verify.cpp)
4. **Both search policies** can then try the reduced singular locus. The C++ kernel forms the Jacobian ideal `(f, ∂f/∂x₁, …, ∂f/∂xₙ)` and computes its radical. If that radical presents a smooth complete intersection through the origin, of positive dimension and codimension at least two, its generators define a candidate blowup. The kernel saturates each standard open and checks the transforms and crossings. Embedded mode can certify a single exceptional divisor when the centre is a polynomial graph over free coordinates. Other centres and intrinsic requests go through geometric component analysis. Eligibility includes three to eight variables and four to 128 input terms. The native trial has a 100 ms deadline. A failed proof or budget limit returns control to the existing search. [Native centre and chart proofs](../native/ideal.cpp) · [Trial limits](../engine/reduced_centre.py) · [Record construction](../engine/general_resolution.py)
5. **Quick search** uses bounded parsing and a 55-second CLI worker deadline. Eligible inputs try a regular Newton modification, including coordinate changes, monomial ideals or repairs of failed crossings. An unresolved singular three-variable input can try blowups from the origin. In four to eight variables the quick origin attempt is limited to the repeated-tangent rule. The final search tests priority blowup ideals with bounded effort. Exhausting the quick search yields an inconclusive record. [CLI deadline](../engine/resolve_cli.py) · [Newton](../engine/newton_backend.py) · [Origin blowups](../engine/origin_backend.py) · [Ideal search](../engine/general_resolution.py)
6. In the exhaustive worker search, Atlas can rule out Newton on embedded ternary inputs of total degree at most 16, with at most 128 terms and 64-bit rational numerators and denominators. The test requires a higher-order term and a lowest-degree part exactly equal to `c*L^d`, with `d ≥ 2` and all three coefficients of the linear form `L` nonzero. All axial intercepts then have degree `d`. The only coordinate choice in Atlas's Newton search gives the ordinary origin blowup. Its exceptional plane meets the strict transform along the repeated line `L=0`. That line meets the dense torus in a curve, beyond the scope of Newton's finite-point repair. With this proof, Atlas starts the origin-blowup worker immediately and starts ideal certification if that worker declines, fails, or is still running after 250 ms. [Native tangent test](../native/resolution_candidate.cpp) · [Newton repair](../engine/newton_repair.py) · [Worker coordination](../engine/exhaustive_resolution.py)
7. In the usual parallel search Newton gets a two-second head start. Origin blowups and general ideal certification join as soon as Newton gives no result or its head start expires. The first successful worker supplies the charts for verification. Outside the parallel case, eligible inputs try Newton serially before general ideal enumeration. The exhaustive scheduler increases allowances and retries undecided candidates. For HTTP requests with no search policy, the server tries quick first and can return a background exhaustive job. [Candidate schedule](../engine/resolution_search.py) · [HTTP policy](../engine/app.py)

For example, set `A=x+y+z`, `B=x-y` and `T=z`. The equation `A²−TB²` has Jacobian ideal `(A, TB, B²)`, whose radical is the line `(A,B)`. On the two opens of its blowup, the strict transforms are `u²−T=0` for `u=A/B` and `1−Tv²=0` for `v=B/A`. They agree where `uv=1`. A smooth centre alone cannot certify this outcome. For `(y−x²)²−z⁵`, the reduced singular locus is also smooth, but the chart with pivot `z` has strict equation `u²−z³=0` and the search continues. [Positive and negative regression cases](../tests/test_reduced_centre_geometry.py)

When a producer reports a component repeated through the origin, embedded mode can resolve the reduced support and lift each original multiplicity back to the charts. Intrinsic resolution requires a reduced germ. [Weighted divisors](../engine/weighted_record.py)

For a **Newton modification**, each maximal cone of a regular fan gives an affine chart. Its rays specify a monomial substitution for the original coordinates, and factoring the exceptional monomial from the pullback gives the strict-transform equation. **Successive smooth blowups** use a generator of the centre as pivot on each standard open, express the other generators as ratios times that pivot, and transform the strict ideal and existing divisors there. For an **ideal blowup**, saturating the graph equations by the pivot gives the ambient chart ideal. Adjoining the pulled-back hypersurface equation to the ambient chart ideal and saturating that ideal by the pivot gives the strict transform. Comparing boundary components on overlapping opens determines which labels glue. [Newton chart construction](../engine/newton_backend.py) · [Blowups of smooth centres](../engine/local_blowups.py) · [Graph reconstruction](../engine/graph_record.py) · [Boundary gluing](../engine/general_resolution.py)

## What the record certifies

Each owned chart specifies its projection, ambient equations, components, origin fibre, and the region and exclusions that assign points to that chart. These assignments prevent counting an overlap twice. The dispatcher can reuse a native monomial construction that already verified the same record. Otherwise it tries dedicated verifiers for identity, plane, ideal-blowup, point and fan charts, then reconstructs the general atlas when those verifiers decline. [Attachment](../engine/resolve.py) · [Native monomial construction](../native/monomial_resolution.cpp) · [Verifier dispatch](../engine/strata.py) · [General chart checks](../engine/atlas.py)

To compare divisor pieces on affine charts, Atlas equates their original coordinates and saturates away from the exceptional divisors to obtain the closure of the overlap. From this graph it can recover the rational coordinate changes in both directions. Two pieces glue when these changes are regular at their generic points, carry each generic point to the other, and respect the loci removed from both charts. A meeting confined to a smaller closed subset therefore leaves the components distinct. [Overlap construction](../engine/atlas_gluing.py) · [Native coordinate comparisons](../native/primary_decomposition.cpp)

Embedded mode verifies normal crossings of the total transform on the owned fibre and computes incidence for ambient exceptional divisors. Intrinsic mode verifies the smooth strict transform and computes incidence on that transform. If an intrinsic atlas cannot supply embedded strata for the requested zetas, Atlas resolves the same polynomial in embedded mode and can store the verified `embedded_companion.atlas`. [Mode and companion handling](../engine/resolve.py) · [Chart conventions](resolution-records.md#chart-and-complex-conventions)

Both zetas use the same compactly supported Euler characteristics of disjoint strata. A’Campo’s formula uses the open strata on individual components and their multiplicities. The topological zeta function uses all strata, their multiplicities and the orders of the pulled-back volume form. [Monodromy](../engine/monodromy.py) · [Topological zeta](../engine/topological.py)

## Replaying a saved resolution

```sh
./atlas resolve 'x^2-y^2*(z+t)' --vars x,y,z,t --compute cpu --monodromy --json > resolution.json
./atlas recompute-dual-complex resolution.json
./atlas monodromy resolution.json --json
```

Replay starts from the saved equation and charts and skips the search for a modification. The `recompute-dual-complex` command recomputes incidence in the saved mode. Zeta replay checks embedded strata, using the saved companion when present. Some older records without reconstructible chart ownership can supply their archived certified numerical data and strata. A record that cannot establish the required embedded strata yields an unavailable zeta. [Incidence replay](../engine/resolution_complex.py) · [Strata replay and archive fallback](../engine/strata.py) · [Session zeta replay](../engine/session_values.py)
