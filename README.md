# Resolution Atlas

Atlas computes resolutions of hypersurface germs at the origin over the algebraic closure of the rationals. It returns affine charts, maps to the input coordinates, strict transforms and the geometric dual complex of the exceptional divisor. The CLI and local HTTP API use the same solver.

C++17, GMP, FLINT and libnormaliz handle point blowups, exact linear algebra and polyhedral computations. Python assembles the charts and drives the general ideal-blowup search. Each candidate centre is decided by an exact native kernel: Gröbner bases over the rationals, saturation, ideal quotients, the Jacobian smoothness test and the charts of a blowup. When coefficients swell it switches to a Hilbert-driven computation for lexicographic and elimination orders, or to images modulo many primes whose lift it certifies over the rationals. [Algorithms and proofs](native/KERNEL.md). FLINT handles factorization over the rationals, exact algebraic roots, and arithmetic in the marked number fields used by the verifier. General blowups, intrinsic chains and archived number-field charts exchange sparse polynomials with the kernel. Python schedules the work, transports coefficient and exponent rows, and assembles the result. Optional Triton and OpenCL kernels handle chart substitutions, modular matrix updates and Gröbner replays. Atlas runs independently of Singular and SageMath.

For a comparison of the two systems and a phased roadmap for bringing more algebra to the Atlas CLI and browser, see [Atlas and Singular](README-singular-comparison.md).

Factor a polynomial over the rationals with `./atlas algebra factor --vars x --poly 'x^4-1'`, with `factor(f)` in a session, or with **Polynomial factorization** in the browser’s Algebra view. Each returns the rational unit, the irreducible factors, and their multiplicities. [Factorization examples](examples/algebra-api.md#polynomial-factorization)

Use the **Algebra** view or `./atlas algebra` for global quotient bases and for local quotient bases, Jacobian algebras, and Milnor and Tjurina numbers at any rational point. For example, `./atlas algebra local-jacobian-algebra --vars x,y --point '1/2,0' --poly 'y^2-(x-1/2)^3'` gives dimension 2 and basis $1,x-1/2$. Omitting the point computes at the origin. [Examples for the shell, prompt and browser](examples/algebra-api.md#quotient-bases-and-local-hypersurface-invariants)

Local standard bases and normal forms also accept any rational point. For example, `./atlas algebra local-normal-form --vars x,y --point '1/2,-2' --gen 'x-1/2' --gen 'y+2' --poly 'x+y'` returns $-3/2$. [Point coordinates, saved bases and local units](examples/algebra-api.md#local-standard-bases-and-normal-forms-at-a-rational-point)

The **Hilbert–Samuel function** and **Hilbert–Samuel multiplicity** operations apply to any ideal at a rational point. For the cusp $y^2=(x-1/2)^3$, the local lengths in degrees 0, 1 and 2 are 1, 3 and 5 at $(1/2,0)$ and 1, 2 and 3 at the smooth point $(3/2,1)$. The multiplicities are 2 and 1 respectively. [CLI and browser examples](examples/algebra-api.md#hilbert-samuel-functions)

## Build and run

Install a C++17 compiler, Make, pkg-config, Python 3.10 or later and the development libraries for GMP (including gmpxx), FLINT, JsonCpp and libnormaliz. The project was tested on Linux with FLINT 3.6, SymPy 1.14 and Python 3.14. The optional OpenCL plugins need OpenCL headers, a loader and a compatible device driver. The Triton path uses a GPU-enabled PyTorch installation and a matching Triton compiler in the Python environment that runs Atlas.

~~~sh
make -j4
~~~

Run `./atlas` to open a terminal session. The welcome screen shows the
folded Atlas logo in ASCII. Each launch also suggests a freshly resolved
two-branch plane curve. Run the displayed `resolve` command at the `atlas>`
prompt to inspect its charts and dual graph. Type `help resolve` to see its
options or `quit` to leave.

You can also run commands directly from your shell.

~~~sh
./atlas resolve 'y^2-x^3' --vars x,y
./atlas resolve 'x^5+z*y^3+z^15+t^4*x+t^8' --vars x,y,z,t --mode intrinsic --json
./atlas resolve 'x^2-y^2*z' --vars x,y,z,t --mode embedded --exhaustive --json
./atlas serve
~~~

Choose distinct variable names beginning with a letter. Use rational coefficients, explicit multiplication, and nonnegative integer exponents with `^` or `**`. Division by a nonzero rational constant is supported. The polynomial must vanish at the origin. For a germ at a rational point, translate that point to zero in the input.

The local interface is available at <http://127.0.0.1:8765>. CLI exit code 0 means resolved, 2 means an input or computation error and 3 means the quick search was inconclusive. Use `--json` to retain the equations and complete incidence records.

## Working at the prompt and in scripts

At `atlas>`, define a polynomial in its own coordinates and give the resolution
a name. Assignment is quiet. Enter a name or a field to inspect its value.

~~~text
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
atlas> e = E.equation_in(C)
atlas> e
u
atlas> total = C.pullback(f)
atlas> total == u^6 * v^3 * C.strict_transform
True
atlas> total.multiplicity(e)
6
~~~

`coordinates("x", "y", coeffs=QQ)` explicitly selects the default input
coefficient field. Fractions such as `(1/2)*x` are exact. Each call to
`coordinates` creates a separate coordinate space. Resolution defaults to
embedded mode and the same exhaustive search as the command-line solver.
`resolve(f, mode="intrinsic")` selects intrinsic mode.

The standalone prompt and `./atlas run FILE` accept and print exact
integers and fractions beyond Python’s usual 4,300-decimal-digit threshold.
An explicit `PYTHONINTMAXSTRDIGITS` or `python3 -X int_max_str_digits=...`
setting controls decimal input and output for these sessions. Applications
embedding the session use their host interpreter’s policy.

`C.pullback(f)` substitutes the ambient chart’s map into a polynomial in
the input coordinates. In `B3.u`, the images of `x` and `y` are `u^2*v` and
`u^3*v^2`, so the pullback is `u^6*v^3*(v-1)`. The local equation of
`E3` is `u`. Its multiplicity is the largest power of `u` dividing that
pullback. In `B3.v`, the local equation of the same divisor is `v`.

`R.coordinates` returns the original generators. `C.coordinates`,
`C.map["x"]`, `C.strict_transform` and `E.equation_in(C)` use the chart’s
own polynomial coordinates. Use `E.ideal_in(C)` for a list of generators.
A strict transform that needs several generators also returns a list.
On charts presented with ambient relations, these values are polynomial
representatives. Factor multiplicity currently requires a polynomial ring.

`polynomial(source, variables=...)` reads a quoted expression using the
names in an existing coordinate space. For example, the intersection of
`E3` and the strict transform in `B3.u` is given by `u = 0` and `v = 1`.
Eliminating `u` leaves an ideal in `v`. Use its coordinates for subsequent
calculations in that ring.

~~~text
atlas> s = polynomial("v-1", variables=C.coordinates)
atlas> s == C.strict_transform
True
atlas> J = eliminate(ideal(e, s), u)
atlas> q = polynomial("v^2+v+1", variables=J.coordinates)
atlas> normal_form(q, J)
3
~~~

Run this calculation with `./atlas run examples/chart-elimination.atlas`.
[Chart and elimination script](examples/chart-elimination.atlas)

Run `./atlas run examples/cusp.atlas` to execute the [cusp script](examples/cusp.atlas),
or `./atlas run -` to read a script from standard input. Files use the same
assignments, operations, and field access as the prompt. Use `print(...)` to
choose script output and `for chart in E.charts:` to visit several charts.
At the prompt, finish an indented block with a blank line. In a script, an
error reports the filename and line and stops execution. Type `help script`
at the prompt for a short reference.

Type `:save "my session.atlas"` at the prompt to write the successful
mathematical statements in order, including assignments and explicit
`print(...)` calls. Saving again replaces the file with the current session.
Replay it with `./atlas run "my session.atlas"`. If a statement fails, earlier
successful statements on the same line are still saved. An interrupted or
failed `for` or `if` block restores the variable bindings from before the block.

## Resolution algorithms

| Input or method | Computation |
|---|---|
| Smooth germ | A nonzero linear part certifies the identity map. |
| Divisor with normal crossings at the origin, in 3–8 variables | Its components through the origin are smooth with independent differentials there (multiplicities allowed): the identity is an embedded resolution near the origin. |
| Reduced plane curve | Blow up each singular geometric branch, using exact algebraic numbers for centers. Embedded mode continues until the strict curve meets the exceptional divisor transversely and avoids exceptional crossings. |
| Ordinary multiple point in 3–8 variables | A smooth projective tangent cone certifies one ordinary point blowup. Atlas checks projective smoothness using diagonal structure or an exact Jacobian rank certificate. |
| A-series surface | For a nondegenerate binary quadratic plus a pure power, use a quadratic splitting field when necessary and explicit point and line blowup charts. |
| Diagonal Newton surface | For a diagonal compact principal face, refine the weighted fan and track the components defined by each edge polynomial. Higher weighted terms are allowed. |
| Binomial with disjoint monomial supports | Blow up smooth coordinate strata over the singular locus. On a final chart the strict equation either has a unit term or is linear in a coordinate outside the exceptional divisor. For nonprimitive exponent differences the roots of a primitive torus character label the separate branches. |
| General Newton polyhedron in 2–8 variables | Try rational linear coordinate changes, construct the normal fan, triangulate it compatibly and refine to lattice determinant one. Accept when the map is an isomorphism off the origin and every compact face passes an exact torus nondegeneracy test. When a face is degenerate, the modification is kept and the points over the origin where normal crossings fail are blown up until they hold (see below). |
| Reduced hypersurfaces in 3–8 variables for which the rows above give no answer | Blow up smooth centres starting at the origin, with the centres of the complete rule described below (the invariant of Bierstone and Milman; in three variables, once the strict transform is smooth, the curves that the divisors cut on it). There is no hypothesis on the Newton polyhedron: $(x+y)^2+z^3+x^3$ and $(xy-z^2)^2+x^5+y^5$ are resolved this way. The run stops when normal crossings are certified at every point over the origin. |
| Remaining reduced hypersurfaces | Test projective blowups of ideals. Construct their affine graph charts by saturation, certify smoothness and normal crossings by exact ideal calculations, and glue geometric intersection components on chart overlaps. |

The CLI and Python API continue searching by default until they certify a resolution or the computation is interrupted. Use `--quick` at the CLI or `"search":"quick"` in a request to opt into a bounded attempt. Quick parsing allows up to eight variables, degree 128, 4,096 expanded terms and 8,192 input characters. The quick CLI and HTTP worker have a 55-second wall-clock limit. The default HTTP request returns a finished result for fast computations. If the bounded attempt is inconclusive, it automatically starts a cancellable background search and returns HTTP 202 with the job ID.

Embedded mode also resolves effective divisors with repeated components. It resolves the reduced support, then computes the original strict-transform ideal and the multiplicities of the pullback along each exceptional prime. For example, the exceptional multiplicities for `(y^2-x^3)^2` are 4, 6 and 12. Intrinsic resolution requires a reduced germ. A repeated global factor that is a unit at the origin poses no restriction.

The default exhaustive mode accepts rational polynomials in any finite positive number of variables and removes the quick parser and wall-clock limits. After the structured algorithms it enumerates finitely generated blowup ideals over the rationals and proposed geometric boundary ideals over number fields. Each candidate receives repeated attempts with increasing verification time and algebraic effort. Under idealized unbounded resources there is a finite certificate for a reduced hypersurface in characteristic zero. Embedded desingularization exists over the base field, a sequence of blowups over affine space can be presented as one blowup of an ideal, and its finitely many geometric stratum components are defined over a finite extension. These facts justify the search space. Priority candidates include the Jacobian locus, coordinate centers and successive blowups along repeated rational tangents. The integer enumeration can take impractically long even on a small polynomial. Ctrl-C stops an exhaustive CLI calculation. [Bierstone and Milman](https://arxiv.org/abs/alg-geom/9709028) · [Composition of blowups, Stacks Project](https://stacks.math.columbia.edu/tag/080B)

### Coefficient and exponent sizes

Exhaustive resolution and the algebra CLI use exact rational coefficients. For example, `./atlas resolve '2^20000*x+y' --vars x,y` handles a coefficient with 6,021 decimal digits. Sparse monomials can have exponents beyond a machine word. The packed Gröbner engine switches to wide arithmetic above total degree 32,767. These inputs can still be expensive when an operation has to expand a power or produce many charts. In two variables, $(x+y)^N$ has $N+1$ terms.

Quick resolution accepts at most 256 decimal digits in an integer literal, degree 128 (both the written exponent and intermediate total degree), and 4,096 bits in each numerator or denominator during expansion. A quick attempt can therefore be inconclusive for an input that the default CLI resolves. Exhaustive polynomial parsing also supports deeply nested input. At the prompt or in scripts, pass deeply nested formulas as text to `polynomial("…", variables=[x,y])`, since raw script expressions also pass through Python's parser. [Input parser](engine/polynomial.py)

The exhaustive Newton fan accepts arbitrary-precision lattice coordinates. For example, `./atlas resolve 'x^4097+y^4097+z^4097' --vars x,y,z --exhaustive` certifies the embedded resolution in three charts. If the first Newton-face check exceeds its coefficient, exponent, time or Gröbner basis budget, exhaustive resolution retries with an exact rational GCD or a Gröbner basis on the torus. Fan construction still has a 35-second budget, 16,000 rays and 120,000 intermediate cone records. If this shortcut gives up, the resolver can continue searching for a certificate by other methods. [Fan construction](native/newton_fan.cpp) · [Newton-face tests](engine/newton_faces.py)

With `--compute gpu`, Newton and binomial charts can also transform exponents beyond the device's 32-bit range. Atlas splits the coordinates into 14-bit pieces, computes their dot products on the device, and verifies the reconstructed integers against the exact ray valuations. CPU and automatic mode can reuse the valuations directly. The diagonal surface of degree $2^{80}+5$ produced the same three charts on a Radeon 680M and on the CPU. [Chart arithmetic](engine/chart_batch.py) · [Device regression tests](tests/test_gpu_resolution.py)

Euler projection stores only occupied powers and uses arbitrary-precision exponents and fibre counts. For example, it computes the compactly supported Euler characteristic of the complex curve $x^N+y^N=1$ as $N(2-N)$ even for $N=2^{64}$. The production solver also verifies an embedded resolution of $x^N+y^N=0$ in two charts at this degree. Some optional SymPy diagnostic checks use dense polynomials and can run out of memory on the same input. Polynomial expansion, large intermediate bases and the number of output charts remain practical limits for any run. [Euler projection tests](tests/test_constructible_euler.py) · [Plane resolution tests](tests/test_plane_exhaustive.py)

## The four-variable example

For `x^5+z*y^3+z^15+t^4*x+t^8`, the degree-four tangent form is `z*y^3`. The ordinary point blowup still has a singular strict transform. The Newton algorithm makes the linear change `z = Z+y`, which allows the regular toric modification to preserve the complement of the origin. The result has 598 affine charts. In intrinsic mode the exceptional divisor has 99 geometric components, 251 pairwise intersection components and 153 triple intersection components. In embedded mode the ambient exceptional divisor gives 169 vertices, 593 edges, 720 triangles and 295 tetrahedra. These counts refer to the chosen regular refinement. The integral cellular boundary matrices of the intrinsic complex have ranks 98 and 153. Their product vanishes, and Smith reduction shows that the complex has the integral homology of a point. [Complete intrinsic atlas](examples/four-variable-intrinsic.json)

## Chart and complex conventions

**Embedded** mode returns the ambient exceptional divisor. **Intrinsic** mode returns the reduced exceptional divisor on the smooth strict transform. For the general ideal-blowup charts, retain the strata whose closures meet the fiber over the origin when localizing to the germ. Each vertex represents a geometric irreducible component. Each cell represents one geometric component of a multiple intersection. Distinct cells can have the same vertices. The general Newton and ideal-blowup outputs give each cell's ID and its codimension-one faces.

Every chart gives the original variables in chart coordinates. A toric chart includes its cone, pullback factor and strict-transform equation. Substituting its map in the original polynomial gives the pullback factor times the strict transform. Edge components correspond to the roots of the polynomial in the primitive lattice character recorded in `root_families`. For higher-dimensional nondegenerate faces, connectedness of the torus section and smoothness give geometric irreducibility. [Danilov and Khovanskii, §3](https://www.math.toronto.edu/askold/1986-Izvestiya-5-english-s-Danilovym.pdf)

An ideal-blowup chart additionally gives `ambient_ideal`. Its coordinates satisfy all those equations. Its `strict_transform` is a list of extra equations imposed in this coordinate ring. The original coordinates remain among the graph coordinates, and the projective ratios describe the blowup. When the solver removes a nonconstant unit, its inverse belongs to the chart's base domain.

Plane charts modified by later blowups are retained on the complement of the replaced points. A child's `base_domain` refers to the parent's base domain before applying the parent's `exclude_points`. Exact algebraic centers are given in `algebraic_numbers`. For example, the cusp `y^2-x^3` has pullback `u^2*(v^2-u)` in the first chart. Intrinsic resolution stops once the strict curve is smooth. Embedded resolution continues through tangency and produces three exceptional curves of total multiplicities 2, 3 and 6.

## Local monodromy zeta function

~~~sh
./atlas monodromy examples/xy2z3-t6-embedded.json
./atlas resolve 'y^2-x^3' --vars x,y --monodromy --json
~~~

The first command computes directly from a saved resolution. Use `-` to read a JSON record from stdin. The second includes `monodromy_zeta` in a new resolution. The Python and HTTP resolve APIs accept the boolean option `"monodromy": true`. In Python, `engine.monodromy.monodromy_zeta(record)` also accepts a previously loaded record. If an intrinsic resolution stops before the total transform has normal crossings, a new resolution request computes a companion embedded model. The saved-record command reports when the recorded data alone are insufficient.

Atlas uses the Lefschetz convention on the **unreduced** cohomology of the complex Milnor fiber at the origin:

$$
\zeta_{f,0}(T)
=\prod_q \det(I-T h^*\mid H^q(F_0;\mathbf Q))^{(-1)^{q+1}}
=\prod_i (1-T^{N_i})^{-\chi_c(E_i^\circ\cap\pi^{-1}(0))}.
$$

Here $N_i$ is the vanishing order of the original function along a component of the total transform. Its open stratum $E_i^\circ$ omits every other component, including the strict transform when $E_i$ is exceptional. For a smooth germ the strict component itself contributes, giving $1/(1-T)$. The JSON records the multiplicities, the Euler characteristic of each contributing stratum (including computed zero values), the combined factor exponents, and $\chi(F_0)=\sum_i N_i\chi_c(E_i^\circ\cap\pi^{-1}(0))$. [A’Campo, Théorème 3](https://www.e-periodica.ch/cntmng?pid=com-001%3A1975%3A50%3A%3A27)

For the cusp $y^2-x^3$, the three exceptional curves have multiplicities $2,3,6$ and open Euler characteristics $1,1,-1$. Thus $\zeta=(1-T^6)/((1-T^2)(1-T^3))$. For $xy^2z^3-t^6$, summing the open strata over the origin leaves a single factor $1/(1-T^6)$. Toric atlases use orbit decomposition and normalized lattice volumes. General ideal blowups partition the projective fiber by its first nonzero coordinate and compute Euler characteristics by exact polynomial projection. This last calculation can be expensive.

## Topological zeta function

With `--monodromy` (or `"monodromy": true`) a resolution also carries `topological_zeta`, the local topological zeta function of Denef and Loeser at the origin:

$$
Z_{\mathrm{top},0}(s)=\sum_{I}\chi\bigl(E_I^\circ\cap\pi^{-1}(0)\bigr)\prod_{i\in I}\frac{1}{N_i s+\nu_i}.
$$

The sum is over the sets $I$ of components of the total transform, strict transform included; $E_I^\circ$ is the set of points lying on the components of $I$ and on no other, and $\nu_i-1$ is the order of the pulled-back volume form along $E_i$. The record lists the numerical data $(N_i,\nu_i)$, the strata with their Euler characteristics, the poles with their orders, and whether $e^{2\pi i s_0}$ is a zero or pole of the monodromy zeta function at the origin.

Every kind of resolution provides the strata ([engine/strata.py](engine/strata.py)): torus orbits of a fan, with $\nu$ the sum of the coordinates of the ray; the dual graph of a plane curve, with $\nu$ from the adjunction formula; the smooth tangent hypersurface of an ordinary point; the fan read off from the monomial charts of an A-series surface; and, for the blowup of an ideal, exact Euler characteristics on its graph charts, with $\nu-1$ the order of the Jacobian of the projection, computed as the order of a Fitting ideal. Divisors with repeated components give each factor its own multiplicity. Two identities are checked on every answer, and it is withheld if one fails: $Z_{\mathrm{top},0}(0)=1$, and the strata on a single component must give the monodromy zeta function. An intrinsic resolution uses a companion embedded one when needed.

For the cusp $y^2-x^3$ the result is $(4s+5)/((s+1)(6s+5))$; for $z^3+y^4+x^7$ it is $(48s+61)/((s+1)(84s+61))$.

## Blowups where normal crossings fail

When a Newton face is degenerate, the strict transform can fail normal crossings on a torus orbit. The toric charts retain the complement of the affected centres. The local blowup charts cover their inverse images. The two sets of charts form one atlas, with disjoint owned regions for counting strata. The same blowup rule can start directly at the origin. Normal crossings over the compact origin fibre then hold over a neighbourhood of the origin. [Newton repairs](engine/newton_repair.py) · [Blowups from the origin](engine/origin_backend.py)

The centres are chosen by a complete rule in two phases. While the strict transform is singular at a point where normal crossings fail, the centre is the locus where the desingularization invariant of Bierstone and Milman $(\nu_1,s_1;\nu_2,s_2;\dots)$ is largest among those points ([engine/centres.py](engine/centres.py)). It is computed from orders of derivatives on successive hypersurfaces of maximal contact, with the exceptional divisors counted as old or new according to the invariant of the centre that created them; by their theorem it decreases over each centre and every sequence of such blowups is finite. Once the strict transform $X$ is smooth at all those points, in three variables, the procedure of [engine/surface.py](engine/surface.py) takes over. It works with the curves that the exceptional divisors cut on $X$: blow up the points where their union is worse than nodal; then the curves along which the divisors are tangent to $X$ or share a curve, the most tangent first; then the nodes where one divisor crosses itself. The file states why each centre has normal crossings with the divisors and why the procedure ends. It needs far fewer blowups than the invariant, which treats the divisors of the fan as old and blows up along their intersections with $X$.

In the invariant, a hypersurface of maximal contact must be transverse to the new exceptional divisors through the point considered; a divisor that does not pass through it imposes nothing, and a divisor that is not a coordinate hyperplane of the chart becomes one on the graph of its equation, whose extra variable is eliminated when the equation is linear in some variable. On an implicit hypersurface of maximal contact $N$, the quotient of a function $f$ by the coordinate $x$ of a divisor is the normal form of $tf$ modulo $I(N)+(tx-1)$ for an order that compares $t$ first; it has no $t$ exactly when $x$ divides $f$ on $N$.

After a blowup the chart that was blown up is kept, for its points off the centre (the record lists the centre under `blown_up`), and each chart of the blowup owns only its part of the new exceptional divisor. Every point over the origin thus belongs to exactly one chart, and a chart is asked only about its own points: what lies far from a centre keeps the equations it had, instead of being rewritten in the coordinates of the blowup, where old divisors stop being coordinate hyperplanes and degrees grow. A chart whose own points do not meet a centre is left as it is; it is still a chart of the new space near those points. When the parts of a centre in different charts would give the new divisor different numerical data, only the parts on the largest set of divisors are blown up at that stage.

The embedded dual complex counts geometric intersections of exceptional divisors. Intrinsic mode restricts the divisors to the smooth strict transform. Distinct local components are identified on chart overlaps, including compatible marked number fields. Regular torus charts identify conjugate components by their common primitive character, including after a linear coordinate change. When the incidence cannot be certified, the record carries a diagnostic. For $z^3+y^4+\tfrac1{432}x^6+xy^2z+x^3yz+xyz^2$ the embedded complex has 7 vertices, 8 edges and 2 triangles. The intrinsic graph has 5 vertices and 5 edges, including two edges with the same endpoints. [Unified atlas](engine/atlas.py) · [Overlap computation](engine/atlas_gluing.py)

Native operations compute the Jacobian tests, open-frame ideals and derivatives used by the invariant, divisor orders, finite-centre equations, Euler integrals and topological zeta reduction. Python schedules these operations and assembles the incidence graph from the verified chart overlaps. Marked coefficients retain their exact embedding throughout blowups and saved-record replay. [Kernel proofs](native/KERNEL.md#9-operations-for-the-blowups)

Each rational component of the set of bad points is resolved in its own run. Conjugate points are made rational by adjoining a parameter with its minimal polynomial, so that centres are linear in the coordinates; the blowup of a linear centre is written out by substitution and division, without ideal computations. A short geometric rule is kept as a last resort. Whatever the rule, each centre is checked to be smooth and normal crossings are verified in every final chart by the Jacobian criterion, so an answer does not depend on the choice of centres being right.

For $z^3+y^4+\tfrac1{432}x^6+xy^2z+x^3yz+xyz^2$ the face of the ray $(2,3,4)$ has a node; one point and one curve are blown up, with numerical data $(13,11)$ and $(26,20)$, and $\zeta=(1-T^6)^2/((1-T^3)(1-T^4)(1-T^{12})^2(1-T^{13}))$, $Z_{\mathrm{top},0}=(77s^2+175s+90)/(3(s+1)(4s+3)(13s+10))$. For $x^3+y^3+z^3-3xyz+x^4+y^4+z^4$, whose tangent cone is three planes that only split over $\mathbf Q(\sqrt{-3})$, five blowups over three points give $\zeta=1/((1-T^4)(1-T^6)^2)$ and $Z_{\mathrm{top},0}=(7s^2+14s+12)/(12(s+1)^3)$. Started at the origin, $x^2+y^3+z^5$ takes 16 blowups, one divisor having $(N,\nu)=(30,31)$, and both zeta functions agree with those of its Newton resolution ([tests/test_origin.py](tests/test_origin.py)).

## API and parallel computation

~~~sh
curl -s http://127.0.0.1:8765/api/resolve -H 'Content-Type: application/json' \
  -d '{"polynomial":"x*y-z^5","variables":["x","y","z"],"mode":"intrinsic","compute":"cpu"}'
./atlas devices
./atlas resolve 'x^5+z*y^3+z^15+t^4*x+t^8' --vars x,y,z,t --mode intrinsic --gpu --json
~~~

For large requests, select **Exhaustive search** in the resolution form or **Exhaustive mode** in the algebra form. The HTTP equivalents accept the same JSON bodies as the ordinary routes. `POST /api/searches` starts a resolution job. Poll with `GET /api/searches/{id}` and cancel with `DELETE /api/searches/{id}`. `POST /api/algebra/exhaustive` returns the algebra result on the same connection. Both routes accept requests above the usual 32,768-byte limit. The exhaustive algebra route also lifts the ordinary 55-second calculation deadline. The algebra CLI already uses that policy. Resolution downloads preserve large multiplicities as exact JSON integers. Older browsers without lossless JSON support may prompt for a browser update. [HTTP interface](engine/app.py) · [Browser regression tests](tests/test_browser_exhaustive.py)

Resolution defaults to CPU and rational ideal arithmetic defaults to auto. On a machine with a working Triton toolchain, the ordinary native build (`make`) is sufficient. Use `make gpu` to also build the OpenCL plugins. With `--gpu` (equivalently `--compute gpu`, or `"compute":"gpu"` over HTTP) Atlas attempts eligible work on the selected device. By default it tries Triton, then OpenCL, then the exact CPU implementation. Identity resolutions can finish entirely on the CPU. The `execution.stages` entries identify the chart and rank certificates and the actual modular replay batches from the solve and its verification workers. The primary certificate retains its original backend fields.

`./atlas devices` lists backend/device pairs. Use `RESOLUTION_GPU_BACKEND=triton` or `opencl` to select a backend and `RESOLUTION_GPU_INDEX` to select an ordinal in its list. Triton discovery compiles and executes an integer test on the requested device. OpenCL uses the GPUs enumerated by its driver. For automatic size-based dispatch, use `--compute auto`. The initial thresholds are 50,000 chart–monomial pairs and 256 matrix columns. Adjust them with `RESOLUTION_GPU_MIN_PAIRS` and `RESOLUTION_GPU_MIN_COLS`. Modular replays in automatic mode use the native CPU threads by default.

For Gröbner traces, `--compute auto` uses measured row-reduction times for the selected device, CPU and kernel build. To calibrate, run the following from a built checkout with a working GPU driver. Substitute `triton` for `opencl` to profile that backend.

~~~sh
python3 benchmarks/groebner_trace_profile.py --backend opencl --index 0 --sizes 6 7 8 \
  --write-auto-profile "${XDG_CACHE_HOME:-$HOME/.cache}/resolution-atlas/groebner-trace-opencl-0.json"
~~~

`auto` considers batches with a measured prime, row count, column count and reducer-entry count. The GPU must save at least 10% at the median and its upper-quartile time must beat the CPU's lower quartile. A cold process also charges the observed startup cost. Unmeasured batches use native threads. Set `ATLAS_TRACE_PROFILE` to read a calibration from another path. CPU-only requests bypass calibration and GPU discovery.

| Operation | Independent GPU work | Exactness |
|---|---|---|
| Toric chart transforms, dimensions 2–8 | Compute exponents for many terms and charts. Triton reuses a ray–monomial table when cones share enough rays. | The host bounds every dot product before the launch, checks nonnegative strict powers, and retains the exact coefficients. |
| Projective Jacobian rank | Find a pivot, normalize its row, then update tiles of the trailing matrix in parallel. Successive pivots stay ordered and the matrix stays on the device. | All products are integer residues. C++ verifies the proposed maximal minor independently. |
| Modular Gröbner bases | Reduce independent S-pairs of the same degree on the selected GPU. Replay across primes, using native threads for small batches after a GPU trace. The execution report records trace pairs and replay primes separately. | The native solver lifts by CRT and rational reconstruction. Independent S-pair and generator checks share a reduction plan and run on CPU threads with arbitrary-precision integers. |
| Normal form and membership modulo a monomial ideal | For an explicit GPU request, Triton or OpenCL finds a divisor for each removable term. | The host checks each witness, checks every surviving term for irreducibility and retains its exact rational coefficient. |

All eight rational ideal operations share the request's compute policy across intermediate bases. For example, the following lexicographic elimination computes a basis using the selected device, lifts and checks the basis on the CPU, then projects to the surviving variable. On the Radeon 680M the basis replay used four OpenCL batches totaling 12 primes. The returned ideal in QQ[y] is generated by `y^3-y^2/2147483629+4/2147483629`.

~~~sh
./atlas algebra eliminate --vars x,y --order lex --gen 'x^2+2147483629*y-1' --gen 'x*y-2' --eliminate x --compute gpu --json
~~~

Each work record in `execution.stages` includes its actual backend. Intermediate bases in a derived operation also carry a label. `groebner_certification` reports the host's lift and verification times and the number of certified bases. A failed launch records its reason in `execution.fallback_reasons` and the exact CPU computation supplies the result. [Phase 1 CPU/GPU comparison](benchmarks/README.md#phase-1-rational-ideal-arithmetic)

To compare devices, run `python3 benchmarks/charts.py --backend triton` and `python3 benchmarks/gpu.py --backend triton`, or select `opencl`. Both drivers measure warm solves and fresh CLI invocations, check geometry against the CPU, and require the requested backend to have run. [Benchmarks](benchmarks/README.md)

### GPU implementation plan

The performance target is elapsed time from input polynomial to certified atlas and requested invariants. Measure cold CLI runs and warmed workers separately, including packing, transfers, compilation and CPU verification. Keep a CPU baseline for each workload. Start with ordinary points, the 191- and 598-chart fans, plane branches over a marked field, a repaired fan, and hard Gröbner ideals. Measure both the common small jobs and the large batches.

1. **Establish cross-device baselines.** Run the same correctness and timing suite on a supported NVIDIA CUDA GPU, AMD ROCm GPU and an integrated GPU. Include Intel XPU where the installed backend passes the integer probe, and OpenCL where available. Use ordinary Triton for the shared kernels. Select tile sizes and thresholds from measured workload shapes. Compare a Gluon specialization against this baseline if profiling shows that explicit layouts or shared memory could improve full solves.
2. **Share exact work across charts.** In Newton, diagonal toric and binomial engines, compute ray–monomial valuations, minima and face membership in batches. Reuse those data across incident charts and minimize readbacks. Keep the independent verifier's own pullback, ownership and face checks. Fan refinement and the next binomial blowup depend on the previous subdivision.
3. **Accelerate the costly algebra phases.** The Katsura-6, -7 and -8 benchmarks measure full certified CPU, GPU and automatic solves, with separate trace, replay, lift and verification timings. Use the matching trace profiles to evaluate automatic dispatch on other devices. For Jacobian rank, batch matrices of similar size and test blocked pivot panels. [Profiles and full solves](benchmarks/README.md#profiling-katsura-6--7-and--8)
4. **Batch within each blowup step.** In plane, chain, origin, repaired Newton and general-search paths, group independent child-chart substitutions, jets and Jacobian tests with compatible coefficient fields. Synchronize before selecting the global centre and before constructing the next dependent generation. Keep marked embeddings, the order of blowups and the owned complement of each centre intact.
5. **Extend the shared atlas verifier.** Batch independent chart–divisor subsets, local incidence and Euler computations. Merge overlap evidence and form the exact zeta functions on the host. Validate against fresh and saved records in both modes, including labelled attachments, conjugate branches, companion embedded resolutions and partial results. Promote a path to automatic dispatch after repeated end-to-end timings show which sizes benefit on that device.

## Verification and comparison

~~~sh
make test
make test-gpu
python3 benchmarks/published_comparison.py --gpu-target --output /tmp/atlas-comparison.json
python3 benchmarks/groebner.py --singular
python3 benchmarks/groebner_hard.py --singular --gpu
~~~

The tests check pullback identities in every chart of the four-variable example, fan unimodularity and coverage, nondegenerate faces, root labels, face attachments, graph saturation and overlap gluing. General-search tests include a curved center for a Whitney umbrella and a single ideal representing two successive blowups of a tacnode. GPU tests compare exact results with the CPU.

Install the optional comparison dependency with `python3 -m pip install -r requirements-test.txt` to run the test suite or the SymPy reference modes. The kernel's routines return the same generators as the SymPy code they replace, so the two can be compared exactly. The tests do this on 796 Gröbner bases recorded from the verifier, on saturations, smooth loci and blowup charts, and on whole candidate verifications. Set `ATLAS_GROEBNER=check` to compute everything with both and stop on any difference, or `ATLAS_GROEBNER=sympy` to use SymPy alone. For standalone replay diagnostics, `ATLAS_GROEBNER_GPU=1` selects OpenCL, `triton` selects Triton, and `reference` selects the NumPy interpreter. An explicit CPU solve uses the native CPU replay. The two benchmark commands above time the kernel against Singular's `std` and `modStd` when Singular is installed. [Kernel timings](benchmarks/README.md#gröbner-kernel)

The benchmark includes a published Singular tutorial example, a binomial with published Singular results, and the four-variable example in both modes. Atlas and Singular reproduce the same six-curve intersection graph for the tutorial example. For `x*y^2*z^3-t^6`, the Atlas binomial resolver uses eight smooth blowups and twelve final charts. Its ambient complex has 8 vertices, 15 edges and 8 triangles, and its intrinsic complex has 7 vertices, 12 edges and 6 triangles. [Embedded atlas](examples/xy2z3-t6-embedded.json) · [Intrinsic atlas](examples/xy2z3-t6-intrinsic.json) Times from the published study and local times were measured on different machines. [Report and reproduction steps](benchmarks/README.md) · [Published comparison, Table 1](https://arxiv.org/pdf/1109.1655#page=10)
