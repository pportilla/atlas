# Resolution Atlas

Atlas computes resolutions of hypersurface germs at the origin over the algebraic closure of the rationals. It returns affine charts, maps to the input coordinates, strict transforms and the geometric dual complex of the exceptional divisor. The CLI and local HTTP API use the same solver.

C++17, GMP, FLINT and libnormaliz handle point blowups, exact linear algebra and polyhedral computations. Python assembles the charts and drives the general ideal-blowup search. Each candidate centre is decided by an exact native kernel: Gröbner bases over the rationals, saturation, ideal quotients, the Jacobian smoothness test and the charts of a blowup. When coefficients swell it switches to a Hilbert-driven computation for lexicographic and elimination orders, or to images modulo many primes whose lift it certifies over the rationals. [Algorithms and proofs](native/KERNEL.md). SymPy handles number fields, factorization and the geometric components of a candidate that has passed. The optional OpenCL kernels handle independent chart substitutions and modular matrix updates. Atlas runs independently of Singular and SageMath.

## Build and run

Install a C++17 compiler, Make, pkg-config, Python 3.10 or later, SymPy, and the development libraries for GMP (including gmpxx), FLINT, JsonCpp and libnormaliz. The project was tested on Linux with FLINT 3.6, SymPy 1.14 and Python 3.14. The optional GPU build needs OpenCL headers, a loader and a compatible device driver.

~~~sh
python3 -m pip install -r requirements.txt
make -j4
./atlas resolve 'y^2-x^3' --vars x,y
./atlas resolve 'x^5+z*y^3+z^15+t^4*x+t^8' --vars x,y,z,t --mode intrinsic --json
./atlas resolve 'x^2-y^2*z' --vars x,y,z,t --mode embedded --exhaustive --json
./atlas serve
~~~

Choose distinct variable names beginning with a letter. Use rational coefficients, explicit multiplication, and nonnegative integer exponents with `^` or `**`. Division by a nonzero rational constant is supported. The polynomial must vanish at the origin. For a germ at a rational point, translate that point to zero in the input.

The local interface is available at <http://127.0.0.1:8765>. CLI exit code 0 means resolved, 2 means an input or computation error and 3 means the quick search was inconclusive. Use `--json` to retain the equations and complete incidence records.

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

When a Newton face is degenerate, the strict transform fails normal crossings at the singular points of the cut it makes on a torus orbit. [engine/local_blowups.py](engine/local_blowups.py) blows up over those points until normal crossings hold, and the strata over them are added to those of the fan ([engine/newton_repair.py](engine/newton_repair.py)). The same blowups, started at the origin of the germ with no divisor given, are an engine of their own ([engine/origin_backend.py](engine/origin_backend.py)): failing normal crossings is a closed condition and the fibre over the origin is compact, so normal crossings at every point over the origin give them over a neighbourhood of the origin.

The centres are chosen by a complete rule in two phases. While the strict transform is singular at a point where normal crossings fail, the centre is the locus where the desingularization invariant of Bierstone and Milman $(\nu_1,s_1;\nu_2,s_2;\dots)$ is largest among those points ([engine/centres.py](engine/centres.py)). It is computed from orders of derivatives on successive hypersurfaces of maximal contact, with the exceptional divisors counted as old or new according to the invariant of the centre that created them; by their theorem it decreases over each centre and every sequence of such blowups is finite. Once the strict transform $X$ is smooth at all those points, in three variables, the procedure of [engine/surface.py](engine/surface.py) takes over. It works with the curves that the exceptional divisors cut on $X$: blow up the points where their union is worse than nodal; then the curves along which the divisors are tangent to $X$ or share a curve, the most tangent first; then the nodes where one divisor crosses itself. The file states why each centre has normal crossings with the divisors and why the procedure ends. It needs far fewer blowups than the invariant, which treats the divisors of the fan as old and blows up along their intersections with $X$.

In the invariant, a hypersurface of maximal contact must be transverse to the new exceptional divisors through the point considered; a divisor that does not pass through it imposes nothing, and a divisor that is not a coordinate hyperplane of the chart becomes one on the graph of its equation, whose extra variable is eliminated when the equation is linear in some variable. On an implicit hypersurface of maximal contact $N$, the quotient of a function $f$ by the coordinate $x$ of a divisor is the normal form of $tf$ modulo $I(N)+(tx-1)$ for an order that compares $t$ first; it has no $t$ exactly when $x$ divides $f$ on $N$.

After a blowup the chart that was blown up is kept, for its points off the centre (the record lists the centre under `blown_up`), and each chart of the blowup owns only its part of the new exceptional divisor. Every point over the origin thus belongs to exactly one chart, and a chart is asked only about its own points: what lies far from a centre keeps the equations it had, instead of being rewritten in the coordinates of the blowup, where old divisors stop being coordinate hyperplanes and degrees grow. A chart whose own points do not meet a centre is left as it is; it is still a chart of the new space near those points. When the parts of a centre in different charts would give the new divisor different numerical data, only the parts on the largest set of divisors are blown up at that stage.

Because the charts partition the points over the origin, the components of an intersection $E_J$ of exceptional divisors are counted once each: an irreducible component is dense in the part of it seen by the first chart that meets it, and that part lies in what the chart owns; so the number of components of $E_J$ is the sum, over the charts, of the components of $E_J$ on the chart that lie in its own part ([engine/local_blowups.py](engine/local_blowups.py), `cells`). This gives the dual complex after the blowups: the cells of the fan stay (a toric intersection through a blown-up point is a curve or a surface, and its strict transform still meets the fibre over the point), and the cells with a new divisor are added. It is reported as complete when every number is determined and no intersection with several components is a face of another; otherwise it is partial and the record says why (for instance a divisor created over conjugate points, whose components are not defined over $\mathbf Q$). For $z^3+y^4+\tfrac1{432}x^6+xy^2z+x^3yz+xyz^2$ the complex has 7 vertices, 8 edges and 2 triangles, with Euler characteristic 1, as it must be for the exceptional divisor of a composition of blowups of a smooth point.

The polynomial computations are done in the C++ kernel: the normal crossings test, the blowup of a linear centre, the step of the surface procedure and the Euler characteristics of the strata ([native/KERNEL.md](native/KERNEL.md), section 9, with the proofs). The invariant itself is still computed in Python on top of the kernel's Gröbner bases.

Each rational component of the set of bad points is resolved in its own run. Conjugate points are made rational by adjoining a parameter with its minimal polynomial, so that centres are linear in the coordinates; the blowup of a linear centre is written out by substitution and division, without ideal computations. A short geometric rule is kept as a last resort. Whatever the rule, each centre is checked to be smooth and normal crossings are verified in every final chart by the Jacobian criterion, so an answer does not depend on the choice of centres being right.

For $z^3+y^4+\tfrac1{432}x^6+xy^2z+x^3yz+xyz^2$ the face of the ray $(2,3,4)$ has a node; one point and one curve are blown up, with numerical data $(13,11)$ and $(26,20)$, and $\zeta=(1-T^6)^2/((1-T^3)(1-T^4)(1-T^{12})^2(1-T^{13}))$, $Z_{\mathrm{top},0}=(77s^2+175s+90)/(3(s+1)(4s+3)(13s+10))$. For $x^3+y^3+z^3-3xyz+x^4+y^4+z^4$, whose tangent cone is three planes that only split over $\mathbf Q(\sqrt{-3})$, five blowups over three points give $\zeta=1/((1-T^4)(1-T^6)^2)$ and $Z_{\mathrm{top},0}=(7s^2+14s+12)/(12(s+1)^3)$. Started at the origin, $x^2+y^3+z^5$ takes 16 blowups, one divisor having $(N,\nu)=(30,31)$, and both zeta functions agree with those of its Newton resolution ([tests/test_origin.py](tests/test_origin.py)).

## API and parallel computation

~~~sh
curl -s http://127.0.0.1:8765/api/resolve -H 'Content-Type: application/json' \
  -d '{"polynomial":"x*y-z^5","variables":["x","y","z"],"mode":"intrinsic","compute":"cpu"}'
make gpu
./atlas devices
./atlas resolve 'x^5+z*y^3+z^15+t^4*x+t^8' --vars x,y,z,t --mode intrinsic --compute gpu --json
~~~

Use `POST /api/searches` with the same body for an exhaustive background job. Get its state and result with `GET /api/searches/{id}` or `GET /api/searches/current`. Cancel with `DELETE /api/searches/{id}`.

CPU is the default. The OpenCL 1.2 chart kernel processes one chart–monomial pair per work item in dimensions 2–8. The modular rank kernel processes independent entries in parallel and the CPU verifies the resulting nonzero minor. The CPU builds the fan, proves face nondegeneracy, computes the dual complex and verifies general blowups. Use `--compute auto` for threshold-based dispatch and `--compute gpu` to attempt OpenCL even for small eligible batches. The `execution`, `certificate.chart_batch` and `certificate.rank_backend` fields record the actual backend. Environment variables `RESOLUTION_GPU_INDEX`, `RESOLUTION_GPU_MIN_COLS` and `RESOLUTION_GPU_MIN_PAIRS` select the device and adjust the automatic thresholds.

The modular Gröbner route replays one recorded computation modulo many primes; every prime runs the same instructions on its own coefficients. [engine/gpu_replay.py](engine/gpu_replay.py) breaks that replay into five small data-parallel parts over (primes) × (terms): the pivot factor of a reduction step, the update `work += factor * row`, the zero check, the modular inverse of a leading coefficient, and the scaling of a new row. They are written twice: as [Triton](https://triton-lang.org) kernels (`ATLAS_GROEBNER_GPU=triton`; needs PyTorch with CUDA and the `triton` package) and with NumPy (`ATLAS_GROEBNER_GPU=reference`), which is what the tests run: the kernel traces a Gröbner computation, replays it for a batch of primes on its own threads and through the backend, and counts the primes on which they differ ([tests/test_gpu_replay.py](tests/test_gpu_replay.py)). The Triton kernels have the same decomposition and arithmetic but have not been run: this machine has no CUDA device. In the Triton backend the host walks the program and launches one kernel per instruction; fusing consecutive instructions is the next step before timing it.

## Verification and comparison

~~~sh
make test
make test-gpu
python3 benchmarks/published_comparison.py --gpu-target --output /tmp/atlas-comparison.json
python3 benchmarks/groebner.py --singular
python3 benchmarks/groebner_hard.py --singular --gpu
~~~

The tests check pullback identities in every chart of the four-variable example, fan unimodularity and coverage, nondegenerate faces, root labels, face attachments, graph saturation and overlap gluing. General-search tests include a curved center for a Whitney umbrella and a single ideal representing two successive blowups of a tacnode. GPU tests compare exact results with the CPU.

The kernel's routines return the same generators as the SymPy code they replace, so the two can be compared exactly. The tests do this on 796 Gröbner bases recorded from the verifier, on saturations, smooth loci and blowup charts, and on whole candidate verifications. Set `ATLAS_GROEBNER=check` to compute everything with both and stop on any difference, or `ATLAS_GROEBNER=sympy` to use SymPy alone. `--compute gpu` (or `ATLAS_GROEBNER_GPU=1`), after `make gpu`, runs the modular replays on an OpenCL device; the result is certified on the CPU either way. The two benchmark commands above time the kernel against Singular's `std` and `modStd` when Singular is installed. [Kernel timings](benchmarks/README.md#gröbner-kernel)

The benchmark includes a published Singular tutorial example, a binomial with published Singular results, and the four-variable example in both modes. Atlas and Singular reproduce the same six-curve intersection graph for the tutorial example. For `x*y^2*z^3-t^6`, the Atlas binomial resolver uses eight smooth blowups and twelve final charts. Its ambient complex has 8 vertices, 15 edges and 8 triangles, and its intrinsic complex has 7 vertices, 12 edges and 6 triangles. [Embedded atlas](examples/xy2z3-t6-embedded.json) · [Intrinsic atlas](examples/xy2z3-t6-intrinsic.json) Times from the published study and local times were measured on different machines. [Report and reproduction steps](benchmarks/README.md) · [Published comparison, Table 1](https://arxiv.org/pdf/1109.1655#page=10)
