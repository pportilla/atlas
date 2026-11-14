# Atlas and Singular

This README compares the Atlas capabilities at the start of the project with Singular 4.4.1. The [phased roadmap](#phased-roadmap) records subsequent implementation progress.

Atlas now exposes all eight Phase 1 rational ideal operations through the CLI, prompt, scripts, and browser. The [algebra API examples](examples/algebra-api.md) show the operands, exact results, and execution details. The comparison tables below describe the baseline before this interface was added.

## What each system lets you do

Atlas answers a specific geometric question. Give it one polynomial with rational coefficients and it resolves the hypersurface germ at the origin. You can inspect the blowup charts, maps back to the original coordinates, strict transforms, exceptional components, and their geometric dual complex. You can also request the local monodromy and topological zeta functions. Atlas accounts for conjugate branches over algebraic extensions, so the certified charts also resolve the complex germ. [Atlas resolution scope](README.md#build-and-run) · [Chart conventions and invariants](README.md#chart-and-complex-conventions)

Singular lets you start with a ring, define several polynomials, ideals, matrices or modules in it, and reuse the result of one computation in the next. Desingularization is one library in that environment. The largest difference for an Atlas user is the ability to ask an intermediate question directly, such as “What is the ideal of the singular locus?” or “Which relations hold among these generators?” [Singular data types](https://www.singular.uni-kl.de/Manual/4-4/sing_72.htm) · [Core algorithms](https://www.singular.uni-kl.de/Manual/4-4/sing_33.htm) · [Singularity invariants](https://www.singular.uni-kl.de/Manual/4-4/sing_2038.htm)

This comparison uses the Singular 4.4.1 manual and the Atlas working tree reviewed on 14 November 2026. An operation counts as available to an Atlas user when the documented CLI or browser accepts the relevant input and returns the result. Where the resolver already uses part of an algorithm, the tables identify it. Coverage is by mathematical workflow. Individual library procedures, coefficient domains and optional dependencies have their own requirements. [Singular manual](https://www.singular.uni-kl.de/index.php/singular.pdf) · [Library catalog](https://www.singular.uni-kl.de/Manual/4-4/sing_1018.htm) · [Atlas commands](engine/app.py)

## The gaps closest to Atlas

| Workflow | What Singular offers | What Atlas offers today |
| --- | --- | --- |
| Resolve a variety given by several equations | In characteristic zero, `resolve.lib` takes an ideal, can resolve across an affine variety, and returns intermediate and final charts. A separate operation blows up a center supplied by the user. [Resolution library](https://www.singular.uni-kl.de/Manual/4-4/sing_1784.htm) | Embedded or intrinsic resolution of one hypersurface germ at the origin. Atlas constructs and checks ideal blowups internally. Its public input has no field for several defining equations, a global resolution request, or a chosen center. |
| Describe curve branches | Hamburger–Noether and Puiseux developments, branch parametrizations, and libraries for space curves. [Singularity libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_1867.htm) | Plane-curve charts distinguish geometric branches and record where they meet the exceptional divisor. Users cannot request a branch's Puiseux series or a space-curve parametrization. |
| Compute local invariants and study families | Singular-locus ideals, tangent cones, Milnor and Tjurina numbers under their hypotheses, classification of several classes of singularities, miniversal deformations, and equisingularity strata. [Local invariants](https://www.singular.uni-kl.de/Manual/4-4/sing_2038.htm) · [Classification and deformations](https://www.singular.uni-kl.de/Manual/4-4/sing_1867.htm) | Atlas uses tangent cones and Jacobian tests to select and verify blowups. Its benchmarks can recover Milnor numbers for isolated hypersurfaces from zeta data. The public interface has no general command for the local algebras, classification, or deformations. [Atlas Milnor benchmark](benchmarks/milnor_battery.py) |
| Study monodromy beyond its zeta function | For isolated hypersurface singularities, Singular can compute monodromy Jordan blocks, spectral pairs, the singularity spectrum, the Gauss–Manin system, and the Bernstein–Sato polynomial. [Gauss–Manin library](https://www.singular.uni-kl.de/Manual/4-4/sing_1963.htm) | Atlas returns the monodromy zeta function and the local topological zeta function, with contributions from resolution strata. The monodromy zeta function alone cannot distinguish different Jordan block sizes with the same eigenvalue multiplicities. |

The geometric overlap has a qualification worth preserving. Singular's `resolve(J)` can take several generators, but when it encounters nonreduced or nonequidimensional input it warns and proceeds with the reduced top-dimensional part. Atlas's **embedded** mode retains the multiplicities of a hypersurface divisor with repeated components. Its **intrinsic** mode requires a reduced germ. Which input and output a procedure preserves matters when comparing the two. [Singular procedure source](https://github.com/Singular/Singular/blob/Release-4-4-1p5/Singular/LIB/resolve.lib#L3956-L3964) · [Atlas multiplicity convention](README.md#resolution-algorithms)

## The wider mathematical gaps

| Area | What a Singular user can do | Atlas coverage |
| --- | --- | --- |
| Coefficient domains, rings and orders | Select rational or finite fields, algebraic or transcendental extensions, integer or residue rings, or approximate real or complex coefficients. Work in quotient rings and localizations. Choose global, local, weighted, or block monomial orders. Available algorithms depend on the ring. [Manual, §3.3](https://www.singular.uni-kl.de/index.php/singular.pdf) | Rational input. Exact number fields, selected orders, and intermediate reductions modulo primes serve the resolver. Users cannot declare a base ring or choose a local term order. |
| Polynomial and ideal arithmetic | Factor polynomials, compute GCDs and resultants, form Gröbner or local standard bases, reduce modulo an ideal, test membership, eliminate variables, intersect ideals, and compute quotients and saturations. [Core algorithms](https://www.singular.uni-kl.de/Manual/4-4/sing_33.htm) · [Commutative algebra libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_1249.htm) | **Substantial internal overlap.** Atlas already computes rational Gröbner bases, normal forms, membership, elimination, intersections, quotients, and saturation. These are used during verification. Users cannot submit an ideal to these operations through the CLI or browser. [Atlas ideal kernel](native/KERNEL.md#3-operations-on-ideals) |
| Structure of ideals and rings | Compute dimension, degree, multiplicity, and Hilbert series. Find radicals, associated primes, and primary decompositions. Compute normalization and integral closure in supported settings. [Core algorithms](https://www.singular.uni-kl.de/Manual/4-4/sing_33.htm) · [Commutative algebra libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_1249.htm) | Atlas computes dimensions and uses Hilbert-series methods internally. It also identifies geometric components for its charts. General decomposition, normalization, and Hilbert-series requests have no public entry point. |
| Matrices, modules and homological algebra | Enter polynomial matrices and module presentations. Compute syzygies, free resolutions, graded Betti tables, and Hom, Ext and Tor. [Matrix libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_1190.htm) · [Core algorithms](https://www.singular.uni-kl.de/Manual/4-4/sing_33.htm) · [Homological algebra](https://www.singular.uni-kl.de/Manual/4-4/sing_1324.htm) | Atlas uses matrices for rank, Jacobian, and incidence calculations. It has no public module or chain-complex input. Singular's `res` computes a **free resolution** describing algebraic relations. Atlas's `resolve` gives a **geometric resolution** by modifying a variety. |
| Solve systems of equations | Solve zero-dimensional polynomial systems, compute triangular decompositions, isolate real roots, determine realizable sign conditions, and solve over finite fields. [Solving libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_2148.htm) | Exact algebraic roots locate branches and blowup centers. There is no general systems solver with a user-selected solution domain. |
| Algebraic geometry and invariant theory | Compute with divisors on curves, Riemann–Roch spaces, sheaf and de Rham cohomology, intersection theory, and invariant rings of group actions. [Algebraic geometry](https://www.singular.uni-kl.de/Manual/4-4/sing_1601.htm) · [Invariant theory](https://www.singular.uni-kl.de/Manual/4-4/sing_2074.htm) | Atlas reports exceptional divisors, their intersections, and Euler characteristics of resolution strata. The broader computations require inputs such as a sheaf, a divisor on a curve, or a group action. |
| Noncommutative algebra and D-modules | Work in algebras such as Weyl and exterior algebras. PLURAL, SCA, and LETTERPLACE support different classes of algebras. D-module libraries include annihilator, localization, and integration computations. [Noncommutative subsystems](https://www.singular.uni-kl.de/Manual/4-4/sing_431.htm) · [PLURAL libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_501.htm) | The Atlas input language describes commutative polynomials. It has no user-defined multiplication relations, left ideals, or D-module operations. |
| Tropical, toric and polyhedral geometry | Work with cones, fans and polytopes as objects, compute tropical varieties and GIT fans, and use interfaces to external polyhedral software. [Data types](https://www.singular.uni-kl.de/Manual/4-4/sing_72.htm) · [Tropical libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_2484.htm) | Atlas already constructs Newton fans and toric modifications for resolution. A user cannot request the tropical variety of an ideal or run a general fan computation. |
| Other application libraries | Algebraic-geometric codes and decoding, algebraic systems and control theory, hyperplane arrangements, teaching libraries, surface rendering and TeX output. The catalog also identifies experimental packages and interfaces to external software. [Library catalog](https://www.singular.uni-kl.de/Manual/4-4/sing_1018.htm) · [Visualization](https://www.singular.uni-kl.de/Manual/4-4/sing_2251.htm) | Atlas provides resolution-specific visualization and JSON export. These additional application families lie outside its public workflows. |

## One calculation through the two interfaces

For the cusp `y^2-x^3`, both programs can answer geometric questions. Singular can also compute the dimension of the local Jacobian algebra directly. The derivative ideal is `(x^2,y)` over the rationals, so the quotient has basis `1,x`. This session returns that basis and dimension **2**, the Milnor number of the cusp. [Singular standard-basis commands](https://www.singular.uni-kl.de/Manual/4-4/sing_33.htm)

```singular
ring R = 0,(x,y),ds;
poly f = y^2-x^3;
ideal J = jacob(f);
ideal G = std(J);
kbase(G);
vdim(G);
```

Running the following Atlas command on the same equation returned an embedded resolution with exceptional multiplicities **2, 3, 6** and monodromy zeta function **(1−T⁶)/((1−T²)(1−T³))**. Its JSON contains the chart maps and intersection data. These two examples were run locally. [Atlas zeta convention](README.md#local-monodromy-zeta-function)

```sh
./atlas resolve 'y^2-x^3' --vars x,y --mode embedded --monodromy --json
```

Singular's interactive session retains the ring, the polynomial, the ideal, and the basis. A user can inspect any of them, change one input, save the commands as a script, write procedures with loops and conditionals, or load a library. Singular also documents a Jupyter interface. Atlas's prompt currently dispatches `resolve`, `monodromy`, `devices`, and `serve`. Its browser already shares the solver and adds forms, chart and dual-complex views, progress and cancellation, and JSON downloads. [Singular manual, chapter 3](https://www.singular.uni-kl.de/index.php/singular.pdf) · [Singular in Jupyter](https://www.singular.uni-kl.de/public/pages/graphical-interface.html) · [Atlas prompt](engine/app.py) · [Atlas browser](web/index.html)

For the proposed alternative UI, the cusp gives a concrete comparison. A terminal user could name a Jacobian ideal and ask for its quotient basis. A browser user could select the same polynomial, choose **Local Jacobian algebra**, and see the generators, the basis, and the dimension, together with the equivalent CLI calculation. That algebra workflow is a future capability. The guided resolution workflow already exists. Providing equivalent mathematical operations and accepting existing Singular scripts are separate compatibility goals.

## Phased roadmap

Phases 1 and 2 have working public examples. Phase 3 has begun with polynomial GCD over the rationals. The remaining milestones follow their mathematical dependencies.

### Phase 1. Expose rational ideal arithmetic

Start with ideals in a global polynomial ring over $\mathbb Q$. Offer a reduced Gröbner basis, the normal form of a polynomial, and ideal membership first. These three operations share a basis computation and already have callable implementations. Then expose elimination, intersection, ideal quotient, saturation by one polynomial, and Krull dimension. Some of these currently live inside other kernel operations and need a public adapter. Use the general expression parser so that zero, constants, and generators such as `x^2-1` are valid algebra inputs. [Polynomial parser](engine/polynomial.py) · [Basis and remainder routines](engine/groebner.py) · [Ideal algorithms](native/KERNEL.md#3-operations-on-ideals)

Both interfaces would submit one operation with the same ordered variables, coefficient field, monomial order, and operands. A polynomial belongs in the normal-form field, a second ideal in an intersection or quotient, and a variable to remove in an elimination. The result records its type and ring, including the actual order of a returned basis and the surviving variables after elimination. Initially, offer lexicographic and graded reverse lexicographic order for basis computations. A normal-form result is the exact remainder. The current native routine does not compute the coefficients expressing the reduction in terms of the generators. [Kernel interface](engine/kernel.py) · [Normal form implementation](engine/groebner.py)

**CLI.** Use the same commands at the shell and at the `atlas>` prompt. Allow repeatable generators and JSON output for reuse.

```sh
./atlas algebra normal-form --vars x,y --order lex --gen 'x^2-1' --gen 'y-x' --poly 'x^2+y/2' --json
```

**Browser.** In an **Algebra** view, choose **Normal form**, enter the variables in order, and add one row per generator. The example above should display the remainder $1+y/2$ in $\mathbb Q[x,y]$, a copyable CLI command, and the same downloadable JSON result as the terminal. Both entry points should use the same parser and operation handler. If the kernel cannot complete a request, both should report that outcome explicitly so it cannot be mistaken for failed membership or the zero ideal.

**Phase 1 acceptance examples.** Run each case through the CLI and browser, compare their structured results, and check the mathematics against Singular over the same ring and order. Compare ideals by reduced basis or mutual membership so that equivalent choices of generators are accepted. [Singular basis operations](https://www.singular.uni-kl.de/Manual/4-4/sing_33.htm) · [Existing Atlas comparison corpus](benchmarks/README.md#the-verifiers-bases)

| Request in $\mathbb Q[x,y]$ | Expected result |
| --- | --- |
| With $x>y$ in lex order, take $I=(x^2-1,y-x)$ | The reduced basis consists of $x-y$ and $y^2-1$. Eliminating $x$ gives $(y^2-1)$ in $\mathbb Q[y]$. |
| Reduce $x^2+y/2$ modulo $I$, then test membership of $y^2-1$ and $y$ | Remainder $1+y/2$. The membership answers are true and false, respectively. |
| For $D=(x^2y)$ compare quotient by $(x)$ with saturation by $x$ | $D:(x)=(xy)$ and $D:x^\infty=(y)$. Intersecting $(x)$ and $(y)$ gives $(xy)$. |
| Ask for dimension and normal forms at the boundary cases | $\mathbb Q[x,y]/(0)$ has Krull dimension 2 and reduction preserves the input. Modulo $(1)$ every remainder is zero and the variety is empty. |

### Phase 2. Make results reusable across a session

**Verified in the working tree.** The prompt’s `:save FILE.atlas` records successful statements for replay. In the browser, a named Gröbner basis supplied two normal forms, and the exported session reproduced the results through `./atlas run`. The HTTP server can reuse a verified basis in the same ordered ring. [Transcript tests](tests/test_session_transcript.py) · [Browser replay tests](tests/test_browser_algebra_session.py) · [Basis references](examples/algebra-api.md#reuse-a-basis-over-http)

Give rings, polynomials, ideals and bases names. Let CLI users save and replay a sequence of operations. Let browser users select an earlier result as the next operand and export the equivalent script. A user should be able to define $I$ once, compute its basis, reuse it for two normal forms, and reproduce the session in the other interface. Each object carries its ring so a change of field or variable order requires an explicit conversion.

### Phase 3. Extend commutative algebra and the supported rings

**Available now:** monic polynomial GCD over $\mathbb Q$ through the shell, prompt, scripts, HTTP and guided browser. The result is a polynomial in the declared ordered ring and can be reused in another calculation. The regression cases cover zero operands, rational coefficients with thousands of digits, and a sparse exponent of $2^{64}+17$. A physical Radeon 680M produced the same exact result as direct FLINT on the measured 25-term example. Direct FLINT was faster for that example and is the current automatic path. [Usage and conventions](examples/algebra-api.md#polynomial-gcd) · [Exactness and device measurements](native/KERNEL.md#monic-polynomial-gcd-over-the-rationals) · [Browser replay tests](tests/test_browser_algebra_gcd.py)

Remaining milestones include user-facing factorization, resultants, quotient bases, Hilbert series and multiplicities. Extend to radicals, primary decomposition, associated primes and normalization. Introduce local standard bases and localizations, quotient rings, weighted and block orders, and user-declared coefficient domains in tested combinations. Cover finite fields, algebraic and transcendental extensions, integer and residue rings, and approximate domains where Singular supports the operation. The CLI would declare the ring and the browser would offer the corresponding valid choices.

Check the distinction between local and global computation on $f=x^3-3x^2+y^2$. Its global Jacobian quotient over $\mathbb Q$ has vector-space dimension 2 because there are two critical points. At the origin the local quotient has dimension 1. Both interfaces should return the answer for the ring the user selected. A second check factors $x^2+1$ over $\mathbb Q$ and over $\mathbb Q(i)$ to exercise field-dependent output. [Singular core algorithms](https://www.singular.uni-kl.de/Manual/4-4/sing_33.htm) · [Commutative algebra libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_1249.htm)

### Phase 4. Support modules, maps and homological algebra

Introduce ring maps, polynomial matrices, module presentations and complexes in both interfaces. Use these objects for kernels, syzygies, graded free resolutions, Betti tables, Hom, Ext and Tor. As a reference example over $R=\mathbb Q[x,y]$, present $R/(x,y)$, verify the relation $(-y,x)$ between the ideal generators, and recover the free ranks $1,2,1$ in degrees $0,1,2$ of its minimal resolution. Browser users could open the differential matrices from the Betti table. [Homological algebra](https://www.singular.uni-kl.de/Manual/4-4/sing_1324.htm)

### Phase 5. Extend geometric resolution and singularity analysis

Accept multi-generator defining ideals, expose individual blowups with user-specified centers, and distinguish the germ at a chosen point from a global affine calculation. Extend branch data to parametrizations and Puiseux expansions. Add local singularity invariants and classification, followed by deformations, spectra and monodromy Jordan data as their algebraic prerequisites become available. Show the same center, chart maps, local hypotheses and invariants in the CLI record and browser.

One reference calculation would resolve the space curve $(z,y^2-x^3)$ and verify that projection to $(x,y)$ recovers the cusp germ and its intrinsic branch data. For a global example with several singular points, verify smoothness above each one. Different valid blowup sequences can produce different chart counts, so use the maps, smoothness, and required crossing conditions to compare results. [Singular resolution library](https://www.singular.uni-kl.de/Manual/4-4/sing_1784.htm) · [Singularity libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_1867.htm)

### Phase 6. Cover the remaining mathematical libraries

Advance these subjects independently as their prerequisites become available. Each would have a CLI operation and a browser view suited to its output.

| Family | Reference calculation |
| --- | --- |
| Equation solving and real algebra | Solve a zero-dimensional ideal, isolate its real roots, and display intervals with their defining polynomials. Distinguish exact and numerical output. [Solving libraries](https://www.singular.uni-kl.de/Manual/4-4/sing_2148.htm) |
| Algebraic geometry and invariant theory | Compute a Riemann–Roch space on a curve, a sheaf cohomology table, and generators of an invariant ring for a specified group action. [Algebraic geometry](https://www.singular.uni-kl.de/Manual/4-4/sing_1601.htm) · [Invariant theory](https://www.singular.uni-kl.de/Manual/4-4/sing_2074.htm) |
| Toric and tropical geometry | Accept an ideal and export its tropical fan with cones and weights. The browser can draw the same fan in low dimensions. [Tropical geometry](https://www.singular.uni-kl.de/Manual/4-4/sing_2484.htm) |
| Noncommutative algebra and D-modules | Introduce the algebra and its defining relations before offering left ideals, noncommutative bases, and differential-operator calculations. A Weyl-algebra example should recover $\partial x-x\partial=1$. Cover PLURAL, SCA, and LETTERPLACE according to their respective input algebras. [Noncommutative subsystems](https://www.singular.uni-kl.de/Manual/4-4/sing_431.htm) |
| Applied and supporting libraries | Work through the catalog entries for coding, control theory, teaching, visualization and miscellaneous applications. For example, compute a code's minimum distance and reproduce a documented surface-rendering example with its external dependency identified. [Library catalog](https://www.singular.uni-kl.de/Manual/4-4/sing_1018.htm) |

### Phase 7. Close gaps against a versioned Singular inventory

One verified input-range advantage is sparse normal form over $\mathbb Q[x,y]$. In lex order, reduce `x+y^N` modulo `(x)`. Both systems return `y^40000` for $N=40000$. For $N=2^{64}$, Atlas returns `y^18446744073709551616`. Singular 4.4.1p5 rejects the same exponent literal in ordinary `poly ^ int` syntax. Reproduce the Atlas result with the command below. This comparison measures accepted input and exact output for one operation. Track execution time and peak memory separately on identical inputs, including examples where Singular currently wins. [Reference tests](tests/test_algebra_groebner_singular.py)

~~~sh
./atlas algebra normal-form --vars x,y --order lex --gen x \
  --poly 'x+y^18446744073709551616' --json
~~~

Starting with Phase 1, record each covered manual command or library procedure, its input objects and supported rings, the equivalent Atlas CLI and UI workflows, and a reproducible reference example. Distinguish internal-only, partial, and publicly verified capabilities. Record external dependencies and experimental procedures explicitly. For outputs with multiple valid presentations, compare mathematical properties such as ideal equality, module isomorphisms in the reference examples, or verified resolution maps.

Use that inventory to identify the remaining procedures and address interpreter features such as control flow, user-defined procedures, loadable libraries, contextual help, and external interfaces. The inventory is complete when every entry in the selected Singular release has a coverage status. Full functional coverage requires every applicable entry to pass a reference calculation within its documented domains. Entries marked partial or unsupported remain open. If running unmodified Singular scripts becomes a target, track parser and interpreter compatibility with a separate script corpus. [Singular manual](https://www.singular.uni-kl.de/index.php/singular.pdf) · [Library catalog](https://www.singular.uni-kl.de/Manual/4-4/sing_1018.htm)
