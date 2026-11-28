# Performance log: resolve() and canonical blowups

This log follows the cost of successive blowups above the origin of the surface
`z^2 - (x^2*(x+2*y)^2 + 4*y^3*(x+y)^2)*(x^2*(x+2*y)^2 + 4*y*(x+y)^2)`.
The measurements use the in-process resolution routine on one machine.
The numbered entries record changes to basis computation, substitutions and
centre selection. The correction below identifies which observations at the
ninth blowup must be discarded. For complete resolutions with comparable
inputs and recorded trial conditions, see the benchmark reports. [Resolution and zeta timings](../../benchmarks/resolution-timings.md) · [Resolution algorithms](../../native/resolution-algorithms.md)

> **Correction (later the same day).** Item 4 had a bug, now fixed.
>
> - **The bug:** when `flint_q::load` brought a value over a common denominator, it did not
>   scale the integer coefficients. For example, `3a² − ½ab + 7b⁵ + ⅓` was loaded as
>   `½a² − ½ab + 7/6·b⁵ + ⅓`. Every product or substitution above the FLINT threshold that mixed
>   integer and fractional coefficients was therefore wrong.
> - **What was wrong as a result:**
>   - The B9 observations made with it: the 3,000-bit unit-ideal basis in item 5, the B9
>     timings, and the NoContact analysis at the end. These are artifacts and are struck below.
>   - The corpus records and the earlier replay hashes did not reveal it, because they rarely
>     reach that path with mixed denominators.
> - **The fix and checks:**
>   - Fixed in `native/chart_expr.hpp`.
>   - Regression test `native/test_chart_expr.cpp`, in `make -f groebner.mk test-verify`:
>     products and substitutions over the threshold against term-by-term arithmetic, on mixed,
>     integer and fractional inputs. It fails 57 of 180 checks with the bug reintroduced.
>   - A scratch differential run of 2,000 checks also agrees.
>   - Corpus records match clean HEAD again with the fix.

## 1. Moving merges in the Gröbner kernel (`native/groebner.cpp`, `native/groebner.hpp`)

- **Hot spot:** in the post-B8 normal-crossings check, `merge` copied the first operand on every
  step of fraction-free pseudo-division, reduction, head computation, multiplication and the
  minors sum. Cost was quadratic in the length of long polynomials.
- **Change:**
  - `merge` now has an overload that takes the first operand by rvalue and moves its
    coefficients (`fmpz_swap`, scaling in place).
  - When one operand is more than 8 times longer than the other, a galloping search
    bulk-moves the untouched runs.
  - The call sites in `reduce_terms`, division, `head`, `multiply`, `Quotient::divide`, the
    minors sum, row elimination and `substitute` now move their temporaries.
- **Why generic:** same arithmetic, fewer copies.
- **Effect:**

  | Measure | Before | After |
  |---|---|---|
  | Slowest crossings request after B8 | 78.5 s | 5.35 s |
  | Whole check after B8 | > 140 s | 8–10 s |

  Output is byte-identical to the unmodified kernel (sha of the replayed request).

## 2. One basis per subset in `op_crossings` (`native/ideal.cpp`)

- **Hot spot:** the failing-subset branch rebuilt bases of the same ideal.
- **Change:**
  - Compute the basis of the subset locus once.
  - Reduce the Jacobian rows by it.
  - Extend it with the minors using the `known` prefix, so the S-pairs among the basis are
    not formed again.
- **Why generic:** identical ideals, fewer Gröbner computations.

## 3. Substitution with one accumulator and cached powers (`native/chart_expr.hpp`)

- **Hot spot:** `chart::substitute` added the image of each term to a running `Expr`,
  renormalizing a polynomial of about 6000 terms at every step, and recomputed the powers of
  each image.
- **Change:**
  - Sum all images in one map and normalize once.
  - Obtain each power of an image from the largest cached power below it.
- **Effect:** B7 centre choice 15 s → 6.6 s.

## 4. FLINT arithmetic for large chart polynomials (`native/chart_expr.hpp`)

- **Hot spot:** `Expr::operator*` and `substitute` used `std::map<vector<mpz>, mpq>`, so every
  partial product canonicalized an `mpq` with a GMP gcd (35% of self time in `__gmpn_hgcd2`).
- **Change:**
  - Above 256 partial products, load the values into `fmpq_mpoly`: one rational content
    times a primitive integer polynomial, built from a common denominator.
  - Use `fmpq_mpoly_mul` / `fmpq_mpoly_compose_fmpq_mpoly`.
  - Store the result back as the same canonical `Expr` (active coordinates, sorted terms).
  - Negative exponents fall back to the term-by-term route.
- **Why generic:** exact arithmetic on the same polynomials. FLINT itself uses GMP; the gain
  comes from avoiding the per-term rational normalization and the map nodes.
- **Effect:** B7 centre choice 6.6 s → 5.6 s. `substitute` disappears from the B9 profile.

## 5. Basis in the frame before carrying a centre back (`native/centre_maximum.cpp`, `engine/centres.py`)

- **Hot spot:** in `maximum`, the centre of each maximal item was carried back to the chart
  coordinates together with the frame rows (strict-transform equations), and only then was a
  basis computed.
  - ~~On the B9 search, this produced one Gröbner basis in 3 variables from 6 generators:
    23,407 terms with 3,000-bit coefficients, while the answer was the unit ideal.~~ That
    input came from the arithmetic bug in item 4 (see the correction above).
  - The change below is exact in any case and stays. Its effect on correct inputs remains to
    be measured.
- **Change:** `back` is a substitution, i.e. a ring homomorphism, so the images of any
  generating set generate the image ideal.
  - Compute the reduced basis in the frame ring first, where the centre is simple.
  - Carry that basis back.
  - Native route: falls back to the raw generators if the frame ring is too wide or the
    basis is declined.
  - The Python stepwise route does the same.
- **Why generic:** same ideal; the final reduced basis in the chart is unique, so the result is
  identical.
- **Effect:**

  | Measure | Before | After |
  |---|---|---|
  | B7 centre choice | 5.6 s | 1.9 s |

  (The B9 figures that stood here were measured with the item 4 bug and are withdrawn.)

## 6. Coordinate orders on implicit charts (`native/field_wide.cpp`)

- **Hot spot:** `implicit_coordinates` (`remove_coordinate_powers`) computes the order of a
  function along each divisor. It divides repeatedly in sparse `std::map` polynomials with
  rational coefficients; the comparator recomputed total degrees with `fmpz` adds.
  - In one B9 request, two localized membership tests of a 5,974-term polynomial took 7.4 s.
- **Changes:**
  - **Early exit.** `divide(p, G).empty()` is replaced by `reduces_to_zero(p, G)`, which stops
    at the first term set aside for the remainder. Such a term is never touched again, so the
    answer is exactly the same for any divisors.
  - **Packed remainders.** The chart basis and the cuts are reduced Gröbner bases for grevlex,
    so a remainder does not depend on how the division is carried out. `PackedRemainders`
    divides them in FLINT `fmpq_mpoly` (degrevlex). Rows beyond the packing range fall back to
    the sparse map.
  - The elimination-order division (`over`) stays on the sparse route.
- **Effect** (single replays, same output hash; machine loaded by other jobs):

  | Request | Before | After |
  |---|---|---|
  | `maxreq.2` | 27–33 s | 19 s |
  | `maxreq.3` | 12.7 s | 9.1 s |

## 7. Jet layers in parallel (`native/field_wide.cpp`)

- **Hot spot:** an `implicit_jet` of order 6 of a 435-term function on a 6-variable chart took
  44.8 s of a 59.5 s B9 request.
  - The time went to 63 reductions of products of about 79,000 terms by the 33-element chart
    basis (`fmpq_mpoly_divrem_ideal`).
- **Change:**
  - The derivatives of one layer are independent, so a layer with at least 2048 input terms is
    computed on `hardware_concurrency` threads.
  - The deduplication and the output then run in the sequential order, and the first failing
    index's exception is rethrown.
  - `PackedDivisors` no longer shares scratch quotients, so reductions are const and
    thread-safe.
- **Why generic:** same derivatives, same order of output, same values.
- **Effect:** that jet request 52 s → 13.5 s under load (same output hash).

## 8. No repeated contact attempts when no shear is possible (`native/centre_maximum.cpp`, `engine/centres.py`)

- **Hot spot:** in the contact search, when the only free coordinate is also the only base,
  `c → c + a·base` leaves every coordinate in place. Each later attempt reran the same search
  until the eighth.
- **Change:** raise NoContact at once if all of these hold:
  - the attempt yielded nothing;
  - `free == bases == {c}`;
  - no banned choice names this attempt.
- **Why exact:** the state is unchanged, every later shear is also empty, and a later attempt
  may only try a subset of the same choices. So every remaining attempt fails the same way.
- **Effect:** about 4 s saved on a request that, on the buggy build (item 4), repeated such
  an attempt 9 times. The rule is exact independently of that bug.

## Verification

The checks after the item 4 fix:

- **Corpus records:** identical to a fresh clean HEAD build on all 24 cases.
- **`make test`:** 1,313 tests, OK (38 skipped).
- **`make -f groebner.mk test-verify`:** passes, including the new `test_chart_expr`.

Earlier checks, run before the fix:

- **Corpus records** (24 cases, `atlas resolve --json`): identical between clean HEAD
  (bd68f89) and the working tree, ignoring `*elapsed_ms` timing fields. The comparison script
  missed the newer `native_elapsed_ms` field at first.
- **Test suite:** `make test` ran 1,313 tests with 1 failure,
  `test_session_panel_terminal …test_frame_follows_the_window_while_waiting_for_input`. It is
  a terminal-timeout test that fails under heavy load and passes alone on both trees.
- **`make check-native-boundary`:** OK.
- **B9 replays** (`maxreq.2`–`maxreq.5`) give the same output hash before and after each step.
- **Timing check, mixed4 CLI:** 7 alternating trials under load, median 698 ms → 539 ms.
  Canonical cases run in-process showed no regression.

## Tried and rejected

- **Skipping Newton repairs for the five-variable cubic cone.** For
  `f4-cone-cubic-sum-mixed`, all eight convenient linear frames produce
  the ordinary blowup of the origin, with five cones and the new ray
  `(1,1,1,1,1)`. At the direction of the singular `u`-axis, write
  `(x,y,z,t)=u*(a,b,c,d)`. The strict equation is
  `a²+b²+c²+d²+a*b*c+c³+d³`, independent of the radial variable `u`.
  A Newton repair can choose the smooth line `a=b=c=d=0` through its
  isolated point on the exceptional fibre. Blowing up this line gives
  the smooth exceptional quadric in projective three-space. In each
  affine chart the restriction is `1+A²+B²+C²`; Singular returns the
  unit ideal for this equation and its three derivatives. Thus this
  repair can remove the defect while meeting the old exceptional
  divisor transversely. The solver retains all linear frames and their
  repair searches. The five-trial fresh compute medians in the broader
  comparison were 2,398 ms against Singular's 191 ms embedded and
  3,066 ms against 177 ms intrinsic.

- **Multimodular normal form** (word-size `nmod_mpoly_divrem_ideal` images, CRT of quotients
  and remainder, exact check `f = Σ qᵢgᵢ + r`). It is sound for a Gröbner basis whose leading
  coefficients are units mod p, but slower on the jet reductions:
  - the dividends have about 79,000 terms and the quotients about 107,000;
  - each reduction needs about 32 primes of 62 bits;
  - CRT and the exact check cost more than FLINT's exact division (more than 120 s against
    52 s).
  - Along the way: `fmpz_mpoly_interp_reduce_p` (FLINT-internal) needs
    `nmod_mpoly_fit_length_reset_bits` on the target first, or it writes out of bounds.
- **Pre-reducing the generators of the B9 centre ideal before homogenizing.** The homogeneous
  basis keeps 97–98 elements mod p, so this gives no gain. Superseded by item 5.

## Stage timeline on the test surface

| Point | Original | Now |
|---|---|---|
| B1–B8 chosen and blown up | ≈ 45 s | 8.6 s (load average about 4) |
| Crossings check after B8 | > 140 s | 9.4 s |
| B9 centre search | never finished | first `maximum` requests take 13.5 s and 6.5 s; the next is stuck in a frame basis (see open items) |

## Open items

- **NoContact on B9 was an artifact, not a counterexample.**
  - **What the buggy build showed:** the B9 search raised "No hypersurface of maximal contact
    was found".
  - **The exact local check** (independent rational power series, no Atlas code): at the
    remaining point P of the failing implicit frame, the element had order 6 with initial
    form `−480/169·v⁶` (`v = divisor_B8`). So every hypersurface of maximal contact was tangent
    to B8, which made the failure look forced.
  - **The giveaway:** the outer state after its shears was inconsistent. The piece pinned down
    P, but P was not in the locus. The shears are linear and fix P, so this can't happen with
    correct arithmetic. It was traced to the loader bug in item 4.
  - **With the fix, NoContact does not occur there.** The B9 search goes on instead.
- **Stuck frame basis in B9 (open).** The next B9 `maximum` request spends its time
  (over 6 min, then cut) in `Frame::over`.
  - That is the basis of `rows + (t·divisor_B8 − 1)` in the "first" elimination order, in 7
    variables. One row is a contact hypersurface with 681 terms, total degree 39 and
    118-digit coefficients.
  - It is needed to divide by the divisor coordinate in `remove_coordinate_powers`.
  - Singular `std` does not finish it within 120 s, in this order or in grevlex.
  - A generic route that avoids the elimination basis: obtain the quotient by the coordinate
    from a cofactor in the `cut` basis (`rows + divisor_B8`), which `implicit_coordinates`
    already has.
- **Modular CRT.** `native/modular.cpp`:
  - `reconstruct_coefficients` lifts one prime at a time, which is O(k²) per coefficient in
    the number of primes k.
  - `monic_image` reduces every input coefficient modulo each prime separately.
  - Remainder or product trees would make both quasi-linear.

## Fresh CLI startup and reuse of a preloaded engine

A Linux prompt, script or HTTP server advertises its preloaded worker image to
standalone `atlas resolve` commands in the same checkout. Each command forks
a clean worker and passes its arguments, standard streams and working
directory. The CLI supervisor enforces the ordinary deadline. A subreaper
beside the worker kills and waits for its descendants on completion,
cancellation or client disconnect. Incompatible environments, changed code
and unavailable engines take the usual fresh Python path.
`ATLAS_PRELOADED_CLI=0` selects that path explicitly.
`ATLAS_PRELOADED_CLI=trace` prints the worker choice on standard error.

The baseline on the AMD Ryzen 7 6800H used Python 3.14.4, one logical CPU
(CPU 2), nine fresh CLI processes per polynomial and no running Atlas session.
The three commands used embedded exhaustive CPU resolution with JSON output.
The median fixed costs, computed per trial as wall time minus the record's
`elapsed_ms`, were 61.15 ms for the ordinary double point, 63.28 ms for the
inclined pinch and 59.88 ms for the coordinate pinch.

The comparison below alternated standalone and preloaded invocations within
each pair on the same CPU. An existing session supplied the worker image in
the preloaded column. Each of the 54 commands started a new CLI process and
produced the complete resolution. The driver verified each record against the
other trials after removing the timing fields. Host load was uncontrolled.

| Germ | Standalone computation / wall / fixed (ms) | Preloaded computation / wall / fixed (ms) |
| --- | ---: | ---: |
| Ordinary double point | 5.52 / 50.37 / 44.60 | 4.57 / 25.31 / 20.71 |
| Inclined pinch | 41.24 / 92.46 / 50.57 | 23.51 / 44.80 / 21.22 |
| Coordinate pinch | 6.38 / 51.45 / 45.08 | 4.31 / 25.28 / 20.91 |

These are medians of the three measurements separately. The 30 ms fixed-cost
target holds with a running engine. All three standalone medians exceed it.
Reproduce the samples and comparison hashes with
`python3 benchmarks/resolve_cli_overhead.py --cpu 2 --trials 9`.

On the baseline, `PYTHONPROFILEIMPORTTIME=1` showed that the resolve CLI
imported `tempfile` and its compression dependencies (5.96 ms cumulative in
one run). On the ordinary double point it also loaded the full smooth-centre
implementation through a shared exception (16.46 ms cumulative). The
supervised CLI now opens its existing progress file directly. Its proposal
verifiers share a lightweight exception module. A standalone one-blowup
command also avoids importing process-launch support solely for exception
matching. A traced final write on the ordinary double point was followed by
about 5 ms of Python shutdown. The supervisor woke on the child's exit
notification and finished cleanup within 0.4 ms under tracing.

For Zoo S15, `LD_DEBUG=files,statistics` and a separate timed trace of
`ctypes.CDLL` on the baseline gave the following dynamic loads on this host.

| Library or probe | Reason | Time in timed trace (ms) |
| --- | --- | ---: |
| `libgroebner.so` | Shared polynomial, chart and centre kernel, including FLINT | 2.35 |
| `libresolution.so` | Ordinary tangent-cone proposal | 0.17 |
| Second `libgroebner.so` handle | Graph verifier helper | 0.02 |
| Current process | Restore C streams in forked helpers | 0.02 |
| `libgomp.so.1`, `libopenblas.so.0` | Four probes for already loaded thread setters | 0.05–0.10 each |
| `libnewton_fan.so` | Newton candidate actually requesting a full fan | 1.63 |
| `libsmall_newton_fan.so` | Later bounded fan request | 0.21 |
| `libface_gb.so` | Persistent face checker for that solve | 0.19 |

The full fan loaded Normaliz, e-antic and their system FLINT dependency in
the Newton worker. A preloaded server now leaves that library for the first
full fan request. The graph verifier shares the existing kernel handle. The
ordinary shared library builds with hidden visibility, explicitly exports its
nine C entry points and uses the ELF linker's `--as-needed` and `-O1`.

## Exact Taylor coefficients on rational points and Galois orbits

The point kernel expands a polynomial through a specified total degree in
the free local coordinates. It returns exact coefficients in
`QQ[theta]/(mu)`. On a smooth implicit frame it inverts the dependent
Jacobian once and cancels the residual of every equation one homogeneous
degree at a time. It verifies that the resulting coordinate series solve
every frame equation through the requested degree. At rational points it
uses the existing truncated smooth-point reducer with prescribed pivots.
A zero jet through degree d proves order at least d+1. A singular Jacobian,
reducible field polynomial, or exhausted resource budget declines the certificate.

The table gives microseconds per computation on the AMD Ryzen 7 6800H, CPU 2,
with five alternating trials of 100 computations each. Atlas and Singular
4.4.1 each warmed their prepared equation and field once. Singular computed
`jet(shift(f),d)` for the unconstrained Taylor polynomial. For implicit
frames it reduced the translated target by a lexicographic standard basis
of the translated frame equations and the degree-(d+1) power of the maximal
ideal, eliminating the dependent increments first. Every trial compared the
entire resulting polynomial in the free increments, including all field
coefficients. These measurements exclude parsing, process startup and output.

| Frame, target and point | Degree | Atlas median (µs) | Singular median (µs) |
| --- | ---: | ---: | ---: |
| y=x², z=xy, target z at (2,4,8) | 4 | 19.17 | 375.28 |
| Affine plane, target x³y³−y+x² at (2,√2) | 3 | 3.33 | 6.45 |
| y²=x, target y at (2,√2) | 4 | 41.53 | 89.85 |
| y³+y=x, target y at (−1,θ), θ³+θ+1=0 | 3 | 114.22 | 133.95 |
| y²=x+z², target yz+x at (2,√2,0) | 5 | 126.66 | 290.03 |

The unconstrained plane initially took 24.60 µs in Atlas against 6.24 µs
in Singular. Expanding the binomial columns directly reduced it to 10.50 µs.
For integral monomial coordinates in a field with θᵉ=c, checked machine
integer products and sums reduced it to the 3.33 µs above. An overflow or
a mixed field coordinate uses exact rational vectors. The tests include a
90-bit coefficient, a wide exponent, opposite conjugates, off-frame points,
and rational linear combinations of powers of a cubic-field generator.

Reproduce with `make -C native -f groebner.mk build/bench_point_jet`
and `python3 benchmarks/point_jets.py --cpu 2`. The separate differential
test `python3 -m unittest tests.test_recorded_point_jets` compares the point
kernel with the archived symbolic jets after specializing the original
derivative fields at all 41 regular integer points in the box [−4,4]³
across the four recorded requests. The
unrestricted symbolic jet on a positive-dimensional chart still uses the
existing global calculation when a point witness cannot settle the question.

## Canonical invariant over the S15 fibre

The reference checkout was `785be93`. Its complete S15 search spent about two
seconds giving the Newton worker priority on one CPU. A separate quick search
on the same polynomial spent 4.44 seconds in three calls to `maximum`, two of
which exhausted their budgets. These measurements concern different search
policies.

With the final repair check, S15's eligible linear frames have isolated
orbit defects. The ordinary Newton worker tried the first repair for over
100 seconds in a separate diagnostic run. In the default one-core race it
keeps its two-second head start. The five fresh computation medians were
2,308 ms for Atlas and 269 ms for Singular. Warm medians were 2,279 and
259 ms. The canonical result has the same five charts and two blowups.

The native trace measured the working set, the maximal-order locus in the
chart, and their intersection. In both distinct slow-search frames the three
dimensions were 0, 1 and 0. In the second request, 13.85 seconds of a
15-second deadline went into `saturate_grevlex`. The first request spent
35.9 ms in `smooth_point_order`. Timing only the enclosing JSON call hides
this distinction.

For a double point on a coordinate exceptional divisor in a smooth
threefold, a nonsingular tangent quadratic on the divisor gives two
independent directions of maximal contact. The previous birth record
determines when the divisor enters the presentation. The shortcut checks the
closure in the original coordinates. It verifies the maximal ideal of the
point using the equation of the divisor and the Jacobian when the divisor
enters early. For the line case it verifies vanishing to order two along the
line and its exact global Jacobian ideal. Other presentations continue through
the existing descent. In particular, a single point of the working fibre can
select an entire line as its centre.

The three fully evaluated S15 requests give the following medians in
milliseconds. Each trial starts a fresh process, runs the three chart requests
in order, and compares the entire native reply with the frozen reference.
These calls use 120-second budgets so that the reference can finish. Ratios
of these times describe the kernel only.

| Request | Frozen Atlas | Atlas with point certificate |
| --- | ---: | ---: |
| First point on the divisor | 50.67 | 0.52 |
| Point with a different divisor history | 16,237.39 | 0.22 |
| Global line centre | 6,704.75 | 1.53 |

To repeat the measurement, use a built reference checkout and one available CPU.

```sh
python3 benchmarks/canonical_maximum.py --cpu 14 --reference /path/to/reference
python3 tools/replay_requests.py benchmarks/canonical-s15-maximum.jsonl.gz
```

The two original quick-search deadline responses are observable changes.
The optimized computation returns a verified answer within a budget that
formerly expired. The longer-budget fixture above compares their mathematical
answers, including generator order. The archived successful ABI replies can
be checked separately.

```sh
python3 tools/replay_requests.py benchmarks/canonical-requests.jsonl.gz
```

Set `ATLAS_RECORD_REQUESTS` to a directory to record the actual ABI and inner
calls. Centre profiles in that directory include per-operation timings and
locus dimensions. Turn recording off for comparisons. `ATLAS_LINEAR_ALGEBRA=0`
selects the reference algorithms. `ATLAS_TRACE_CERTIFICATES=1` logs accepted
and inconclusive predicates by operation.

## Exact certificates in the canonical solver

- Ideal membership, emptiness on an open, and radical inclusion use bounded
  Macaulay systems. Modular pivots select an exact rational solve. FLINT
  checks the reconstructed cofactor identity before the native caller may use
  it. Failed searches return control to the original ideal routine.
- A bounded finite quotient supplies multiplication matrices. Exact
  nilpotence checks on each coordinate identity certify that the separating
  form represents the entire support. Factoring its squarefree polynomial
  gives the conjugate orbits on which to evaluate equations and open units.
  This completeness check also detects a linear form that identifies two
  different points.
- The centre search can compute the order of the strict transform, residual
  orders and contact-direction exclusions by jets on certified finite
  supports. A zero truncation supplies a lower bound. Higher degrees or the
  original search decide the remaining cases. Global derivative ideals
  still define the published centres. The profiled working sets did not
  require sampling a positive-dimensional working set for pruning.
- A nonzero tangential jet on a divisor, or a nonzero remainder on the open
  divisor, proves coordinate order zero. This avoids building the inverse
  variable basis for `Frame::over` for that divisor. Positive orders that
  need a quotient still use the original normal form.
- A triangular graph or an invertible scalar block proves that the centre
  generators are polynomial coordinates. The blowup can omit the graph
  saturation under this certificate and retains the same ambient equations,
  maps and generator order for saved records.
- On a monomial fibre, terms in the square of its radical have zero value and
  first differential. The crossing certificate uses the remaining terms
  and a unit-ideal proof. A possible tangency goes to the existing
  per-stratum calculation, which returns the actual failure loci.

## Two blowups along a graph and a point

For a verified pair of blowups, the first exceptional divisor is a projective
bundle over its geometrically irreducible centre. A second blowup at one
rational point preserves the irreducibility of that divisor. Its intersection
with the new exceptional divisor is the projective tangent hyperplane at the
point. Atlas checks the first centre's polynomial graph equations, the
second centre's point equations and the existence of a point of both
exceptional divisors on an owned part of the origin fibre. Stored records
use the centres reconstructed during replay.

On `s2b-hidden-line-shear-z`, the centre ideals are `(x+z,y)` and
`(x,z,r1_2+1)`. The result has five final charts and one edge joining
the two exceptional surfaces. A separate calculation with the general
gluing routine returns the same labelled edge, including the face order.
Seven alternating fresh and seven warm trials on each CPU gave these
medians in milliseconds, with Singular 4.4.1.

| Logical CPU | Fresh computation Atlas / Singular | Fresh wall Atlas / Singular | Warm computation Atlas / Singular | Warm wall Atlas / Singular |
| --- | ---: | ---: | ---: | ---: |
| 11 | 35.735 / 37.993 | 78.763 / 58.620 | 17.890 / 37.270 | 18.046 / 37.366 |
| 15 | 36.645 / 38.939 | 80.538 / 59.750 | 18.979 / 37.348 | 19.172 / 37.463 |

The fresh wall times include launching the command, serializing and
writing its output, and exiting. The five Atlas charts passed Singular's
independent final-chart checks. The reference and optimized Atlas records
have the same digest after timing fields are removed.

```sh
python3 benchmarks/fallback_resolution.py --cpu 11 --trials 7 --zoo /path/to/resolution-zoo \
    --cases zoo/s2b-hidden-line-shear-z/embedded --output /tmp/hidden-line.json
```

## Finite loci, membership and crossings

These comparisons used one AMD Ryzen 7 6800H logical CPU (9), FLINT
3.6 and Singular with rational coefficients. Each figure is the median
of five alternating trials and excludes process startup.

| Finite support | Atlas (µs) | Singular (µs) |
| --- | ---: | ---: |
| Two nonreduced rational points | 172.75 | 4772.81 |
| One quadratic orbit on the open `x ≠ 1` | 47.70 | 3632.31 |

Both systems received the same equations and units. Atlas computed all
orbit polynomials and coordinates. Singular computed the minimal associated
primes with `minAssGTZ`, after saturation for the open case. Outside the
timing window, Singular eliminated each Atlas parameter and compared the
resulting ideals by mutual containment against every minimal prime.
Each trial used 32 calls on prepared input.
[Finite-locus samples and orbits](../../benchmarks/finite-loci-ryzen-6800h.json)

| Canonical predicate | Atlas (µs) | Singular (µs) |
| --- | ---: | ---: |
| Radical inclusion | 181.33 | 31.71 |
| Element of a principal ideal | 88.07 | 16.64 |
| Element outside an ideal | 61.22 | 16.13 |
| Empty principal open | 56.70 | 26.85 |
| Unit ideal | 93.57 | 17.19 |
| Zero target | 54.45 | 14.83 |
| Transverse strict transform | 78.00 | 26.31 |
| High powers on a monomial fibre | 76.72 | 34.02 |
| Tangency on the exceptional divisor | 143.11 | 21.70 |

Singular was faster on each of these small decision calls. The Atlas
timing includes the shared JSON kernel interface, result decoding and
warm per-process caches. Singular used `std` and `reduce` on prepared
ideals in the interpreter. Each trial used 256 calls and checked both
answers against the expected Boolean result. These timings measure a
decision through the respective interfaces.
[Predicate samples](../../benchmarks/canonical-predicates-ryzen-6800h.json)

```sh
make -C native -f groebner.mk build/bench_finite_locus
python3 benchmarks/finite_loci.py --cpu 9 --trials 5 --repeats 32
taskset -c 9 python3 benchmarks/canonical_matrix_decisions.py --tests 5 --batch 256
```

## Inclined pinch and the preliminary toric proposal

For the three-variable preliminary toric trial, Atlas now checks whether the
homogeneous pieces of degrees two and three factor as A·L² and B·M²·C over
the rationals. It re-expands the factors, checks that L, M, C are independent,
and checks that every coordinate of the direction L×M is nonzero. The
irreducible surface is singular along the line L=M=0. Even after one
origin-preserving translation of one axis, the other two coordinates are
generically nonzero on that line. A toric subdivision is an isomorphism
there. Properness brings the closure of the singular strict transform to
the origin fibre. This rules out every fan used by this early trial.
The later general coordinate searches and chart verifiers still run.

On CPU 2 of the Ryzen 7 6800H, the unchanged record for
`(x+y+z)^2-z*(x-y)^2` used engine `verified-ideal-blowup`. Before this
change, median fresh Atlas compute/wall times were 60.32/136.99 ms across
five trials. In nine alternating trials after the change, fresh Atlas took
30.52/82.82 ms and fresh Singular 4.4.1 took 36.77/59.63 ms. With a warm
process for each solver, their compute/wall medians were 9.05/9.21 ms and
34.95/35.10 ms respectively. Standalone Atlas still had the higher fresh
wall time. The 18 Atlas runs matched the frozen record and the saved-chart
dual complex and zeta replay hashes. Reproduce after building with
`ATLAS_PRELOADED_CLI=0 python3 benchmarks/fallback_resolution.py --cases inclined_pinch --cpu 2 --trials 9 --replay --output /tmp/atlas-inclined.json`.
`ATLAS_LINEAR_ALGEBRA=0` restores the original preliminary toric trial.

## Testing unrepaired toric charts on the cone over the lemniscate

For binary homogeneous H and K of degrees d and d−2 with d≥4, Atlas
can certify that the cone H(x,y)+z²K(x,y) has an irreducible geometric
strict transform which stays singular on a punctured coordinate line.
It checks that H and K share no root, that K has a simple root, and that
H has a repeated nonzero root. The first two conditions give −H/K an
odd pole and prove geometric irreducibility. The last condition proves
that the positive-weight Newton face H fails torus transversality. The
preliminary test requires a persistent boundary ray of the source fan.
Atlas currently repairs degenerate Newton faces only on fans supported
over the origin. The existing chart tests must also exclude every trial
fan that can modify the punctured line. A candidate ramification
translation keeps the full Newton search in place.

For `(x²+y²)²-z²(x²-y²)`, the blowup of the z-axis still has singular
strict equation t²(1+s²)²−z²(1−s²). Each of the four other trial supports
that can modify the punctured z-axis also has an exact chart obstruction.
The intrinsic chart test checks the strict polynomial and all three first
derivatives. It detects singular points at toric fixed points and on
one-dimensional orbits mapping to the origin. The embedded variant also
detects smooth tangencies to the boundary. These checks apply to the
particular fan before any further modifications.

Newton can repair an isolated orbit point using a sequence that includes
blowups along curves. The scheduler therefore checks each eligible linear
frame before changing its priority. A repeated nonmonomial initial factor
on the two-dimensional orbit of an interior fan ray gives a curve of defects
there. Newton classifies these orbits before beginning isolated point
repairs. The lemniscate has repairable isolated defects in a convenient
frame and keeps the original Newton priority.

On 121 unrepaired charts from this cone, two Steiner equations and a
smooth tangency control, Singular independently removed monomial factors
from each pullback and tested the Jacobian ideal together with the three
equations of the origin fibre. All 104 singular and 17 smooth chart
decisions agreed. Across these input groups the five-trial medians ranged
from 35 to 42 µs per chart for Atlas and 61 to 86 µs for Singular on one
Ryzen 7 6800H logical CPU. Reproduce with
`python3 benchmarks/fan_obstructions.py --cpu 9 --trials 5 --batch 16 --output timings.json`.
[Strict chart samples](../../benchmarks/strict-fan-obstructions-ryzen-6800h.json)

For the repair-orbit test, Singular independently extracts the initial
form and computes the dimension of its Jacobian ideal in the torus.
Both systems classified two repeated factors, an isolated node and a
repeated coordinate monomial identically. On the same CPU with five
trials of 64 calls, the median Atlas times ranged from 68 to 124 µs,
and Singular took 56 to 69 µs. Singular was faster on each input.
Run the same benchmark with `--repair-orbits --batch 64`.
[Repair-orbit samples](../../benchmarks/repair-orbit-obstructions-ryzen-6800h.json)

## Intrinsic fallback on a surface singular along a monomial curve

On a nonconvenient weighted homogeneous surface, Atlas tests for a singular
point with coordinates ±1. Substitution and the weighted Euler action then
give a whole singular curve. Three distinct positive weights ensure that the
curve spans the ambient linear space. Geometric irreducibility in this test
follows from a quadratic presentation with constant leading coefficient and
a discriminant that stays nonsquare over the algebraic closure. The curve
meets the torus in every linear frame. After a single coordinate translation
it still has two generically nonzero coordinates. A toric modification is an
isomorphism there. The same argument applies to a polynomial graph with one
auxiliary coordinate over the original torus. A second test obtains possible
weights from equally spaced exponent triples. It substitutes into the full
polynomial and all three logarithmic derivatives, matching each weighted
degree independently. Thus two binomial squares can have different weighted
degrees. This intrinsic test also checks squarefreeness.

An initial Newton defect at an isolated orbit point can lead to blowups of
positive-dimensional centres. Before giving the canonical worker priority,
Atlas checks every eligible linear frame. A curve of defects on an exceptional
orbit rules out that frame's point repair. A frame with inconclusive orbit
data retains its priority. In particular the frozen Newton patch records
for `s2-space-469-3-2` and `s2-space-cone-cusp-3-2` pass through this check.

After a single verified canonical blowup, the restricted exceptional divisor
can be nonreduced even on a smooth surface. Atlas checks the whole reduced
section on each chart whose strict transform meets the fibre. If just one
such chart exists and its reduced section is a polynomial graph on an affine
line meeting the fibre, that section gives one geometric vertex of dimension
one. The other charts must have empty strict fibre, certified by the unit
ideal. Multiple visible charts, holes, marked fields, and inconclusive
bounded radical computations use the existing gluing procedure.

On the two developable charts, this native certificate took 220 µs per
pair (rounded). Singular took 4.20 ms to check the radical by two ideal
inclusions against the parametrized line and to test both origin fibres.
Five alternating trials each used sixteen pairs with one pinned CPU.
[Exact timing samples](../../benchmarks/intrinsic-curve-section-ryzen-6800h.json)
and [reproduction driver](../../benchmarks/intrinsic_curve_section.py).

The curve test matched twelve equations across the 5,091-record catalogue.
Ten have archived canonical results and two have archived Newton repairs.
Fresh before/after complete record hashes agreed on all twelve. The two
trivariate single-blowup entries on the screening list also matched. On
`s2-developable-123` the fresh computation medians after the repair check
were 68.58 ms for Atlas and 99.39 ms for Singular. The warm medians were
41.06 and 97.61 ms. This comparison used CPU 13, five fresh processes and five warm calls per
solver. The timings, hashes, and earlier observations are in the suite report.

For marked curve sections in a canonical chart, Atlas now tries a bounded
unit-ideal calculation on the short equations before asking for their Euler
characteristic. A unit in this subset proves emptiness on all sheets and all
opens of the chart. Other intersections retain their ordinary Euler integral.
Polynomial composition over a marked field uses its live parameter and a
single native modular pullback. For a wide power modulo a cyclotomic
polynomial the affine kernel uses the existing exact period calculation
before division. The regression includes a `2**72+1` power and a 91-bit
coefficient in a field with the same printed parameter name as a coordinate.

## Catalogue checks and the frozen reference

The 5,091-record local-invariant comparison checks Milnor and Tjurina
numbers independently in Singular. All agree. For the two local standard
bases that exceeded the ten-second direct limit, Singular's homogenized
`locstd` computation completed the comparisons. The four-variable octic
has Milnor number 2,526 and Tjurina number 1,751.

Two catalogue regeneration assertions compare against a Newton subdivision
that predates the frozen Atlas checkout. For `m-s-varying-cusp-slope`, the
archive has 21 charts and both the reference and updated builds produce 19.
For `laufer-1`, the archive has 172 and both builds produce 171. The full
regenerated records from the two builds agree, including their zeta
expressions. Rebuilding the fan implementation from the parent of
`5a2a1fe` reproduces the archived rays and cones in both modes for both
records. The external records and the two exact assertions are preserved.
With this catalogue installed, the full `make test` run executed 1,715
tests and reported those two assertion failures and 58 skips.

## Canonical fallback and Singular across the resolution suite

The timing report combines earlier snapshots with repeated comparisons on the corrected scheduling gate. The gate preserves priority for Newton point repairs that can choose a curve centre later. Its decision changed on 19 of the 244 timed cases, and 19 of those have corrected-gate timing series in this report. The archive labels individual rechecks and retains earlier series. On the corrected gate, S15, both cone-lemniscate modes and all four Steiner modes lose to Singular in fresh and warm solves.

2026-11-27, Ryzen 7 6800H, GCC 15 and FLINT 3.6, Singular 4.4.1. The 244 measured germ/mode pairs comprise 24 standard controls and 220 Resolution Zoo entries. Each comparison used one logical CPU, with five fresh trials and five in-process calls after a warm-up. Some independent confirmation runs used seven trials. Atlas reports elapsed milliseconds and Singular reports microsecond-resolution `rtimer` around `resolve(ideal(f))`. Some comparisons ran on separate cores concurrently with other host work. The compressed report includes raw compute and wall samples, run fingerprints, CPU IDs, and superseded measurements. [Raw measurements](../../benchmarks/fallback-resolution-ryzen-6800h.json.gz) · [Benchmark and commands](../../benchmarks/fallback_resolution.py)

Of the 239 pairs with finite medians in both systems, Atlas was faster on 122 fresh and 126 warm. Including process startup and output, Atlas had the faster fresh wall median on 103 and the faster warm wall median on 126. Singular exceeded the 120-second limit on 5 fresh and 5 warm comparisons. The warm Atlas series on those five inputs were repeated after stopping the timed-out Singular process. 166 comparisons produced canonical Atlas records. Across the 163 with finite paired medians, the median Singular/Atlas compute ratios were 0.77× fresh and 0.83× warm.

| Germ and mode | Atlas charts / blowups | Atlas fresh (ms) | Singular fresh (ms) | Atlas warm (ms) | Singular warm (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| zoo_S15_open | 5 / 2 | 2307.53 | 268.89 | 2279.48 | 258.74 |
| shared_boundary | 84 / 24 | 897.23 | 6012.56 | 681.66 | 5136.58 |
| D4 | 15 / 4 | 498.87 | >120,000 | 144.60 | >120,000 |
| inclined_pinch | 2 / 1 | 40.76 | 56.13 | 12.04 | 53.55 |
| s2b-newton-17 embedded | 9 / 4 | 111.31 | 124.88 | 67.73 | 102.01 |
| s2b-newton-45 embedded | 5 / 3 | 149.90 | 166.69 | 97.13 | 168.16 |
| s2-chain-13214 embedded | 5 / 3 | 138.48 | 184.23 | 94.07 | 188.95 |
| s2-chain-23212 embedded | 11 / 5 | 158.92 | 413.95 | 80.60 | 296.24 |
| s2-cone-lemniscate embedded | 6 / 2 | 2264.08 | 148.66 | 2271.08 | 147.15 |
| s2-developable-123 intrinsic | 2 / 1 | 68.58 | 99.39 | 41.06 | 97.46 |
| s2b-hidden-line-shear-z embedded | 5 / 2 | 36.65 | 38.94 | 18.98 | 37.35 |
| f4-cone-cubic-sum-mixed embedded | 8 / 2 | 2398.25 | 191.09 | 2365.32 | 170.97 |
| f4-cone-cubic-sum-mixed intrinsic | 8 / 2 | 3066.37 | 177.00 | 3240.12 | 178.87 |
| f4-loop-1111-2221-3 intrinsic | 8 / 2 | 1854.82 | 229.91 | 1671.17 | 180.00 |
| f4-sigma3 intrinsic | 24 / 2 | 3146.70 | 715.52 | 3103.44 | 736.73 |
| f5-chain-11213-12112-4 embedded | 22 / 4 | 2770.54 | 302.80 | 2747.31 | 294.25 |
| f5-chain-11213-12112-4 intrinsic | 22 / 4 | 3146.73 | 329.69 | 3068.92 | 303.56 |
| s2-chain-31334 intrinsic | 31 / 9 | 5247.21 | 899.00 | 4571.89 | 806.57 |
| s2-cone-lemniscate intrinsic | 6 / 2 | 2643.97 | 156.73 | 2463.36 | 153.46 |
| s2-space-cone-cusp-2-2 intrinsic | 16 / 6 | 5055.40 | 608.01 | 4541.16 | 566.61 |
| s2-steiner-222-x-2 embedded | 10 / 3 | 2274.01 | 385.68 | 2237.99 | 383.36 |
| s2-steiner-222-x-2 intrinsic | 10 / 3 | 2542.36 | 375.81 | 2456.39 | 340.62 |
| s2-steiner-223-sum-2 embedded | 13 / 4 | 3065.78 | 381.56 | 2952.05 | 398.47 |
| s2-steiner-223-sum-2 intrinsic | 13 / 4 | 3540.32 | 379.70 | 3842.28 | 438.78 |
| s2b-hidden-line-shear-z intrinsic | 5 / 2 | 289.35 | 47.99 | 273.82 | 46.62 |
| s2b-line-line-4 intrinsic | 10 / 4 | 2903.79 | 544.29 | 2935.07 | 536.19 |
| s2b-loop-41-32-3 intrinsic | 6 / 3 | 2229.84 | 287.73 | 2203.71 | 288.07 |
| s2b-newton-16 intrinsic | 10 / 4 | 3416.84 | 458.94 | 3425.19 | 455.47 |
| t3-chain-3211213 embedded | 33 / 6 | 4250.85 | 268.50 | 5204.75 | 250.40 |
| t3-chain-3211213 intrinsic | 3 / 1 | 2184.44 | 251.58 | 2145.05 | 231.99 |
| t3-suspension-image-quadruple-10-2 embedded | 7 / 2 | 655.89 | 124.85 | 694.80 | 130.95 |
| t3-suspension-image-quadruple-10-2 intrinsic | 7 / 2 | 599.12 | 116.17 | 605.39 | 115.86 |
| t3b-newton-33 intrinsic | 2 / 1 | 458.80 | 69.36 | 458.55 | 67.62 |
| t3b-suspension-ns-cover-2-2-cross-node-2 intrinsic | 19 / 4 | 3038.77 | 519.78 | 3051.86 | 489.42 |
| s2b-loop-41-32-3 embedded | 6 / 3 | 2213.47 | 277.34 | 2159.35 | 238.34 |
| s2b-chain-23-13-5 embedded | 18 / 8 | 2342.50 | 1439.01 | 2504.81 | 1395.76 |
| s2-chain-31334 embedded | 31 / 9 | 3368.43 | 913.39 | 3200.84 | 844.01 |
| s2-chain-42213 embedded | 17 / 7 | 2192.40 | 875.23 | 2150.62 | 850.37 |
| s2b-chain-13-21-5 embedded | 12 / 6 | 2760.05 | 914.36 | 2656.50 | 931.45 |
| s2b-loop-32-12-5 embedded | 40 / 14 | 3981.32 | 1907.80 | 3957.66 | 1852.84 |
| s2b-chain-14-23-5 embedded | 12 / 6 | 2271.49 | 1248.58 | 2173.37 | 901.73 |
| s2-cone-d6-x3-1-b embedded | 9 / 5 | 2281.53 | 1073.10 | 2233.78 | 1023.05 |
| s2-cone-tangent-conics embedded | 10 / 5 | 2355.37 | 2017.46 | 2251.24 | 1997.38 |
| s2b-loop-23-21-5 embedded | 20 / 9 | 2515.15 | 1490.34 | 2397.75 | 1533.52 |
| f4-sigma3 embedded | 24 / 2 | 2818.60 | 737.62 | 2748.59 | 724.70 |
| f5-loop-12132-12112-3 embedded | 23 / 3 | 3132.07 | 1397.66 | 3126.74 | 1462.65 |
| f5-loop-12132-12112-3 intrinsic | 23 / 3 | 2912.41 | 983.97 | 2848.58 | 988.75 |
| f5-sigma3 embedded | 35 / 2 | 4139.72 | 1178.57 | 4099.87 | 1203.41 |
| f5-sigma3 intrinsic | 35 / 2 | 5148.60 | 1196.53 | 5294.83 | 1225.74 |
| s2-chain-21326 embedded | 16 / 6 | 2203.11 | 651.89 | 2194.31 | 796.08 |
| s2-chain-21326 intrinsic | 16 / 6 | 2918.94 | 1053.02 | 2631.72 | 775.46 |
| s2-chain-42213 intrinsic | 14 / 5 | 2458.09 | 864.04 | 2450.08 | 994.58 |
| s2-cone-d6-x3-1-b intrinsic | 4 / 2 | 2235.78 | 1068.49 | 2225.22 | 1068.76 |
| s2-cone-d6-xyxy-1-b intrinsic | 6 / 3 | 3011.87 | 649.07 | 3156.28 | 710.39 |
| s2-cone-quadrifolium embedded | 12 / 6 | 3243.33 | 1829.28 | 3175.64 | 1777.14 |
| s2-cone-quadrifolium intrinsic | 9 / 4 | 3101.19 | 1758.76 | 3059.48 | 1811.77 |
| s2-cone-tangent-conics intrinsic | 10 / 5 | 2434.95 | 1777.30 | 2417.87 | 1761.71 |
| s2-developable-123 embedded | 5 / 3 | 217.03 | 71.42 | 192.23 | 68.33 |
| s2-modulus-e13-1 intrinsic | 4 / 3 | 331.93 | 246.31 | 294.63 | 241.38 |
| s2-modulus-e13-2 intrinsic | 4 / 3 | 300.34 | 202.02 | 194.72 | 149.67 |
| s2-modulus-e19-1 intrinsic | 5 / 4 | 234.05 | 219.11 | 214.80 | 212.24 |
| s2-modulus-e19-2 intrinsic | 5 / 4 | 232.06 | 213.56 | 210.19 | 217.15 |
| s2-modulus-w13-1 intrinsic | 3 / 2 | 302.62 | 222.93 | 270.07 | 218.40 |
| s2-modulus-w13-2 intrinsic | 3 / 2 | 323.03 | 228.58 | 293.43 | 228.15 |
| s2-modulus-w17-1 intrinsic | 3 / 2 | 218.04 | 155.82 | 192.44 | 153.88 |
| s2-modulus-w17-2 intrinsic | 3 / 2 | 221.62 | 205.26 | 193.86 | 176.68 |
| s2-modulus-z18-1 intrinsic | 4 / 3 | 223.72 | 164.47 | 209.38 | 163.68 |
| s2-modulus-z18-2 intrinsic | 4 / 3 | 237.68 | 169.82 | 224.31 | 163.01 |
| s2-ruled-4-2-8-2 intrinsic | 2 / 1 | 140.34 | 52.50 | 25.07 | 49.57 |
| s2-space-cone-cusp-2-2 embedded | 16 / 6 | 2437.34 | 458.82 | 2425.83 | 428.80 |
| s2-trans-d4-d4-s intrinsic | 9 / 4 | 571.30 | 422.98 | 597.30 | 453.47 |
| s2-trans-d4-node-s intrinsic | 7 / 3 | 345.88 | 266.19 | 321.81 | 267.58 |
| s2-trans-d4-node intrinsic | 11 / 4 | 3165.62 | 1894.55 | 3116.99 | 1751.02 |
| s2-trans-d6-e6-s intrinsic | 22 / 8 | 2032.24 | 1765.91 | 2029.56 | 1759.42 |
| s2b-chain-23-13-5 intrinsic | 18 / 8 | 3269.11 | 982.64 | 3142.35 | 972.50 |
| s2b-hidden-bp-shear-xy intrinsic | 11 / 4 | 446.82 | 396.92 | 418.43 | 414.73 |
| s2b-invertible-c2f-527 embedded | 37 / 12 | 5149.85 | 930.58 | 5119.94 | 942.77 |
| s2b-invertible-c2f-527 intrinsic | 37 / 12 | 5152.17 | 904.22 | 5131.22 | 913.42 |
| s2b-invertible-l3-255 embedded | 28 / 9 | 5287.12 | 874.13 | 5188.41 | 916.19 |
| s2b-invertible-l3-255 intrinsic | 18 / 7 | 5304.15 | 935.13 | 5216.62 | 928.66 |
| s2b-line-line-3 embedded | 10 / 4 | 2265.35 | 1035.07 | 2371.36 | 1000.68 |
| s2b-line-line-3 intrinsic | 10 / 4 | 2437.72 | 1112.09 | 2476.89 | 1226.34 |
| s2b-line-line-4 embedded | 10 / 4 | 2224.66 | 602.09 | 2383.90 | 556.00 |
| s2b-loop-13-32-3 embedded | 12 / 4 | 2810.19 | 864.53 | 2463.84 | 729.27 |
| s2b-loop-13-32-3 intrinsic | 5 / 2 | 2317.99 | 741.83 | 2289.69 | 726.95 |
| s2b-loop-23-21-5 intrinsic | 20 / 9 | 3049.43 | 1508.53 | 2767.79 | 1483.68 |
| s2b-loop-32-12-5 intrinsic | 19 / 8 | 4211.84 | 1727.42 | 4134.04 | 1793.15 |
| s2b-newton-17 intrinsic | 9 / 4 | 352.38 | 122.55 | 326.62 | 119.23 |
| s2b-newton-45 intrinsic | 4 / 2 | 321.62 | 173.66 | 189.92 | 106.18 |
| s2b-newton-57 intrinsic | 9 / 4 | 473.32 | 345.58 | 592.86 | 501.40 |
| s2b-product-h1-tacnode2 intrinsic | 14 / 5 | 2643.00 | 695.25 | 2558.87 | 679.97 |
| t3-chain-1212222 embedded | 13 / None | 215.39 | 101.67 | 197.06 | 100.13 |
| t3-chain-1212222 intrinsic | 13 / None | 256.06 | 111.36 | 235.00 | 110.22 |
| t3-chain-1212312 embedded | 17 / None | 392.83 | 184.24 | 374.85 | 179.62 |
| t3-chain-1212312 intrinsic | 17 / None | 417.14 | 176.19 | 340.47 | 166.11 |
| t3-chain-1221312 embedded | 37 / None | 204.91 | 132.53 | 191.14 | 128.83 |
| t3-chain-1221312 intrinsic | 37 / None | 269.16 | 148.17 | 243.32 | 142.69 |
| t3-chain-1222312 embedded | 20 / None | 240.29 | 150.50 | 338.26 | 237.69 |
| t3-chain-1222312 intrinsic | 20 / None | 376.31 | 239.84 | 248.31 | 152.71 |
| t3-chain-2221112 embedded | 6 / None | 168.52 | 124.56 | 130.79 | 119.51 |
| t3-chain-2221112 intrinsic | 6 / None | 128.98 | 79.78 | 139.31 | 122.10 |
| t3-chain-3121222 embedded | 49 / None | 267.65 | 231.30 | 290.01 | 330.60 |
| t3-chain-3121222 intrinsic | 49 / None | 345.80 | 306.72 | 318.62 | 302.80 |
| t3-join-nonreduced-x2yxy-a1 embedded | 29 / None | 86.86 | 69.29 | 63.50 | 66.64 |
| t3-join-nonreduced-x2yxy-a1 intrinsic | 29 / None | 160.63 | 120.99 | 128.21 | 118.23 |
| t3-join-nonreduced-x3y-d4 embedded | 26 / None | 205.59 | 37.81 | 212.64 | 36.74 |
| t3-join-nonreduced-x3y-d4 intrinsic | 26 / None | 212.16 | 36.93 | 196.02 | 34.51 |
| t3-modulus-q11-1 embedded | 13 / None | 261.13 | 203.47 | 235.00 | 204.02 |
| t3-modulus-q11-1 intrinsic | 13 / None | 451.72 | 376.64 | 439.43 | 377.85 |
| t3-modulus-q11-2 embedded | 13 / None | 306.19 | 234.96 | 285.70 | 226.72 |
| t3-modulus-q11-2 intrinsic | 13 / None | 293.09 | 233.42 | 270.84 | 229.25 |
| t3-modulus-z12-2 intrinsic | 15 / None | 325.53 | 305.96 | 250.16 | 294.60 |
| t3-newton-221 embedded | 4 / None | 311.93 | 190.23 | 302.35 | 188.36 |
| t3-newton-221 intrinsic | 4 / None | 320.04 | 207.22 | 307.35 | 188.69 |
| t3-suspension-chain-12125-2 embedded | 17 / None | 262.85 | 112.32 | 244.84 | 112.77 |
| t3-suspension-chain-12125-2 intrinsic | 17 / None | 262.83 | 113.92 | 247.04 | 113.89 |
| t3-suspension-ruled-4-2-3-1-2 embedded | 7 / 2 | 2221.46 | 377.04 | 2212.90 | 372.64 |
| t3-suspension-ruled-4-2-3-1-2 intrinsic | 7 / 2 | 2248.99 | 380.24 | 2238.68 | 397.52 |
| t3-suspension-ruled-4-3-5-2-2 embedded | 5 / None | 252.83 | 130.24 | 225.97 | 128.09 |
| t3-suspension-ruled-4-3-5-2-2 intrinsic | 5 / None | 209.91 | 117.18 | 191.50 | 115.30 |
| t3-union-1 intrinsic | 8 / 2 | 1776.02 | 1132.69 | 1405.51 | 1090.25 |
| t3b-chain-111-123-3 intrinsic | 3 / 1 | 2573.37 | 743.72 | 2528.37 | 650.66 |
| t3b-suspension-ns-cover-2-2-cross-node-2 embedded | 19 / 4 | 2417.04 | 498.83 | 2582.83 | 785.96 |
| t3b-suspension-ns-sum-z2-three-double-lines-2 intrinsic | 22 / 4 | 659.31 | 519.26 | 650.76 | 512.97 |
| t3b-suspension-s2-ruled-4-5-3-2-2 embedded | 7 / 2 | 2227.12 | 592.90 | 2206.01 | 580.30 |
| t3b-suspension-s2-ruled-4-5-3-2-2 intrinsic | 7 / 2 | 2248.34 | 572.91 | 2245.11 | 606.41 |
| s2-trans-d4-node embedded | 11 / 3 | 2390.44 | 2706.86 | 2354.57 | 1994.03 |

Singular separately checked the saved chart equations and Jacobian conditions on all 951 charts in the 66 trivariate embedded canonical comparisons.

Against the frozen `785be93` checkout, five alternating fresh CLI trials gave matching public-record hashes on all 24 controls. 0 control medians exceeded a 5% regression. The original axis-certificate sweep returned 80/80 matching old and updated embedded Zoo records. Its single-trial median baseline/updated compute ratio was 2.44× before the scheduling correction. A fresh frozen/corrected comparison covers the 80 historical inputs together with the 24 mode/input pairs whose gate changed, a union of 89. The archived 89 pairs include 89 matching public records. On the common 80-case set the corrected single-trial median ratio was 2.21×. All 9 additional intrinsic control pairs also matched. Across additional stable Zoo confirmations, 187/187 independent frozen-build CLI checks reproduced the timed snapshot’s full public-record hash.

The broader screen covered all 748 Zoo modes. It grouped the 406 distinct equations by input text and variable order, reused Singular results for 88 of them and tested the other 318 in fresh processes with a 1.75 s wall deadline. There were 124 completed Singular observations, 273 censored equations and nine without usable Singular output. A diagnostic rerun on the nine preserved the original observation and recorded the library error or new deadline. Each previously unmeasured Atlas mode with a completed Singular counterpart was timed once in a separate pinned process with a 12 s wall limit. All 189 such Atlas runs completed. Singular was faster on 108 of the 189 single observations. The 33 original Zoo comparisons were supplemented with 24 targeted high-ratio or distinct-family confirmations, one of which produced different records across trials. The remaining 165 modes with a completed Singular screen were scheduled for the same repeated protocol. All 165 have checkpoints. Across the Zoo, 220 modes have repeated comparison results or 120 s censoring, and 2 have unstable public records. The other 509 modes had a censored short Singular screen and 17 an unusable Singular result. The report includes the result or censoring status for every equation and mode. [Screening program](../../benchmarks/screen_fallback_resolution.py)

The embedded and intrinsic modes of s2b-join-e19-z6 each changed producer across fresh calls. On the intrinsic mode, the complete normalized hashes identify the same 38-chart repaired Newton record and 55-chart canonical record on both builds. The frozen solve returned the Newton record on one CPU and the canonical record on six logical CPUs. The updated solve returned both on one CPU. Two intrinsic attempts and one embedded attempt encountered the record-stability guard. Both modes are excluded from median comparisons.

Both switches reproduced all 6,480 archived native replies byte for byte. Baseline and candidate also reproduced the complete records, saved dual complexes, and saved zeta outputs on the separate historical 24-case corpus. On the timing controls, stripped saved-record replay matched on 23 of 24. The S15 dual complex recomputed, while its stripped zeta computation exceeded 120 seconds on both builds. [Historical cases](../../benchmarks/canonical-record-corpus.csv) · [Exact request archive](../../benchmarks/canonical-requests.jsonl.gz)

The median across per-case fresh wall-minus-compute medians was 60.8 ms. The 30 ms fresh-overhead target remains open.

## Local centre certificates on Newton repair charts

The reference for these measurements is Atlas `10beb05` on a Ryzen 7 6800H. On the captured S15 quartic, the expensive operation was the global vanishing test for a coefficient after the second maximal contact. Its Gröbner basis had two input equations, with 8 and 32 terms. In three reference runs the enclosing test consumed 15.0–16.0 seconds and exhausted the native deadline. The support and off-piece saturations each had zero calls. The profiles now count those two saturations and the global vanishing test separately. Their timers include their child operations.

There are genuine saturation bottlenecks on other charts. On the marked cone over the lemniscate, two support saturations took 1.94 seconds in a reference run, and 90 off-piece saturations took 13 ms. On Steiner, six support saturations took 1.86 seconds, and 138 off-piece saturations took 67 ms. Both had finite working sets. Across the 76 archived maximum requests and the three earlier S15 requests, the native profile measured 56 zero-dimensional working sets and 23 one-dimensional working sets. In 12 of those 23 the maximal-order locus met the working set in dimension zero. These are counts of recorded calls, with repeated inputs included.

A frame with an empty equation list is the entire affine chart and already has a direct grouping rule. On an implicit curve, Atlas can certify that the germ through a working orbit is a coordinate line. The normal Jacobian must have full rank at the point and every frame equation must vanish as a polynomial on the whole line. Atlas then restricts the coefficient to the line over the exact residue field. This distinguishes a zero germ from a coefficient whose first several derivatives vanish. The lemniscate example uses `QQ(i)`. Steiner uses `QQ`.

The original working ideal can be kept when all of its geometric points lie in the current locus and open chart, and all give the same decision for that coefficient. Such pieces go into the corresponding group with their original generators and ordering. This preserves what later point searches, contact restrictions and emptiness tests see. An inconclusive orbit, rank test, size bound or mixed decision within a single piece goes through the original ideal computation. `ATLAS_LINEAR_ALGEBRA=0` selects the reference path. [Two captured requests and their full reference replies](../../benchmarks/canonical-split-maxima.jsonl.gz)

For a smooth affine threefold with one coordinate boundary, Atlas can sometimes certify the entire point centre from the polynomial on that boundary. If the restriction is homogeneous of degree `d` and its order-`d−1` derivatives span both transverse linear coordinates, the order-`d` locus on the boundary is the reduced origin globally. The divisor history must put the boundary into the first or second level of the invariant. The remaining coefficient orders then come from the two transverse parameters and the mark-one boundary. A working piece supported at the origin may contain powers of the coordinates.

In the S15 repair the certified invariant is `(4,0,1,1,1,0,∞)`. An independent check used the original polynomial and the original descent. On the contact chart the two defining equations are a polynomial matrix times the two transverse coordinates. The determinant at the working point is −7776, so the coordinate equations generate exactly the same ideal on that open set. The reference descent on this presentation returned all the same invariant entries, the empty tie-breaking key and the ordered point ideal. The unsimplified reference also exhausted a 500-second native budget. [Captured S15 request](../../benchmarks/canonical-s15-repair.json) · [Three paired trials](../../benchmarks/canonical-s15-repair-ryzen-6800h.json)

A pure power on the boundary needs a different calculation. Write the polynomial as `Σ f_j(s,e) t^j`, with boundary `e=0` and restriction `c t^d`. On maximal contact, the order-`d−1` derivative gives `t=f_(d−1) W` with `W` regular near the entire boundary intersection. Atlas bounds the orders of every term in each coefficient of the presentation. One coefficient must attain both the exceptional and the residual order. Its leading boundary polynomial `P(s)` supplies a global check. If `a` is its multiplicity at zero, the gcd of `P` and its derivatives through order `a−1` must generate `(s)`. An equally multiple conjugate root makes this test decline. Once the history makes the divisor old at the second level, the equations generate the reduced origin on the boundary. This also handles some positive-dimensional working sets contained in the boundary. The residual orders are 2 in the loop repair and 4/3 in the first chain repair. [Loop request](../../benchmarks/canonical-loop-repair.json) · [Chain request](../../benchmarks/canonical-chain-repair.json)

The marked lemniscate supplies a limit on local reconstruction. In a later chart the centre also has components outside the working fibre. The line through the working point only accounts for one of them. Local tests may decide which coefficients to keep while the existing global descent still computes that centre. The accepted shortcuts use direct identities or derivative ideals of bounded degree. They need no choice of an interpolation degree, no local standard basis and no cache shared between solver lanes.

| Captured native maximum call | Reference median | Updated median |
| --- | ---: | ---: |
| S15 quartic | 15 s deadline, 3/3 | 0.40 ms |
| Loop | 15 s deadline, 3/3 | 0.43 ms |
| Chain, first repair | 15 s deadline, 3/3 | 9.38 ms |
| Chain, second repair | 15 s deadline, 3/3 | 0.45 ms |
| Cone over the lemniscate | 1,474.70 ms | 154.47 ms |
| Steiner surface | 1,532.51 ms | 289.13 ms |

The measurements alternate builds in fresh processes pinned to CPU 14, with the default linear-algebra setting enabled on both. Each split used five trials. The first chain repair took 0.36–9.43 ms. A native deadline is a censored result. The shorter calls can still leave a slower operation later in the same Newton repair. [Loop samples](../../benchmarks/canonical-loop-repair-ryzen-6800h.json) · [First chain samples](../../benchmarks/canonical-chain-repair-ryzen-6800h.json) · [Second chain samples](../../benchmarks/canonical-chain-gcd-repair-ryzen-6800h.json) · [Split samples](../../benchmarks/canonical-split-maxima-ryzen-6800h.json)

In an exhaustive search, the choice between two certified resolutions can depend on which worker finishes in time. Enabling the coordinate-line split everywhere changed the Steiner embedded record from 10 canonical charts to 41 Newton charts. The competing Newton worker therefore uses the original saturation for that split. The canonical workers and direct centre requests use the new split. The boundary certificates are available to all workers. The scheduler retains the original grace period.

All 244 public resolution records matched their archived hashes across the guarded timing trials. Singular independently verified the equations and smoothness or normal crossings on 951 final charts in the embedded canonical trivariate cases. It reached the 120-second limit on five inputs. Among the other 239, Atlas had the shorter compute median in 114 fresh and 122 warm comparisons. Singular had the shorter compute median in 125 fresh and 117 warm comparisons. Fresh wall time including CLI startup favored Atlas on 105 inputs. The archive identifies each slower case and records the raw times and censoring. [Full comparison](../../benchmarks/local-centre-resolution-ryzen-6800h.json.gz)

| Embedded germ | Atlas fresh compute (ms) | Singular fresh compute (ms) |
| --- | ---: | ---: |
| S15 | 2,493 | 199 |
| Cone over the lemniscate | 2,939 | 184 |
| Steiner surface | 2,279 | 399 |
| Loop 41-32-3 | 2,274 | 310 |
| Chain 31334 | 3,335 | 988 |

Each named mode used five fresh processes and five measured calls to a warm interpreter after one warm-up. The other modes used three trials per series. Atlas and Singular each ran on one logical CPU. Multiple physical cores ran independent series concurrently. The two-second Newton head start and subsequent work still dominate the embedded examples above.

We also alternated the frozen and updated Atlas CLIs on each CPU to compare the cost of producing the same record. Across all 244 inputs, the median ratio of updated to frozen compute time was 0.994. Eighteen inputs improved by more than 5%, 15 slowed by more than 5%, and 211 were within 5%. The wall-time ratio was 0.993. Separate runs of the two full builds had suggested a 3.4% increase in fresh compute time. The alternating measurements better control for changes in host load. For the nine targeted modes the median compute ratio was 0.998. The intrinsic chain 31334 improved from 4,883 to 3,422 ms and retained its 31-chart canonical resolution. S15 embedded took 2,384 and 2,397 ms.

The intrinsic loop 32-12-5 took 56% of its previous compute time. The embedded loop 13-32-3 had a reproducible slowdown on dedicated CPU 4, from 2,877 to 3,511 ms in a further five alternating trials. Profiling found that its competing Newton worker reached a third expensive maximum earlier. The canonical worker that returned the record used essentially the same CPU time in isolation, 230 and 234 ms. In three trials on a separate, shared CPU the wall-time difference reversed. The additional five-trial check of the intrinsic mixed cubic cone also reversed its initial slowdown, from 3,214 to 3,009 ms. The timings of competing workers depend on core and load.

The offline catalogue check covered all 748 previously screened Zoo modes. The 220 in the timing suite retained their archived public records. Of the other 528, 523 produced identical complete records on the first completed exhaustive pair, including runs extended beyond the initial 20-second screen. Three more produced the same complete record on cross-build repeats after their first pair differed: the embedded join E19–Z6 and both modes of the hidden E7 with two shears. All 52 modes with additional previously audited hashes reproduced those hashes on both builds.

The embedded W13 cusp and Z11 tacnode exceeded 420 seconds on both builds. With 20-second and 120-second quick-search budgets they returned matching inconclusive records. The historical catalogue lists resolved models with 138 and 78 charts, respectively. Those two resolved models remain unverified in this comparison. Separately, both switch settings reproduced all 6,485 archived native replies byte for byte. The 24 historical reference and updated resolutions and all 24 saved dual-complex and zeta replays matched. The broad catalogue comparison is an offline benchmark. The new split regression test reads just two captured native calls.

In a separate comparison that alternated the frozen and updated Atlas CLIs, the intrinsic chain 31334 improved from 4,883 to 3,422 ms. All five updated trials finished faster than all five reference trials and reproduced the same 31-chart resolution. The other eight named modes ranged from a 4.6% decrease to a 1.1% increase. The median change across the nine named modes was a 0.2% decrease.

To reproduce the native measurements, set `before` to a built checkout of `10beb05` and choose a CPU available on the host. The public comparison uses the equations and modes saved in the earlier timing archive. For an individual result, add `--cases NAME` and use the CPU and trial count in that case's row. The separate 24-case check reopens the reference build's charts to recompute dual complexes and zeta functions.

```sh
for repair in s15 loop chain chain-gcd; do
    python3 benchmarks/canonical_maximum.py --cpu 14 --trials 3 --reference "$before" --reference-switch 1 --repair "$repair"
done
python3 benchmarks/canonical_maximum.py --cpu 14 --reference "$before" --reference-switch 1 --corpus benchmarks/canonical-split-maxima.jsonl.gz
python3 benchmarks/fallback_resolution.py --cpu 4 --trials 3 --measured-suite benchmarks/fallback-resolution-ryzen-6800h.json.gz --output comparison.json
python3 benchmarks/fallback_resolution.py --cpu 4 --record-corpus "$before" --replay --output records.json
```
