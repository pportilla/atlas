# Performance log: resolve() and canonical blowups

This log follows the cost of successive blowups above the origin of the surface
`z^2 - (x^2*(x+2*y)^2 + 4*y^3*(x+y)^2)*(x^2*(x+2*y)^2 + 4*y*(x+y)^2)`.
The measurements use the in-process resolution routine on one machine.
The numbered entries record changes to basis computation, substitutions and
centre selection. The correction below identifies which observations at the
ninth blowup must be discarded. For complete resolutions with comparable
inputs and recorded trial conditions, see the benchmark reports. [Resolution and zeta timings](../benchmarks/resolution-timings.md) · [Resolution algorithms](../native/resolution-algorithms.md)

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
