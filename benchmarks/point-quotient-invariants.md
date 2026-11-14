# Local quotient bases and hypersurface invariants at rational points

Every timed request starts from rational coordinates and the original input expressions. The basis operations must produce the actual basis in the input coordinates. The Jacobian algebra must also return every original partial derivative, including zero derivatives. The scalar operations return the local Jacobian or Tjurina colength. Off-hypersurface inputs exercise the public error path. Positive-dimensional quotients have infinite vector-space dimension, and a unit ideal has dimension zero.

## Measurements on the Ryzen 7 6800H

Atlas returned the expected result on all 44 requests across 19 fixtures. Singular 4.4.1 completed 39 on the original equations. Comparing Atlas with the fastest successful Singular median on each request, Atlas had the lower median on 39 of 39. The median of those 39 time ratios was 1.33×. The earlier Atlas executable completed 30 of 44 requests under the same limits. [Complete inputs, exact outputs, checks, individual timings and binary hashes](point-quotient-invariants-ryzen-6800h.json)

Selected fresh-process medians are in milliseconds. A dash marks an exponent that the installed Singular rejects in the original input.

| Input | Operation | Exact result | Atlas | Fastest Singular |
| --- | --- | ---: | ---: | ---: |
| Cusp at (0, 0) | Milnor number | 2 | 6.23 | 7.42 |
| Node at (1, 0) | Milnor number | 1 | 6.76 | 9.45 |
| Smooth point | Milnor number | 0 | 8.28 | 9.30 |
| Off the hypersurface | Tjurina number | Rejected | 8.33 | 10.15 |
| Fractional shift, unequal numbers | Milnor number | 16 | 7.47 | 10.38 |
| Fractional shift, unequal numbers | Tjurina number | 15 | 8.74 | 10.93 |
| Three-variable length 7, lex | Quotient basis | 7 | 6.97 | 8.22 |
| Three-variable length 7, grevlex | Quotient basis | 7 | 7.21 | 9.61 |
| Cusp, power 1,000,001 | Jacobian algebra | 2 | 6.25 | 70.29 |
| Cusp, power 2⁶⁴+7 | Jacobian algebra | 2 | 9.71 | — |
| Unequal numbers, power 1,001 | Milnor number | 16 | 6.45 | 22.71 |
| Unequal numbers, power 1,001 | Tjurina number | 15 | 6.93 | 18.92 |
| Colength 2⁶⁵ | Milnor number | 2⁶⁵ | 7.57 | — |
| Three coupled axes, power 1,001 | Quotient basis | ∞ | 7.49 | 8.96 |
| Million-degree coordinate slice | Quotient basis | 5 | 6.30 | 20.01 |
| Oblique double divisor | Jacobian algebra | ∞ | 7.65 | 41.56 |

Singular’s sparse Taylor calculation supplied all three direct million-degree cusp results. The coordinate-line test certified the infinite quotient on the coupled axes, and factoring the source hypersurface certified the oblique double divisor. In those cases the direct dense-translation procedure reached its memory or time limit. The three operations on the cusp at exponent 2⁶⁴+7 and the two scalar operations of colength 2⁶⁵ required independent certificates. They are excluded from the 39 time ratios.

The earlier build encountered a polynomial-pullback overflow on three requests for the 2⁶⁴+7 cusp. Eight requests hit the address-space limit, and three requests reported that the worker exited before returning a result.

A separate 21-trial run confirmed the three-variable grevlex quotient basis. Its median was 6.26 ms with Atlas and 8.40 ms with the fastest Singular method. [All confirmation samples](point-quotient-invariants-confirmation-ryzen-6800h.json)

## Methods

Atlas requests CPU explicitly. Singular runs with one Singular thread and one FLINT thread. Five same-input Singular methods compute the same mathematical outputs. Each first checks incidence and recognizes smooth points and unit ideals. Jacobian generators come from `jacob` of the source polynomial. Returned quotient bases are mapped into the source variables before printing. The reference ring raises Singular's exponent budget for large inputs.

1. Translate the original ideal to the point, compute `std`, and call `vdim` or `kbase` as appropriate.
2. Compute a global standard basis of the source generators before the same local calculation.
3. Factor each generator and discard factors that are units at the point before computing the local basis.
4. Test whether an entire coordinate line through the point lies in the zero set of the ideal. For a hypersurface in at least two variables, also factor the source polynomial and test whether a repeated factor passes through the point. Either condition certifies an infinite-dimensional quotient. Otherwise continue to the translated standard basis.
5. Use exact derivatives at the source point to construct fourth-order Taylor polynomials. Compute the quotient lengths modulo the fourth and fifth powers of the maximal ideal. If they agree, Nakayama's lemma gives the full local length and the fifth-power quotient supplies a basis. Otherwise continue to the full translated standard basis. This gives a direct finite result for the million-degree power cusp without a dense translation.

Each program starts fresh, builds its inputs, solves, and prints its result. The reported time includes all of those costs. A separate verification process checks every distinct polynomial output after timing. Each timed process has three seconds and 768 MiB of virtual address space, including the Atlas worker subprocess. A timeout yields a lower bound on wall time. An allocation failure or an unsupported exponent has no inferred time. Solver order rotates across three repetitions, and further repetitions of a failed method are skipped. The driver checks Singular’s affinity after its interpreter has started. The installed Singular selects logical CPU 0 on startup, so this experiment also pins Atlas and its subprocesses to CPU 0. Other host jobs can still share that core.

## Exact output checks

For finite quotients of length `d>0` over `QQ`, the `d`th power of the maximal ideal vanishes. The verifier takes Taylor coefficients up to degree `d−1` of the **source** ideal and actual output basis. In a separate global jet quotient, Singular checks the length, the dimensions of the coefficient matrix, and its nonzero determinant. This proves both spanning and linear independence of the source-coordinate basis. For the three-variable order-sensitive fixture it also checks which of two tied quadratic monomials survives. Source derivatives are independently differentiated coefficient by coefficient and compared as exact polynomials, with a zero retained for each unused variable.

The hypersurface `y²−x³(x−1)²` has a cusp at `(0,0)`, a node at `(1,0)`, and a smooth point at `(4,24)`. The deformation `u⁵+v⁵+u³v³` has Milnor number 16 and Tjurina number 15. It appears both in affine coordinates shifted by fractions and in the étale coordinate `u=x^1001−1`. In general, the power charts `u=x^N−p^N` are valid here because `N*p^(N−1)` is nonzero. The Jacobian ideals agree under an étale coordinate change by the chain rule. Singular computes the local lengths in those low-degree coordinates, while basis rank still uses the actual source quotient. For the oblique square the rational polynomial `x^1001*y^1001+x+y−3` has a nonzero differential at `(1,1)`. Its square is a double smooth divisor, and the local Jacobian quotient is positive-dimensional.

The scalar-only test `f=x^(2^65+1)+(y−1)^2` checks a colength beyond 64 bits. At `(0,1)` its Jacobian ideal is `(x^(2^65),y−1)`, and adjoining `f` leaves that ideal unchanged. The quotient has rank `2^65` by successive monic quadratic extensions in `x, x², x⁴, …`. The verifier checks the source derivatives exactly and uses Singular to compute the rank-four two-stage model and the 65-step product in arbitrary-precision integer arithmetic. This avoids requesting a list of 36,893,488,147,419,103,232 basis polynomials.

## Reproduction

Build Atlas and use the logical CPU selected by the installed Singular. The recorded run used CPU 0.

```sh
make -C native -f groebner.mk build/local_algebra
python3 benchmarks/point_quotient_invariants.py --cpu 0 --seconds 3 --memory-mib 768 --repeats 3 --output /tmp/point-quotients.json
```

Add `--baseline-root /path/to/previous/build` to include a separate earlier executable and its library. The JSON retains complete input equations and points, per-method outputs, check descriptions, individual samples and SHA-256 build hashes.
