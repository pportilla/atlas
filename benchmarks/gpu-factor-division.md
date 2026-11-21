# Triton factorization, GCD, cancellation and polynomial division

Factoring the degree-90 polynomial $\prod_{i=1}^{90}(x-i)$ took 3.330 ms on the Radeon 680M, 6.677 ms on the Atlas CPU and 4.704 ms in Singular. The Atlas CPU was faster than Triton on the smaller controls. Timings are medians of seven warm trials in milliseconds. [Exact input expressions, software versions and individual timings](gpu-factor-division-680m.json)

| Workload | Atlas CPU | Atlas Triton | Singular |
|---|---:|---:|---:|
| Rational factorization, degree 12 | 0.409 | 0.619 | 0.085 |
| Absolute factorization, degree 12 | 1.625 | 1.871 | 1.596 |
| Square-free factorization | 0.305 | 0.573 | 0.039 |
| Polynomial GCD | 0.358 | 0.743 | 0.022 |
| Extended polynomial GCD | 0.401 | 0.636 | 0.035 |
| Rational function cancellation | 0.349 | 0.600 | 0.040 |
| Quotient and remainder by a binomial | 1.675 | 2.015 | 0.119 |
| Normal form by a binomial | 0.984 | 1.622 | 0.090 |
| Rational factorization, 90 integral roots | 6.677 | 3.330 | 4.704 |

Run `python benchmarks/triton_factor_division.py --trials 7` with a working Triton device at index zero and Singular on the path. Atlas times the public algebra dispatcher, including reading the input, device transfers, exact native verification and producing its result record. It alternates CPU and GPU order after a warmup. Each timed GPU sample requires a recorded launch on the physical device and exact equality with the CPU result. Singular uses one thread, the same monomial order and formulas, and its internal timer. It materializes native polynomial results, with comparison against Atlas outside the timer. For absolute factorization the Singular call also serializes one representative per conjugate orbit. The check compares orbit norms, distinct factors, multiplicities and reconstruction in the actual coefficient field. [Benchmark driver](triton_factor_division.py)

The root sieve evaluates coordinate sections on all 8191 residues with one lane per residue. Exact division tests every candidate over the rationals. Roots lost by specialization, or whose coefficients do not reconstruct, stay in the polynomial passed to the full solver. For absolute factorization the number-field work runs on the CPU. The sieve currently considers coordinate sections of degree 4 through 512, with at least eight input terms and at most sixteen coordinates.

For division by one binomial, Triton computes how many times its leading monomial rewrites each term. The native kernel checks the end of every rewrite, combines equal monomials over $\mathbb Q$ and uses an exact division to obtain the quotient when requested. Packing the leading monomial, tail and term matrix into one transfer and avoiding overlapping quotient expansions reduced the 16th-power GPU division control from 2.029 ms to 1.972 ms and normal form from 1.760 ms to 1.575 ms in the optimization run. This stage covers eight through 65,536 terms with up to 32 coordinates and input exponents up to 1,073,741,823. Other inputs retain full native division. The bounds concern GPU buffers and do not restrict the exact public operations.
