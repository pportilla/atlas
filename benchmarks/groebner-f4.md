# The F4 trace of the modular Gröbner route

For the graded reverse lexicographic order, the modular route now traces one prime with F4 matrices, reduces random combinations of the rows of each degree, and checks each degree against the Hilbert bound. The other primes replay the recorded matrices. The lift and the certificate over Q are unchanged. [Algorithm and proofs](../native/KERNEL.md#75-the-f4-trace) · [Driver](groebner_f4.py)

**Input.** Katsura-$n$ from `groebner_hard.py`: $n+1$ variables, $n$ quadrics and one linear form, $2^n$ solutions. msolve and several papers call this system Katsura-$(n+1)$.

**References.** msolve 0.10.1, giac 2.0.0 and Groebner.jl 0.10.11 (Julia 1.13.1). Singular is not used. Groebner.jl is installed in a project of its own, [`groebner-systems/julia`](groebner-systems/julia), and driven by [`groebner_systems.py`](groebner_systems.py).

**Machine.** 13th Gen Intel Core i7-1355U (2 performance and 8 efficiency cores, 12 threads), AVX2, 31 GB. Other jobs were running on it during every measurement: the load average was between 3 and 8 before each run. Times are seconds; "best" is the smallest of the runs and the median is next to it.

```bash
python3 benchmarks/groebner_f4.py --size 11 --parts prime --repeats 3 --limit 1500 --out benchmarks/groebner-f4-katsura-11-prime-i7-1355u.json
python3 benchmarks/groebner_f4.py --size 11 --parts rational --limit 1500 --out benchmarks/groebner-f4-katsura-11-rational-i7-1355u.json
```

## Katsura-11 modulo one prime

The reduced basis modulo 2147483629 has 1050 elements and 1,481,345 terms. Atlas reduced 1050 combinations in 12 matrices, none of them dependent, and every degree met its Hilbert bound, so the basis is proved complete without a final check.

| System | 1 thread, best (median) | 12 threads, best (median) | Runs |
|---|---:|---:|---:|
| Atlas F4 trace | 2.84 (3.08) | 2.19 (2.21) | 3 |
| msolve F4, probabilistic linear algebra (`-l 42`) | 5.14 (6.05) | 4.27 (4.48) | 3 |
| msolve F4, exact linear algebra (`-l 2`) | 22.21 (23.00) | 11.96 (12.00) | 3 |
| giac `gbasis`, default settings | 324.5 | 124.0 | 1 |

A replay of the recorded matrices for another prime takes 1.67 s on one thread (median 1.79 s). Four primes replayed at once on the 12 threads take 3.83 s together.

**Agreement.** msolve's four bases equal the Atlas basis term by term. giac's basis has the same 1050 elements; its elements were not compared at this size, because evaluating them takes giac longer than computing them. On Katsura-8 giac's basis agrees element by element (below). [All runs](groebner-f4-katsura-11-prime-i7-1355u.json)

## Katsura-11 over Q

The certified basis has 1050 elements and 1,481,345 terms, from 63 primes with none rejected. All runs use 12 threads; each external run was allowed 1500 s.

| System | Seconds |
|---|---:|
| Atlas certified, F4 trace | 1967.3 |
| — trace of one prime | 2.0 |
| — replays of 62 primes | 42.0 |
| — lift | 5.3 |
| — certificate over Q | 1911.8 |
| Atlas candidate before the certificate (trace, replays, lift) | 49.3 |
| giac `gbasis`, probabilistic | 135.8 |
| giac `gbasis`, certified (`proba_epsilon:=0`) | over 1500 |
| msolve solve (parametrization and real roots, no basis over Q) | over 1500 |
| Atlas with the Buchberger trace (before) | gives up after 26.9 |

The certificate takes 97% of the certified time. **Agreement.** The certified basis reduced modulo 2147483629 equals msolve's basis there term by term. giac's probabilistic basis has the same 1050 elements; its elements were not compared at this size. [All runs](groebner-f4-katsura-11-rational-i7-1355u.json)

## Katsura-8, every comparison

The same driver on a size where every system finishes and every basis is compared in full. The basis has 143 elements and 26,135 terms; Atlas used 31 primes.

| Modulo 2147483629 | 1 thread, best | 12 threads, best |
|---|---:|---:|
| Atlas F4 trace | 0.018 | 0.017 |
| msolve F4, probabilistic | 0.044 | 0.045 |
| msolve F4, exact | 0.066 | 0.089 |
| giac `gbasis`, default settings | 0.747 | 0.448 |

| Over Q, 12 threads | Total | Trace | Replay | Lift | Certificate |
|---|---:|---:|---:|---:|---:|
| Atlas certified, F4 trace | 0.504 | 0.023 | 0.076 | 0.034 | 0.344 |
| Atlas certified, Buchberger trace (before) | 0.695 | 0.106 | 0.030 | 0.041 | 0.480 |
| giac `gbasis`, probabilistic | 0.385 | | | | |
| giac `gbasis`, certified (`proba_epsilon:=0`) | 13.044 | | | | |
| msolve solve (parametrization and real roots, no basis over Q) | 1.202 | | | | |

All bases agree: msolve's modular bases term by term, the certified basis modulo the trace prime with msolve's basis there, giac's three bases through the values of every element at two points, and the two Atlas traces through the complete text. [All runs](groebner-f4-katsura-8-i7-1355u.json)

## Katsura-9 against Groebner.jl and msolve

The current kernel, on a quieter machine (load average 5 at the start, 2 at the end). Best of 5 runs, the median in parentheses. Groebner.jl solves a small system first, so compilation is not timed.

| Modulo 2147483629 | 1 thread | 12 threads |
|---|---:|---:|
| Atlas F4 trace | 0.070 (0.072) | 0.069 (0.070) |
| Groebner.jl, randomized linear algebra | 0.071 (0.083) | 0.045 (0.054) |
| msolve, probabilistic | 0.155 (0.165) | 0.142 (0.146) |
| Groebner.jl, deterministic | 0.353 (0.360) | 0.144 (0.166) |
| msolve, exact | 0.363 (0.395) | 0.234 (0.245) |

One further prime, on one thread: the Atlas replay takes 0.021 s, Groebner.jl's apply 0.029 s after a learn of 0.357 s.

| Over Q, 12 threads | Seconds |
|---|---:|
| Atlas candidate (trace, 30 replays, lift) | 0.564 |
| Groebner.jl, `certify=false` | 0.737 |
| Atlas certified | 9.801 |
| Groebner.jl, `certify=true` | over 240 |

All bases agree term by term, over Q modulo $2^{61}-1$. Groebner.jl's `certify=true` fails with its default linear algebra in this version, so it was run with `linalg=:deterministic` and `homogenize=:yes`. [All runs](groebner-systems-katsura-9-i7-1355u.json)

## Atlas alone on Katsura 8 to 10

The current kernel by itself, on a quiet machine (load average 1 at the start). Best of 5 runs for the trace and the replay, one run over Q. The other programs were not run again: their columns repeat the measurements above and in the next section.

| Modulo 2147483629, 1 thread | Katsura-8 | Katsura-9 | Katsura-10 |
|---|---:|---:|---:|
| Atlas F4 trace | 0.016 | 0.073 | 0.366 |
| Atlas F4 trace, 12 threads | 0.016 | 0.072 | 0.339 |
| Atlas replay of one prime | 0.003 | 0.021 | 0.157 |
| Groebner.jl, randomized (recorded) | not measured | 0.071 | 0.625 under load |
| msolve, probabilistic (recorded) | 0.044 | 0.155 | 1.481 under load |
| msolve, exact (recorded) | 0.066 | 0.363 | 3.639 under load |

| Over Q, 12 threads | Katsura-8 | Katsura-9 | Katsura-10 |
|---|---:|---:|---:|
| Atlas candidate (trace, replays, lift) | 0.121 | 0.513 | 4.107 |
| Atlas certified | 0.486 | 8.678 | 104.8 |
| Primes | 31 | 31 | 47 |
| Groebner.jl, `certify=false` (recorded) | not measured | 0.737 | 5.497 under load |

The Katsura-10 references were recorded on 2026-11-16 with other jobs running. In that same run the Atlas trace took 0.610 s, against 0.366 s here, so the load alone accounts for a factor of about 1.7. Like for like, Katsura-10 modulo the prime was 0.610 s for Atlas, 0.625 s for Groebner.jl and 1.481 s for msolve. The Katsura-9 references were recorded in one quiet run with Atlas at 0.070 s. [All runs](groebner-systems-katsura-atlas-i7-1355u.json)

## Generators that are not homogeneous

Atlas homogenizes the generators for its certificate. When the homogenized generators are no complete intersection, their ideal needs a larger basis than the given one: 2587 elements and 5.0 million terms against 227 and 195 thousand for the Reimer system in 7 variables. msolve and Groebner.jl compute the smaller basis directly. The kernel now chooses: such generators can be traced as they are, with a certificate that needs no homogenization. [Algorithm, choice and proof](../native/KERNEL.md#76-generators-that-are-not-homogeneous-the-direct-trace)

**Recorded with the homogenized trace only** (2026-11-16, 3 runs, machine under load; the run stopped after these systems). Seconds modulo the prime on one thread, best of the runs.

| System | Homogenized basis | Reduced basis | Atlas | Groebner.jl, randomized | msolve, probabilistic |
|---|---:|---:|---:|---:|---:|
| Katsura-10 | 537 | 537 | 0.610 | 0.625 | 1.481 |
| Katsura-11 | 1050 | 1050 | 4.550 | 4.378 | 6.406 |
| noon-7 | 495 | 495 | 0.308 | 0.194 | 0.350 |
| noon-8 | 1338 | 1338 | 2.468 | 1.075 | 3.217 |
| henrion-6 | 90 | 90 | 0.039 | 0.017 | 0.075 |
| cyclic-7 | 443 | 209 | 0.116 | 0.054 | 0.175 |
| cyclic-8 | 1182 | 372 | 1.305 | 1.006 | 1.542 |
| eco-11 | 758 | 389 | 0.367 | 0.494 | 0.479 |

With 12 threads Groebner.jl's randomized mode was the fastest on every one of these systems (Katsura-11: 2.62 s against 3.53 s for Atlas). [All runs](groebner-systems-homogenized-i7-1355u.json)

**With the direct trace** (2026-11-17, 3 runs, one thread, quiet machine; msolve only). These runs were made while the direct trace was developed, with every such system sent to it. The final rule first makes both traces modulo one prime to choose, which adds to the trace at most a homogenized trace limited to twice the recorded structure of the direct one. They were not repeated with that rule.

| System | Atlas trace, homogenized | Atlas trace, direct | msolve, probabilistic | msolve, exact | Atlas certified over Q, homogenized | Atlas certified over Q, direct |
|---|---:|---:|---:|---:|---:|---:|
| reimer-6 | 0.27 | 0.049 | 0.036 | 0.053 | not measured | 0.38 |
| reimer-7 | 19.3 | 0.861 | 0.818 | 1.105 | not measured | 16.0 |
| cyclic-7 | 0.044 | 0.055 | 0.105 | 0.130 | 1.32 | 0.88 |
| cyclic-8 | 0.94 | 0.817 | 1.095 | 1.957 | not measured | 37.2 |
| eco-11 | 0.29 | 0.208 | 0.311 | 0.657 | 31.3 | 14.9 |

The direct trace times include the trace given up at its missed bound and the selection of rows for the certificate. The homogenized trace times are single runs of the trace alone. With the direct trace the basis modulo the prime equals msolve's term by term; with the homogenized one it only reduces to it.

**Where the direct trace is not taken.** A chart of a resolution in the test suite gives five generators of degree up to 17 in four variables that generate the unit ideal. The homogenized route proves it in 0.07 s from a basis of 186 elements and 7678 terms. The direct trace has 366 elements and 41,422 terms before it reaches 1, with coefficients so large that their lift was not stable after two minutes. The choice compares these sizes modulo one prime and keeps the homogenized route there.

## Remaining slower cases

- With 12 threads Groebner.jl's randomized F4 is faster than the Atlas trace modulo a prime (Katsura-9: 0.045 s against 0.069 s; Katsura-11: 2.62 s against 3.53 s in the earlier run). Symbolic preprocessing in the Atlas trace runs on one thread, and a degree has few blocks of lanes.
- On noon-8, henrion-6 and cyclic-7 Groebner.jl was about twice as fast on one thread in the earlier run. Those systems meet the Hilbert bound or were measured before the direct trace.
- A replay costs about as much as the elimination of the trace. The Buchberger replay program repeats only the useful reductions and is cheaper per prime: on Katsura-8 the replays take 0.076 s against 0.030 s. The faster trace outweighs it there, and the Buchberger trace cannot finish Katsura-10 at all.
- For lexicographic and elimination orders the Buchberger trace is kept. On Katsura-5 in the lexicographic order the F4 replays of 94 primes took 0.117 s against 0.011 s for the 126 primes of the replay program, for a trace of 0.030 s against 0.045 s.
- giac's probabilistic basis over Q is faster than the certified Atlas basis on Katsura-8 (0.385 s against 0.504 s). The Atlas candidate before certification takes 0.133 s.
- The certificate over Q dominates every certified time: 97% for Katsura-11, 93% for Katsura-9.
- The choice between the two traces rests on two thresholds fitted to the few systems above. A complete intersection with solutions at infinity is not compared at all and keeps the homogenized trace, although its homogenized basis is larger (cyclic-5: 38 elements against 20).
- A direct lift that is not stable after 255 primes is given up for the homogenized trace. The Reimer system in 4 variables with coefficients near $2^{100}$ does that: its basis needs 2719 primes, and both traces and the exact engine take 21 to 28 s.
- A solve that asks for a device keeps the Buchberger trace, since the devices run its replay program.
