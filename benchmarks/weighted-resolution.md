# Weighted resolution against Atlas's other engines

`benchmarks/weighted_resolution.py` times `resolve(f, method="weighted")`
against `resolve(f, method="auto")` on the same inputs. Atlas is its own
benchmark here: the comparison is with the engines the weighted resolution can
replace. Each request runs in Atlas's isolated resolution worker, in embedded
mode with the exhaustive search. The table gives the solver's elapsed time
(`elapsed_ms` of the record), median over 3 trials; every listed
result is a certified resolution (the weighted one certifies normal crossings
over the origin, `auto` returns its own verified atlas). `auto` stopped at
60 seconds per trial.

Conditions: 13th Gen Intel Core i7-1355U, 12 threads, Python 3.14.7, 2026-11-16. Raw trials:
[`weighted-resolution-i7-1355u.json`](weighted-resolution-i7-1355u.json).

| Case | Polynomial | weighted (ms) | auto (ms) | auto's engine |
| --- | --- | ---: | ---: | --- |
| E8 | `x^2+y^3+z^5` | 17.3 | 15.9 | native-toric |
| A20 | `x^2+y^2+z^21` | 12.1 | 34.0 | native-a-series |
| T444 | `x^4+y^4+z^4+x*y*z` | 96.2 | 45.6 | newton-polyhedron |
| canon_x2y_y4_z3 | `x^2*y+y^4+z^3` | 19.9 | 43.4 | verified-monomial-modification |
| whitney | `x^2-y^2*z` | 18.1 | 9.4 | binomial-coordinate-blowups |
| double_conic | `(x*y-z^2)^2+x^5+y^5` | 27.4 | 224.0 | canonical-blowups |
| double_conic_norat | `(x^2+y^2+z^2)^2+x^5` | 50.2 | 41072.9 | canonical-blowups |
| double_elliptic | `(x^3+y^3+z^3)^2+x^7` | 116.7 | 50464.6 | canonical-blowups |
| triple_conic | `(x^2+y^2-z^2)^3+x^7` | 202.1 | no result in 60 s | — |
| binomial4 | `x*y^2*z^3-t^6` | 93.3 | 9.9 | binomial-coordinate-blowups |
| double_quadric3 | `(x*y-z*w)^2+x^5+y^5+z^5+w^5` | 6612.5 | 385.6 | canonical-blowups |
| B9 | `z^2 - (x^2*(x+2*y)^2 + 4*y^3*(x+y)^2)*(x^2*(x+2*y)^2 + 4*y*(x+y)^2)` | 46.5 | no result in 60 s | — |

The weighted time includes about 15 ms for forking the native helper, which
Atlas stops at its deadline; the native work for E8 takes under a millisecond.

Slower cases: `auto` is faster where a specialised engine applies (the Newton
polyhedron for T444, coordinate blowups for whitney and binomial4), and for
`double_quadric3`, where the weighted engine spends its time on the points of
charts with ambient equations that are defined over a field of degree 4.
