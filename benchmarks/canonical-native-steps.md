# Canonical blowup steps in one native call

The canonical (origin) lane blows up centres chosen by the invariant of Bierstone and Milman and, on a smooth surface, by the surface procedure. Each stage, its presence and crossing checks, and each centre choice on a chart are now one native call that composes the native operations the Python modules called one at a time. The comparison is between Atlas builds: `main` at 6009247 before, and the same tree with the native steps after. [Blowup stage](../native/canonical_blowup.cpp) · [Centre choice](../native/centre_maximum.cpp) · [Proof references](../native/KERNEL.md#9-operations-for-the-blowups)

## Full resolutions

Median wall time in milliseconds of a fresh `./atlas resolve --mode embedded --compute cpu --json` process over seven runs after one warm-up round, on logical CPU 2 (a performance core) of an Intel Core i7-1355U. The two builds ran in alternating order. Every run of both builds produced the same JSON record except fields ending in `_ms`. The zeta columns add `--monodromy`. [Samples and records](canonical-native-steps-i7-1355u.json.gz) · [Driver](canonical_native_steps.py)

| Input | Engine | Before | After | Before with zetas | After with zetas |
| --- | --- | ---: | ---: | ---: | ---: |
| `(x+y)^2+z^3+x^3` | canonical-blowups | 649.3 | 414.0 | 650.6 | 415.6 |
| `(x*y-z^2)^2+x^5+y^5` | canonical-blowups | 771.3 | 460.0 | 773.8 | 462.2 |
| `x^2+(y+z)^3+y^4` | canonical-blowups | 759.2 | 469.8 | 761.7 | 474.9 |
| `(x+2*y)^2+(y+z)^3+z^4` | canonical-blowups | 857.6 | 521.8 | 890.4 | 561.3 |
| `x^2*y+y^4+z^3` | verified-monomial-modification | 194.7 | 164.8 | 218.8 | 153.3 |

With one CPU the racing lanes share it, so the time saved in every lane that runs the canonical loop (the origin lane, its tangent trial and the Newton repairs) adds up. Without pinning, five alternating runs per build gave 25–44% shorter medians on the same ten commands.

## The origin lane

The lane alone, in a fresh interpreter with its modules imported, split by phase. `cold` is the first call, as in a new resolution worker. `warm` follows one call on `x^2+y^3+z^5`, so the lazy imports and caches of the step-by-step route are in place.

| Input | Start | Before total | After total | Centre choice | Blowups | Presence and crossings |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `(x+y)^2+z^3+x^3` | cold | 39.6 | 36.1 | 9.7 → 10.8 | 10.5 → 6.5 | 2.9 → 4.9 |
| `(x+y)^2+z^3+x^3` | warm | 19.8 | 17.1 | 3.0 → 3.4 | 10.2 → 6.3 | 2.7 → 4.8 |
| `(x*y-z^2)^2+x^5+y^5` | cold | 87.6 | 51.9 | 48.7 → 21.4 | 17.3 → 9.1 | 4.4 → 6.8 |
| `(x*y-z^2)^2+x^5+y^5` | warm | 71.3 | 34.6 | 42.0 → 14.7 | 19.9 → 9.3 | 4.5 → 6.9 |
| `x^2+(y+z)^3+y^4` | cold | 75.3 | 55.3 | 43.4 → 21.0 | 6.2 → 9.3 | 4.9 → 7.9 |
| `x^2+(y+z)^3+y^4` | warm | 47.7 | 29.0 | 31.6 → 13.0 | 5.8 → 5.8 | 4.7 → 7.1 |
| `(x+2*y)^2+(y+z)^3+z^4` | cold | 106.7 | 77.1 | 69.1 → 36.3 | 10.2 → 10.3 | 7.0 → 12.3 |
| `(x+2*y)^2+(y+z)^3+z^4` | warm | 84.5 | 52.7 | 57.7 → 26.3 | 11.5 → 9.9 | 7.3 → 11.8 |
| `x^2*y+y^4+z^3` | cold | 73.8 | 54.2 | 40.0 → 21.4 | 7.1 → 7.6 | 6.0 → 9.3 |
| `x^2*y+y^4+z^3` | warm | 52.1 | 35.7 | 30.5 → 14.5 | 6.8 → 7.7 | 6.3 → 9.5 |

The last column now also contains the presence test of each new chart, which the earlier build ran before it; their sum is still 1–5 ms larger. Stages made of linear blowups, which were already a single native operation, and the centre choice of the first input, which the affine descent answers alone, gained one more request with its transport and are slower by up to 3 ms. Every total is shorter.

## Exact comparisons

- The full records of the ten commands above agree between the builds, field for field, except timing fields.
- `tests/test_native_canonical_stage.py` runs whole resolutions through the native steps and through the step-by-step route and compares the charts, centres and strata. It also compares both routes on every centre choice along four resolutions, including the errors they raise. During development the same comparison ran on every centre choice of 33 three-variable singularities and of the centre, surface and blowup tests, without a difference.

## Reproduce

```sh
python3 benchmarks/canonical_native_steps.py --before /path/to/baseline --after /path/to/candidate \
    --cpu 2 --trials 7 --output /tmp/canonical-native-steps.json.gz
```
