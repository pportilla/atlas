# Ideal arithmetic at the prompt and in resolution charts

## Declaring ideals and computing with them

Median microseconds on a Ryzen 7 6800H with CPython 3.14.4 and Singular 4.4.1. The four input polynomials already exist in one copy of `QQ[x,y]` with graded reverse lexicographic order. Each iteration constructs the required ideals and immediately computes a sum, product or square. A square needs only the left ideal. The Atlas columns call the public ideal constructor and operator directly. Singular reassigns the ideals and the result in its interpreter. Trials use a single CPU core, and each reported time is the median of five batches.

| Inputs | Operation | Previous Atlas, 16 calls | Atlas, 16 | Singular, 16 | Atlas, 256 | Singular, 256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Monomials | Sum | 16.42 | 3.12 | 4.00 | 2.71 | 2.88 |
| Monomials | Product | 17.39 | 3.55 | 4.06 | 2.99 | 2.85 |
| Monomials | Square | 12.36 | 5.40 | 8.81 | 4.65 | 7.41 |
| Rational forms | Sum | 27.26 | 3.20 | 4.06 | 2.89 | 3.41 |
| Rational forms | Product | 29.66 | 4.81 | 5.94 | 3.98 | 5.00 |
| Rational forms | Square | 49.76 | 37.43 | 66.06 | 35.03 | 65.57 |

The monomial inputs are `I = (x², y)` and `J = (x, y²)`. The rational inputs are `I = ((x+2y)⁸, (18446744073709551629/7)x⁶y)` and `J = ((2x−y)⁷, xy⁵/13)`. Atlas returns a reduced Gröbner basis for the square, so Singular runs `std(I^2)` with `option(redSB)`. Outside the timer, each trial verifies mutual ideal containment in Singular. [Driver](ideal_operators.py) · [Previous 16-call measurements](https://github.com/pportilla/atlas/blob/7907dfa/benchmarks/ideal-operators-ryzen-6800h.json) · [All 16-call trials](ideal-operators-ryzen-6800h.json) · [All 256-call trials](ideal-operators-batch256-ryzen-6800h.json)

### First declaration and script execution

The first Atlas column uses polynomials prepared in separate fresh rings before timing. The constructor has never seen the polynomial objects or assembled an ideal in that ring. The script column binds each input ideal with `I = ideal(a, b)` and assigns the result with the named function. The script evaluator handles all the assignments. Both columns include ideal construction and computation. The Singular column runs the same assignment sequence in its interpreter. Each trial contains 256 iterations.

| Inputs | Operation | Atlas, first declaration | Atlas script | Singular |
| --- | --- | ---: | ---: | ---: |
| Monomials | Sum | 3.41 | 4.19 | 2.88 |
| Monomials | Product | 3.98 | 4.37 | 2.85 |
| Monomials | Square | 5.31 | 6.12 | 7.41 |
| Rational forms | Sum | 4.00 | 4.45 | 3.41 |
| Rational forms | Product | 5.79 | 5.76 | 5.00 |
| Rational forms | Square | 36.28 | 37.20 | 65.57 |

Atlas had a lower median on all 6 cases at 16 calls and 5 of the 6 cases at 256 calls in the original declaration-inclusive test. The fresh-polynomial condition had a lower median on 2 of the 6 cases, and the script on 2 of the 6 cases.

## Cost of declaring two ideals

On the current build, a live polynomial in a rational polynomial ring retains a native representation tied to that ring. The public ideal constructor checks the parent identity before assembling generators. Eligible native ideals share the retained polynomials and their ring context. Earlier builds encoded generators as sparse text at each ideal declaration and reconstructed their native form. Singular copies the named polynomials into the ideal and normalizes those copies in the current ring. [Atlas constructor](../engine/session_algebra.py) · [Native assembly](../native/prepared_ideal.cpp) · [Singular constructor](https://github.com/Singular/Singular/blob/Release-4-4-1p5/Singular/ipassign.cc#L2353-L2386)

These medians cover the two positional `ideal(f, g)` calls together over seven trials of 256 pairs. Existing polynomials in the first-declaration columns have never appeared in an ideal. Singular includes reassignment and loop bookkeeping. The recorded trials also give the explicit `ring=` and `order=` variant.

| Inputs | Previous Atlas, first declaration | Atlas, first declaration | Atlas, repeated declaration | Singular |
| --- | ---: | ---: | ---: | ---: |
| Monomials | 22.71 | 1.65 | 1.46 | 3.56 |
| Rational forms | 48.14 | 2.73 | 1.41 | 2.32 |

### Including polynomial creation

A separate script evaluates all four polynomial expressions from `x` and `y` and optionally declares both ideals. The timers include polynomial arithmetic, assignment and preparation of the resulting values. Ring and session construction happen before timing. Each Atlas script uses a fresh ring. Singular evaluates the expressions afresh in an existing ring. These medians use seven trials of 32 iterations.

| Inputs | Work | Previous Atlas | Atlas | Singular |
| --- | --- | ---: | ---: | ---: |
| Monomials | Four polynomials | 65.47 | 80.84 | 5.22 |
| Monomials | Four polynomials and two ideals | 108.64 | 89.62 | 6.81 |
| Rational forms | Four polynomials | 511.40 | 604.76 | 7.59 |
| Rational forms | Four polynomials and two ideals | 584.31 | 617.73 | 8.81 |

The declaration driver verifies the ordered generators against the original expressions in Singular. The public `ideal` calls and the scripts are timed separately from those checks. [Driver](ideal_declarations.py) · [Previous build, all trials](ideal-declarations-before-native-polynomials-ryzen-6800h.json) · [Current build, all trials](ideal-declarations-ryzen-6800h.json)

## Operating on saved ideals at the prompt

The Atlas prompt column includes routing the submitted line, computing and saving the ideal, and recording the command in the transcript. Input comes from memory. Screen painting, terminal I/O and formatting the result for display take place outside the measurement. These medians use five trials of 16 operations on the same saved ideals.

| Inputs | Operation | Original Atlas prompt | Atlas C++ arithmetic | Atlas prompt | Singular assignment |
| --- | --- | ---: | ---: | ---: | ---: |
| Monomials | Sum | 512.12 | 0.09 | 1.73 | 2.94 |
| Monomials | Product | 515.24 | 0.52 | 2.11 | 3.00 |
| Monomials | Square | 76,134.63 | 1.29 | 3.57 | 8.00 |
| Rational forms | Sum | 757.67 | 0.10 | 1.74 | 2.94 |
| Rational forms | Product | 877.69 | 1.59 | 3.21 | 4.94 |
| Rational forms | Square | 72,558.28 | 28.66 | 31.93 | 64.94 |

At 256 calls per trial, the named prompt assignment had a lower median than Singular on all 6 cases. Small powers compute a complete reduced basis in the session under a work limit. Larger inputs and reductions that reach the limit use the cancellable worker. A saved result can feed another operation immediately. Displaying its generators or exporting the replay record formats the native data on demand. [C++ driver](ideal_operators_kernel.cpp) · [Original build](ideal-operators-before-ryzen-6800h.json) · [256-call results](ideal-operators-batch256-ryzen-6800h.json)

### Starting a new command

Starting a fresh `./atlas algebra` process, parsing the input, writing the JSON record and exiting took 4.13–4.94 ms across the six cases. Singular took 5.54–6.52 ms to start, compute and print the generators. The comparison uses five launches per case on core 6. [Driver](ideal_operators_process.py) · [All process trials](ideal-operators-process-ryzen-6800h.json)

### Repeating the comparison

Build with the headers for the CPython interpreter that will run Atlas. On Debian or Ubuntu these are usually in `python3-dev`. With headers, the build creates a stable-ABI extension. On a headerless build, Atlas calls the same native functions through ctypes. The measurements above use the extension.

```sh
make -j4
make -C native -f groebner.mk build/bench_ideal_operators
python3 benchmarks/ideal_operators.py --cpu 5 --trials 5 --batch 16
python3 benchmarks/ideal_operators.py --cpu 5 --trials 5 --batch 256
python3 benchmarks/ideal_operators_process.py --cpu 6 --trials 5
python3 benchmarks/ideal_declarations.py --cpu 6 --trials 7 --batch 256
```

## Products of strict components on a resolution chart

Each measurement times construction of the strict-transform ideal, including the final choice of generators. Charts have three private native coordinates over `QQ` and an empty ambient ideal. Each batch uses fresh coordinates on 20 charts with integer parameter `i` ranging from 1 to 20. Chart construction happens before the timer starts. After discarding one warm-up batch, the driver reports all seven trials and their median in milliseconds per chart. The measurements below used the Ryzen 7 6800H with ordinary process affinity. [Chart inputs and timing driver](strict_component_products.py)

| Strict component ideals | Individual polynomial products (ms/chart) | Batched native product (ms/chart) |
| --- | ---: | ---: |
| `(x³+y²+iz)` | 0.080 | 0.077 |
| `(x³+y²+iz), (y−z), (x+iy), (z+ix)` | 0.443 | 0.169 |
| `(x+iz,y), (y+ix,z), (z+iy,x)` | 1.838 | 0.832 |

Build the native kernel and run the following command from the repository root. To repeat the baseline on the earlier checkout (`702548e`), use the same driver with that checkout on `PYTHONPATH` and its native kernel built.

```sh
python3 -m benchmarks.strict_component_products
```
