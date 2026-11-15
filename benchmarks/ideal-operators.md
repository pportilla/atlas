# Ideal arithmetic at the prompt and in resolution charts

## Declaring ideals and computing with them in scripts

Each iteration starts with four existing polynomials in one `QQ[x,y]` ring and reassigns the ideals and result. For a sum or product, Atlas executes these three lines.

```atlas
I = ideal(a, b)
J = ideal(c, d)
answer = ideal_sum(I, J)
```

For the product the third line calls `ideal_product(I, J)`. The square declares only `I` and calls `ideal_power(I, 2)`. Singular executes equivalent assignments inside its interpreter loop, and uses `std(I^2)` with `option(redSB)` because Atlas returns a reduced Gröbner basis. Both use graded reverse lexicographic order. Each trial checks mutual ideal containment in Singular outside the timer.

Median microseconds per complete iteration, from five trials on a single Ryzen 7 6800H core with CPython 3.14.4 and Singular 4.4.1. The repeated-script columns submit the complete assignment block through the Atlas script evaluator for each iteration. The last column runs the same lines inside a single Atlas `for` loop. An untimed iteration primes the evaluator for each script. Ring and polynomial construction, process launch and result formatting are outside these timers.

| Inputs | Operation | Previous Atlas, 256 | Atlas, 16 | Singular, 16 | Atlas, 256 | Singular, 256 | Atlas `for` loop, 256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Monomials | Sum | 3.95 | 2.40 | 4.06 | 2.30 | 2.91 | 2.31 |
| Monomials | Product | 4.20 | 2.60 | 4.06 | 2.43 | 2.89 | 2.50 |
| Monomials | Square | 5.42 | 4.89 | 8.50 | 4.48 | 7.38 | 4.40 |
| Rational forms | Sum | 3.96 | 2.34 | 4.00 | 2.18 | 3.17 | 2.18 |
| Rational forms | Product | 5.26 | 3.79 | 6.00 | 3.43 | 4.89 | 3.43 |
| Rational forms | Square | 32.90 | 35.71 | 65.69 | 33.06 | 61.25 | 33.31 |

Both Atlas script measurements had a lower median than Singular on all six cases at both batch sizes. [Driver](ideal_operators.py) · [Before the change](ideal-script-before-ryzen-6800h.json) · [All 16-iteration trials](ideal-operators-ryzen-6800h.json) · [All 256-iteration trials](ideal-operators-batch256-ryzen-6800h.json)

The monomial inputs are `I = (x², y)` and `J = (x, y²)`. The rational inputs are `I = ((x+2y)⁸, (18446744073709551629/7)x⁶y)` and `J = ((2x−y)⁷, xy⁵/13)`.

### Time spent evaluating script statements

The direct control calls the same public Atlas functions, reads the same named operands and writes the results into the same workspace dictionary. Subtracting that median from the repeated-script median estimates the cost of dispatching the Atlas statements. Timings are in microseconds at 256 iterations.

| Inputs | Operation | Previous direct | Previous script | Current direct | Current script | Current native arithmetic alone |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Monomials | Sum | 2.44 | 3.95 | 1.53 | 2.30 | 0.06 |
| Monomials | Product | 2.71 | 4.20 | 1.71 | 2.43 | 0.26 |
| Monomials | Square | 4.00 | 5.42 | 3.65 | 4.48 | 1.16 |
| Rational forms | Sum | 2.46 | 3.96 | 1.44 | 2.18 | 0.06 |
| Rational forms | Product | 3.65 | 5.26 | 2.66 | 3.43 | 1.10 |
| Rational forms | Square | 31.34 | 32.90 | 32.02 | 33.06 | 27.53 |

Evaluating and binding the statements added 1.49–1.62 µs above the direct calls on the four sum and product examples before this change. Atlas now prepares groups of simple assignments once, then resolves the current names and calls the original functions in order. Newly declared ideals own their native generators directly, removing an intermediate wrapper from each declaration. The remaining dispatch time was 0.72–0.78 µs on the same examples. [Script evaluator](../engine/session_expressions.py) · [Native call dispatch](../native/python_session.cpp) · [Ideal constructor](../native/python_ideal.cpp)

### Repeated and fresh input polynomials

The original direct-operator benchmark supplies the generators, parent and order to the public ideal constructor on every iteration. The fresh-input control performs the very first ideal declaration from each set of polynomials in its own independent parent. Both exclude creating the polynomials. Singular uses the same polynomials in the same ring across iterations. These medians use 256 iterations.

| Inputs | Operation | Atlas direct, repeated inputs | Atlas direct, first declaration | Singular, repeated inputs |
| --- | --- | ---: | ---: | ---: |
| Monomials | Sum | 1.83 | 2.60 | 2.91 |
| Monomials | Product | 2.06 | 2.79 | 2.89 |
| Monomials | Square | 4.04 | 4.81 | 7.38 |
| Rational forms | Sum | 1.75 | 2.58 | 3.17 |
| Rational forms | Product | 2.96 | 4.20 | 4.89 |
| Rational forms | Square | 31.91 | 33.97 | 61.25 |

For the monomial product, Atlas took 2.79 µs with fresh polynomials and parents, compared with 2.89 µs for Singular with reused inputs.

## Cost of declaring two ideals

On the current build, a live polynomial in a rational polynomial ring retains a native representation tied to that ring. The public ideal constructor checks the parent identity before assembling generators. Eligible native ideals share the retained polynomials and their ring context. Earlier builds encoded generators as sparse text at each ideal declaration and reconstructed their native form. Singular copies the named polynomials into the ideal and normalizes those copies in the current ring. [Atlas constructor](../engine/session_algebra.py) · [Native assembly](../native/prepared_ideal.cpp) · [Singular constructor](https://github.com/Singular/Singular/blob/Release-4-4-1p5/Singular/ipassign.cc#L2353-L2386)

These medians cover the two positional `ideal(f, g)` calls together over seven trials of 256 pairs. Existing polynomials in the first-declaration columns have never appeared in an ideal. Singular includes reassignment and loop bookkeeping. The recorded trials also give the explicit `ring=` and `order=` variant.

| Inputs | Previous Atlas, first declaration | Atlas, first declaration | Atlas, repeated declaration | Singular |
| --- | ---: | ---: | ---: | ---: |
| Monomials | 1.65 | 1.04 | 0.40 | 3.62 |
| Rational forms | 2.73 | 1.69 | 0.68 | 2.33 |

### Including polynomial creation

A separate script evaluates all four polynomial expressions from `x` and `y` and optionally declares both ideals. The timers include polynomial arithmetic, assignment and preparation of the resulting values. Ring and session construction happen before timing. Each Atlas script uses a fresh ring. Singular evaluates the expressions afresh in an existing ring. These medians use seven trials of 32 iterations.

| Inputs | Work | Previous Atlas | Atlas | Singular |
| --- | --- | ---: | ---: | ---: |
| Monomials | Four polynomials | 80.84 | 80.65 | 5.28 |
| Monomials | Four polynomials and two ideals | 89.62 | 89.85 | 7.00 |
| Rational forms | Four polynomials | 604.76 | 620.80 | 7.75 |
| Rational forms | Four polynomials and two ideals | 617.73 | 647.83 | 8.91 |

The rational expressions produce 12 intermediate polynomial operations. On first evaluation in a fresh ring, Atlas formats operands for each native request and decodes the sparse result back into Python terms. In a separate profile of these four expressions, decoding took about 227 µs and formatting operands took about 89 µs. These costs occur before the first ideal declaration. [Polynomial arithmetic and decoding](../engine/native_polynomial.py)

The declaration benchmark verifies the ordered generators against the original expressions. [Driver](ideal_declarations.py) · [Previous build, all trials](https://github.com/pportilla/atlas/blob/fa9e3d5/benchmarks/ideal-declarations-ryzen-6800h.json) · [Current build, all trials](ideal-declarations-ryzen-6800h.json)

## Operating on saved ideals at the prompt

The prompt benchmark times a named assignment on already declared ideals and saves the command to the transcript. Input comes from memory. Screen painting, terminal I/O and result formatting take place outside the timer. These medians use five trials of 16 operations.

| Inputs | Operation | Atlas C++ arithmetic | Atlas prompt | Singular assignment |
| --- | --- | ---: | ---: | ---: |
| Monomials | Sum | 0.09 | 1.84 | 3.00 |
| Monomials | Product | 0.52 | 1.93 | 3.00 |
| Monomials | Square | 1.29 | 3.57 | 7.75 |
| Rational forms | Sum | 0.09 | 1.66 | 2.81 |
| Rational forms | Product | 1.55 | 3.14 | 4.94 |
| Rational forms | Square | 29.11 | 32.57 | 66.56 |

[C++ driver](ideal_operators_kernel.cpp) · [Original build](ideal-operators-before-ryzen-6800h.json) · [All 16-call trials](ideal-operators-ryzen-6800h.json)

### Starting a new command

Starting a fresh `./atlas algebra` process, parsing the input, writing the JSON record and exiting took 4.70–5.50 ms across the six cases. Singular took 6.20–6.71 ms to start, compute and print the generators. The comparison uses five launches per case on core 6. [Driver](ideal_operators_process.py) · [All process trials](ideal-operators-process-ryzen-6800h.json)

### Repeating the comparison

Build with the headers for the CPython interpreter that will run Atlas. On Debian or Ubuntu these are usually in `python3-dev`. With headers, the build creates a stable-ABI extension. On a headerless build, Atlas calls the same native functions through ctypes. The measurements above use the extension. To reproduce the matched before-control, run the current operator driver in a checkout of the [previous build](https://github.com/pportilla/atlas/tree/fa9e3d54fd17bdce19ffef0b39e1fe836b20f368) with the same compiler and interpreter.

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
