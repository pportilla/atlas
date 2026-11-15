# Ideal arithmetic at the prompt and in resolution charts

## Assigning an ideal at the prompt

Median microseconds per assignment on a Ryzen 7 6800H, using one CPU core, CPython 3.14.4 and Singular 4.4.1. Both systems start with the same ideals over `QQ[x,y]` in graded reverse lexicographic order. Each cell is the median of five trials of 16 operations. The Atlas prompt column includes routing the submitted line, computing and saving the ideal, and recording the command in the transcript. Input comes from memory. Screen painting, terminal I/O and formatting a result for display are outside this measurement.

| Inputs | Operation | Previous Atlas prompt | Atlas C++ arithmetic | Atlas prompt | Singular assignment |
| --- | --- | ---: | ---: | ---: | ---: |
| Monomials | Sum | 512.12 | 0.09 | 1.70 | 2.88 |
| Monomials | Product | 515.24 | 0.54 | 2.17 | 2.94 |
| Monomials | Square | 76,134.63 | 1.23 | 3.56 | 7.81 |
| Rational forms | Sum | 757.67 | 0.09 | 1.69 | 2.81 |
| Rational forms | Product | 877.69 | 1.46 | 3.15 | 4.62 |
| Rational forms | Square | 72,558.28 | 28.17 | 31.59 | 64.31 |

The monomial inputs are `I = (x², y)` and `J = (x, y²)`. The rational inputs are `I = ((x+2y)⁸, (18446744073709551629/7)x⁶y)` and `J = ((2x−y)⁷, xy⁵/13)`. Atlas computes a reduced Gröbner basis for an ordinary power, so the Singular square is `std(I^2)` with `option(redSB)`. Each trial computes fresh results and checks mutual ideal containment in Singular. [Timing driver](ideal_operators.py) · [C++ timing driver](ideal_operators_kernel.cpp) · [Previous build](ideal-operators-before-ryzen-6800h.json) · [All 16-call trials](ideal-operators-ryzen-6800h.json)

With 256 operations per trial, both named and infix assignments also had lower medians than Singular on all six cases. The closest comparison was the monomial product at 1.68 µs for `answer = ideal_product(I, J)`, 1.61 µs for `answer = I * J`, and 1.79 µs in Singular. [All 256-call trials](ideal-operators-batch256-ryzen-6800h.json)

### Where the time goes

Previously, sums and products converted saved generators to text, parsed them again, and eagerly built a full replay record. Ordinary powers also started a separate worker. Small declared ideals now retain their native generators. Each sum and product computes a new native result. Small powers compute a complete reduced basis in the running session under a work limit. Larger inputs or reductions that reach the limit use the cancellable worker. A saved result can feed another operation immediately. Displaying its generators or exporting its replay record formats the native data on demand.

Preparation and presentation have separate costs. In the 16-call runs, constructing the ideals from existing polynomials **and** computing a sum or product took 16–30 µs in Atlas and 4–6 µs in Singular. For a monomial square, declaration and computation took 12.4 µs in Atlas and 8.3 µs in Singular. For the rational square they took 49.8 µs and 64.0 µs. Exporting a complete Atlas record from an already computed result took a further 45–90 µs, including the command and script to reproduce it. The raw trial files distinguish these measurements from the arithmetic on saved inputs.

### Why Singular constructs these ideals faster

Take four polynomials that already exist in one ring, and form two ideals with two generators each. Singular copies the named polynomials into the generator slots and normalizes them within the current ring. Atlas checks the live parents, checks each term against its size limits and assembles a sparse text representation. The C++ ideal constructor parses that text, creates a native ring context for each ideal and converts the generators into FLINT polynomials. Subsequent operations reuse those native generators. [Singular's ideal assignment](https://github.com/Singular/Singular/blob/Release-4-4-1p5/Singular/ipassign.cc#L2353-L2386) · [Singular's polynomial copy](https://github.com/Singular/Singular/blob/Release-4-4-1p5/Singular/subexpr.cc#L470-L472) · [Atlas preparation](../engine/session_ideal_native.py) · [Atlas native constructor](../native/prepared_ideal.cpp)

The following medians are in microseconds for **both declarations together**, over seven trials of 256 pairs on core 6. Polynomial expansion, ideal arithmetic and process startup take place outside the timer. The columns for Atlas measure the public constructor, including disposal, using the CPython extension. Singular's timings include reassignment and its interpreter loop.

| Existing generators | Atlas, first encoding | Atlas, encoding already cached | Singular |
| --- | ---: | ---: | ---: |
| Monomials | 22.07 | 12.78 | 2.17 |
| Rational forms | 46.28 | 21.20 | 2.29 |

For the rational forms on first use, separately instrumented stages took 25.0 µs to encode the coefficients and exponents, 10.0 µs in native preparation, 8.1 µs to check input sizes, and 3.5 µs to validate parents and wrap the ideals. Each stage includes the overhead of its timers. Retaining the native polynomial when it is first created would let the ideal constructor reuse its generators and their shared ring context. [Declaration benchmark](ideal_declarations.py) · [All trials and stage timings](ideal-declarations-ryzen-6800h.json)

### Starting a new command

Starting a fresh `./atlas algebra` process, parsing the input, writing the full JSON result and exiting took 5.69–6.04 ms across the six cases. Singular took 7.25–8.12 ms to start, compute and print the generators. Atlas had the lower median in each pair. Before the change, a standalone sum or product started both Python and an algebra worker and took about 165 ms. This comparison uses five fresh launches on core 6 and excludes the separate verification process. [Fresh-process driver](ideal_operators_process.py) · [Inputs and individual times](ideal-operators-process-ryzen-6800h.json)

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
