# Marked number-field polynomial arithmetic on the Radeon 680M

Seven warmed trials compared the exact output of the Atlas CPU and Triton modes with Singular 4.4.1 on the same inputs in QQ(i)[x,y]. The script alternates the Atlas mode order and requires every GPU run to report a physical Triton stage. Singular checked equality in its algebraic extension with minimal polynomial a²+1 after every timed calculation. All checks passed. [Inputs and benchmark driver](triton_marked_arithmetic.py) · [Individual samples](triton-marked-arithmetic-680m.json)

| Operation | Atlas CPU (ms) | Atlas Triton (ms) | Singular (ms) |
|---|---:|---:|---:|
| Add | 0.477 | 0.796 | 0.011 |
| Subtract | 0.495 | 0.730 | 0.011 |
| Multiply | 0.705 | 1.174 | 0.021 |
| Multiply with a 256-bit coefficient | 0.370 | 0.673 | 0.009 |
| Divide by 1+i | 0.556 | 0.855 | 0.065 |
| Differentiate | 0.347 | 0.577 | 0.008 |
| Eighth power | 0.538 | 1.337 | 0.017 |
| Simultaneous substitution | 0.579 | 0.806 | 0.015 |

Times are medians on an AMD Ryzen 7 6800H with Radeon 680M, ROCm 7.1, and Triton 3.5.1. Atlas timings include public dispatch, parsing, execution records, and device transfers. Singular timings include parsing the same polynomial expressions and calculation inside a running process with one thread. Each system initializes its field and warms up before measurement. No timed workload ran faster on the GPU than on the CPU. Automatic dispatch uses the CPU.

For a substitution that maps each coordinate to a monomial with a number-field coefficient, Atlas now caches powers of those coefficients and sends their products to Triton together. The median time for the substitution above fell from 11.659 ms to 0.806 ms. The optimized CPU path took 0.579 ms. General polynomial images use repeated multiplication with exact reduction modulo the minimal polynomial between steps. [Native implementation](../native/triton_polynomial.cpp)

Run `python benchmarks/triton_marked_arithmetic.py --trials 7` in an environment with Singular on PATH and a working Triton device at index zero.
