# Absolute polynomial factorization over the algebraic closure of Q

The comparison runs Atlas's public `absolute_factor(f, order="lex")` and
Singular's `absfact.lib::absFactorize(f)` with the same rational input in
the ordered coordinates $x,y,z$. Atlas returns every individual geometric
factor with coefficients in a common number field. Singular returns one
representative of each orbit under conjugation, its minimal polynomial and
its multiplicity. The two representations can use different field generators.
[Benchmark driver](absolute_factor.py) · [Exact comparison](../tests/absolute_factor_oracle.py)

| Case | Input | Distinct geometric factors |
|---|---|---:|
| Two lines | $x^2+y^2$ | 2 |
| Irreducible quadric | $x^2+yz$ | 1 |
| Non-normal cubic extension | $(x+yz)^3-2(y+z)^3$ | 3 |
| Two quadratic fields | $-\tfrac23(x^2+y^2)^2((x+yz)^2-2(y+z)^2)(x^2+yz)$ | 5 |
| Three quadratic translates | $(z^2-D)((z-A)^2-D)((z-A-B)^2-D)$, where $D=y^3-x^5$, $A=y^2-x^3$, $B=x^4+y^3$ | 3 |

The cubic uses the three roots of $t^3-2$ and has a degree-six splitting
field. In the fourth case two factors have multiplicity two, and the
coefficient field contains both $i$ and $\sqrt{2}$. Each quadratic of the
last case is absolutely irreducible since $D$ has odd degree in $y$.

Every trial verifies the reconstruction of the original polynomial over
the respective fields. For each Atlas factor the independent checker
computes its norm to $\mathbb Q$ and identifies the rational orbit product.
It compares the number of distinct normalized factors in each orbit and
their multiplicities with the known decomposition and with Singular's
reported orbit. The checker also multiplies Singular's orbit products with
their multiplicities and constant unit. These exact checks run outside
the timers.

A fresh run includes starting the process, loading libraries and reading,
factoring, printing and collecting its polynomial. A warm run sends a new
expression to an initialized interpreter, requests the same full answer and
reads its output. Each solver receives one untimed warm-up per input. The
driver alternates solver order across five trials and includes serialization
in both timings. Singular uses seeds 1 through 5 in fresh runs and seed 1
at the start of its warm session. Atlas uses its shared prompt and script
session. Repeat with the following command from the repository root.

```sh
taskset -c 5 python3 benchmarks/absolute_factor.py --trials 5
```

## Same-host timings

On 2026-11-19 the benchmark ran on logical CPU 5 of an AMD Ryzen 7 6800H
in Linux x86-64, with Python 3.14.4 and Singular 4.4.1. All 100 timed
outputs passed the exact checks, as did both untimed warm-ups per input.

| Case | Session | Solver | Five wall times (ms) | Median (ms) |
|---|---|---|---|---:|
| two-lines | fresh | Atlas | 137.938, 139.561, 138.481, 137.615, 142.281 | 138.481 |
| two-lines | fresh | Singular | 8.628, 8.143, 8.502, 7.873, 8.450 | 8.450 |
| two-lines | warm | Atlas | 1.253, 0.923, 0.835, 0.894, 0.827 | 0.894 |
| two-lines | warm | Singular | 0.900, 0.791, 0.830, 0.780, 0.810 | 0.810 |
| irreducible-quadric | fresh | Atlas | 136.592, 131.582, 136.784, 144.987, 139.889 | 136.784 |
| irreducible-quadric | fresh | Singular | 7.937, 7.256, 9.657, 9.019, 7.914 | 7.937 |
| irreducible-quadric | warm | Atlas | 0.933, 0.778, 0.689, 0.663, 0.625 | 0.689 |
| irreducible-quadric | warm | Singular | 0.894, 0.686, 0.725, 0.687, 0.713 | 0.713 |
| non-normal-cubic | fresh | Atlas | 153.905, 153.134, 151.707, 155.253, 150.293 | 153.134 |
| non-normal-cubic | fresh | Singular | 15.461, 14.844, 16.107, 14.487, 14.855 | 14.855 |
| non-normal-cubic | warm | Atlas | 5.892, 5.770, 5.903, 5.861, 5.947 | 5.892 |
| non-normal-cubic | warm | Singular | 6.942, 7.401, 6.624, 7.879, 8.060 | 7.401 |
| two-quadratic-fields | fresh | Atlas | 149.667, 150.789, 152.032, 154.800, 151.761 | 151.761 |
| two-quadratic-fields | fresh | Singular | 12.281, 12.472, 12.065, 12.412, 12.224 | 12.281 |
| two-quadratic-fields | warm | Atlas | 4.724, 4.818, 4.747, 4.709, 4.817 | 4.747 |
| two-quadratic-fields | warm | Singular | 4.445, 4.087, 4.464, 4.289, 4.209 | 4.289 |
| three-quadratic-translates | fresh | Atlas | 152.003, 149.622, 146.132, 145.486, 145.548 | 146.132 |
| three-quadratic-translates | fresh | Singular | 11.304, 10.622, 10.076, 10.134, 10.143 | 10.143 |
| three-quadratic-translates | warm | Atlas | 4.739, 4.653, 4.336, 4.580, 4.341 | 4.580 |
| three-quadratic-translates | warm | Singular | 3.636, 2.690, 3.821, 3.494, 3.309 | 3.494 |

The initial warm Atlas implementation launched a worker for each expression.
Its median for the irreducible quadric was 71.4 ms. Reusing the native CPU
operation in the live session reduced this to 0.689 ms. On the non-normal
cubic, the marked-field factorization initially took about 83 ms inside
the native operation. Computing its norm by translating only the selected
separating coordinate reduced this native time to about 4.2 ms. Atlas's
final warm public median on that cubic was 5.892 ms, compared with 7.401 ms
for Singular. The warm irreducible quadric also finished slightly faster in
Atlas, at 0.689 ms versus 0.713 ms. Singular was faster on the two-line,
two-field, and three-quadratic-translate warm controls and on all five
fresh-process controls.
