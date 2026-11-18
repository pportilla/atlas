# Methods on saved values

A saved value carries the ring or resolution it came from. The methods below use that owner when interpreting their inputs. Each section starts in a fresh Atlas session. At the prompt, type `help` followed by a saved name to inspect that particular value.

## Quotient rings and residue classes

Let `K` be a quotient ring and `a` a class in it. The relation $x^2=0$ makes $1+x$ invertible since $(1+x)(1-x)=1$ in the quotient.

```atlas
R = ring("x")
K = quotient_ring(ideal(x^2))
a = K.element(x)
print(K(x) == a, K(x^2) == 0)                 # True True
print(a.is_unit(), (1+a).is_unit())           # False True
print((1+a).inverse() == 1-a)                 # True
power = K.apply("power", 1+a, exponent=-2, compute="cpu")
print(power == 1-2*a, power.parent == K)      # True True
```

| Call | Inputs and result |
| --- | --- |
| `K.element(polynomial)` or `K(polynomial)` | Project an ambient polynomial, a rational scalar, or a class from the chain of defining quotients into `K`. The result has parent `K` and a reduced ambient `.representative`. Use `polynomial("x", ring=K)` to read a formula string. |
| `a.is_unit()` | Takes no arguments. Returns a Boolean indicating whether the class has a multiplicative inverse. |
| `a.inverse()` | Takes no arguments. Returns the inverse in the same quotient ring. A nonunit raises an error. |
| `K.apply(action, *values, exponent=None, compute=None)` | Run a named quotient operation and return its structured result. Pass operands separately as classes in `K`, ambient polynomials or rational scalars. `compute` inherits the ring’s declaration policy. An explicit `"cpu"`, `"auto"` or `"gpu"` selects the policy for this call. |

The accepted `action` strings determine how many operands to supply.

| Action | Operands | Result |
| --- | --- | --- |
| `"declare"` | None | The same live quotient ring with a recorded declaration. |
| `"reduce"`, `"negate"`, `"inverse"` | One | A class in `K`. Inversion requires a unit. |
| `"unit"` | One | A result with the invertibility test in `.value`. |
| `"add"`, `"subtract"`, `"multiply"`, `"divide"` | Two, in order | A class in `K`. Division requires the second operand to be a unit. |
| `"equal"` | Two | A result with the equality test in `.value`. |
| `"power"` | One, plus a signed integer `exponent` | A class in `K`. A negative power requires a unit. Decimal strings also work for the exponent. |

## Local fractions

In the localization at $P=(x)$, the denominator $y$ is invertible. The fraction $x/y$ belongs to the extended ideal $(xy)$ because $x/y=(xy)/y^2$.

```atlas
R = ring("x", "y")
L = localize(ideal(x*y), ideal(x))
q = L.fraction(x, y)
print(q.member, q.unit)                       # True False
print(L.fraction(y).unit)                     # True
```

`L.fraction(f, s=1, *, compute=None)` takes a numerator `f` and denominator `s` in the original polynomial ring, including exact rational scalars. The denominator must lie outside `L.prime`. By default the calculation uses the compute policy of `L`. Choose `"auto"`, `"cpu"` or `"gpu"` to override it. The result records `.numerator`, `.denominator`, membership in the extended ideal as `.member`, and invertibility in the local ring as `.unit`.

## Ring homomorphisms

Use `phi(value)` to apply a saved map. The input is an element of `phi.domain` or a rational scalar. The result lies in `phi.codomain` and uses all the ordered generator images simultaneously. The same call applies to a quotient projection, an elimination inclusion, a normalization map and a chart pullback map. These calls take one argument.

```atlas
R = ring("x", "y")
S = ring("t")
phi = hom(R, S, images=(t^2, t^3))
print(phi(y^2-x^3) == 0)                      # True
print(phi(x+y) == t^2+t^3)                    # True
```

## Resolution charts, divisors and polynomial multiplicity

The third exceptional divisor in the embedded resolution of the cusp has multiplicity six. Read its equation in a chart, pull the original cusp equation back to that chart, and count how many times the divisor equation divides it.

```atlas
R = ring("x", "y")
f = y^2-x^3
Z = resolve(f)
C = Z.chart("B3.u")
E = Z.divisor("E3")
g = E.equation_in(C)
total = C.pullback(f)
print(E.ideal_in(C) == [g], total.parent == C.parent)   # True True
print(total.multiplicity(g))                             # 6
```

| Call | Inputs and result |
| --- | --- |
| `Z.chart(id)`, `Z.divisor(id)` | Supply a string ID from this resolution, such as `"B3.u"` or `"E3"`. Returns the linked chart or exceptional divisor. Inspect `Z.charts` or `Z.divisors` to find IDs. |
| `C.pullback(polynomial)` | Substitute the chart map into a polynomial in the resolution’s original ring. The result is a polynomial in the chart ring. |
| `E.ideal_in(chart)` | Use a chart of the same resolution containing local equations for the divisor. Returns their polynomial generators as a list in chart coordinates. |
| `E.equation_in(chart)` | Use a chart on which the recorded divisor ideal has one generator. Returns that generator. Use `ideal_in` to read several generators. |
| `f.multiplicity(factor)` | For nonzero `f`, returns the largest integer $n$ such that the nonconstant polynomial `factor` raised to $n$ divides `f`. Both polynomials must share a parent with no ambient relations. |

The cusp walkthrough follows the branch into these charts and identifies the equations for both the exceptional divisor and the strict transform. [Resolve a plane cusp](resolving-a-cusp.md#look-inside-a-chart)
