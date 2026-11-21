# Methods on saved values

Once you have named a ring, a map or a resolution, you can use its methods in later calculations. For example, `K(f)` takes a residue class in the particular ring `K`, and `phi(f)` applies the particular map `phi`. At the prompt, `help K` or `help phi` describes the saved value and its available fields.

## Quotient rings and residue classes

Consider the dual numbers $K=\mathbb Q[x]/(x^2)$, and write $a$ for the class of $x$. To take the class of an ambient polynomial, apply `K` to it. Arithmetic on the resulting class uses the relation $a^2=0$.

```atlas
R = ring("x")
K = quotient_ring(ideal(x^2))
a = K(x)
print(K(x^2) == 0)                                      # True
print(a.is_unit(), (1+a).is_unit())                     # False True
print((1+a).inverse())                                  # -x+1
power = K.apply("power", 1+a, exponent=-2, compute="cpu")
print(power == 1-2*a, power.parent == K)                # True True
```

The inverse of $1+a$ is $1-a$. Its square is $1-2a$, which explains the last line. In a script where the operation itself is an input, `K.apply` accepts its name as a string.

| Method | Inputs and result |
| --- | --- |
| `K(polynomial)` or `K.element(polynomial)` | Give an ambient polynomial, a rational scalar or a class from the chain of defining quotients. Returns its class in `K`. The field `.representative` gives the reduced ambient polynomial. For a formula supplied as a string, use `polynomial("x", ring=K)`. |
| `a.is_unit()` | Returns `True` if `a` has a multiplicative inverse in its parent ring. Takes no arguments. |
| `a.inverse()` | Returns that inverse as a class in the same ring. A nonunit raises an error. Takes no arguments. |
| `K.apply(action, *values, exponent=None, compute=None)` | Supply an action below and its operands in order. Each operand can be a class in `K`, an ambient polynomial or a rational scalar. Omit `compute` to inherit the ring’s policy, or pass `"cpu"`, `"auto"` or `"gpu"`. |

| Action | Operands | Result |
| --- | --- | --- |
| `"declare"` | None | The ring `K` with a recorded declaration. |
| `"reduce"`, `"negate"`, `"inverse"` | One | A class in `K`. Inversion requires a unit. |
| `"unit"` | One | An invertibility test in `.value`. |
| `"add"`, `"subtract"`, `"multiply"`, `"divide"` | Two | A class in `K`. The divisor must be a unit. |
| `"equal"` | Two | An equality test in `.value`. |
| `"power"` | One and a signed integer `exponent` | A class in `K`. A negative exponent requires a unit. The exponent can also be a decimal string. |

## Local fractions

`L.fraction(f, s=1, *, compute=None)` constructs $f/s$ in the local ring described by `L`. With $P=(x)$, for example, $y$ is invertible in $R_P$. Does $x/y$ belong to the extension of the ideal $(xy)$ there?

```atlas
R = ring("x", "y")
L = localize(ideal(x*y), ideal(x))
q = L.fraction(x, y)
print(q.member, q.unit)                                 # True False
print(L.fraction(y).unit)                               # True
```

Indeed $x/y=(xy)/y^2$, so `q.member` is true. The numerator belongs to $P$, so `q.unit` is false. With one argument the denominator defaults to $1$.

Supply polynomials from the original ring or rational scalars. The denominator must lie outside `L.prime`. The result retains `.numerator` and `.denominator`. The optional `compute` policy defaults to that of `L` and accepts `"auto"`, `"cpu"` or `"gpu"`. [Localization at a prime](../tutorial/quotient-rings.md#localize-at-a-prime)

## Ring homomorphisms

The parametrization $x=t^2$, $y=t^3$ gives a homomorphism $\mathbb Q[x,y]\to\mathbb Q[t]$. Specify the images in the order of the source generators. Applying the map substitutes both at once.

```atlas
R = ring("x", "y")
S = ring("t")
phi = hom(R, S, images=(t^2, t^3))
print(phi(x+y))                                         # t^3+t^2
print(phi(y^2-x^3))                                     # 0
```

The second output verifies that the image lies on the cusp. `phi(value)` takes one argument, an element of `phi.domain` or a rational scalar, and returns its image in `phi.codomain`. Use the same syntax for a quotient projection, an inclusion from elimination, a normalization map or a chart pullback map. [Defining and applying a ring map](../tutorial/rings.md#move-a-polynomial-between-rings)

## Resolution charts, divisors and polynomial multiplicity

How many times does the third exceptional divisor occur in the total transform of the cusp? Work in one chart that meets it, read its local equation and pull back the cusp polynomial.

```atlas
R = ring("x", "y")
f = y^2-x^3
Z = resolve(f)
C = Z.chart("B3.u")
E = Z.divisor("E3")
g = E.equation_in(C)
total = C.pullback(f)
print(g)                                                # u
print(total)                                            # u^6*v^4-u^6*v^3
print(total.multiplicity(g))                            # 6
```

In this chart the divisor is $u=0$. The pullback factors as $u^6v^3(v-1)$, so the divisor occurs six times. For the other charts and the strict transform of the curve, follow the full resolution example. [Resolve a plane cusp](../tutorial/resolving-a-cusp.md#look-inside-a-chart)

| Method | Inputs and result |
| --- | --- |
| `Z.chart(id)`, `Z.divisor(id)` | Give a string ID from this resolution, such as `"B3.u"` or `"E3"`. Returns the corresponding chart or exceptional divisor. The fields `Z.charts` and `Z.divisors` list the choices. |
| `C.pullback(polynomial)` | Give a polynomial from the original ring of the resolution. Returns its pullback expressed in the chart ring. |
| `E.ideal_in(chart)` | Give a chart of the same resolution with local equations for this divisor. Returns their polynomial generators as a list in the chart ring. |
| `E.equation_in(chart)` | Give a chart on which the recorded divisor ideal has one generator. Returns that generator. Use `ideal_in` for a list of generators. |
| `f.multiplicity(factor)` | Give a nonconstant factor in the same polynomial ring as the nonzero polynomial `f`. Returns the largest integer $n$ such that `factor` raised to $n$ divides `f`. The ring must have no ambient relations. |
