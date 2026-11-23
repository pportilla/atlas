# Methods on saved values

Once you have saved a polynomial, a ring or a map under a name, you can use its fields and methods in later calculations. For example, `K(f)` takes a residue class in the quotient ring `K`, and `phi(f)` applies the ring map `phi`. At the prompt, `help K` or `help phi` describes the saved value and the fields available on it.

This page covers polynomials, quotient rings, local fractions and ring homomorphisms. A resolution, its charts and divisors, and its zeta functions are described in [Resolution objects and zeta functions](resolution-values.md). Each section is a complete script whose output appears in the comments.

## Polynomials

Every polynomial remembers its ring. Compare polynomials with `==` and combine them with `+`, `-`, `*` and `^`, provided they belong to the same ring or one of them is a rational number.

```atlas
R = ring("x", "y")
f = (x - y)^3 * (x + y)
print(f.parent == R)                                    # True
print(f.variables)                                      # ('x', 'y')
print(f.multiplicity(x - y))                            # 3
print(f.multiplicity(x + y), f.multiplicity(x))         # 1 0
```

| Field or method | Value |
| --- | --- |
| `f.parent` | The ring of `f`. Use it as `ring=f.parent` to read a formula in the same ring. |
| `f.variables` | The names of the coordinates of that ring, in order. |
| `f.coeffs` | The coefficient field, `QQ` unless the ring was declared over an algebraic extension. |
| `f.multiplicity(g)` | The largest $n$ such that $g^n$ divides the nonzero polynomial `f`. The factor `g` must be a nonconstant polynomial of the same ring, and the ring must have no relations. On a resolution chart, `C.pullback(f).multiplicity(u)` gives the order of a divisor $u=0$. [Exceptional divisors](resolution-values.md#exceptional-divisors) |

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
