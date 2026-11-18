# Polynomials and ideals

Run these examples in the Atlas console or save a block in an `.atlas` file and run it with `./atlas run`. Each section declares its own rational coordinates. The monomial order defaults to graded reverse lexicographic (`grevlex`). Use `order="lex"` for lexicographic order with the first declared variable highest.

## Reduce a polynomial modulo an ideal

Work in $\mathbb Q[x,y]/(x^2-1,y-x)$. In this quotient $x=y$ and $y^2=1$, so the class of $x^2+y/2$ is $1+y/2$. Compute a Gröbner basis once and reuse it for reductions and membership.

```atlas
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(x^2-1, y-x, order="lex")
B = groebner(I)
r = normal_form(x^2+y/2, B)
print(r.polynomial == 1+y/2)           # True
print(membership(y^2-1, B).value)     # True
print(membership(y, B).value)         # False
```

An ideal and its basis retain their coordinates. `I.parent` is the live ambient ring and `I.ring` records the field, variable names and monomial order. In that ring, `ideal(ring=I.parent)` gives the zero ideal and `ideal(1, ring=I.parent)` gives the unit ideal. If you select this ring with `use_ring(I.parent)`, then `ideal()` and `ideal(1)` give the same ideals. [Choosing an active ring](../docs/sessions.md#name-select-and-delete-rings) · [Runnable basis example](groebner.atlas) · [Reduction and membership tests](../tests/test_session_algebra.py) · [Gröbner kernel](../native/groebner.cpp)

## Factor and reconstruct a polynomial

Recover a polynomial from its rational unit and its irreducible factors. Each factor is monic in the selected order. Here $2x-y$ contributes a factor of $2$ twice, so the unit is $-12/5$.

```atlas
x, y = coordinates("x", "y")
f = -3/5*(2*x-y)^2*(x^2+1)
F = factor(f, order="lex")
product = F.unit
for term in F.factors:
    product = product * term.polynomial^term.multiplicity

print(F.unit, product == f)          # -12/5 True
```

A nonzero constant has an empty factor list and is its own unit. Factoring zero raises an input error. [Factorization checks](../tests/test_factor_session.py) · [C++ factorization](../native/polynomial_factors.cpp)

## Find a common factor

```atlas
x, y = coordinates("x", "y")
d = gcd((x+y)*(x+1), (x+y)*(y+1), order="lex")
print(d.polynomial == x+y, d.order)  # True lex
```

A nonzero GCD is monic in the recorded order. Thus `gcd(0,f)` gives the monic associate of $f$, while `gcd(0,0)` gives zero. For two constant operands, pass `variables=(x,y)` to keep the result in the desired ring. [GCD tests](../tests/test_polynomial_gcd.py) · [C++ polynomial arithmetic](../native/affine.cpp)

## Eliminate a coordinate

The equations $t^2=x$ and $t^3=y$ parametrize a cusp. The signed Sylvester resultant eliminates $t$ from this pair. To eliminate coordinates from an entire ideal, use `eliminate(I, t)`. Each result carries the smaller ring and an inclusion back into the original ring.

```atlas
t, x, y = coordinates("t", "x", "y")
r = resultant(t^2-x, t^3-y, t)
u, v = r.coordinates
print(r.polynomial == v^2-u^3)       # True
print(r.inclusion(r) == y^2-x^3)     # True

J = eliminate(ideal(t^2-x, t^3-y), t)
a, b = J.coordinates
print(membership(b^2-a^3, J).value)  # True
```

Resultants retain sign and rational content. For example, eliminating $t$ from $(t-x,t-y)$ gives $x-y$. A zero operand gives zero. Two nonzero operands independent of $t$ give one. With just one independent operand, the answer is that operand raised to the degree in $t$ of the other. Eliminating the sole variable gives a constant over $\mathbb Q$. [Resultant tests](../tests/test_resultant_session.py) · [Inclusion tests](../tests/test_session_maps.py) · [C++ resultant](../native/affine.cpp) · [Ideal elimination](../native/ordered_ideals.cpp)

## Intersect, divide and saturate ideals

For $I=(x^2,xy)$, the quotient $I:(x)$ is $(x,y)$. Requiring $fx^k\in I$ for *some* $k\ge 0$ gives the saturation $I:x^\infty$. It contains $1$ because $x^2\in I$.

```atlas
x, y = coordinates("x", "y")
I = ideal(x^2, x*y)
Q = quotient(I, ideal(x))
print(membership(y, Q).value, membership(1, Q).value)  # True False
print(membership(1, saturation(I, x)).value)            # True
J = intersection(ideal(x), ideal(y))
print(J.generators == (x*y,))                           # True
```

Use `quotient(I,J)` for the ideal of all $f$ with $fJ\subset I$. To declare a ring of residue classes, use the [quotient ring guide](quotient-algebras.md#compute-with-residue-classes). [Ideal-operation tests](../tests/test_ideal_operations.py) · [C++ ideal operations](../native/ordered_ideals.cpp)

## Recover the reduced support

The radical of $I$ contains every $f$ for which a positive power of $f$ lies in $I$. For $I=(y,y-x^2)$, the difference of its generators gives $x^2\in I$. The radical is $(x,y)$.

```atlas
x, y = coordinates("x", "y")
I = ideal(y, y-x^2)
J = radical(I)
print(membership(x, I).value, membership(x, J).value)  # False True
```

The radical is returned over $\mathbb Q$ even when the support has no rational point. For instance, the radical of $((x^2+1)^2)$ is $(x^2+1)$. [Radical tests](../tests/test_radical_interface.py) · [C++ radical calculation](../native/primary_decomposition.cpp)

## Reconstruct an ideal from its primary components

For $I=(x^2,xy)$, one primary decomposition is $(x)\cap(x^2,y)$. The associated primes are $(x)$, defining the line, and $(x,y)$, defining the embedded origin. Inspect each pair and check that the intersection recovers $I$.

```atlas
x, y = coordinates("x", "y")
I = ideal(x^2, x*y)
D = primary_decomposition(I)
a, b = D.components
print(a.primary.generators, a.prime.generators)          # (x,) (x,)
print(b.primary.generators, b.prime.generators)          # (x^2, y) (x, y)
J = intersection(a.primary, b.primary)
print(groebner(J).generators == groebner(I).generators)  # True
```

Different choices of embedded primary ideals can give the same intersection. Each associated prime is the radical of its primary component. The zero ideal has a single component in the polynomial ring and the unit ideal has an empty decomposition. [Component and reconstruction tests](../tests/test_primary_decomposition_interface.py) · [C++ primary decomposition](../native/primary_decomposition.cpp)

## Compare ordinary and symbolic powers

The three coordinate axes have ideal $I=(xy,xz,yz)$. Every generator of the ordinary square has degree four. The product $xyz$ vanishes to order two along each axis, so it lies in the symbolic square.

```atlas
x, y, z = coordinates("x", "y", "z")
I = ideal(x*y, x*z, y*z)
print(membership(x*y*z, ideal_power(I, 2)).value)      # False
print(membership(x*y*z, symbolic_power(I, 2)).value)   # True
```

An exponent must be a nonnegative integer. Exponent zero gives the unit ideal. If you select this ring with `use_ring(I.parent)`, then `ideal()` and `ideal(1)` give the same ideals. [Choosing an active ring](../docs/sessions.md#name-select-and-delete-rings) · Symbolic powers require a radical input, including at exponents zero and one. The result is the intersection of $I^n R_P\cap R$ over the minimal primes $P$ of $I$. Each power returns a reduced basis in the original coordinates and order. [Power tests](../tests/test_ideal_power_interface.py) · [C++ ideal powers](../native/ideal_powers.cpp)
