# Quotient rings, localization and normalization

The relation $y^2=0$ changes what it means for two polynomials to be equal. It can even make two different ideals equal. We start in a polynomial ring, impose this relation, and compare the ideals on either side of the quotient map.

## Compare ideals before and after quotienting

Let $R=\mathbb Q[x,y]$, and consider the principal ideals $I=(x)$ and $J=(x(1+y))$. The generator of $J$ is a multiple of $x$. Could the reverse inclusion hold as well?

```atlas
R = ring("x", "y", coeffs=QQ)
f = x
g = x*(1+y)
I = ideal(f)
J = ideal(g)
print(membership(g, I).value)                          # True
print(membership(f, J).value)                          # False
```

Thus $J\subsetneq I$. There is also a quick way to see the failure of the second inclusion by hand. Set $y=-1$. Every element of $J$ then becomes zero, while $x$ remains $x$.

Now pass to $K=R/(y^2)$. The names `x` and `y` still denote the original polynomials. Let `u` and `v` denote their residue classes. Create the two ideals again, this time inside $K$.

```atlas
K = quotient_ring(ideal(y^2))
u, v = K.coordinates
IK = ideal(u)
JK = ideal(g, ring=K)
print(IK.parent == K, JK.parent == K)                  # True True
print(membership(u*(1+v), IK).value)                   # True
print(membership(u, JK).value)                         # True
print((1+v)*(1-v) == 1)                                # True
```

These ideals are equal. The extra relation makes $1+v$ a unit with inverse $1-v$. Consequently $u=u(1+v)(1-v)$ lies in `JK`.

Both constructors are useful. In `ideal(u)` the generator already has parent `K`. In `ideal(g, ring=K)` Atlas takes the class of the ambient polynomial `g`. To test equality for ideals with more generators, test every generator for membership in the other ideal, in both directions. The expression `IK == JK` compares saved ideal objects. [Ideals in a quotient ring](rings.md#create-an-ideal-in-a-quotient-ring)

## Compute with residue classes

On the hyperbola $xy=1$, what is the inverse of $x$? In the quotient, arithmetic with `u` and `v` uses the defining equation automatically.

```atlas
x, y = coordinates("x", "y")
A = quotient_ring(ideal(x*y-1))
u, v = A.coordinates
print(u.inverse())                                     # y
print(A(x^2*y+1))                                      # x+1
print((u^3/u).representative)                          # x^2
```

The first line of output is a class in `A`, where $y$ really is an inverse of $x$. The last is a polynomial in `A.ambient`. Its spelling can look identical to that of a class. Use `A(f)` to project an ambient polynomial and `a.representative` to get the reduced ambient polynomial for a class `a`. The chosen monomial order determines this representative.

Division and negative powers require a unit. For example, in the dual numbers $D=\mathbb Q[s]/(s^2)$, $s$ is nilpotent and $1+s$ is invertible.

```atlas
s = coordinates("s")
D = quotient_ring(ideal(s^2))
a, = D.gens
print(a^2 == 0, a.is_unit())                           # True False
print((1+a).inverse())                                 # -s+1
```

Suppose we would like to substitute $s=z^2$ in a residue class of `D`. For substitution to be well-defined, the proposed image of $s$ must square to zero. This holds in $\mathbb Q[z]/(z^3)$. Give `hom` one image for each source generator, in order.

```atlas
z = coordinates("z")
B = quotient_ring(ideal(z^3))
b, = B.gens
phi = hom(D, B, images=(b^2,))
print(phi(a) == b^2, phi(a^2) == 0)                     # True True
```

The map checks the defining relation and can be applied to any element of `D`. The built-in projection `A.projection` is itself a ring map, from `A.ambient` to `A`. [Ring maps and substitutions](rings.md#move-a-polynomial-between-rings) · [Methods on residue classes](../reference/saved-values.md#quotient-rings-and-residue-classes)

## Localize at a prime

Work at the prime $P=(x)$ in $R=\mathbb Q[x,y]$. Every polynomial outside $P$ becomes invertible in $R_P$, including $y$. Consequently $(x^2,xy)R_P$ contains $x=(xy)/y$. Which polynomials of $R$ belong to this extended ideal?

```atlas
x, y = coordinates("x", "y")
P = ideal(x)
L = localize(ideal(x^2, x*y), P)
print(L.contraction.generators)                        # (x,)
q = L.fraction(x, y)
print(q.member, q.unit)                                # True False
print(localize(P).fraction(y+1, y).unit)               # True
```

The contraction is $(x)$. The fraction $x/y$ belongs to the extended ideal and is a nonunit in the local ring. The fraction $(y+1)/y$ is a unit since its numerator also lies outside $P$.

Supply a proper prime for `P`. Each denominator must lie outside it. Localizing at the zero prime gives the rational function field. If you already have a fraction at the time of construction, pass `numerator=` and `denominator=` to `localize` and read the result in `L.element`. [Local fractions](../reference/saved-values.md#local-fractions) · [Localization algorithm](../../native/ideal-algorithms.md#localization-at-a-prime-over-the-rationals)

## Normalize the cusp

The cusp $y^2=x^3$ has parametrization $(t^2,t^3)$. How can we recover the parameter from its coordinate ring? In the function field, $t=y/x$ satisfies the monic equation $t^2-x=0$, so $t$ is integral. Adjoining it gives the polynomial ring $\mathbb Q[t]$, the normalization of the cusp.

```atlas
x, y = coordinates("x", "y")
N = normalization(ideal(y^2-x^3))
C = N.components[0]
print(C.coordinates, C.relations.generators)           # (n1,) ()
print(C.images)                                        # (n1^2, n1^3)
print(C.fractions[0].numerator, C.fractions[0].denominator)  # y x
```

Atlas calls the parameter `n1`. The empty tuple of relations means it satisfies no polynomial relation over $\mathbb Q$. The images of $x$ and $y$ are its square and its cube, and the last line recovers it as the fraction $y/x$.

For a reducible input, read one entry of `N.components` per irreducible component. Each `C.prime` defines the source component, and `C.map` sends its residue classes to the algebra presented by `C.coordinates` and `C.relations`. Atlas normalizes the reduction, with `N.input_reduced` recording whether the input was already reduced. The algorithm notes describe the special treatment of monomial curves and the general method using the Grauert–Remmert criterion. [Atlas normalization](../../native/ideal-algorithms.md#normalization-of-affine-algebras-over-the-rationals) · [Greuel, Laplagne and Seelisch, *Normalization of Rings*](https://arxiv.org/abs/0904.3561)

## Read the graded Hilbert series

Give $x$ and $y$ degree one. In $\mathbb Q[x,y]/(x^2,y^3)$ a monomial survives precisely when its exponent of $x$ is at most one and its exponent of $y$ is at most two. Grouping these six monomials by degree gives the series $1+2t+2t^2+t^3$.

```atlas
x, y = coordinates("x", "y")
H = hilbert_series(ideal(x^2, y^3))
print(H)                                               # (1 - t^2 - t^3 + t^5)/(1 - t)^2
print(H.dimension, H.multiplicity)                     # 0 6
```

Atlas displays the numerator over $(1-t)^2$, one denominator factor for each ambient variable. Canceling the factors of $1-t$ gives our polynomial. The quotient has Krull dimension zero and length six. To use the coefficients in a script, read the degree and coefficient of each term in `H.numerator` and the exponent in `H.denominator_power`.

The ideal must be homogeneous for the standard grading. Its particular generators may be inhomogeneous if they generate a homogeneous ideal, as in $(x+y^2,y^2)=(x,y^2)$. [Graded Hilbert series algorithm](../../native/local-algorithms.md#hilbert-series-of-homogeneous-quotients)
