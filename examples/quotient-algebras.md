# Quotient rings, localization and normalization

Start with an ideal in $\mathbb Q[x_1,\dots,x_n]$. Declare a quotient to compute with residue classes, localize to make chosen denominators invertible, or normalize to describe the integral closure on each irreducible component.

## Compare ideals before and after quotienting

Can two distinct ideals have the same image in a quotient ring? Start in the
polynomial ring $R=\mathbb Q[x,y]$ with the principal ideals $I=(x)$ and
$J=(x(1+y))$. The declaration below names the ring and binds its coordinates
in the given order. Enter both blocks in the same session or save them together
as an Atlas script.

```atlas
R = ring("x", "y", coeffs=QQ)
f = x
g = x*(1+y)
I = ideal(f)
J = ideal(g)
print(membership(g, I).value)           # True
print(membership(f, J).value)           # False
```

Each ideal here has one generator, so the two membership tests check both
inclusions. The first says $J\subseteq I$. The second gives a witness that
the inclusion is strict. Substituting $y=-1$ sends every multiple of $g$ to
zero and sends $f$ to $x$.

Now impose $y^2=0$. The names `x` and `y` still belong to $R$, while
`K.coordinates` gives their classes in $K=R/(y^2)$. Apply `K` to take the
class of a polynomial, or pass `ring=K` to `ideal` to project the generators
as you build the ideal.

```atlas
K = quotient_ring(ideal(y^2))
u, v = K.coordinates
IK = ideal(K(f))
JK = ideal(g, ring=K)
print(IK.parent == K, JK.parent == K)  # True True
print(membership(K(g), IK).value)      # True
print(membership(K(f), JK).value)      # True
print((1+v)*(1-v) == 1)                # True
```

Both inclusions now hold, so $IK=JK$. The last line explains the change.
Since $v^2=0$, the factor $1+v$ has inverse $1-v$ in $K$. Multiplying the
generator $u(1+v)$ by $1-v$ recovers $u$. For an ideal with several
generators, check membership of **each** generator in the other ideal in
both directions. The expression `IK == JK` checks whether two names refer
to the same saved ideal object. [Quotient ideal operations](../docs/sessions.md#create-an-ideal-in-a-quotient-ring)

## Compute with residue classes

In $\mathbb Q[x,y]/(xy-1)$ the classes of $x$ and $y$ are inverses. In $\mathbb Q[x,y]/(x^2)$ the class of $x$ is nilpotent, and $1+x$ has inverse $1-x$.

```atlas
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(x*y-1)
A = I.quotient_ring
u, v = A.coordinates
print(u.inverse() == v)                  # True
print(A(x^2*y+1) == u+1)                 # True
print((u^3/u).representative == x^2)     # True

D = quotient_ring(ideal(x^2))
t = D(x)
print(t^2 == 0, t.is_unit())             # True False
print((1+t).inverse() == 1-t)            # True
```

`A.coordinates` and `A.gens` give the residue classes in the quotient. Use `A.ambient.coordinates` for the original polynomial generators. In arithmetic with a class, Atlas projects scalars and polynomials from `A.ambient` into `A`. For equality, project with `A(f) == u` or `A.projection(f) == u`. The projection has domain `A.ambient` and codomain `A`. Each class has a `.parent` and a canonical ambient `.representative` for the chosen monomial order. Division and negative powers require a unit. In the zero ring $0=1$ and its sole element is its own inverse. [Arithmetic and projection checks](../tests/test_session_quotient_ring.py) · [C++ quotient ring](../native/quotient_ring.cpp)

To move elements through a ring map, use `hom(domain, codomain, images=...)` and give the images of the domain generators in order. A map out of a quotient must send its defining relations to zero. For example, $a\mapsto b^2$ defines a map $\mathbb Q[s]/(s^2)\to\mathbb Q[z]/(z^3)$ because $b^4=0$.

```atlas
s = coordinates("s")
z = coordinates("z")
A = quotient_ring(ideal(s^2))
B = quotient_ring(ideal(z^3))
a, = A.gens
b, = B.gens
phi = hom(A, B, images=(b^2,))
print(phi(a) == b^2, phi(a^2) == 0)    # True True
```

In the first example `D(u.representative)` evaluates a chosen representative in another quotient of the same polynomial ring. To define a map on *all* residue classes, supply images that satisfy the relations as above. [Ring-map tests](../tests/test_session_maps.py) · [Polynomial ring maps in a session](../docs/sessions.md#move-a-polynomial-between-rings)

## Localize at a prime

For $R=\mathbb Q[x,y]$, localization at the prime $P=(x)$ makes $y$ invertible. The ideal $I=(x^2,xy)$ then generates $(x)$ in $R_P$. Atlas returns the contraction $IR_P\cap R$ in the original coordinates and can test fractions in the local ring.

```atlas
x, y = coordinates("x", "y")
P = ideal(x)
L = localize(ideal(x^2, x*y), P)
print(L.contraction.generators == (x,))  # True
q = L.fraction(x, y)
print(q.member, q.unit)                  # True False
print(localize(P).fraction(y+1, y).unit) # True
```

`member` tests membership in $IR_P$, and `unit` tests invertibility in $R_P$. Atlas verifies that $P$ is proper and prime. Each denominator must lie outside $P$. At the zero prime the local ring is the rational function field. Pass `numerator=` and `denominator=` to the original localization to record a fraction in `L.element`. The spelling `localization(I,P)` invokes the same function. [Fraction and contraction tests](../tests/test_localization_interface.py) · [C++ localization](../native/localization.cpp)

## Normalize the cusp

In the fraction field of $\mathbb Q[x,y]/(y^2-x^3)$, let $t=y/x$. It satisfies $t^2=x$ and $t^3=y$. Adjoining $t$ gives the normalization $\mathbb Q[t]$.

```atlas
x, y = coordinates("x", "y")
N = normalization(ideal(y^2-x^3))
C = N.components[0]
X, Y = C.images
print(N.input_reduced)                            # True
print(membership(Y^2-X^3, C.relations).value)     # True
print(membership(y^2-x^3, C.prime).value)         # True
t = C.coordinates[0]
phi = C.map
print(phi(phi.domain(x)) == phi.codomain(t^2))    # True
print(C.fractions[0].numerator, C.fractions[0].denominator)  # y x
```

The product of the returned component algebras is the normalization of the reduction. For each component, `C.prime` is in the source coordinates, and `C.coordinates` modulo `C.relations` present the target algebra. The live homomorphism `C.map` sends classes modulo `C.prime` to classes modulo `C.relations`. The coordinate images follow source generator order. `C.fractions` expresses the target generators as numerator/denominator pairs on that source component. A denominator may vanish at points of the component but is nonzero in its function field. For an input with nilpotents, `N.input_reduced` is false. The zero ring has no components. [Cusp image and fraction tests](../tests/test_normalization_interface.py) · [Normalization map tests](../tests/test_session_ring_ownership.py) · [C++ normalization](../native/normalization.cpp)

## Read the graded Hilbert series

With all variables in degree one, the coefficient of $t^d$ counts degree-$d$ classes of a homogeneous quotient. In $\mathbb Q[x,y]/(x^2,y^3)$ the six standard monomials have degrees $0,1,1,2,2,3$, so the Hilbert series is $1+2t+2t^2+t^3$.

```atlas
x, y = coordinates("x", "y")
H = hilbert_series(ideal(x^2, y^3))
print(H.dimension, H.multiplicity, H.denominator_power)  # 0 6 2
print(H.numerator[1].degree, H.numerator[1].coefficient) # 2 -1
```

Atlas records the numerator against $(1-t)^n$ for $n$ ambient variables, before cancellation. Here it is $1-t^2-t^3+t^5$. The zero ideal has numerator $1$, and the unit ideal has an empty numerator, dimension $-1$ and multiplicity $0$. The ideal must be homogeneous. For example, $(x+y^2,y^2)=(x,y^2)$ qualifies even with the first generating set. [Graded series checks](../tests/test_session_hilbert_series.py) · [C++ monomial counts](../native/monomial_quotient.cpp)
