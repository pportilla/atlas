# Local algebras and map germs

A polynomial equation may describe several points when only one is of interest. Give a local operation a rational point `point=(...)` in coordinate order, or omit it to work at the origin. A factor that is nonzero at the chosen point becomes invertible in its local ring.

## Count global and local lengths

The equations $x^2(x-1)=0$ and $y=0$ define a double point at $(0,0)$ and a simple point at $(1,0)$. The global quotient sees them both.

```atlas
x, y = coordinates("x", "y")
I = ideal(x^2*(x-1), y, order="lex")
G = quotient_basis(I)
L = local_quotient_basis(I)
print(G.basis, L.basis)                                    # (1, x, x^2) (1, x)
print(G.dimension, L.dimension)                            # 3 2
print(local_quotient_basis(I, point=(1,0)).dimension)       # 1
```

Globally $x^3=x^2$. At the origin $x-1$ is a unit, so the relation simplifies to $x^2=0$. At $(1,0)$ it is $x$ that becomes a unit, leaving $x=1$. The two local lengths add up to the global length $3$.

`quotient_basis(I).dimension` counts basis monomials. `multiplicity(I).value` returns that count without listing the monomials. `dimension(I).value` gives the Krull dimension, which is zero in this example. For a quotient of infinite vector-space dimension the basis result has `finite=False` and both `basis` and `dimension` are `None`. A zero algebra has an empty basis. [Quotient bases and lengths](../native/local-algorithms.md#quotient-bases-and-lengths-at-the-origin)

## Compare the cusp and its parametrization

For the cusp $y^2=x^3$ at the origin, the Milnor and Tjurina numbers are both two. We can see where that answer comes from by asking for a basis of the local Jacobian algebra.

```atlas
x, y = coordinates("x", "y")
f = y^2-x^3
A = local_jacobian_algebra(f)
print(A.generators)                                        # (-3*x^2, 2*y)
print(A.basis)                                             # (1, x)
print(milnor_number(f).value, tjurina_number(f).value)      # 2 2
```

The two partial derivatives kill $x^2$ and $y$, leaving the classes of $1$ and $x$. Adding $f$ itself to this ideal leaves the same quotient, which explains why the Tjurina number agrees here.

The same curve is the image of the map germ $t\mapsto(t^2,t^3)$. To study deformations of this *map*, enter its two components as polynomials in the source coordinate $t$.

```atlas
t = coordinates("t")
r = ae_codimension(t^2, t^3)
print(r.finite, r.value)                                   # True 1
```

The extended left-right codimension is one. There is one independent infinitesimal deformation of the parametrization modulo changes of source and target coordinates. Mond discusses this quotient and works out the same example in Section 1.1. [Mond, *Some open problems in the theory of singularities of mappings*](https://www.journalofsingularities.com/volume12/mond.pdf)

Pass components in target coordinate order, separately or as a list, including any zero components. All must vanish at the origin. Read `r.finite` alongside `r.value`. A value of `True` certifies a finite codimension and `False` certifies infinity. If `r.finite` is `None`, the calculation is inconclusive. The fields `r.lower_bound`, `r.jet_degree` and `r.reason` report what was established. For larger examples, the notes describe the weighted and filtered calculations, including the corank-two germ $\widehat B_7$ of codimension $837$ at $\alpha=-1$. [Map-germ algorithms](../native/map-germ-algorithms.md) · [Published examples and timings](../benchmarks/map-germ-timings.md#published-map-germs)

## Evaluate one curve at three points

The plane curve $y^2=(x-1/2)^3(x+1)^2$ contains two singular points. At the first, the right-hand side vanishes to order three. At the second, it vanishes to order two. Compare these points with a smooth point of the same curve.

```atlas
x, y = coordinates("x", "y")
f = y^2-(x-1/2)^3*(x+1)^2
for p in [(1/2,0), (-1,0), (3/2,5/2)]:
    print(milnor_number(f, point=p).value, tjurina_number(f, point=p).value)
```

| Point in loop order | Milnor number | Tjurina number |
| --- | ---: | ---: |
| $(1/2,0)$, cusp | 2 | 2 |
| $(-1,0)$, node | 1 | 1 |
| $(3/2,5/2)$, smooth | 0 | 0 |

At the smooth point a partial derivative is a unit, so the local Jacobian algebra is zero. At each singularity a nontrivial quotient remains. The two numbers agree at all three points in this example, but they can differ. For $x^4+y^5+x^2y^3$ at the origin they are $12$ and $11$, respectively.

Use a point on the hypersurface. The field `finite` on the corresponding local algebra distinguishes a finite-dimensional algebra from one with infinite dimension, such as the Jacobian algebra of $x^2$ in two variables. [Hypersurface invariants at rational points](../native/local-algorithms.md#quotients-and-hypersurface-invariants-at-rational-points)

## Reduce at a rational point

For $I=((x-1/2)(x-2),y+2)$, what is $x+y$ modulo $I$ near $(1/2,-2)$? Save a local standard basis at that point if you intend to reduce several expressions there.

```atlas
x, y = coordinates("x", "y")
I = ideal((x-1/2)*(x-2), y+2)
B = local_standard_basis(I, point=(1/2,-2))
h = local_normal_form(x+y, B)
print(h.remainder, h.unit)                                 # -3/2 1
print(local_normal_form(x+y, I, point=(2,-2)).remainder)    # 0
```

At either point the local quotient is just $\mathbb Q$, so the remainder is the value of $x+y$ there. In a positive-dimensional quotient, a reduction can also produce a nonconstant unit.

```atlas
h = local_normal_form(x, ideal((1-y)*x-y^2))
print(h.remainder, h.unit)                                 # y^2 -y+1
```

The local relation is $(1-y)x=y^2$. Since $1-y$ is invertible at the origin, the pair describes $x$ by the fraction $y^2/(1-y)$. In general the fields `remainder` and `unit` give $r$ and $u$ with $uf-r$ in the localized ideal and $u(p)=1$. A zero remainder tests membership. With `reduction="weak"`, only the leading monomial of a nonzero remainder is guaranteed to be irreducible. For details on local division, see the algorithm notes and the introduction by Mora, Pfister and Traverso. [Atlas local normal forms](../native/local-algorithms.md#local-standard-bases-and-normal-forms-at-rational-points) · [*An Introduction to the Tangent Cone Algorithm*](https://www.numdam.org/item/PSMIR_1989___4_133_0/)
