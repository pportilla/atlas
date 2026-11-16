# Geometry at a rational point

Choose a point in the declared coordinate order. Use `local_dimension(I)` to find the dimension at the origin. To distinguish smooth and singular points, compare the local ring with its tangent space. To retain higher-order information, compute the tangent cone and its graded dimensions.

## Compare local dimension and smoothness

The ideal $I=(xy,xz)$ cuts out a plane $x=0$ and a line $y=z=0$ in affine three-space. Each component is smooth away from the origin. At their intersection the local ring has dimension two, and its tangent space is all of affine three-space.

```atlas
x, y, z = coordinates("x", "y", "z", coeffs=QQ)
I = ideal(x*y, x*z)
for p in [(0,1/2,0), (2/3,0,0), (0,0,0)]:
    s = local_smoothness(I, point=p)
    print(s.local_dimension, s.embedding_dimension, s.value)
```

| Point in loop order | Local dimension | Embedding dimension | Smooth |
| --- | ---: | ---: | --- |
| $(0,1/2,0)$ on the plane | 2 | 2 | True |
| $(2/3,0,0)$ on the line | 1 | 1 | True |
| $(0,0,0)$ at the intersection | 2 | 3 | False |

Every generator must vanish at the selected point. The smoothness result also gives the Jacobian rank. A saved local standard basis can supply the ideal and its point. For a saved tangent cone, use its vertex. [Plane and line tests](../tests/test_local_dimension_interface.py) · [C++ dimension calculation](../native/point_geometry.cpp)

## Compute the tangent space

Take the translated cusp $I=((y-2)^2-(x-1/2)^3)$. The Zariski tangent space consists of directions $v$ with $df_p(v)=0$ for every ideal generator $f$. Its dimension is the embedding dimension. At the cusp $p=(1/2,2)$ the gradient vanishes, so both coordinate directions survive. At $q=(3/2,3)$, the gradient gives $3\Delta x-2\Delta y=0$.

```atlas
x, y = coordinates("x", "y")
I = ideal((y-2)^2-(x-1/2)^3)
V = zariski_tangent_space(I, point=(3/2,3))
vx, vy = V.basis[0]
print(V.dimension, 3*vx-2*vy == 0)                       # 1 True
print(embedding_dimension(I, point=(1/2,2)).value)       # 2
```

`V.equations` gives independent linear equations in the direction coordinates `V.coordinates`. Each entry of `V.basis` is a vector of exact rationals in that order. Equations retain scheme structure. For example, $(x^2)$ and $(x)$ have embedding dimensions one and zero respectively at the origin of the affine line. [Tangent vector checks](../tests/test_session_zariski_tangent.py) · [C++ point and Jacobian arithmetic](../native/point_geometry.cpp)

## Compute the tangent cone

The tangent cone has coordinate ring $\operatorname{gr}_{\mathfrak m}(A)$ for the local ring $A$ at the point. For the translated cusp at $(1/2,2)$, the lowest-degree term gives the double line $(\Delta y)^2=0$. Its tangent space in the preceding example had no linear constraints, while the cone detects multiplicity two.

```atlas
x, y = coordinates("x", "y")
I = ideal((y-2)^2-(x-1/2)^3)
C = tangent_cone(I, point=(1/2,2))
u, v = C.coordinates
print(C.generators[0] == v^2, C.dimension, C.multiplicity)  # True 1 2
S = tangent_cone_hilbert_series(C)
print(S.numerator[1].degree, S.numerator[1].coefficient)     # 2 -1
print(S.denominator_power)                                 # 2
```

The cone's generator prints as `y^2` because Atlas reuses the coordinate names in its tangent ring. Compare it to $v^2$ formed with `C.coordinates`. A cone passed to a local operation uses the vertex as its point. Its series is $(1-t^2)/(1-t)^2$. At the smooth point $(3/2,3)$ the cone is a single line and the series is $(1-t)/(1-t)^2$. [Cone and series tests](../tests/test_session_tangent_cone.py) · [C++ initial forms](../native/point_geometry.cpp) · [C++ series calculation](../native/monomial_quotient.cpp)

## Sample the Hilbert-Samuel function

For the maximal ideal $\mathfrak m$ of the local ring $A$, the cumulative Hilbert–Samuel function is $H(n)=\dim_{\mathbb Q} A/\mathfrak m^{n+1}$. Its successive differences $h(n)=H(n)-H(n-1)$, with $H(-1)=0$, are the coefficients of the Hilbert series of the tangent cone. Read `values` for $H$ and `graded_values` for $h$.

```atlas
x, y = coordinates("x", "y")
I = ideal((y-2)^2-(x-1/2)^3)
H = hilbert_samuel_function(I, point=(1/2,2), degrees=[5,0,1])
print(H.values, H.graded_values)                            # [11, 1, 3] [2, 1, 2]
print(hilbert_samuel_multiplicity(I, point=(3/2,3)).value)  # 1
```

| Point | $H(0),H(1),H(2),H(5)$ | $h(0),h(1),h(2),h(5)$ | Multiplicity |
| --- | --- | --- | ---: |
| $(1/2,2)$, cusp | 1, 3, 5, 11 | 1, 2, 2, 2 | 2 |
| $(3/2,3)$, smooth point | 1, 2, 3, 6 | 1, 1, 1, 1 | 1 |

The default degrees run from zero through five. You can supply nonnegative degrees in any order, including repeated degrees. An empty list returns dimension and multiplicity with empty value lists. For local dimension $d$, the multiplicity is $d!$ times the leading coefficient of the eventual polynomial for $H(n)$. At dimension zero the eventual value equals the local length. Outside the support Atlas gives dimension $-1$, multiplicity zero and all function values zero. [Hilbert–Samuel value tests](../tests/test_session_hilbert_samuel.py) · [C++ graded counts](../native/monomial_quotient.cpp)
