# Geometry at a rational point

Where is a variety singular, and what do its equations look like near a singular point? We begin with two components of different dimensions, then follow one plane cusp from its tangent space to its Hilbert–Samuel function. Give points in the order of the declared coordinates.

## Compare local dimension and smoothness

The equations $xy=xz=0$ cut out a plane $x=0$ together with the line $y=z=0$. Take one point on each component and then their intersection at the origin. A smooth point should have tangent space of the same dimension as its component.

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

At the intersection, both equations have zero linear part. Every direction is tangent there, even though the largest component has dimension two. If you only want the local dimension, call `local_dimension(I, point=p)`. For smoothness, `p` must lie on the scheme. A saved local standard basis supplies both the ideal and its point. [Local dimension and smoothness algorithms](../../native/local-algorithms.md#local-dimension-and-smoothness)

## Compute the tangent space

Consider the translated cusp $(y-2)^2=(x-1/2)^3$. The point $(3/2,3)$ is smooth. Linearizing the equation there gives $-3\Delta x+2\Delta y=0$. Ask for the equations or for a basis of tangent directions.

```atlas
x, y = coordinates("x", "y")
I = ideal((y-2)^2-(x-1/2)^3)
V = zariski_tangent_space(I, point=(3/2,3))
print(V.equations)                                         # (x-2/3*y,)
vx, vy = V.basis[0]
print(vx, vy)                                              # 2/3 1
print(embedding_dimension(I, point=(1/2,2)).value)         # 2
```

The output variables `x` and `y` in `V.equations` are direction coordinates. The vector $(2/3,1)$ spans the tangent line. At the cusp $(1/2,2)$ both partial derivatives vanish and the tangent space is the entire plane. Its dimension, the embedding dimension, depends on the local scheme structure. For instance, $(x^2)$ and $(x)$ define the same point on the affine line and have embedding dimensions one and zero there. [Tangent space calculation](../../native/local-algorithms.md#zariski-tangent-spaces)

## Compute the tangent cone

The tangent space at the cusp allowed every direction. The first *nonzero* homogeneous part of its equation tells us more. With the cusp as origin the equation is $v^2-u^3$, whose lowest-degree part is $v^2$.

```atlas
x, y = coordinates("x", "y")
I = ideal((y-2)^2-(x-1/2)^3)
C = tangent_cone(I, point=(1/2,2))
print(C.generators)                                        # (y^2,)
print(C.dimension, C.multiplicity)                         # 1 2
print(tangent_cone_hilbert_series(C))                      # (1 - t^2)/(1 - t)^2
```

The equation `y^2` describes a double line. Atlas has reused the coordinate names in a new ring, with origin at the vertex. Use `u, v = C.coordinates` to work with those directions. The graded coordinate ring is $\mathbb Q[u,v]/(v^2)$. It has one independent class in degree zero and two in each positive degree, as the displayed series confirms.

For an ideal with several equations, the cone uses the lowest-degree forms of **all** elements of the ideal. Taking just the initial forms of the given generators can miss relations. For general ideals Atlas computes a local standard basis to find these extra relations. [Tangent cones and graded series](../../native/local-algorithms.md#tangent-cones-and-graded-hilbert-series)

## Sample the Hilbert-Samuel function

How many classes survive if we discard terms above degree $n$ at a point? If $A$ is the local ring and $\mathfrak m$ its maximal ideal, this number is $H(n)=\dim_{\mathbb Q} A/\mathfrak m^{n+1}$. At the cusp degree zero contributes one class and each positive degree contributes two. We therefore expect $H(n)=1+2n$.

```atlas
x, y = coordinates("x", "y")
I = ideal((y-2)^2-(x-1/2)^3)
H = hilbert_samuel_function(I, point=(1/2,2), degrees=[0,1,2,5])
print(H.values, H.graded_values)                           # [1, 3, 5, 11] [1, 2, 2, 2]
print(hilbert_samuel_multiplicity(I, point=(3/2,3)).value)  # 1
```

`values` contains the cumulative lengths in the requested order. `graded_values` gives the contribution in each requested degree, even when some intermediate degrees were omitted. At the smooth point there is one contribution per degree and $H(n)=1+n$. The slopes of these two functions give multiplicities two and one, respectively.

By default `degrees` runs from zero through five. You can request nonnegative degrees in any order, including repeated degrees. In local dimension $d$, the multiplicity is $d!$ times the leading coefficient of the eventual polynomial for $H$. For a zero-dimensional local algebra, $H$ stabilizes at its length. Outside the support the multiplicity and all values are zero. [Local algebra algorithms](../../native/local-algorithms.md)
