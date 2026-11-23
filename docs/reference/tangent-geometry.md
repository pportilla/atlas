# Tangent geometry and smoothness

These functions describe a scheme near a rational point on it: the tangent cone and its graded series, the Zariski tangent space, the embedding and local dimensions, and smoothness. Give the point with `point=(...)`, or omit it to work at the origin. [Geometry at a rational point](../tutorial/local-geometry.md) · [Local algebra algorithms](../../native/local-algorithms.md)

## Look at the cusp near two points

```atlas
R = ring("x", "y")
I = ideal(y^2 - x^3)
print(tangent_cone(I).generators)                     # (y^2,)
print(zariski_tangent_space(I).dimension)             # 2
print(embedding_dimension(I).value, local_dimension(I).value)   # 2 1
print(local_smoothness(I).value)                      # False
print(local_smoothness(I, point=(1, 1)).value)        # True
```

| To obtain | Write |
| --- | --- |
| The tangent cone and its Hilbert series | `tangent_cone(I)`, `tangent_cone_hilbert_series(I)` |
| The Zariski tangent space | `zariski_tangent_space(I)` |
| The embedding dimension and the local dimension | `embedding_dimension(I).value`, `local_dimension(I).value` |
| Whether the scheme is smooth at the point | `local_smoothness(I, point=p).value` |
