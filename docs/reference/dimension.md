# Global dimension, length and Hilbert series

These functions measure the quotient of a polynomial ring by an ideal: its Krull dimension, a vector-space basis and its length when it is finite, and the graded Hilbert series of a homogeneous quotient. For the same questions in the local ring at a point, see [Local algebra at a point](local-algebra.md). [Count global and local lengths](../tutorial/local-algebra.md#count-global-and-local-lengths) · [Read the graded Hilbert series](../tutorial/quotient-rings.md#read-the-graded-hilbert-series) · [Local algebra algorithms](../../native/local-algorithms.md)

## Measure a quotient

```atlas
R = ring("x", "y")
print(dimension(ideal(x*y)).value)                    # 1
B = quotient_basis(ideal(x^2, y^3))
print(B.dimension, B.basis)                           # 6 (1, x, y, x*y, y^2, x*y^2)
print(multiplicity(ideal(x^2, y^3)).value)            # 6
print(hilbert_series(ideal(x^2, y^3)))                # (1 - t^2 - t^3 + t^5)/(1 - t)^2
```

| To obtain | Write |
| --- | --- |
| The Krull dimension of $R/I$ | `dimension(I).value` |
| A basis of $R/I$ over $\mathbb Q$, when it is finite | `quotient_basis(I).basis` |
| The length of $R/I$ | `multiplicity(I).value` |
| The Hilbert series of a homogeneous quotient | `hilbert_series(I)` |
