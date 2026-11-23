# Local algebra at a point

These functions work in the local ring at a rational point. Give the point with `point=(...)` in coordinate order, or omit it to work at the origin. A saved local standard basis remembers its point. [Local algebras and map germs](../tutorial/local-algebra.md) · [Local algebra algorithms](../../native/local-algorithms.md)

## Compare global and local lengths

The ideal $(x^2(x-1),y)$ defines a double point at the origin and a simple point at $(1,0)$. Its global length is three, and its local length at the origin is two.

```atlas
R = ring("x", "y")
I = ideal(x^2*(x - 1), y)
print(quotient_basis(I).dimension)                      # 3
print(local_quotient_basis(I).dimension)                # 2
print(local_quotient_basis(I, point=(1, 0)).dimension)  # 1
C = ideal(y^2 - x^3)
print(hilbert_samuel_function(C, degrees=[0, 1, 2, 3]).values)   # [1, 3, 5, 7]
print(hilbert_samuel_multiplicity(C).value)             # 2
```

| To obtain | Write |
| --- | --- |
| A standard basis in the local ring | `local_standard_basis(I, point=p)` |
| A local remainder and the unit that produces it | `local_normal_form(f, I, point=p)` |
| A basis and the length of the local quotient | `local_quotient_basis(I, point=p)` |
| The Hilbert–Samuel function and multiplicity | `hilbert_samuel_function(I)`, `hilbert_samuel_multiplicity(I)` |
