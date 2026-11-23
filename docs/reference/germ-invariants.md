# Hypersurface and map germ invariants

These functions compute invariants of a hypersurface germ at a rational point, namely its local Jacobian algebra and its Milnor and Tjurina numbers, and the extended left-right codimension of a map germ at the origin. For the resolution and zeta functions of a hypersurface germ, see [Resolution, zeta functions and monodromy](resolution.md). [Compare the cusp and its parametrization](../tutorial/local-algebra.md#compare-the-cusp-and-its-parametrization) · [Local algebra algorithms](../../native/local-algorithms.md#quotients-and-hypersurface-invariants-at-rational-points) · [Certificates for the extended tangent quotient](../../native/map-germ-algorithms.md)

## Invariants of a curve and of its parametrization

```atlas
R = ring("x", "y")
f = y^2 - x^3
print(milnor_number(f).value, tjurina_number(f).value)    # 2 2
print(local_jacobian_algebra(f).basis)                    # (1, x)
g = x^4 + y^5 + x^2*y^3
print(milnor_number(g).value, tjurina_number(g).value)    # 12 11
T = ring("t")
print(ae_codimension(t^2, t^3).value)                     # 1
```

| To obtain | Write |
| --- | --- |
| The Milnor number $\mu$ | `milnor_number(f, point=p).value` |
| The Tjurina number $\tau$ | `tjurina_number(f, point=p).value` |
| A basis of the local Jacobian algebra | `local_jacobian_algebra(f).basis` |
| The $\mathcal A_e$-codimension of a map germ | `ae_codimension(f1, f2, ...).value` |
