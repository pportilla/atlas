# Components, localization and normalization

These functions describe an ideal over $\mathbb Q$ through its components: the radical, a primary decomposition, the localization at a prime and the normalization of the quotient. [Recover the reduced support](../tutorial/polynomials-and-ideals.md#recover-the-reduced-support) · [Quotient rings, localization and normalization](../tutorial/quotient-rings.md) · [Polynomial, ideal and quotient algorithms](../../native/ideal-algorithms.md)

## Decompose an ideal

The ideal $(xy,x^2)$ defines the line $x=0$ with an embedded point at the origin.

```atlas
R = ring("x", "y")
J = ideal(x*y, x^2)
print(radical(J).generators)                          # (x,)
D = primary_decomposition(J)
print(len(D.components), D.components[1].primary.generators)   # 2 (x^2, y)
print(localize(ideal(x*y), ideal(x)).contraction.generators)   # (x,)
print(normalization(ideal(y^2 - x^3)).components[0].images)    # (n1^2, n1^3)
```

| To obtain | Write |
| --- | --- |
| The radical | `radical(I)` |
| Primary components and their associated primes | `primary_decomposition(I).components`, then `.primary` and `.prime` |
| The extension of an ideal to a local ring and its contraction | `localize(I, P)`, then `.contraction` |
| Whether a fraction lies in the localized ideal | `localize(I, P).fraction(f, s).member` |
| The normalization and its maps | `normalization(I).components` |
