# Coordinates, polynomials and ring maps

A ring owns its ordered coordinates, and every polynomial remembers the ring it belongs to. The functions on this page declare polynomial rings over $\mathbb Q$ and their quotients, name and select them in the workspace, read, divide and factor polynomials, and define maps between rings. [Rings, coordinates and maps](../tutorial/rings.md) · [Polynomials and ideals](../tutorial/polynomials-and-ideals.md)

## Declare a ring and compute in it

`ring` declares the ring and binds its coordinates to the same names. Polynomials are then written directly.

```atlas
R = ring("x", "y")                          # Q[x, y], with x and y bound
f = (x + y)^2 * (x - y)
print(f.parent == R)                        # True
print(factor(f))                            # 1*(x-y)*(x+y)^2
print(gcd(f, x^2 - y^2))                    # x^2-y^2
print(polynomial("x^2 + y/2", ring=R))      # x^2+1/2*y
S = ring("t")
phi = hom(R, S, images=(t^2, t^3))          # x -> t^2, y -> t^3
print(phi(y^2 - x^3))                       # 0
```

| To obtain | Write |
| --- | --- |
| A polynomial ring and its coordinates | `R = ring("x", "y")` or `x, y = coordinates("x", "y")` |
| A polynomial from a string | `polynomial("x^2 + y", ring=R)` |
| The ring of a polynomial | `f.parent` |
| A default ring for strings and numbers | `use_ring(R)`, `active_ring()` |
| The saved name of a ring, or its deletion | `ring_name(R)`, `delete_ring(R)` |
| A factorization over $\mathbb Q$ or over $\overline{\mathbb Q}$ | `factor(f)`, `absolute_factor(f)` |
| A greatest common divisor or a resultant | `gcd(f, g)`, `resultant(f, g, x)` |
| Division with remainder | `polynomial_division(f, g)` |
| A quotient ring | `quotient_ring(ideal(f))` |
| A ring map given by the images of the coordinates | `hom(R, S, images=(...))` |
| Exact linear algebra over $\mathbb Q$ | `matrix_analysis(rows)` |
| Arithmetic modulo an integer | `modular_arithmetic(action, values, modulus)` |
