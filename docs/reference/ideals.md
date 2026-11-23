# Gröbner bases and ideal operations

These functions take ideals of a polynomial or quotient ring over $\mathbb Q$ and return a basis, a normal form, a membership test or another ideal of the same ring. A saved Gröbner basis can stand in for its ideal, and the monomial order travels with it. [Polynomials and ideals](../tutorial/polynomials-and-ideals.md) · [The exact kernel](../../native/KERNEL.md) · [Polynomial, ideal and quotient algorithms](../../native/ideal-algorithms.md)

## Reduce, test and eliminate

```atlas
R = ring("t", "x", "y")
I = ideal(x^2 - y, x*y - 1)
print(groebner(I).generators)                         # (x^2-y, x*y-1, y^2-x)
print(normal_form(x^3, I))                            # 1
print(membership(x^3 - 1, I).value)                   # True
print(eliminate(ideal(x - t^2, y - t^3), t).generators)   # (x^3-y^2,)
print(intersection(ideal(x), ideal(y)).generators)    # (x*y,)
```

| To obtain | Write |
| --- | --- |
| An ideal, optionally with a monomial order | `ideal(f, g)`, `ideal(f, g, order="lex")` |
| Its reduced Gröbner basis | `groebner(I).generators` |
| The remainder of a polynomial | `normal_form(f, I)` |
| Whether a polynomial lies in the ideal | `membership(f, I).value` |
| The ideal after eliminating coordinates | `eliminate(I, t)` |
| Intersection, colon ideal and saturation | `intersection(I, J)`, `quotient(I, J)`, `saturation(I, f)` |
| Sum, product and powers | `I + J`, `I * J`, `I^n`, `symbolic_power(I, n)` |
