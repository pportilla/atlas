# Constructions

This page collects short answers to questions of the form “How do I compute … in Atlas?” Each answer is a complete script with its output in comments. Paste it at the prompt in a fresh session, or save it in a file and run it with `./atlas run`. Follow the links for the mathematics behind a calculation and the full description of each function.

## Rings and polynomials

### Declare a polynomial ring

`ring` names the ring and binds its generators. The coefficient field is $\mathbb Q$.

```atlas
R = ring("x", "y", "z")
print(R.coordinates)                                       # (x, y, z)
print(((x + y)^2).parent == R)                             # True
```

[Rings, coordinates and maps](tutorial/rings.md#choose-the-ring-of-a-polynomial)

### Read a polynomial from a string

```atlas
R = ring("x", "y")
f = polynomial("y^2 - x^3", ring=R)
print(f == y^2 - x^3)                                      # True
```

[Choose the ring of a polynomial](tutorial/rings.md#choose-the-ring-of-a-polynomial)

### Substitute polynomials for the coordinates

A ring map sends each generator of its domain to an image, in order.

```atlas
R = ring("x", "y")
S = ring("t")
phi = hom(R, S, images=(t^2, t^3))
print(phi(x + y))                                          # t^3+t^2
print(phi(y^2 - x^3))                                      # 0
```

[Move a polynomial between rings](tutorial/rings.md#move-a-polynomial-between-rings) · [Ring homomorphisms](reference/saved-values.md#ring-homomorphisms)

### Factor a polynomial

```atlas
R = ring("x", "y")
F = factor(-3/5*(2*x - y)^2*(x^2 + 1))
print(F.unit)                                              # -12/5
print(F.factors[0].polynomial, F.factors[0].multiplicity)  # x^2+1 1
print(F.factors[1].polynomial, F.factors[1].multiplicity)  # x-1/2*y 2
```

[Factor and reconstruct a polynomial](tutorial/polynomials-and-ideals.md#factor-and-reconstruct-a-polynomial)

### Find a greatest common divisor or a resultant

```atlas
R = ring("t", "x", "y")
print(gcd((x + y)*(x + 1), (x + y)*(y + 1)).polynomial)    # x+y
print(resultant(t^2 - x, t^3 - y, t).polynomial)           # -x^3+y^2
```

[Find a common factor](tutorial/polynomials-and-ideals.md#find-a-common-factor) · [Eliminate a coordinate](tutorial/polynomials-and-ideals.md#eliminate-a-coordinate)

## Ideals

### Compute a Gröbner basis

The default monomial order is graded reverse lexicographic. Pass `order="lex"` for the lexicographic order.

```atlas
R = ring("x", "y")
I = ideal(x^2 - 1, y - x, order="lex")
print(groebner(I).generators)                              # (x-y, y^2-1)
```

[Reduce a polynomial modulo an ideal](tutorial/polynomials-and-ideals.md#reduce-a-polynomial-modulo-an-ideal)

### Reduce a polynomial or test membership

```atlas
R = ring("x", "y")
I = ideal(x^2 - y)
print(normal_form(x^3, I).polynomial)                      # x*y
print(membership(x^3 - x*y, I).value)                      # True
```

[Reduce a polynomial modulo an ideal](tutorial/polynomials-and-ideals.md#reduce-a-polynomial-modulo-an-ideal)

### Decide whether two ideals are equal

Two ideals of the same ring are equal when their reduced Gröbner bases in the same order agree.

```atlas
R = ring("x", "y")
I = ideal(x^2, x*y)
J = intersection(ideal(x), ideal(x^2, y))
print(groebner(I).generators == groebner(J).generators)    # True
```

To compare ideals of a quotient ring, test each generator of either ideal for membership in the other. [Compare ideals before and after quotienting](tutorial/quotient-rings.md#compare-ideals-before-and-after-quotienting)

### Eliminate a variable

```atlas
R = ring("t", "x", "y")
J = eliminate(ideal(t^2 - x, t^3 - y), t)
print(J.generators)                                        # (x^3-y^2,)
print(J.coordinates)                                       # (x, y)
```

[Eliminate a coordinate](tutorial/polynomials-and-ideals.md#eliminate-a-coordinate)

### Intersect, divide and saturate ideals

```atlas
R = ring("x", "y")
I = ideal(x^2, x*y)
print(intersection(ideal(x), ideal(y)).generators)         # (x*y,)
print(quotient(I, ideal(x)).generators)                    # (x, y)
print(saturation(I, x).generators)                         # (1,)
```

[Intersect, divide and saturate ideals](tutorial/polynomials-and-ideals.md#intersect-divide-and-saturate-ideals)

### Add, multiply and raise ideals to powers

```atlas
R = ring("x", "y")
I = ideal(x^2, y)
J = ideal(x, y^2)
print(groebner(I + J).generators)                          # (x, y)
print(groebner(I * J).generators)                          # (x^3, y^3, x*y)
print(groebner(J^2).generators)                            # (y^4, x*y^2, x^2)
```

[Add, multiply, and raise ideals to powers](tutorial/polynomials-and-ideals.md#add-multiply-and-raise-ideals-to-powers)

### Compute a symbolic power

```atlas
R = ring("x", "y", "z")
I = ideal(x*y, x*z, y*z)
print(membership(x*y*z, ideal_power(I, 2)).value)          # False
print(membership(x*y*z, symbolic_power(I, 2)).value)       # True
```

[Compare ordinary and symbolic powers](tutorial/polynomials-and-ideals.md#compare-ordinary-and-symbolic-powers)

### Find the radical and the primary components

```atlas
R = ring("x", "y")
I = ideal(x^2, x*y)
print(radical(I).generators)                               # (x,)
a, b = primary_decomposition(I).components
print(a.primary.generators, a.prime.generators)            # (x,) (x,)
print(b.primary.generators, b.prime.generators)            # (x^2, y) (x, y)
```

[Recover the reduced support](tutorial/polynomials-and-ideals.md#recover-the-reduced-support) · [Reconstruct an ideal from its primary components](tutorial/polynomials-and-ideals.md#reconstruct-an-ideal-from-its-primary-components)

### Find the dimension of the zero set

```atlas
R = ring("x", "y", "z")
print(dimension(ideal(x*y, x*z)).value)                    # 2
```

[Count global and local lengths](tutorial/local-algebra.md#count-global-and-local-lengths)

## Quotient rings

### Compute modulo relations

```atlas
R = ring("x", "y")
A = quotient_ring(ideal(x*y - 1))
u, v = A.coordinates
print(u.inverse())                                         # y
print(A(x^2*y + 1))                                        # x+1
print(A(x^2*y + 1).representative.parent == R)             # True
```

[Compute with residue classes](tutorial/quotient-rings.md#compute-with-residue-classes) · [Quotient rings and residue classes](reference/saved-values.md#quotient-rings-and-residue-classes)

### Find a basis of a finite-dimensional quotient

```atlas
R = ring("x", "y")
G = quotient_basis(ideal(x^2, y^3))
print(G.dimension)                                         # 6
print(G.basis)                                             # (1, x, y, x*y, y^2, x*y^2)
```

[Count global and local lengths](tutorial/local-algebra.md#count-global-and-local-lengths)

### Compute a Hilbert series

```atlas
R = ring("x", "y")
H = hilbert_series(ideal(x^2, y^3))
print(H)                                                   # (1 - t^2 - t^3 + t^5)/(1 - t)^2
print(H.dimension, H.multiplicity)                         # 0 6
```

[Read the graded Hilbert series](tutorial/quotient-rings.md#read-the-graded-hilbert-series)

### Localize at a prime

```atlas
R = ring("x", "y")
L = localize(ideal(x^2, x*y), ideal(x))
print(L.contraction.generators)                            # (x,)
print(L.fraction(x, y).member)                             # True
```

[Localize at a prime](tutorial/quotient-rings.md#localize-at-a-prime) · [Local fractions](reference/saved-values.md#local-fractions)

### Normalize a curve

```atlas
R = ring("x", "y")
C = normalization(ideal(y^2 - x^3)).components[0]
print(C.coordinates, C.images)                             # (n1,) (n1^2, n1^3)
```

[Normalize the cusp](tutorial/quotient-rings.md#normalize-the-cusp)

## Singularities and local algebra

### Compute Milnor and Tjurina numbers

```atlas
R = ring("x", "y")
f = x^4 + y^5 + x^2*y^3
print(milnor_number(f).value, tjurina_number(f).value)     # 12 11
```

[Compare the cusp and its parametrization](tutorial/local-algebra.md#compare-the-cusp-and-its-parametrization)

### Work at a point other than the origin

Local functions take `point=` in coordinate order.

```atlas
R = ring("x", "y")
f = (y - 2)^2 - (x - 1/2)^3
print(milnor_number(f, point=(1/2, 2)).value)              # 2
print(local_quotient_basis(ideal(x^2*(x - 1), y), point=(1, 0)).dimension)  # 1
```

[Evaluate one curve at three points](tutorial/local-algebra.md#evaluate-one-curve-at-three-points)

### Reduce in the local ring at a point

```atlas
R = ring("x", "y")
h = local_normal_form(x, ideal((1 - y)*x - y^2))
print(h.remainder, h.unit)                                 # y^2 -y+1
```

[Reduce at a rational point](tutorial/local-algebra.md#reduce-at-a-rational-point)

### Compute the codimension of a map germ

Give the components of the map as polynomials in the source coordinates.

```atlas
R = ring("t")
r = ae_codimension(t^2, t^3)
print(r.finite, r.value)                                   # True 1
```

[Compare the cusp and its parametrization](tutorial/local-algebra.md#compare-the-cusp-and-its-parametrization)

### Test smoothness at a point

```atlas
R = ring("x", "y", "z")
s = local_smoothness(ideal(x*y, x*z), point=(0, 0, 0))
print(s.local_dimension, s.embedding_dimension, s.value)   # 2 3 False
```

[Compare local dimension and smoothness](tutorial/local-geometry.md#compare-local-dimension-and-smoothness)

### Compute a tangent space or a tangent cone

```atlas
R = ring("x", "y")
I = ideal(y^2 - x^3)
print(zariski_tangent_space(I).dimension)                  # 2
print(tangent_cone(I).generators)                          # (y^2,)
print(hilbert_samuel_multiplicity(I).value)                # 2
```

[Compute the tangent space](tutorial/local-geometry.md#compute-the-tangent-space) · [Compute the tangent cone](tutorial/local-geometry.md#compute-the-tangent-cone)

## Resolution of singularities

### Resolve a hypersurface germ

```atlas
R = ring("x", "y")
K = resolve(y^2 - x^3)
print(K.divisors)                                          # [E1, E2, E3]
print(K.branches[0].meets)                                 # [E3]
```

Use `K.dual_complex` to display the intersection graph of the exceptional curves. For a resolution of the curve itself, pass `mode="intrinsic"`. [Follow the branch to the exceptional divisor](tutorial/resolving-a-cusp.md#follow-the-branch-to-the-exceptional-divisor) · [Resolution modes](thematic/resolution-records.md#resolution-modes)

### Find the multiplicity of an exceptional divisor

```atlas
R = ring("x", "y")
f = y^2 - x^3
K = resolve(f)
C = K.chart("B3.u")
E = K.divisor("E3")
print(C.pullback(f).multiplicity(E.equation_in(C)))        # 6
```

[Check the multiplicity in the pullback](tutorial/resolving-a-cusp.md#check-the-multiplicity-in-the-pullback)

### Compute the zeta functions of a germ

```atlas
R = ring("x", "y")
K = resolve(y^2 - x^3, monodromy=True)
print(K.monodromy_zeta.expression)                         # (1-T^6)/((1-T^2)*(1-T^3))
print(K.topological_zeta.expression)                       # (4*s + 5)/((s + 1)*(6*s + 5))
```

[Local monodromy zeta function](thematic/resolution-records.md#local-monodromy-zeta-function) · [Topological zeta function](thematic/resolution-records.md#topological-zeta-function)

### Resolve with weighted blowups

```atlas
R = ring("x", "y", "z")
K = resolve(x^2 + y^3 + z^5, method="weighted")
print(K.divisors, K.charts)                                # [E1] [W1.c0, W1.c1, W1.c2]
```

[Resolve with method weighted](thematic/weighted-resolution.md#resolve-with-method-weighted)

## Interface issues

### Run a calculation from the shell

Save the statements in a file and run it. Use `-` to read the script from standard input.

```sh
./atlas run divides.atlas
```

At the prompt, `:save "my session.atlas"` writes the statements you have entered so far. [Run or save a calculation](tutorial/prompt-and-scripts.md#run-or-save-a-calculation)

### Write results as JSON

In a script, `print_json(count=3)` writes the named values as one JSON object. Large integers and rationals keep their exact values. [Run or save a calculation](tutorial/prompt-and-scripts.md#run-or-save-a-calculation)

### Get help on a function or a saved value

Type `help normal_form` at the prompt for the arguments and an example, or `help I` for the ideal saved as `I`. `:workspace` lists the saved values under their rings. [Inspect a saved value](tutorial/prompt-and-scripts.md#inspect-a-saved-value)

### Call Atlas from another program

Start the server with `./atlas serve` and post a JSON request.

```sh
curl -sS http://127.0.0.1:8765/api/algebra \
  -H 'Content-Type: application/json' \
  -d '{"operation": "normal-form", "ring": {"variables": ["x", "y"]},
       "operands": {"ideal": ["x^2-y"], "polynomial": "x^3"}}'
```

[Algebra requests and replay](reference/algebra-requests.md) · [HTTP API and compute options](reference/http-and-compute.md)

### Choose the CPU or the GPU

Pass `compute="cpu"`, `"gpu"` or `"auto"` to a session function, `--compute` on the command line, or `compute` in a JSON request. `./atlas devices` lists the available devices. [Select a device](reference/http-and-compute.md#select-a-device)
