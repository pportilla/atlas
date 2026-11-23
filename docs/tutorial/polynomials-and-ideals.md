# Polynomials and ideals

This chapter starts with single polynomials over $\mathbb Q$: factoring them, finding common factors and dividing. It then turns to ideals: reducing modulo an ideal, eliminating a coordinate, intersecting, saturating, raising to powers and decomposing. The last sections treat exact linear algebra, modular arithmetic and the choice of device for polynomial arithmetic. Each section declares its coordinates, so you can try it in a fresh session.

## Factor and reconstruct a polynomial

A factorization over $\mathbb Q$ records a rational coefficient and the multiplicity of each monic irreducible factor. Here the coefficient of the original expression is $-3/5$. Where does the coefficient $2$ of $x$ in $2x-y$ go?

```atlas
x, y = coordinates("x", "y")
f = -3/5*(2*x-y)^2*(x^2+1)
F = factor(f, order="lex")
print(F.unit)                                              # -12/5
print(F.factors[0].polynomial, F.factors[0].multiplicity)  # x^2+1 1
print(F.factors[1].polynomial, F.factors[1].multiplicity)  # x-1/2*y 2
```

The factors are $x^2+1$ with multiplicity one and $x-y/2$ with multiplicity two. Making $2x-y$ monic contributes $2^2$ to `F.unit`. To recover the original polynomial, start with the unit.

```atlas
product = F.unit
for term in F.factors:
    product = product * term.polynomial^term.multiplicity

print(product == f)                                        # True
```

A nonzero constant contributes only its unit. The zero polynomial has no factorization into irreducibles.

## Factor over the algebraic closure

The polynomial $x^2+y^2$ has one irreducible factor over $\mathbb Q$ and two
over $\overline{\mathbb Q}$, namely $x+iy$ and $x-iy$ with $i^2=-1$.
Use `absolute_factor` to obtain the individual factors with exact algebraic
coefficients. Raising the polynomial to a power preserves the two distinct
factors and changes their multiplicities.

```atlas
x, y = coordinates("x", "y", coeffs=QQ)
f = -3/2*(x^2+y^2)^3
F = absolute_factor(f, order="lex")
print(len(factor(f).factors), len(F.factors))               # 1 2
print(F.unit, F.factors[0].multiplicity)                    # -3/2 3
product = F.unit
for entry in F.factors:
    product = product * entry.polynomial^entry.multiplicity

print(product == F.inclusion(f))                           # True
```

`F.ambient` is the polynomial ring containing all returned factors.
`F.inclusion` sends the original coordinates to that ring in their original
order. The factors are monic in the selected monomial order. A nonzero
constant gives an empty factor list and its value as the unit. Zero has no
factorization. For a coefficient extension, `F.ring["field"]` records a
primitive element, its minimal polynomial, and a choice of complex root.
These data specify the coefficients of every factor in a single number
field. If every monic absolute factor already has rational coefficients,
the result uses the original rational polynomial ring.

Absolute factorization extends the **constants**. For example,
$z^2-y^3+x^5$ remains irreducible in
$\overline{\mathbb Q}[x,y,z]$. Writing its roots as
$\pm\sqrt{y^3-x^5}$ adjoins a function of $x,y$ to the coefficient
field of the univariate polynomial in $z$.

## Group square-free factors

`squarefree_factor(f)` returns the leading coefficient and factors in order of increasing multiplicity. Each reported polynomial is a product of the distinct monic irreducibles occurring to that multiplicity.

```atlas
x = coordinates("x")
F = squarefree_factor(-3*(x-1)^2*(x+1)^2*(x^2+1)^3)
print(F.unit, F.factors[0].polynomial, F.factors[0].multiplicity)  # -3 x^2-1 2
```

## Find a common factor

The two polynomials below both have $x+y$ as a factor. Their remaining factors, $x+1$ and $y+1$, have no common divisor of positive degree.

```atlas
x, y = coordinates("x", "y")
d = gcd((x+y)*(x+1), (x+y)*(y+1), order="lex")
print(d.polynomial)                                        # x+y
```

The greatest common divisor is determined up to multiplication by a nonzero scalar. Atlas chooses the monic associate in the selected order. With one zero input, this gives the monic associate of the other input. With two zeros, the result is zero. When both inputs are constants, pass `variables=(x,y)` to specify their coordinates.

## Divide polynomials and find Bezout coefficients

`polynomial_division(f, g)` returns the quotient and reduced remainder for a nonzero divisor in the chosen monomial order. For two polynomials in one common variable over $\mathbb Q$, `extended_gcd(f, g)` also computes coefficients $u,v$ satisfying $uf+vg=\gcd(f,g)$. Both coefficients are zero when both inputs vanish.

```atlas
x, y = coordinates("x", "y", coeffs=QQ)
D = polynomial_division(x^3 + y, x - 1)
print(D.quotient, D.remainder)                            # x^2+x+1 y+1
E = extended_gcd(x^2 - 1, x^2 - x)
print(E.gcd)                                             # x-1
print(E.left_coefficient*(x^2-1) + E.right_coefficient*(x^2-x) == E.gcd)  # True
```

Here $x^3+y=(x^2+x+1)(x-1)+(y+1)$, and $\gcd(x^2-1,x^2-x)=x-1$.

## Cancel a rational function

`cancel_fraction(f,g)` divides both inputs by their GCD and scales them to give the denominator leading coefficient one for the selected monomial order. The denominator must be nonzero. A zero numerator gives zero over one. Both returned polynomials have the input parent.

```atlas
x, y = coordinates("x", "y")
C = cancel_fraction((x-y)*(x+1), 3*(x-y)*(y+1))
print(C.numerator, C.denominator)  # 1/3*x+1/3 y+1
```

## Reduce a polynomial modulo an ideal

Suppose $x^2=1$ and $y=x$. How do we decide whether two expressions in $x$ and $y$ have the same value subject to those relations? Form the ideal generated by the relations, then compute a Gröbner basis.

```atlas
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(x^2-1, y-x, order="lex")
B = groebner(I)
print(B.generators)                                # (x-y, y^2-1)
print(normal_form(x^2+y/2, B).polynomial)          # 1/2*y+1
print(membership(y^2-1, B).value)                  # True
print(membership(y, B).value)                      # False
```

The first basis element lets us replace $x$ by $y$. The second reduces every power of $y$ to either $1$ or $y$. In particular, every class has a unique expression $a+by$ with $a,b\in\mathbb Q$. Two polynomials represent the same class precisely when their normal forms agree.

The option `order="lex"` selects lexicographic order, with the first declared variable highest. The default order is graded reverse lexicographic (`"grevlex"`). Changing the order can change the normal form. Save the basis if you have several polynomials to reduce in the same order. The parent ring is available as `I.parent`. To construct its zero or unit ideal, use `ideal(ring=I.parent)` or `ideal(1, ring=I.parent)`. [Choosing an active ring](rings.md#name-select-and-delete-rings)

## Eliminate a coordinate

The map $t\mapsto(t^2,t^3)$ parametrizes a plane curve. To find an equation for its image, eliminate $t$ from $t^2-x$ and $t^3-y$.

```atlas
t, x, y = coordinates("t", "x", "y")
r = resultant(t^2-x, t^3-y, t)
print(r.polynomial)                                        # -x^3+y^2
J = eliminate(ideal(t^2-x, t^3-y), t)
print(J.generators)                                        # (x^3-y^2,)
print(r.inclusion(r) == y^2-x^3)                           # True
```

Both computations give the cusp equation $y^2=x^3$. The resultant of the ordered pair retains its sign and scalar factor. Elimination computes the entire ideal of relations in the remaining coordinates, here generated by the monic polynomial $x^3-y^2$. Use `J.coordinates` to work in that smaller ring. The inclusion in the last line sends the resultant back to the original three-variable ring.

If exactly one operand is a nonzero polynomial independent of $t$, the resultant is that polynomial raised to the $t$-degree of the other operand. Two nonzero inputs independent of $t$ give $1$. A zero operand gives zero. [Elimination and its result ring](../../native/ideal-algorithms.md#elimination-and-the-result-ring)

## Intersect, divide and saturate ideals

For $I=(x^2,xy)$, which polynomials become elements of $I$ after multiplication by $x$? The answer is the ideal quotient $I:(x)$. Allowing multiplication by *any power* of $x$ gives the saturation $I:x^\infty$.

```atlas
x, y = coordinates("x", "y")
I = ideal(x^2, x*y)
print(quotient(I, ideal(x)).generators)                     # (x, y)
print(saturation(I, x).generators)                          # (1,)
print(intersection(ideal(x), ideal(y)).generators)          # (x*y,)
```

One multiplication suffices for $x$ and $y$. Two multiplications suffice even for $1$, which explains why the saturation is the whole ring. The intersection in the last line answers a different question. A polynomial divisible by both $x$ and $y$ is divisible by $xy$.

For an arbitrary ideal $J$, `quotient(I,J)` consists of all $f$ such that $fJ\subseteq I$. To do arithmetic in the ring $R/I$, construct `quotient_ring(I)`. [Arithmetic with residue classes](quotient-rings.md#compute-with-residue-classes)

## Add, multiply, and raise ideals to powers

For two ideals in the same ring, `I + J` collects their generators and `I * J` takes all pairwise products. Ask for a reduced Gröbner basis to simplify either generating set.

```atlas
x, y = coordinates("x", "y")
I = ideal(x^2, y)
J = ideal(x, y^2)
print(groebner(I + J).generators)  # (x, y)
print(groebner(I * J).generators)  # (x^3, y^3, x*y)
```

The operators `+`, `*` and `^` and the functions `ideal_sum`, `ideal_product` and `ideal_power` work in both scripts and the prompt. To add the principal ideal generated by `f`, write `I + ideal(f)`. The result uses the left ideal's monomial order. [Read the ideal arithmetic script](../../examples/ideal-arithmetic.atlas)

## Compare ordinary and symbolic powers

The union of the three coordinate axes has ideal $I=(xy,xz,yz)$. At a general point on the $x$-axis, $x$ is a unit and $y,z$ both vanish. Thus $xyz$ vanishes to order at least two there. The same argument applies along the other axes. Does $xyz$ belong to the square of $I$?

```atlas
x, y, z = coordinates("x", "y", "z")
I = ideal(x*y, x*z, y*z)
print(membership(x*y*z, ideal_power(I, 2)).value)           # False
print(membership(x*y*z, symbolic_power(I, 2)).value)        # True
```

The ordinary square is generated by products of two quadratic generators, all of degree four. The symbolic square tests the order along each component separately, so it contains $xyz$.

Both functions accept a nonnegative integer exponent. Exponent zero gives the unit ideal. Symbolic powers require a radical ideal, including for exponents zero and one. For exponent $n$ the calculation intersects $I^nR_P\cap R$ over the minimal primes $P$ of $I$. [Power algorithms](../../native/ideal-algorithms.md#ordinary-and-symbolic-powers-of-rational-ideals)

## Recover the reduced support

The equations $y=0$ and $y=x^2$ meet at the origin. Their intersection ideal remembers the tangency there. Taking the radical gives the ideal of the underlying reduced point.

```atlas
x, y = coordinates("x", "y")
I = ideal(y, y-x^2)
J = radical(I)
print(J.generators)                                        # (x, y)
print(membership(x, I).value, membership(x, J).value)       # False True
```

The class of $x$ in the original quotient is nonzero but its square vanishes. Taking the radical kills exactly these nilpotent classes. Everything is computed over the declared field. For instance, $((x^2+1)^2)$ has radical $(x^2+1)$ over $\mathbb Q$.

## Reconstruct an ideal from its primary components

The zero set of $(x^2,xy)$ is the line $x=0$. Along the line the equation $xy=0$ forces $x=0$ wherever $y$ is invertible. At the origin the ideal retains extra structure. A primary decomposition records both pieces.

```atlas
x, y = coordinates("x", "y")
I = ideal(x^2, x*y)
D = primary_decomposition(I)
a, b = D.components
print(a.primary.generators, a.prime.generators)            # (x,) (x,)
print(b.primary.generators, b.prime.generators)            # (x^2, y) (x, y)
J = intersection(a.primary, b.primary)
print(groebner(J).generators == groebner(I).generators)    # True
```

Each row gives a primary ideal followed by its radical. The first component defines the line, and the second is supported at its origin. An embedded primary component can have different choices of generators or even a different primary ideal. Intersecting the components recovers $I$.

For the general method over $\mathbb Q$, see the algorithm notes and the paper by Gianni, Trager and Zacharias. [Atlas primary decomposition](../../native/ideal-algorithms.md#primary-decomposition-over-the-rationals) · [Gröbner Bases and Primary Decomposition of Polynomial Ideals](https://doi.org/10.1016/S0747-7171(88)80040-3)

## Exact rational linear algebra

`matrix_analysis` treats the rows as a matrix with entries in `QQ`. The pivot indices
are zero-based, `.image` lists the corresponding original columns, and `.kernel` lists
a basis of column vectors annihilated by the matrix. The reduced row echelon matrix
has the same shape as the input. A rectangular matrix has determinant `None`. The
empty square matrix has determinant one. Use `columns=n` to specify a matrix with
zero rows and `n` columns.

```atlas
M = matrix_analysis([[1, 2], [2, 4]], rhs=[3, 6])
print(M.rank, M.determinant, M.pivot_columns)  # 1 0 (0,)
print(M.solution.status)                       # underdetermined
print(M.solution.particular[0])                # 3
print(M.kernel[0][0])                          # -2
```

With `rhs`, the solution status is `unique`, `underdetermined`, or `inconsistent`.
When consistent, all solutions are the particular vector plus linear combinations
of the kernel vectors. For an inconsistent system the particular vector is `None`.
Exact coefficients may be integers, rational strings, or session rational values.
The scalar field is independent of the active polynomial ring.

With `compute="gpu"`, Atlas selects rows modulo a prime on the GPU and certifies the result over `QQ`. The `.execution` record lists the stages and the device. The command line gives each row with `--row`. [CPU, Triton and Singular timings](../../benchmarks/matrix-triton.md)

```sh
./atlas algebra matrix-analysis --row 1,2 --row 2,4 --rhs 3,6
```

## Compute with residues and lift roots

`modular_arithmetic(action, values, modulus)` returns the ordered `.values` and the output `.modulus`. Arithmetic accepts any integer modulus of at least two. A prime modulus gives prime-field arithmetic. Use `other=[...]` for corresponding operands of `add`, `subtract`, `multiply` or `divide`. A divisor or an input to `inverse` must be a unit. `reduce` returns the least nonnegative representatives. For `crt`, give the second residue vector and its coprime `other_modulus`.

For extension fields, pass `field_polynomial` with the coefficients of a monic irreducible polynomial over a prime field. Elements are coefficient vectors, constant first. The empty vector represents zero. For example, in $\mathbb F_2[a]/(a^2+a+1)$ the vector `[0,1]` represents $a$, and squaring it gives `(1,1)`. The result records the defining polynomial.

For `hensel`, give integer polynomial coefficients constant first. Each input root modulo $m$ must have invertible derivative. The returned root reduces to the input and solves the equation modulo $m^2$. For `reconstruct`, Atlas returns the unique reduced fraction with numerator magnitude and positive denominator at most $\lfloor\sqrt{(m-1)/2}\rfloor$, or `None` when none exists. With `compute="gpu"`, eligible batches run on the GPU. [Exact arithmetic on Triton](../reference/http-and-compute.md#exact-arithmetic-on-triton)

```atlas
r = modular_arithmetic("hensel", [3, 4], 7, other=[-2, 0, 1])
print(r.values, r.modulus)  # (10, 39) 49
c = modular_arithmetic("crt", [2, 1], 3, other=[3, 0], other_modulus=5)
print(c.values, c.modulus)  # (8, 10) 15
e = modular_arithmetic("multiply", [[0,1]], 2, other=[[0,1]], field_polynomial=[1,1,1])
print(e.values, e.field_polynomial)  # ((1, 1),) (1, 1, 1)
```

## Lift two coprime polynomial factors

`hensel_factors(f,g,h,p)` takes integer polynomials in one common variable. Modulo the prime $p$, the two monic polynomials must be coprime and their product must equal $f$. The target modulo $p^2$ must be monic of the same total degree. Atlas returns the unique monic lifts of the original degrees in the same order, with coefficient representatives from zero through $p^2-1$.

```atlas
x = coordinates("x")
H = hensel_factors(x^2-2, x-3, x+3, 7)
print(H.left, H.right, H.modulus)  # x+39 x+10 49
```

Here $x^2-2\equiv(x-3)(x+3)$ modulo $7$, and the lifts satisfy $x^2-2\equiv(x+39)(x+10)$ modulo $49$. Each output polynomial keeps the input ring.

## Choose a backend for polynomial arithmetic

`polynomial_arithmetic` accepts `add`, `subtract`, `multiply`, `divide`, `power`, `differentiate` and `substitute` over rational coefficients or the exact marked number field of a saved factorization. Division takes a nonzero element of the coefficient field. Substitution replaces all the coordinates simultaneously, in declaration order. Use rational images to evaluate at a point. The result retains the input ring, including when it is constant. Its `.execution` records the stages that ran on each device.

```atlas
x, y = coordinates("x", "y", coeffs=QQ)
p = polynomial_arithmetic("substitute", x^2+y, images=[y, x], compute="gpu")
print(p == y^2+x)  # True
d = polynomial_arithmetic("differentiate", x^3*y, variable=x, compute="cpu")
print(d == 3*x^2*y)  # True
```

The following calculation takes place over the same marked field that Atlas chose for the factorization. You can also pass `F.ambient.coeffs` as the coefficient field of a new coordinate declaration. The scalar generator stays fixed under substitution and differentiation.

```atlas
t = coordinates("t")
F = absolute_factor(t^2+1)
g = F.factors[0].polynomial
h = polynomial_arithmetic("multiply", g, g, compute="gpu")
print(h == g^2)  # True
print(h.parent == F.ambient)  # True
```

The CLI and HTTP accept a field description with a generator name and exact embedding. Atlas checks any supplied minimal polynomial, degree, and root index against the embedding.

```sh
./atlas algebra polynomial-arithmetic --vars x --field '{"generator":"a","embedding":"I"}' --action multiply --poly 'x+a' --other-poly 'x-a' --gpu
```
