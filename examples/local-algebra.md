# Local algebras and map germs

A global quotient includes every component of the ideal. To study a germ at a rational point $p$, work in the local ring $O=\mathbb Q[x_1,\ldots,x_n]_{\mathfrak m_p}$. A polynomial with nonzero value at $p$ is a unit there. Point arguments follow the declared variable order. Omitting the point selects the origin.

## Count global and local lengths

The scheme defined by $(x^2(x-1),y)$ consists of a double point at $(0,0)$ and a simple point at $(1,0)$. Compare its global length with the local length at either point.

```atlas
x, y = coordinates("x", "y")
I = ideal(x^2*(x-1), y, order="lex")
G = quotient_basis(I)
L = local_quotient_basis(I)
print(G.dimension, L.dimension)                    # 3 2
print(local_quotient_basis(I, point=(1,0)).dimension) # 1
print(multiplicity(I).value, dimension(I).value)   # 3 0
```

`G.basis` contains $1,x,x^2$. At the origin `L.basis` contains $1,x$. At a general $p$, basis monomials are in $x_i-p_i$ and are expressed using the original variables. The selected order breaks ties between terms of the same local degree.

Use `dimension(I)` for **Krull dimension**, `multiplicity(I)` for the total vector-space dimension over $\mathbb Q$, and `quotient_basis(I)` when you also want the monomials. The total counts extension residue fields with their degrees. For an infinite-dimensional quotient, a basis result has `finite=False`, `basis=None` and `dimension=None`. A multiplicity result has `value=None` in that case. The zero quotient has dimension zero as a vector space and has an empty basis. [Global and local length tests](../tests/test_local_algebra_session.py) · [Multiplicity tests](../tests/test_session_multiplicity.py) · [C++ local quotient](../native/local_point_quotient.cpp)

## Compare the cusp and its parametrization

The plane curve $f=y^2-x^3$ has Jacobian ideal $J(f)=(-3x^2,2y)$. Its local Jacobian algebra has basis $1,x$, so the Milnor number $\dim_{\mathbb Q}O/J(f)$ is two. Adding $f$ to this ideal gives the Tjurina algebra and leaves the length unchanged for this cusp.

```atlas
x, y = coordinates("x", "y")
f = y^2-x^3
A = local_jacobian_algebra(f)
print(A.dimension, A.generators[0] == -3*x^2, A.generators[1] == 2*y) # 2 True True
print(milnor_number(f).value, tjurina_number(f).value)               # 2 2
```

The parametrization $t\mapsto(t^2,t^3)$ is itself a map germ. To measure its infinitesimal deformations modulo changes of source and target coordinates, apply `ae_codimension(t^2, t^3)` to its ordered components in the source ring. This gives the extended left-right ($\mathcal A_e$) codimension. A stable germ has codimension zero.

```atlas
t = coordinates("t")
r = ae_codimension(t^2, t^3)
print(r.finite, r.value)                   # True 1
```

Pass the components separately or as a list. They must vanish at the origin. Keep repetitions and zero components because each one contributes a target coordinate. Read `r.finite` before using `r.value`. A true value certifies an exact integer. False certifies infinity. With `None`, the calculation is inconclusive, and `r.lower_bound`, `r.jet_degree` and `r.reason` describe the partial result. Advanced cases record the proof method and, where available, the last weight checked. [Cusp and finiteness tests](../tests/test_ae_codimension_interface.py) · [C++ map-germ solver](../native/ae_codimension.cpp) · [Kernel derivation](../native/map-germ-algorithms.md)

The weighted example $\widehat B_7$ with $\alpha=-1$ in the published-case tests has components $(x,y^2+xz,z^2-xy,y^7+y^6z+yz^6-z^7)$. The tests check the value $837$, contact order $3$ and last weight $27$. [Published-case tests](../tests/test_ae_literature.py) · [Recorded map-germ timings](../benchmarks/map-germ-timings.md#published-map-germs)

## Evaluate one curve at three points

Set $f=y^2-(x-1/2)^3(x+1)^2$. The same equation has a cusp, a node and a smooth point. For a hypersurface invariant, the point must satisfy $f(p)=0$.

```atlas
x, y = coordinates("x", "y")
f = y^2-(x-1/2)^3*(x+1)^2
for p in [(1/2,0), (-1,0), (3/2,5/2)]:
    print(milnor_number(f, point=p).value, tjurina_number(f, point=p).value)
```

| Point in loop order | Milnor number | Tjurina number |
| --- | ---: | ---: |
| $(1/2,0)$, cusp | 2 | 2 |
| $(-1,0)$, node | 1 | 1 |
| $(3/2,5/2)$, smooth | 0 | 0 |

For isolated germs the two numbers can also differ. For example, $x^4+y^5+x^2y^3$ at the origin has $\mu=12$ and $\tau=11$. Requesting a number alone avoids enumerating a large basis. For a nonisolated critical locus such as $x^2$ in two variables, the local Jacobian algebra has infinite dimension. [Unequal-invariant tests](../tests/test_local_algebra_session.py) · [Rational-point tests](../tests/test_session_point_invariants.py) · [Large shifted and wide inputs](../tests/test_large_point_quotient.py)

## Reduce at a rational point

Let $I=((x-1/2)(x-2),y+2)$. At $p=(1/2,-2)$ the factor $x-2$ is a unit, so $x+y$ reduces to $-3/2$. At $(2,-2)$ the same polynomial reduces to zero. Compute `local_standard_basis(I, point=p)` once to reuse the point and basis for subsequent reductions.

```atlas
x, y = coordinates("x", "y")
I = ideal((x-1/2)*(x-2), y+2)
B = local_standard_basis(I, point=["1/2", "-2"])
h = local_normal_form(x+y, B)
print(h.remainder == -3/2, h.unit == 1)                    # True True
print(local_normal_form(x+y, I, point=(2,-2)).remainder)   # 0
```

If $r$ is the remainder and $u$ the returned unit, then $u(p)=1$ and $uf-r$ lies in the localized ideal. The class of the input is $r/u$, and $r=0$ tests local membership. In a finite-dimensional quotient the reduction has $u=1$. For a positive-dimensional example, reducing $x$ modulo $((1-y)x-y^2)$ at the origin gives $r=y^2$, $u=1-y$. Even `reduction="standard"` can have a nonconstant unit. A `"weak"` reduction guarantees that the leading monomial of a nonzero remainder is irreducible.

For a wide-exponent example at $(1,0,0)$, reduce $z$ modulo $x^N z-y^2$ with $N=2^{64}+7$. The unit is $x^N$ and the remainder is $y^2$.

```sh
./atlas algebra local-normal-form --vars x,y,z --point 1,0,0 \
  --gen 'x^18446744073709551623*z-y^2' --poly z --json
```

[Certificate and wide-exponent tests](../tests/test_local_standard_basis_session.py) · [C++ point standard basis](../native/local_point_basis.cpp) · [C++ point normal form](../native/local_point_normal_form.cpp)
