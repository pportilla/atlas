# Rational polynomial and ideal arithmetic

Send an operation to `POST /api/algebra` with `Content-Type: application/json`. The variables are ordered from highest to lowest priority. The defaults are `field: "QQ"`, `order: "grevlex"` and `compute: "auto"`.

## Example request

```json
{
  "operation": "normal-form",
  "ring": {"field": "QQ", "variables": ["x", "y"], "order": "lex"},
  "operands": {"ideal": ["x^2-1", "y-x"], "polynomial": "x^2+y/2"},
  "compute": "cpu"
}
```

## Example success

This HTTP 200 response was produced by the example request. Elapsed times and execution stages depend on the machine and computation. The replay forms carry the same operands and requested execution mode.

```json
{
  "status": "ok",
  "operation": "normal-form",
  "result": {
    "type": "polynomial",
    "ring": {
      "field": "QQ",
      "variables": [
        "x",
        "y"
      ],
      "order": "lex"
    },
    "polynomial": "1/2*y+1"
  },
  "execution": {
    "requested": "cpu",
    "backend": "cpu",
    "device": "CPU",
    "stages": [
      {
        "operation": "groebner_basis",
        "backend": "cpu",
        "algorithm": "direct",
        "order": "lex",
        "generators": 2,
        "known": 0
      },
      {
        "operation": "normal_form",
        "backend": "cpu",
        "divisors": 2
      }
    ],
    "fallback_reasons": [],
    "elapsed_ms": 2.398
  },
  "replay": {
    "command": "./atlas algebra normal-form --vars x,y --field QQ --order lex --compute cpu --gen 'x^2-1' --gen y-x --poly 'x^2+y/2' --json",
    "script": "x, y = coordinates(\"x\", \"y\", coeffs=QQ)\nI = ideal(x^2-1, -x+y, order='lex')\nresult = normal_form(x^2+1/2*y, I, compute='cpu')\nprint(result.json)\n"
  }
}
```

## Example error

Submitting the same request with `"order": "local"` returns HTTP 400.

```json
{
  "status": "error",
  "error": {"code": "invalid_request", "message": "Unsupported monomial order. Choose lex or grevlex."}
}
```

## Operands and results

The ideal operations below take an `ideal`, an array of polynomial strings. Use `[]` or `["0"]` for the zero ideal. Constant generators and generators with nonzero constant terms are valid. Use `ring.variables: []` for the field itself. Polynomial strings accept integer literals, rational division by a nonzero constant, multiplication with `*`, and powers with `^` or `**`.

| Operation | Additional operand | Result in its recorded ring |
| --- | --- | --- |
| `ideal` | | Ideal declaration with the original generator sequence, written as exact polynomials |
| `groebner` | | Reduced basis, in `generators` |
| `normal-form` | `polynomial` | Exact remainder, in `polynomial` |
| `membership` | `polynomial` | Boolean `value` |
| `eliminate` | `variables`, an array of names to remove | Ideal, in the surviving variables |
| `radical` | | Radical ideal, given by its reduced Gröbner basis in the original ring and order |
| `ideal-power` | `exponent`, a nonnegative integer | Ordinary ideal power, as a reduced basis |
| `symbolic-power` | `exponent`, a nonnegative integer | Symbolic power of a prime or radical ideal, as a reduced basis |
| `localization` | `prime` ideal, optional `numerator` and `denominator` | Contracted ideal, verified prime and optional fraction membership and unit flags |
| `normalization` | | Normalized component algebras, coordinate maps and fraction-field generators |
| `primary-decomposition` | | Primary components and their associated primes in the input ring |
| `intersection` | `other_ideal`, an array of polynomials | Ideal, in `generators` |
| `quotient` | `other_ideal` | Ideal quotient, in `generators` |
| `quotient-ring` | `action` and its operands | Ring declaration, canonical residue class or boolean |
| `saturation` | `polynomial` | Saturation by one polynomial, in `generators` |
| `dimension` | | Krull dimension of the quotient as an integer `value`, with −1 for the empty variety |
| `hilbert-series` | | Sparse numerator, ambient denominator power, dimension and multiplicity of a homogeneous quotient |

### Arithmetic in declared quotient rings over QQ

Declare an ideal in the ambient coordinates, then call `quotient_ring(I)` to work in $A=\mathbb Q[x_1,\ldots,x_n]/I$. The ring retains the chosen monomial order. `A.coordinates` holds the original ambient variables, `A.gens` holds their residue classes, and `A(f)` takes the class of an ambient polynomial or rational number. Every quotient element exposes its `.parent` and its canonical ambient `.representative`.

```python
x, y = coordinates("x", "y", coeffs=QQ)
A = quotient_ring(ideal(x*y - 1))
u, v = A.gens
print(u.inverse() == v)                 # True
print((u^3 / u).representative == x^2)  # True
print(A(x^2*y + 1) == u + 1)           # True

D = quotient_ring(ideal(x^2))
t = D(x)
print(t^2 == 0, t.is_unit())            # True False
print((1 + t)^-1 == 1 - t)              # True
```

Use these expressions at the prompt or in a file passed to `./atlas run`. Rational numbers and ambient polynomials from the original coordinates can appear on either side of an operator. To transfer a representative into a second quotient of the same ambient space, write `D(u.representative)`. Division and negative powers require an invertible denominator or base. In the zero ring, $0=1$ and its only element is its own inverse.

```sh
./atlas algebra quotient-ring --vars x,y --gen 'x*y-1' --action inverse --poly x
./atlas algebra quotient-ring --vars x,y --gen 'x^2' --action power --poly '1+x' --exponent -3 --json
```

Over HTTP, select operation `quotient-ring` and supply `operands.ideal`. The default action is `declare`. For `reduce`, `negate`, `unit`, and `inverse`, pass `polynomial`. For `add`, `subtract`, `multiply`, `divide`, and `equal`, also pass `other_polynomial`. For `power`, pass `polynomial` and a signed integral `exponent`. A decimal string preserves an exponent of any width. All result records contain the reduced defining `relations`. A `quotient-element` also has a `representative`, and a `boolean` has a `value`. In a session, `A.apply("equal", u*v, 1).json` returns the complete predicate record.

### Ordinary and symbolic powers over QQ

Use `ideal_power(I, n)` to form the ideal generated by products of $n$ elements of $I$. For a radical ideal, `symbolic_power(I, n)` returns the intersection of $I^n R_P\cap R$ over the minimal primes $P$ of $I$. When $I$ is prime this requires just the contraction at $I$. Both results are ideals in the original coordinate space with a reduced Gröbner basis in the requested order. Exponent zero gives the unit ideal, including for $I=(0)$. Symbolic powers check that the input is radical even for exponents zero and one.

For the three coordinate axes, $I=(xy,xz,yz)$. Its symbolic square contains $xyz$ because $xyz$ vanishes to order two along each axis. Every generator of the ordinary square has degree four. In fact $I^{(2)}=I^2+(xyz)$.

```python
x, y, z = coordinates("x", "y", "z", coeffs=QQ)
I = ideal(x*y, x*z, y*z)
O = ideal_power(I, 2)
S = symbolic_power(I, 2)
print(membership(x*y*z, O).value)  # False
print(membership(x*y*z, S).value)  # True
```

```sh
./atlas algebra ideal-power --vars x,y,z --gen 'x*y' --gen 'x*z' --gen 'y*z' --exponent 2
./atlas algebra symbolic-power --vars x,y,z --gen 'x*y' --gen 'x*z' --gen 'y*z' --exponent 2
```

The HTTP operations are `ideal-power` and `symbolic-power`, each with operands `ideal` and `exponent`.

### Localization at a prime over QQ

For $R=\mathbb Q[x_1,\ldots,x_n]$ and a prime ideal $P$, the ring $R_P$ makes every polynomial outside $P$ invertible. Atlas checks that $P$ is a proper prime. Given an ideal $I$, it returns the contraction $J=IR_P\cap R$ as a reduced Gröbner basis in your chosen order. This ideal in the original coordinates uniquely represents $IR_P$, since $JR_P=IR_P$.

For example, take $I=(x^2,xy)$ and $P=(x)$. The variable $y$ is invertible in $R_P$, so the equation $xy=0$ in the localized quotient implies $x=0$. The contraction is $(x)$. The fraction $x/y$ belongs to $IR_P$ and is a nonunit in $R_P$. At $P=(x,y)$ the same ideal retains its embedded component and the contraction is $(x^2,xy)$.

```sh
./atlas algebra localization --vars x,y --gen 'x^2' --gen 'x*y' --prime x --numerator x --denominator y
```

```python
x, y = coordinates("x", "y", coeffs=QQ)
P = ideal(x)
L = localize(ideal(x^2, x*y), P)
print(L.contraction.generators)              # (x,)
print(L.fraction(x, y).member)               # True
print(L.fraction(x, y).unit)                 # False
A = localize(P)                              # R localized at P
print(A.fraction(y+1, y).unit)               # True
```

Use `localization` or `localize` at the prompt and in scripts. The fields `.prime` and `.contraction` are ideals in the original coordinate space. Pass `numerator=` and `denominator=` in the original call to receive the same checks in `.element`. A numerator alone uses denominator one. A denominator alone tests its inverse. The denominator supplied must lie outside $P$, including when a common factor could be cancelled. The zero prime is valid and gives the rational function field.

Use the HTTP operation `localization` with operands `ideal`, `prime`, `numerator`, and `denominator`. The JSON response stores the contraction in `generators`, the prime in `prime`, and the optional query in `element`. The `member` flag refers to $IR_P$ and the `unit` flag refers to the ambient local ring $R_P$.

### Normalization of affine algebras over QQ

For the cusp $A=\mathbb Q[x,y]/(y^2-x^3)$, the fraction $t=y/x$ satisfies $t^2=x$ and $t^3=y$. Adjoining $t$ gives the normalization $\mathbb Q[t]$. Atlas returns the images of the original coordinates in the normalized ring, and each normalized generator as a fraction on its source component.

```sh
./atlas algebra normalization --vars x,y --gen 'y^2-x^3'
```

```python
x, y = coordinates("x", "y", coeffs=QQ)
N = normalization(ideal(y^2-x^3))
C = N.components[0]
t = C.coordinates[0]
print(C.images[0] == t^2, C.images[1] == t^3)  # True True
print(C.fractions[0].numerator, C.fractions[0].denominator)  # y x
```

For a reduced algebra with several minimal primes, the normalization is the product of the returned component algebras. Each `C.prime` is an ideal in the original coordinates. The normalized algebra is the polynomial ring on `C.coordinates` modulo `C.relations`. The tuple `C.images` follows the input coordinate order, and `C.fractions` follows the target coordinate order. A denominator is nonzero in the component's function field and may vanish at particular points. For input with nilpotents, Atlas normalizes the reduction and sets `N.input_reduced` to `False`. The unit ideal presents the zero algebra and returns no components.

A saved normalization lets you reuse its component primes and defining relations. The JSON record preserves the component maps and their coordinate rings.

### Primary decomposition over QQ

A primary decomposition expresses an ideal as an intersection of primary ideals. Each component $Q$ comes with its associated prime $P=\sqrt Q$. For example, $(x^2,xy)=(x)\cap(x^2,y)$. The first component is the line $x=0$ and the second is an embedded component at the origin with associated prime $(x,y)$.

```sh
./atlas algebra primary-decomposition --vars x,y --gen 'x^2' --gen 'x*y'
```

At the prompt or in a script, each component exposes two ideals in the original coordinate space.

```python
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(x^2, x*y)
D = primary_decomposition(I)
for component in D.components:
    print(component.primary.generators, component.prime.generators)

J = intersection(D.components[0].primary, D.components[1].primary)
print(groebner(J).generators == groebner(I).generators)  # True
```

The shell’s `--json` output and the HTTP operation `primary-decomposition` return `components`, an array of objects with `primary` and `prime` generator lists. A zero ideal has one component with primary and prime ideals both zero. The unit ideal has an empty decomposition. Embedded primary ideals can have different valid generators or even different valid choices of ideal. Their associated primes and the intersection of the primary components are invariant.

### Radicals of polynomial ideals

The radical $\sqrt I$ consists of the polynomials $f$ for which $f^n\in I$ for some positive integer $n$. For example, take $I=(y, y-x^2)$. Both generators are linear in $y$, but their difference gives $x^2\in I$. The radical is $(x,y)$.

```sh
./atlas algebra radical --vars x,y --gen y --gen 'y-x^2'
```

Use the same function at the prompt and in scripts. Its result is an ideal with generators in the input coordinate space.

```python
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(y, y-x^2)
J = radical(I)
print(membership(x, I).value)  # False
print(membership(x, J).value)  # True
```

For HTTP, send `operation: "radical"` and `operands.ideal`. The result’s `generators` array contains the reduced Gröbner basis in the requested order.

The operation also applies to positive-dimensional and mixed-dimensional ideals. For example, $\sqrt{(x^2,xy)}=(x)$ is the line $x=0$. Over $\mathbb Q$, $\sqrt{((x^2+1)^2)}=(x^2+1)$, even though the equation has no rational root. The zero and unit ideals return `[]` and `["1"]`, respectively.

### Polynomial factorization

`factor` takes one polynomial over the rationals. This example has unit `-12/5`, the factor `x-y/2` with multiplicity 2, and the factor `x^2+1` with multiplicity 1.

```sh
./atlas algebra factor --vars x,y --order lex --poly '-3/5*(2*x-y)^2*(x^2+1)'
```

Use the same calculation at the prompt or in an Atlas script. Each factor polynomial belongs to the input coordinate space.

```python
x, y = coordinates("x", "y", coeffs=QQ)
f = -3/5*(2*x-y)^2*(x^2+1)
F = factor(f, order="lex")
product = F.unit
for term in F.factors:
    product = product * term.polynomial^term.multiplicity

print(product == f)
```

For HTTP, send the operation `factor` with a single `polynomial` operand. The shell’s `--json` output and the HTTP response both include the `ring`, the exact rational `unit` as a string, and a `factors` array. Each entry carries its `polynomial` and positive integer `multiplicity`.

Each factor has leading coefficient 1 in the selected monomial order. The unit carries the sign and scale needed to recover the input. A nonzero constant returns itself as the unit and an empty factor list. Zero has no irreducible factorization and returns an input error.

### Polynomial resultants

The equations `t^2-x` and `t^3-y` parametrize a cusp. Eliminating `t` gives `-x^3+y^2` in `QQ[x,y]`. With the same two inputs, eliminating `x` gives `t^3-y` in `QQ[t,y]`.

```sh
./atlas algebra resultant --vars t,x,y --poly 't^2-x' --other-poly 't^3-y' --eliminate t
```

At the prompt or in a script, pass the elimination coordinate as the third argument. The returned polynomial owns its surviving coordinates, in declaration order.

```python
t, x, y = coordinates("t", "x", "y", coeffs=QQ)
R = resultant(t^2-x, t^3-y, t)
u, v = R.coordinates
print(R.polynomial == v^2-u^3)
```

The HTTP request specifies both polynomials in order and the variable to eliminate.

```json
{
  "operation": "resultant",
  "ring": {"variables": ["t", "x", "y"]},
  "operands": {"polynomial": "t^2-x", "other_polynomial": "t^3-y", "variables": ["t"]}
}
```

Atlas returns the signed Sylvester resultant. For example, `resultant(t-x, t-y, t)` gives `x-y`. A zero input gives zero. Two nonzero inputs independent of the selected variable give one. If only one input is independent of that variable, the resultant is that input raised to the degree of the other. Eliminating the only coordinate returns a constant polynomial over `QQ`.

### Polynomial GCD

`gcd` takes two polynomial strings in the same ring. For example, this request returns `"polynomial": "x+y"` in $\mathbf Q[x,y]$.

```json
{
  "operation": "gcd",
  "ring": {"variables": ["x", "y"], "order": "lex"},
  "operands": {"polynomial": "(x+y)*(x+1)", "other_polynomial": "(x+y)*(y+1)"}
}
```

```sh
./atlas algebra gcd --vars x,y --order lex --poly '(x+y)*(x+1)' --other-poly '(x+y)*(y+1)' --json
```

```python
x, y = coordinates("x", "y", coeffs=QQ)
d = gcd((x+y)*(x+1), (x+y)*(y+1), order="lex")
print(d.polynomial, d.order)
```

A nonzero GCD has leading coefficient one in the recorded order. Thus `gcd(0,f)` is the monic associate of $f$, `gcd(0,0)` is zero, and coprime inputs or a nonzero constant operand give one. Use `variables=(x,y)` to place two constant operands in that ambient ring.

### Extended right-left codimension of a map germ

Use `ae_codimension(f1, ..., fp)` or `ae_codimension([f1, ..., fp])` at
the prompt or in a script. Supply ordered components in one declared source
coordinate space over `QQ`. Each must vanish at the origin. Zero and repeated
components count toward the target dimension.

The $\mathcal A_e$-codimension counts independent infinitesimal deformations
of the map modulo changes of source and target coordinates, including constant
vector fields. A stable map germ has value zero. In Singular's
`classifyMapGerms.lib`, the corresponding value is `coDimMap(f)[2]` with
the default extended convention. [Definition and native algorithms](../native/KERNEL.md#extended-right-left-codimension) · [Singular comparison](../benchmarks/README.md#extended-right-left-codimension-of-map-germs)

For Sharland's $\widehat B_7$ with $\alpha=-1$, inspect the value and the
weight at which the native solver proved that every higher piece vanishes.

```python
x, y, z = coordinates("x", "y", "z", coeffs=QQ)
r = ae_codimension(x, y^2+x*z, z^2-x*y, y^7+y^6*z+y*z^6-z^7)
print(r.finite, r.value)                     # True 837
print(r.method, r.contact_order, r.last_weight)  # weighted_tangent_ranks 3 27
```

| `r.finite` in a session | `r.value` | Interpretation |
| --- | --- | --- |
| `True` | Integer | Exact finite codimension |
| `False` | `None` | Proven infinite codimension |
| `None` | `None` | Inconclusive. Inspect `r.reason`, the proven `r.lower_bound`, and the last completed `r.jet_degree`. |

Every result gives the computation `method`. The optional fields
`contact_order`, `last_weight`, and `jet_degree` describe the proof used.
JSON uses `true`, `false`, and `null` for the three finiteness states.
[Published examples and kernel timings](../benchmarks/README.md#published-map-germs)

At the shell, repeat `--component` in target order. For the parametrized cusp
$t\mapsto(t^2,t^3)$, the following command returns `finite: true` and `value: 1`.

```sh
./atlas algebra ae-codimension --vars t --component 't^2' --component 't^3' --json
```

Send the same map to `POST /api/algebra` with this JSON body. The response
stores the number in `result.value`.

```json
{"operation":"ae-codimension","ring":{"variables":["t"]},"operands":{"components":["t^2","t^3"]}}
```

### Quotient bases and local hypersurface invariants

For an ideal $I$ of $R=\mathbb Q[x_1,\ldots,x_n]$, choose the global quotient to include every component, or the local quotient to study the germ at a rational point $p$. The local calculation takes place in $O=R_{(x_1-p_1,\ldots,x_n-p_n)}$, where every polynomial nonzero at $p$ is a unit. For the three hypersurface operations, supply a polynomial $f$ in at least one variable with $f(p)=0$. Write $J(f)$ for the ideal generated by its partial derivatives in the declared coordinate order. Omitting the point selects the origin.

| Operation | HTTP operand | Returned mathematical object |
| --- | --- | --- |
| `quotient-basis` | `ideal` | Monomial basis of $R/I$ and its dimension over $\mathbb Q$, when finite |
| `multiplicity` | `ideal` | Total multiplicity $\dim_{\mathbb Q}(R/I)$ when the ideal has finite colength |
| `local-quotient-basis` | `ideal` | Monomial basis of $O/IO$ and its dimension over $\mathbb Q$, when finite |
| `local-jacobian-algebra` | `polynomial` | Partial derivatives presenting $O/J(f)$, and its basis and dimension when finite |
| `milnor-number` | `polynomial` | $\mu=\dim_{\mathbb Q} O/J(f)$ |
| `tjurina-number` | `polynomial` | $\tau=\dim_{\mathbb Q} O/(f,J(f))$ |

The basis elements are residue classes, so `1` in a returned basis means the class of the constant polynomial. A zero quotient has dimension zero and the empty basis. An infinite-dimensional quotient returns `finite: false` and JSON `null` in `dimension` and `basis`, or in `value` for a number request. For instance, the smooth germ $f=x+y^2$ has $\mu=\tau=0$, while $f=x^2$ in two variables has an entire critical line and both dimensions are infinite. The separate `dimension` operation computes **Krull dimension**.

For a global basis, `--order` selects the monomial order. At $p$, Atlas compares the lowest total degree in $x_i-p_i$ first and uses the chosen order to break ties. A returned local basis consists of monomials in these centered coordinates, written in the original variables. The Jacobian generators are the ordinary partial derivatives of the input polynomial in those same original variables. [Algorithm at rational points](../native/KERNEL.md#quotients-and-hypersurface-invariants-at-rational-points)

```sh
# Two critical points globally, one at the origin
./atlas algebra quotient-basis --vars x,y --gen '3*x^2-6*x' --gen '2*y'
./atlas algebra local-quotient-basis --vars x,y --gen '3*x^2-6*x' --gen '2*y'
# The cusp has Jacobian basis 1, x and Milnor number 2
./atlas algebra local-jacobian-algebra --vars x,y --poly 'y^2-x^3' --json
# A germ for which the two numbers differ
./atlas algebra milnor-number --vars x,y --poly 'x^5+y^5+x^2*y^2'
./atlas algebra tjurina-number --vars x,y --poly 'x^5+y^5+x^2*y^2'
```

The first pair gives dimensions 2 and 1. The last pair gives $\mu=11$ and $\tau=10$. Request the number alone when a full list of basis elements would be too large.

The same four local operations accept sparse equations whose expansion about the point would exceed one million terms. For example, set $g=x^{1001}y^{1001}+x+y-3$ and $h=x-y$. At $(1,1)$ these are independent local coordinates. The Jacobian algebra of $g^5+h^5+g^3h^3$ has 16 basis monomials, and adding the equation itself leaves 15.

```sh
g='(x^1001*y^1001+x+y-3)'
h='(x-y)'
f="$g^5+$h^5+$g^3*$h^3"
./atlas algebra local-jacobian-algebra --vars x,y --point 1,1 --poly "$f"
./atlas algebra milnor-number --vars x,y --point 1,1 --poly "$f" --compute cpu
./atlas algebra tjurina-number --vars x,y --point 1,1 --poly "$f" --compute cpu
```

A basis request lists every surviving monomial. A Milnor or Tjurina request returns an exact integer even when the length exceeds 64 bits. [CPU measurements and Singular checks](../benchmarks/point-quotient-invariants.md)

```python
x, y = coordinates("x", "y", coeffs=QQ)
A = local_jacobian_algebra(y^2-x^3)
print(A.generators, A.basis, A.dimension)
I = ideal(A.generators, variables=A.coordinates)
print(local_quotient_basis(I).dimension)
print(tjurina_number(y^2-x^3).value)
```

#### One curve at three rational points

Take $f=y^2-(x-1/2)^3(x+1)^2$. The following four commands use the cusp at $(1/2,0)$. Its local Jacobian algebra has basis $1,x-1/2$ and dimension 2. The first command uses the same local defining ideal, with the unit factors removed.

```sh
f='y^2-(x-1/2)^3*(x+1)^2'
./atlas algebra local-quotient-basis --vars x,y --point '1/2,0' --gen '(x-1/2)^2' --gen y
./atlas algebra local-jacobian-algebra --vars x,y --point '1/2,0' --poly "$f"
./atlas algebra milnor-number --vars x,y --point '1/2,0' --poly "$f"
./atlas algebra tjurina-number --vars x,y --point '1/2,0' --poly "$f"
```

| Point on the same curve | Milnor number | Tjurina number |
| --- | ---: | ---: |
| $(1/2,0)$, a cusp | 2 | 2 |
| $(-1,0)$, a node | 1 | 1 |
| $(3/2,5/2)$, a smooth point | 0 | 0 |

At the prompt or in an Atlas script, pass `point=` in the declared variable order. For example, `A = local_jacobian_algebra(f, point=["1/2", 0])` records the point alongside its basis. `local_quotient_basis(ideal(A.generators, variables=A.coordinates), point=A.point)` computes the same dimension. For an ideal presented by a saved local standard basis `B`, `local_quotient_basis(B)` uses its saved point.

For HTTP requests, supply `operands.point` as an ordered array of exact integers or rational strings. An explicit point returns `scope: "point"` and the reduced coordinates as strings. An omitted point preserves the existing `scope: "origin"` response. The global `quotient-basis` command accepts the ideal alone.

For a local calculation, supply the rational point in the declared coordinate order or omit it to use the origin. Pass `--compute gpu` to request eligible device work. The execution record identifies the stages that ran and any CPU fallback.

### Multiplicity of finite-colength quotients

For a finite-dimensional quotient $R/I$ over $\mathbb Q$, its total multiplicity is its vector-space dimension over $\mathbb Q$. For example, $(x^2(x-1),y)$ has a double point at $(0,0)$ and a simple point at $(1,0)$, giving multiplicity 3.

```sh
./atlas algebra multiplicity --vars x,y --gen 'x^2*(x-1)' --gen y
```

At the prompt or in a script, read the integer from the result’s `.value` field. A saved global Gröbner basis can also supply the ideal.

```python
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(x^2*(x-1), y)
M = multiplicity(I)
print(M.finite, M.value)           # True 3
print(multiplicity(groebner(I)))   # 3
```

The count includes points with extension residue fields. For instance, $\mathbb Q[x]/((x^2+1)^2)$ has dimension 4. The unit ideal gives multiplicity 0. A positive-dimensional quotient has infinite colength, returned as `finite: false` and `value: null`. The HTTP operation `multiplicity` takes `operands.ideal` and returns the same global result as the shell’s `--json` option.

### Hilbert series of homogeneous quotients

Give every variable degree one. For a homogeneous ideal $I$ in $S=\mathbb Q[x_1,\ldots,x_n]$, the coefficient of $t^d$ in the Hilbert series is the dimension of the degree-$d$ part of $S/I$ over $\mathbb Q$.

```sh
./atlas algebra hilbert-series --vars x,y --gen 'x^2' --gen 'y^3'
```

This gives $(1-t^2-t^3+t^5)/(1-t)^2 = 1+2t+2t^2+t^3$. The quotient has Krull dimension 0 and multiplicity 6. At the prompt or in an Atlas script, the function returns a value whose fields can be read directly.

```python
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(x^2, y^3)
H = hilbert_series(I)
print(H)
print(H.dimension, H.multiplicity)  # 0 6
print(H.numerator[1])              # {'degree': 2, 'coefficient': -1}
print(H.denominator_power)         # 2
```

The HTTP operation `hilbert-series` takes `operands.ideal`. Both the response and the shell’s `--json` output give the nonzero signed numerator terms in ascending degree over the common denominator $(1-t)^n$, before cancellation. For the zero ideal the series is $1/(1-t)^n$. The unit ideal gives series zero, an empty numerator, dimension $-1$ and multiplicity 0.

Homogeneity is a property of the ideal. For example, $(x+y^2,y^2)=(x,y^2)$ is accepted even with the first set of generators. The ideal $(x+y^2)$ alone gives a homogeneity error.

<a id="hilbert-samuel-functions"></a>

### Hilbert–Samuel functions and multiplicities

Let $A=(\mathbb Q[x_1,\ldots,x_r]/I)_{\mathfrak m_p}$ at the rational point $p$, and let $\mathfrak m$ be its maximal ideal. Atlas returns the cumulative Hilbert–Samuel function $H(n)=\dim_{\mathbb Q}(A/\mathfrak m^{n+1})$ and the graded value $h(n)=\dim_{\mathbb Q}(\mathfrak m^n/\mathfrak m^{n+1})$. Their relation is $h(n)=H(n)-H(n-1)$ with $H(-1)=0$. For local dimension $d$, the Hilbert–Samuel multiplicity is $d!$ times the leading coefficient of the eventual polynomial for $H(n)$. When $d=0$, the function eventually equals the length of the local algebra. A point outside the support gives dimension $-1$, multiplicity 0 and all values 0.

```sh
./atlas algebra hilbert-samuel-function --vars x,y --gen 'y^2-(x-1/2)^3' --point '1/2,0' --degrees 0,1,2,5
./atlas algebra hilbert-samuel-multiplicity --vars x,y --gen 'y^2-(x-1/2)^3' --point '3/2,1'
```

| Point on the cusp equation | $H(0),H(1),H(2),H(5)$ | $h(0),h(1),h(2),h(5)$ | Multiplicity |
| --- | --- | --- | ---: |
| $(1/2,0)$ | 1, 3, 5, 11 | 1, 2, 2, 2 | 2 |
| $(3/2,1)$ | 1, 2, 3, 6 | 1, 1, 1, 1 | 1 |

Omit the point to use the origin. The default function degrees are 0 through 5. Pass `--degrees` with comma-separated nonnegative integers to request specific degrees in any order.

```python
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(y^2-(x-1/2)^3)
H = hilbert_samuel_function(I, point=(1/2,0), degrees=[5,0,1])
print(H.values, H.graded_values)  # [11, 1, 3] and [2, 1, 2]
print(hilbert_samuel_multiplicity(I, point=(3/2,1)).value)  # 1
```

Both functions also accept a saved local standard basis and use its recorded point. The HTTP operations take `operands.ideal` and optional `operands.point`. The function additionally accepts `operands.degrees`, an array of nonnegative integers or exact decimal strings. Its result contains `degrees`, `values`, `graded_values`, `dimension` and `multiplicity`. An explicit empty list returns the dimension and multiplicity with empty value lists. The multiplicity operation returns `dimension` and the integer `value`. Pass `--compute gpu` for eligible Gröbner steps. [Computation and Singular checks](../native/KERNEL.md#hilbert-samuel-computation)

### Tangent cones and their Hilbert series

For an ideal $I$ and a rational point $p$, let $A$ be its local quotient ring and $\mathfrak m$ its maximal ideal. The tangent cone is the affine scheme with coordinate ring $\operatorname{gr}_{\mathfrak m}(A)$. Atlas gives its homogeneous defining equations in **direction coordinates** $\Delta x_i=x_i-p_i$. Its Hilbert series is $\sum_{n\geq0}\dim_{\mathbb Q}(\mathfrak m^n/\mathfrak m^{n+1})t^n$. The coefficients agree with the `graded_values` of the Hilbert–Samuel function.

```sh
./atlas algebra tangent-cone --vars x,y --gen '(y-2)^2-(x-1/2)^3' --point '1/2,2'
./atlas algebra tangent-cone-hilbert-series --vars x,y --gen '(y-2)^2-(x-1/2)^3' --point '1/2,2'
```

| Point on the same curve | Equation in tangent directions | Graded series | Multiplicity |
| --- | --- | --- | ---: |
| $(1/2,2)$, cusp | $(\Delta y)^2=0$ | $(1-t^2)/(1-t)^2$ | 2 |
| $(3/2,3)$, smooth point | $3\Delta x-2\Delta y=0$ | $(1-t)/(1-t)^2$ | 1 |

A tangent cone can be reused as the input to a series calculation at its vertex. Enter these statements at the Atlas prompt, or save them as `cusp.atlas` and run `./atlas run cusp.atlas`.

```atlas
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal((y-2)^2-(x-1/2)^3)
C = tangent_cone(I, point=(1/2,2))
print(C.generators, C.multiplicity)
S = tangent_cone_hilbert_series(C)
print(S.numerator, S.denominator_power)
```

The cone's `generators` use the declared variable names in a separate tangent-coordinate space. In the example the serialized generator is `y^2`, where `y` denotes $\Delta y$. When a saved cone supplies the ideal, its vertex is zero in those coordinates. Both operations also accept a saved local standard basis in the original space at its certified point.

For HTTP, post `operation` equal to `tangent-cone` or `tangent-cone-hilbert-series` with `operands.ideal` and optional `operands.point`. Both results include the local dimension and multiplicity. The cone records `coordinate_system: "tangent"`. The series returns the nonzero signed coefficients of its numerator as ordered `{degree, coefficient}` pairs and `denominator_power` for the denominator $(1-t)^r$, where $r$ is the number of input variables. At a point outside the support, the cone ideal is $(1)$ and the series is zero. [Kernel and verification](../native/KERNEL.md#tangent-cones-and-graded-hilbert-series)

### Local dimension and smoothness

Use `local-dimension` for the Krull dimension of the local ring at a rational point. Use `local-smoothness` for a Boolean result together with the local dimension, embedding dimension and Jacobian rank. Over $\mathbb Q$, a point is smooth exactly when its local and embedding dimensions agree. Each generator must vanish at the point. Omitting the point selects the origin.

For example, the ideal $(xy,xz)$ defines the union of the plane $x=0$ and the line $y=z=0$ in affine three-space.

| Point | Local dimension | Embedding dimension | Smooth |
| --- | ---: | ---: | --- |
| $(0,1/2,0)$ on the plane | 2 | 2 | Yes |
| $(2/3,0,0)$ on the line | 1 | 1 | Yes |
| $(0,0,0)$ at the intersection | 2 | 3 | No |

```sh
./atlas algebra local-dimension --vars x,y,z --gen 'x*y' --gen 'x*z' --point '2/3,0,0'
./atlas algebra local-smoothness --vars x,y,z --gen 'x*y' --gen 'x*z' --point '0,0,0'
```

The following three branches meet at $(1,1,0)$. Atlas returns local dimension 1, embedding dimension 3, Jacobian rank 0, and `smooth = false`. Expanding the monomial $x^{1001}y^{1001}$ about this point alone would produce 1,004,004 terms. Both operations accept the sparse equations directly.

```sh
./atlas algebra local-smoothness --vars x,y,z --point 1,1,0 --compute cpu \
  --gen '(x^1001-1)*(y^1001-1)' --gen '(x^1001-1)*z' --gen '(y^1001-1)*z'
```

```atlas
x, y, z = coordinates("x", "y", "z", coeffs=QQ)
I = ideal(x*y, x*z)
d = local_dimension(I, point=(2/3,0,0))
s = local_smoothness(I)
print(d.value, s.value, s.local_dimension, s.embedding_dimension, s.jacobian_rank)
```

Local dimension and local smoothness accept a saved local standard basis at its recorded point or a saved tangent cone at its vertex. The HTTP operations take `operands.ideal` and optional `operands.point`. The two results expose the same fields in scripts and JSON.

Use `--compute gpu` to request acceleration of eligible modular standard-basis or Jacobian rank calculations. Atlas verifies these results over the rationals. The execution record identifies the stages that used the device. [Ordinary CPU timings against Singular](../benchmarks/local_dimension_cpu.py) · [Large coupled ideal timings](../benchmarks/large-coupled-local.md)

### Zariski tangent spaces and embedding dimensions

For $X=V(I)$ and a rational point $p$ on $X$, a tangent direction $v$ satisfies $df_p(v)=0$ for every generator $f$ of $I$. Atlas returns the independent linear equations and a basis of their solution space in the declared variable order. The embedding dimension is the dimension of this vector space. These operations preserve the scheme structure. For example, $I=(x^2)$ and $I=(x)$ in $\mathbb Q[x]$ have embedding dimensions 1 and 0 at the origin.

```sh
./atlas algebra zariski-tangent-space --vars x,y --gen '(y-2)^2-(x-1/2)^3' --point '3/2,3'
./atlas algebra embedding-dimension --vars x,y --gen '(y-2)^2-(x-1/2)^3' --point '1/2,2'
```

| Point on the same curve | Tangent equation | Basis in $(\Delta x,\Delta y)$ | Embedding dimension |
| --- | --- | --- | ---: |
| $(1/2,2)$, cusp | No linear constraints | $(1,0),(0,1)$ | 2 |
| $(3/2,3)$, smooth | $\Delta x-\tfrac{2}{3}\Delta y=0$ | $(2/3,1)$ | 1 |

At the cusp the tangent cone has equation $(\Delta y)^2=0$. Its quadratic term contributes no linear equation to the Zariski tangent space.

Omitting the point selects the origin. All the generators must vanish at the chosen point. The tangent-space calculation labels its direction coordinates and gives the basis vectors in that order. Pass `--compute gpu` to request modular Jacobian row reduction with exact host verification.

```atlas
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal((y-2)^2-(x-1/2)^3)
V = zariski_tangent_space(I, point=(3/2,3))
print(V.equations, V.basis, V.dimension)
print(embedding_dimension(I, point=(1/2,2)).value)
```

Both functions accept a saved local standard basis at its recorded point or a saved tangent cone at its vertex. In a tangent-space result, `V.coordinates` are directions in their own coordinate space and `V.basis` contains exact rationals. For HTTP, send `operands.ideal` and optional `operands.point`. Both results report `ambient_dimension` and `jacobian_rank`. The tangent-space result carries `dimension`, `equations` and `basis`, with vector entries encoded as rational strings. The embedding-dimension result carries the integer `value`. [Kernel and verification](../native/KERNEL.md#zariski-tangent-spaces)

### Local standard bases and normal forms at a rational point

`local-standard-basis` computes a standard basis of an ideal in the local ring at a rational point $p$. `local-normal-form` reduces a polynomial in that ring. Omit the point to use the origin. Supply its coordinates in the declared variable order as exact integers or fractions. Atlas compares the lowest-degree terms in $x_i-p_i$ first and breaks ties using the selected lex or grevlex order. Returned polynomials use the original variable names and coordinates.

`make groebner` builds the native executable for these two shell commands. The normal `./atlas algebra` invocation uses it for supported CPU work. With `--json`, `execution.implementation` is `native-cli` when that executable computed the answer. Explicit GPU requests and eligible automatic GPU profiles use the shared device dispatcher.

For example, the ideal below has support at $(1/2,-2)$ and $(2,-2)$. At the first point $x-2$ is a unit, so $x=1/2$ and the normal form of $x+y$ is $-3/2$. Repeating the reduction at the second point gives zero.

```sh
./atlas algebra local-standard-basis --vars x,y --point '1/2,-2' \
  --gen '(x-1/2)*(x-2)' --gen 'y+2'
./atlas algebra local-normal-form --vars x,y --point '1/2,-2' \
  --gen '(x-1/2)*(x-2)' --gen 'y+2' --poly 'x+y' --json
```

```python
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal((x-1/2)*(x-2), y+2)
B = local_standard_basis(I, point=["1/2", "-2"])
H = local_normal_form(x+y, B)
print(B.generators, H.remainder, H.unit)
print(local_normal_form(x+y, I, point=[2, -2]).remainder)
```

A saved local standard basis retains its point for subsequent reductions.

Large shifted powers can stay in the original coordinates. For example, let $N=2^{64}+7$ and $I=(x^Nz-y^2)$ at $(1,0,0)$. The factor $x^N$ is a unit there. Reducing $z$ gives remainder $y^2$ and unit $x^N$, so the returned certificate is exactly the defining equation. Both commands take the exponent as an integer.

```sh
./atlas algebra local-standard-basis --vars x,y,z --point 1,0,0 \
  --gen 'x^18446744073709551623*z-y^2'
./atlas algebra local-normal-form --vars x,y,z --point 1,0,0 \
  --gen 'x^18446744073709551623*z-y^2' --poly z
```

Over HTTP, use `operands.ideal`, optional `operands.point` (for example `["1/2", -2]`), and `operands.polynomial` for a normal form. Both operations return `point` as an array of reduced rational strings. The standard basis has a `generators` array. For the normal form, write $r$ for `remainder` and $u$ for `unit`. Then $u(p)=1$ and $uf-r$ belongs to the localized ideal. The class of the input is $r/u$, and $r=0$ tests local membership. For a finite-dimensional quotient Atlas fully reduces the monomials and returns $u=1$. A weak reduction guarantees an irreducible leading monomial when the remainder is nonzero. The `reduction` field identifies the guarantee for the result. Use `--compute gpu` to request device work and inspect the execution record for the stages that ran.

For a positive-dimensional example, reduce $x$ modulo $((1-y)x-y^2)$ at the origin. Atlas returns $r=y^2$ and $u=1-y$. This gives the exact class $x=y^2/(1-y)$. A `standard` result can therefore have a nonconstant unit. A `weak` result may also retain terms divisible by a local basis leader after its first irreducible term. Singular's local reduction likewise allows different representatives. [Singular tutorial, §3.7](https://www.singular.uni-kl.de/DEMOS/SummerSchool-Trieste-09/Singular_Tutorial.pdf)

## Shell, prompt and scripts

```sh
./atlas algebra normal-form --vars x,y --order lex --compute cpu \
  --gen 'x^2-1' --gen 'y-x' --poly 'x^2+y/2' --json
./atlas run examples/algebra-normal-form.atlas
```

At the prompt, start the shell command at `algebra`, or use the same functions as an Atlas script. A named basis can serve multiple reductions.

```python
x, y = coordinates("x", "y", coeffs=QQ)
I = ideal(x^2-1, y-x, order="lex")
B = groebner(I, compute="cpu")
r = normal_form(x^2+y/2, B, compute="cpu")
print(r == 1+y/2)
print(membership(y^2-1, B).value)
print(r.execution.device)
```

Use `ideal(variables=(x,y))` for the empty generating list or `ideal(1, variables=(x,y))` for the unit ideal. Coordinates from separate declarations have distinct identities. To work with the ideal returned by `J = eliminate(I, x)`, unpack `t, = J.coordinates`. Then `membership(t^2-1, J).value` is true in `QQ[y]`. Each result exposes its `.ring`, `.execution`, and full `.json` record.

`polynomial("y^2-1", variables=J.coordinates)` parses a literal in that same surviving coordinate space. Supply `variables=[]` for a rational constant in ℚ. A coordinate name can differ from its console binding, for example `t = coordinates("if")` followed by `f = polynomial("if+1", variables=[t])`.

Use `--compute cpu|auto|gpu` at the shell, or the `compute` keyword in a script. In a GPU run of the normal-form example on an AMD Radeon 680M, the execution record showed one OpenCL replay batch covering two primes, CPU certification of one basis, and CPU reduction by two divisors. If a requested device cannot run a stage, inspect `execution.stages` and `execution.fallback_reasons` for the observed backend and reason.

For HTTP, start `./atlas serve` and POST the example request to `http://127.0.0.1:8765/api/algebra`. HTTP and the shell return the same success and error formats. Invalid mathematical inputs return HTTP 400, a worker limit or unsuccessful calculation returns 422, and an unavailable worker or kernel returns 503. An error response carries no mathematical result.

## Reuse a basis over HTTP

A successful `groebner` response can include an opaque top-level `basis_ref`.
Pass it in place of `operands.ideal` for a subsequent `normal-form`, `membership` or
`multiplicity` calculation in the same ring. The field, variable order, and
monomial order must match. Each request still selects its own compute device.

```json
{
  "operation": "normal-form",
  "ring": {"field": "QQ", "variables": ["x", "y"], "order": "lex"},
  "operands": {"basis_ref": "TOKEN_FROM_GROEBNER_RESPONSE", "polynomial": "x^2+y/2"}
}
```

The server retains a bounded set of verified bases in memory. An oversized
basis can be returned with its ordinary generators alone. Restarting the server
or evicting an entry makes that reference stale. HTTP 410 with
`stale_basis_ref` means to compute the basis again. HTTP 400 with
`basis_ring_mismatch` identifies a ring mismatch. Both command and script
replay in the result contain the mathematical operands and work independently
of the reference.
