# Algebra requests and replay

The algebra endpoint accepts the same ring and polynomials you would use at the Atlas prompt. Each request describes its ring explicitly, so a client can reproduce a calculation without recreating an interactive session.

## Reduce a polynomial

What is the class of $x^2+y/2$ subject to $x^2=1$ and $y=x$? With the Atlas server running (`./atlas serve`), send this request from a second terminal.

```sh
curl -sS http://127.0.0.1:8765/api/algebra \
  -H 'Content-Type: application/json' \
  -d '{
    "operation": "normal-form",
    "ring": {"variables": ["x", "y"], "order": "lex"},
    "operands": {"ideal": ["x^2-1", "y-x"], "polynomial": "x^2+y/2"},
    "compute": "cpu"
  }'
```

The server returns HTTP 200 with `"status": "ok"`. The answer is in `result`.

```json
{
  "type": "polynomial",
  "ring": {"field": "QQ", "variables": ["x", "y"], "order": "lex"},
  "polynomial": "1/2*y+1"
}
```

This is the expected class $1+y/2$. The ring accompanies the polynomial so that a client knows how to interpret its variables and its normal form. For a calculation you want to keep, save the full response. Its `replay.command` and `replay.script` reproduce the calculation from the shell and from an Atlas script. The `execution` fields describe the device and stages used on this particular run.

You can also enter the original calculation directly from the shell. Use `./atlas algebra normal-form --help` to see its options. [Equivalent Atlas script](../../examples/algebra-normal-form.atlas) · [Gröbner bases at the prompt](../tutorial/polynomials-and-ideals.md#reduce-a-polynomial-modulo-an-ideal)

```sh
./atlas algebra normal-form --vars x,y --order lex --compute cpu \
  --gen 'x^2-1' --gen 'y-x' --poly 'x^2+y/2' --json
```

## Specify a ring and its operands

Supply the operation name, the ring, and its operands as three JSON fields. The optional top-level `compute` field selects `"auto"` (default), `"cpu"` or `"gpu"`.

| Ring field | Meaning |
| --- | --- |
| `variables` | Required. An array of distinct names in descending variable priority. An empty array denotes the scalar field. |
| `field` | The coefficient field, currently `"QQ"` (default). |
| `order` | The monomial order, either `"grevlex"` (default) or `"lex"`. |
| `relations` | Equations defining a quotient of the declared polynomial ring. An empty array declares a quotient by the zero ideal. Omit the field to use the polynomial ring itself. |

Write a polynomial as a string. Use `*` for multiplication, `^` or `**` for powers, and division by a nonzero rational constant for exact coefficients. Write an ideal as an array of generators. Both `[]` and `["0"]` describe the zero ideal.

For example, is $x$ a multiple of $x(1+y)$ in $\mathbb Q[x,y]/(y^2)$? Sending the following request body to `POST /api/algebra` gives `true` in `result.value`. Omit `relations` to ask the same question in the ambient polynomial ring, where the answer is `false`.

```json
{
  "operation": "membership",
  "ring": {"variables": ["x", "y"], "relations": ["y^2"]},
  "operands": {"ideal": ["x*(1+y)"], "polynomial": "x"}
}
```

The relation makes $1+y$ a unit with inverse $1-y$. The same quotient presentation works for `ideal`, `ideal-sum`, `ideal-product`, `ideal-power`, `groebner`, `normal-form` and a further `quotient-ring` declaration. On the command line give one `--relation` per equation, or use `--quotient-ring` for the quotient by the zero ideal. [The two ideals at the prompt](../tutorial/quotient-rings.md#compare-ideals-before-and-after-quotienting)

| Operation | Keys in `operands` | Read from `result` |
| --- | --- | --- |
| `polynomial-arithmetic` | `action`, `polynomial` and the action operand (`other_polynomial`, `variable`, `exponent` or `images`) | `polynomial` in the input ring |
| `modular-arithmetic` | `action`, `values`, `modulus`, and where applicable `other`, `other_modulus` or `field_polynomial`. Use an empty variable list. | Ordered `values` and output `modulus`. [Operations and examples](../tutorial/polynomials-and-ideals.md#compute-with-residues-and-lift-roots) |
| `factor` | `polynomial` | Rational `unit` and `factors` with `polynomial` and `multiplicity` |
| `absolute-factor` | `polynomial` | Rational `unit`, individual absolute `factors` with multiplicities, and their common coefficient field in `result.ring.field`. [Example](../tutorial/polynomials-and-ideals.md#factor-over-the-algebraic-closure) |
| `gcd` | `polynomial`, `other_polynomial` | `polynomial` |
| `hensel-factors` | `polynomial`, two `factors`, prime `modulus` | Lifted `left` and `right` factors, squared `modulus` |
| `cancel-fraction` | `polynomial`, `other_polynomial` | Coprime `numerator`, monic `denominator` |
| `squarefree-factor` | `polynomial` | `unit`, `factors` grouped by multiplicity |
| `extended-gcd` | `polynomial`, `other_polynomial` | `gcd`, `left_coefficient`, `right_coefficient` |
| `polynomial-division` | `polynomial`, `other_polynomial` | `quotient`, `remainder` |
| `matrix-analysis` | `rows`, optional `rhs`, `columns`. Use an empty variable list. | Rank, RREF, determinant, kernel, image and solution |
| `resultant` | `polynomial`, `other_polynomial`, `variables` (one elimination name in an array) | `polynomial` in surviving variables |
| `ideal`, `groebner`, `radical` | `ideal` | `generators` for the declaration, reduced basis or radical |
| `normal-form`, `membership` | `ideal`, `polynomial` | `polynomial` or Boolean `value` |
| `eliminate` | `ideal`, `variables` (array of names to remove) | `generators` in surviving variables |
| `intersection`, `quotient` | `ideal`, `other_ideal` | `generators` |
| `ideal-sum`, `ideal-product` | `ideal`, `other_ideal` | `generators` (no Gröbner basis computation) |
| `saturation` | `ideal`, `polynomial` | `generators` |
| `ideal-power`, `symbolic-power` | `ideal`, `exponent` (nonnegative) | `generators` |
| `primary-decomposition` | `ideal` | `components` with `primary` and `prime` generator lists |
| `normalization` | `ideal` | `input_reduced` and component presentations, fractions and coordinate images |
| `localization` | `ideal`, `prime`, optional `numerator`, `denominator` | Contracted `generators`, verified `prime` and optional `element` |
| `quotient-ring` | `ideal`, optional `action` and its operands (below) | `relations` and, for a residue class or predicate, `representative` or `value` |
| `dimension`, `multiplicity` | `ideal` | Krull dimension or total multiplicity in `value` |
| `quotient-basis` | `ideal` | `finite`, `dimension`, `basis` |
| `hilbert-series` | `ideal` | `numerator`, `denominator_power`, `dimension`, `multiplicity` |
| `ae-codimension` | `components` (ordered array of source polynomials vanishing at the origin) | `finite`, `value`, `method` and proof details |

For `quotient-ring` the default action is `declare`. Supply `polynomial` for `reduce`, `negate`, `unit` or `inverse`. Supply both `polynomial` and `other_polynomial` for `add`, `subtract`, `multiply`, `divide` or `equal`. For `power` give `polynomial` and a signed integer `exponent`. A decimal string preserves an exponent wider than a client’s native integer type. [Quotient actions and return values](saved-values.md#quotient-rings-and-residue-classes)

For a parametrization, the variables are those of the source and the ordered components describe the target. The cusp $t\mapsto(t^2,t^3)$ gives `finite: true` and `value: 1`. [The cusp and its parametrization](../tutorial/local-algebra.md#compare-the-cusp-and-its-parametrization)

```json
{"operation":"ae-codimension","ring":{"variables":["t"]},"operands":{"components":["t^2","t^3"]}}
```

## Supply a rational point

For local calculations, `operands.point` is an optional array in declared coordinate order. Use JSON integers or rational strings such as `"1/2"`. Omitting the point selects the origin. To compute the Milnor number at the cusp $(1/2,2)$, send the translated equation together with that point.

```json
{
  "operation": "milnor-number",
  "ring": {"variables": ["x", "y"]},
  "operands": {"polynomial": "(y-2)^2-(x-1/2)^3", "point": ["1/2", 2]}
}
```

The result gives `value: 2` and `point: ["1/2", "2"]`. When a client compares answers at several points, keep `scope` as well as `value`. A local standard basis or normal form always records its point. Other point operations distinguish an omitted origin from an explicitly supplied point through `scope`. [Local calculations at rational points](../tutorial/local-algebra.md#evaluate-one-curve-at-three-points)

| Operation | Other keys in `operands` | Read from `result` |
| --- | --- | --- |
| `local-standard-basis` | `ideal` | `generators` |
| `local-normal-form` | `ideal`, `polynomial` | `remainder`, `unit`, `reduction` |
| `local-quotient-basis` | `ideal` | `finite`, `dimension`, `basis` |
| `local-jacobian-algebra` | `polynomial` | Partial derivatives in `generators`, plus `finite`, `dimension`, `basis` |
| `milnor-number`, `tjurina-number` | `polynomial` | `finite`, `value` |
| `local-dimension`, `local-smoothness` | `ideal` | Integer or Boolean `value`, `local_dimension`. Smoothness also gives `embedding_dimension` and `jacobian_rank` |
| `zariski-tangent-space`, `embedding-dimension` | `ideal` | `ambient_dimension`, `jacobian_rank`, plus tangent `dimension`, `equations` and `basis`, or integer `value` |
| `tangent-cone`, `tangent-cone-hilbert-series` | `ideal` | `dimension`, `multiplicity`, plus tangent `generators` or graded `numerator` and `denominator_power` |
| `hilbert-samuel-function` | `ideal`, optional `degrees` | `degrees`, `values`, `graded_values`, `dimension`, `multiplicity` |
| `hilbert-samuel-multiplicity` | `ideal` | `dimension`, integer `value` |

## Reuse a basis over HTTP

When reducing several polynomials modulo the same ideal, compute the Gröbner basis once. The server returns a `basis_ref` token for a basis it has retained. Send that token in place of `operands.ideal` on the next request. Here is a complete Python client using the standard library.

```python
import json
from urllib.request import Request, urlopen

url = "http://127.0.0.1:8765/api/algebra"
ring = {"variables": ["x", "y"], "order": "lex"}

def calculate(operation, operands):
    payload = {"operation": operation, "ring": ring, "operands": operands}
    request = Request(url, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.load(response)

basis = calculate("groebner", {"ideal": ["x^2-1", "y-x"]})
key = basis["basis_ref"]
reduced = calculate("normal-form", {"basis_ref": key, "polynomial": "x^2+y/2"})
member = calculate("membership", {"basis_ref": key, "polynomial": "y^2-1"})
print(reduced["result"]["polynomial"], member["result"]["value"])  # 1/2*y+1 True
```

You can use a reference with `normal-form`, `membership`, `multiplicity` or `quotient-ring`. Quotient ring and residue class responses can also supply a reference for their relation basis. Keep the field, variable order and monomial order fixed when reusing it. Each call may choose its own compute policy.

The server retains bases in memory and can return a large basis with generators alone. If a response has no token, send the generators in `operands.ideal`. If a token expires, for example after a server restart, the response is HTTP 410 with `stale_basis_ref`. Recompute the basis or send the generators. A saved `replay` already contains the necessary operands.

## Read an error response

Suppose the first request has `"order": "local"`. The server returns HTTP 400 with the following body.

```json
{"status":"error","error":{"code":"invalid_request","message":"Unsupported monomial order. Choose lex or grevlex."}}
```

To reduce in a local ring, choose `local-normal-form` as the operation and supply `operands.point`. The ring order remains `"lex"` or `"grevlex"`.

| HTTP status | What to do |
| --- | --- |
| 400 | Read `error.message` for the invalid field. For `basis_ring_mismatch`, use the same ring as the request that produced the token. |
| 410 | The server no longer has this basis reference. Send the ideal or compute the basis again. |
| 413 | Check that the request body is nonempty. For a large calculation use `POST /api/algebra/exhaustive`. |
| 422 | Inspect `error.code` and `error.message` for the failed computation or limit. |
| 429 | Both computation slots are in use. Retry after one becomes free. |
| 503 | The worker or kernel is unavailable. Check the server before retrying. |

The exhaustive algebra endpoint also lifts the ordinary worker deadline. [Server limits and compute options](http-and-compute.md#submit-an-algebra-request)
