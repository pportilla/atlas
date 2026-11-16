# Algebra requests and replay

Start `./atlas serve` and send JSON to `POST /api/algebra`. For a one-off calculation from a shell, use `./atlas algebra OPERATION --help` for the corresponding flags and add `--json` to get the same record format.

## Reduce a polynomial

In $\mathbb Q[x,y]/(x^2-1,y-x)$, the remainder of $x^2+y/2$ in lex order is $1+y/2$. Paste this request into a second terminal while the server is running.

```sh
curl -s http://127.0.0.1:8765/api/algebra \
  -H 'Content-Type: application/json' \
  -d '{
    "operation": "normal-form",
    "ring": {"variables": ["x", "y"], "order": "lex"},
    "operands": {"ideal": ["x^2-1", "y-x"], "polynomial": "x^2+y/2"},
    "compute": "cpu"
  }'
```

The HTTP 200 response has `status: "ok"`. Its `result` object is

```json
{
  "type": "polynomial",
  "ring": {"field": "QQ", "variables": ["x", "y"], "order": "lex"},
  "polynomial": "1/2*y+1"
}
```

Use `execution.requested`, `execution.stages` and `execution.fallback_reasons` to inspect the device policy and stages used. The top-level `replay.command` and `replay.script` reproduce the mathematical request. Elapsed times depend on the run. These shell commands perform the same calculation directly.

```sh
./atlas algebra normal-form --vars x,y --order lex --compute cpu \
  --gen 'x^2-1' --gen 'y-x' --poly 'x^2+y/2' --json
./atlas run examples/algebra-normal-form.atlas
```

[Runnable script](algebra-normal-form.atlas) · [Algebra interface tests](../tests/test_algebra_interface.py) · [Request parser and dispatch](../engine/algebra.py)

## Specify a ring and its operands

`ring.variables` gives distinct names from highest to lowest priority. Use `[]` for the scalar field. `ring.field` defaults to `"QQ"`, `ring.order` to `"grevlex"`, and `compute` to `"auto"`. The supported orders are `"lex"` and `"grevlex"`. Polynomial strings accept integer literals, division by a nonzero rational constant, multiplication with `*`, and powers with `^` or `**`. An `ideal` is an array of polynomial strings. Either `[]` or `["0"]` declares the zero ideal.

Supply `ring.relations` to work in a quotient. For example, with
`"ring": {"variables": ["x","y"], "relations": ["y^2+x^5"]}`,
the `ideal` operation on `{"ideal": ["y","x^5"]}` returns a
`quotient-ideal` with generators `["y","-y^2"]`. The same presentation
works for `ideal-sum`, `ideal-product`, `ideal-power`, `groebner`,
`membership`, `normal-form` and a further `quotient-ring` declaration.
At the shell, use one `--relation` per equation, or `--quotient-ring` for
a quotient by the zero ideal.

| Operation | Keys in `operands` | Read from `result` |
| --- | --- | --- |
| `factor` | `polynomial` | Rational `unit` and `factors` with `polynomial` and `multiplicity` |
| `gcd` | `polynomial`, `other_polynomial` | `polynomial` |
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

For `quotient-ring` the default action is `declare`. Supply `polynomial` for `reduce`, `negate`, `unit` or `inverse`. Also supply `other_polynomial` for `add`, `subtract`, `multiply`, `divide` or `equal`. Use `polynomial` and a signed integer `exponent` for `power`. Decimal strings preserve exponents wider than a client's native integer representation. [Quotient action and replay tests](../tests/test_quotient_ring_interface.py)

For example, the parametrized cusp from the [map-germ example](local-algebra.md#compare-the-cusp-and-its-parametrization) has this request body. Its result has `finite: true` and `value: 1`.

```json
{"operation":"ae-codimension","ring":{"variables":["t"]},"operands":{"components":["t^2","t^3"]}}
```

## Supply a rational point

Each of the following operations also accepts optional `operands.point`, an ordered array of exact integers or rational strings such as `["1/2", 0]`. An omitted point selects the origin. A local standard basis and a local normal form always return their point. Other point operations distinguish an omitted origin from an explicitly supplied point through `scope`.

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

Use the polynomial and point together for a Milnor or Tjurina number. For a local quotient basis, supply generators of an ideal. [Local algebra requests and expected responses](../tests/test_point_local_algebra_interface.py)

## Reuse a basis over HTTP

A successful Gröbner basis, quotient ring or quotient element response can return a top-level `basis_ref`. Use this token in place of `operands.ideal` for `normal-form`, `membership`, `multiplicity` or `quotient-ring`. The field, variable order and monomial order must match the recorded ring. Each request chooses its own compute policy.

```json
{
  "operation": "normal-form",
  "ring": {"variables": ["x", "y"], "order": "lex"},
  "operands": {"basis_ref": "TOKEN_FROM_GROEBNER_RESPONSE", "polynomial": "x^2+y/2"}
}
```

The server keeps a bounded set of verified bases in memory. An oversized basis can be returned with its generators alone. HTTP 410 with `stale_basis_ref` means the token is stale, for example after eviction or a server restart. Recompute the basis to obtain a new token. HTTP 400 with `basis_ring_mismatch` identifies a ring mismatch. Replay forms carry their own operands and remain usable independently of the token. [Basis reuse and expiry tests](../tests/test_http_basis_refs.py) · [Server basis store](../engine/algebra_basis_store.py)

## Read an error response

Sending the first example with `"order": "local"` returns HTTP 400. To request a local normal form use the operation `local-normal-form` with one of the supported monomial orders.

```json
{"status":"error","error":{"code":"invalid_request","message":"Unsupported monomial order. Choose lex or grevlex."}}
```

HTTP 413 signals an oversized standard request, 422 a computation limit or unsuccessful computation, and 503 an unavailable worker or kernel. Errors carry `error.code` and `error.message`. The exhaustive endpoint accepts larger requests and lifts the ordinary worker deadline. [HTTP endpoints and compute policies](../docs/http-and-compute.md)
