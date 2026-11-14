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
| `intersection` | `other_ideal`, an array of polynomials | Ideal, in `generators` |
| `quotient` | `other_ideal` | Ideal quotient, in `generators` |
| `saturation` | `polynomial` | Saturation by one polynomial, in `generators` |
| `dimension` | | Krull dimension of the quotient as an integer `value`, with −1 for the empty variety |

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

A nonzero GCD has leading coefficient one in the recorded order. Thus `gcd(0,f)` is the monic associate of $f$, `gcd(0,0)` is zero, and coprime inputs or a nonzero constant operand give one. Use `variables=(x,y)` to place two constant operands in that ambient ring. In the browser Algebra view, choose **Polynomial GCD**, enter one polynomial in each field, and run the calculation.

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
Pass it in place of `operands.ideal` for a subsequent `normal-form` or
`membership` calculation in the same ring. The field, variable order, and
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
