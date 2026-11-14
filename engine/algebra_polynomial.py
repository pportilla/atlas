"""Public handler for exact polynomial arithmetic over the rationals."""

from . import native_polynomial as polynomial
from .affine_kernel import _request, _ring
from .algebra_execution import cpu_stage
from .polynomial import format_poly, rational_text
from .polynomial_gcd import monic_gcd


def handle(operation, ring, operands):
    variables = tuple(map(polynomial.Symbol, ring["variables"]))
    if operation == "factor":
        from .polynomial_factorization import factor_polynomial

        unit, factors = factor_polynomial(variables, operands["polynomial"], order=ring["order"])
        return {"type": "factorization", "ring": ring, "unit": rational_text(unit),
                "factors": [{"polynomial": format_poly(polynomial.terms(variables, factor), ring["variables"]),
                             "multiplicity": multiplicity} for factor, multiplicity in factors]}
    if operation == "resultant":
        name, = operands["variables"]
        index = ring["variables"].index(name)
        source = _ring(variables)
        with cpu_stage("polynomial_resultant", algorithm="flint", variable=name):
            row, = _request(["resultant", source.header, str(index), source.text(operands["polynomial"]),
                             source.text(operands["other_polynomial"])])
            result = polynomial.from_row(variables, row)
        ring = {**ring, "variables": [v for v in ring["variables"] if v != name]}
        variables = variables[:index] + variables[index + 1:]
    elif operation == "gcd":
        result = monic_gcd(variables, operands["polynomial"], operands["other_polynomial"],
                           order=ring["order"])
    else:
        raise ValueError(f"Unknown polynomial operation {operation!r}.")
    return {"type": "polynomial", "ring": ring,
            "polynomial": format_poly(polynomial.terms(variables, result), ring["variables"])}
