"""Public handler for exact polynomial arithmetic over the rationals."""

from . import native_polynomial as polynomial
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
    if operation != "gcd":
        raise ValueError(f"Unknown polynomial operation {operation!r}.")
    result = monic_gcd(variables, operands["polynomial"], operands["other_polynomial"],
                       order=ring["order"])
    return {"type": "polynomial", "ring": ring,
            "polynomial": format_poly(polynomial.terms(variables, result), ring["variables"])}
