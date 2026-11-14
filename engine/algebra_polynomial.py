"""Public handler for exact polynomial arithmetic over the rationals."""

from . import native_polynomial as polynomial
from .polynomial import format_poly
from .polynomial_gcd import monic_gcd


def handle(operation, ring, operands):
    if operation != "gcd":
        raise ValueError(f"Unknown polynomial operation {operation!r}.")
    variables = tuple(map(polynomial.Symbol, ring["variables"]))
    result = monic_gcd(variables, operands["polynomial"], operands["other_polynomial"],
                       order=ring["order"])
    return {"type": "polynomial", "ring": ring,
            "polynomial": format_poly(polynomial.terms(variables, result), ring["variables"])}
