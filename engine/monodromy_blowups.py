"""A'Campo strata above the origin for point blowups over the complex numbers."""

from .monodromy import _request
from .polynomial import integer_from_decimal, integer_text


def projective_complement_euler(dimension, degree):
    """Euler characteristic of P^(dimension-1) minus a smooth degree-d hypersurface."""
    # Gauss–Bonnet gives the coefficient of h^(dimension−1) in
    # (1+h)^dimension/(1+degree*h) for the complement.
    return integer_from_decimal(_request("projective_complement", dimension=dimension, degree=integer_text(degree)))
