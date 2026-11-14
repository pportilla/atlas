"""Certify the maximal ideal from sparse linear parts at a rational point."""

from . import native_polynomial as np
from .point_geometry import encode_terms, request
from .polynomial import rational_text


def maximal_ideal_from_linear_parts(variables, shifts, generators):
    return request("maximal", point=list(map(rational_text, shifts)),
                   ideal=[encode_terms(np.terms(variables, f)) for f in generators])
