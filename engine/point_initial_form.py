"""Lowest homogeneous part at a rational point for a sparse principal equation."""

from . import native_polynomial as np
from .point_geometry import decode_terms, point_operation


def principal_initial_form(variables, shifts, f):
    """Return the lowest homogeneous part of f(p+X), up to a rational unit."""
    form = point_operation("initial", shifts, np.terms(variables, f))
    return np.from_terms(variables, decode_terms(form))
