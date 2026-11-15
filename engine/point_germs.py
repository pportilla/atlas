"""Sparse generators of a rational ideal in coordinates centered at a point."""

from . import native_polynomial as np
from .point_geometry import decode_terms, encode_terms, point_operation, request
from .polynomial import integer_from_decimal, rational_text


def _translate(variables, shifts, polynomials):
    if not any(shifts):
        return polynomials
    mapping = {v: v + a for v, a in zip(variables, shifts) if a}
    return tuple(f.xreplace(mapping) for f in polynomials)


def _coprime_columns(bases):
    result = request("columns", point=list(map(rational_text, bases)))
    return (list(map(integer_from_decimal, result["bases"])),
            [list(map(integer_from_decimal, row)) for row in result["columns"]])


def nonzero_terms_at_point(shifts, terms):
    """Decide rational vanishing, combining dependent powers before expansion."""
    return point_operation("nonzero", shifts, terms)


def _large_translation(variables, shifts, f, *, min_terms=65537):
    return point_operation("large", shifts, np.terms(variables, f), limit=min_terms)


def centered_ideal(variables, shifts, generators):
    """Return generators of the translated local ideal, skipping proven unit factors."""
    variables, shifts = tuple(variables), tuple(shifts)
    plan = request("centered", point=list(map(rational_text, shifts)),
                   ideal=[encode_terms(np.terms(variables, f)) for f in generators if f])
    if plan.get("unit"):
        return (np.S.One,)
    if plan.get("maximal"):
        return variables
    centered = []
    for row in plan["rows"]:
        powers = tuple(map(integer_from_decimal, row["powers"]))
        f = np.from_terms(variables, decode_terms(row["residual"]))
        if f != 1 and any(shifts):
            f, = _translate(variables, shifts, (f,))
        if any(powers):
            f = np.from_terms(variables, {powers: 1}) * f
        centered.append(f)
    return tuple(centered)
