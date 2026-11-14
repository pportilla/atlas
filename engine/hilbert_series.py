"""Hilbert series of standard-graded quotients of a rational polynomial ring."""

from . import native_polynomial as np
from .algebra_execution import cpu_stage, stage_label
from .algebra_groebner import reduced_basis
from .hilbert_samuel import monomial_hilbert_series
from .native_library import affine_request
from .quotient_monomials import decode_row, encode_row
from .polynomial import InputError


def _graded_heads(operation, variables, generators, order):
    supports = [";".join(map(encode_row, np.terms(variables, f))) for f in generators]
    status, *rows = affine_request([operation, str(len(variables)), order, *supports])
    if status == "inhomogeneous":
        raise InputError("Hilbert series requires an ideal homogeneous in the standard total degree.")
    if status == "basis":
        return None
    return [decode_row(row) for row in rows]


def _homogeneous_initial(variables, generators, order):
    with cpu_stage("inspect_hilbert_generators", generators=len(generators)):
        heads = _graded_heads("monomial_homogeneous_inspect", variables, generators, order)
    if heads is not None:
        return heads
    # Reduction in the declared order also recognizes homogeneous ideals
    # presented by inhomogeneous generators.
    with stage_label("hilbert_series_basis"):
        basis = reduced_basis(generators, variables, order=order)
    with cpu_stage("graded_initial_monomials", generators=len(basis.exprs)):
        return _graded_heads("monomial_homogeneous_heads", variables, basis.exprs, order)


def handle(operation, ring, operands):
    if operation != "hilbert-series" or ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Hilbert series requires a polynomial ring over QQ with lex or grevlex order.")
    variables = tuple(map(np.Symbol, ring["variables"]))
    generators = tuple(np.coerce(f) for f in operands["ideal"] if f)
    heads = _homogeneous_initial(variables, generators, ring["order"])
    with cpu_stage("graded_hilbert_series", generators=len(heads)):
        series = monomial_hilbert_series(len(variables), heads)
    return {"type": "hilbert-series", "scope": "global", "ring": ring, **series}
