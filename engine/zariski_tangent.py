"""Zariski tangent equations and direction vectors of a QQ scheme at a rational point."""

from fractions import Fraction

from . import native_polynomial as np
from .algebra_execution import cpu_stage
from .point_geometry import encode_terms, point_operation, request
from .polynomial import InputError, format_poly, rational_text
from .rational_point import coordinate_fraction, normalize_point


def _first_jet(terms, point, moving, fixed, factors, columns):
    """Evaluate the differential up to a single nonzero scalar, in the original axes."""
    return list(map(coordinate_fraction, point_operation("jet", point, terms)))


def _equations_and_basis(names, space):
    width = len(names)
    monomials = [(0,)*j + (1,) + (0,)*(width-j-1) for j in range(width)]
    equations = [format_poly({m: coordinate_fraction(c) for m, c in zip(monomials, row) if c != "0"}, names)
                 for row in space["rows"]]
    return equations, space["basis"]


def handle(operation, ring, operands):
    if operation not in ("zariski-tangent-space", "embedding-dimension"):
        raise InputError(f"Unknown tangent space operation {operation!r}.")
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Tangent spaces require QQ and lex or grevlex order.")
    names = ring["variables"]
    variables = tuple(map(np.Symbol, names))
    width = len(names)
    result = {"ring": ring, "scope": "origin", "ambient_dimension": width}
    point = (Fraction(0),)*width
    if "point" in operands:
        coordinates = normalize_point(operands["point"], width)
        result.update(scope="point", point=coordinates)
        point = tuple(map(coordinate_fraction, coordinates))

    generators = [np.coerce(f) for f in operands["ideal"] if f]
    with cpu_stage("tangent_first_jets", generators=len(generators), variables=width):
        rows = request("jets", point=list(map(rational_text, point)),
                       ideal=[encode_terms(np.terms(variables, f)) for f in generators])

    from .tangent_linear import linear_constraints
    dimension_only = operation == "embedding-dimension"
    space = linear_constraints(rows, width, dimension_only=dimension_only)
    rank, dimension = space["rank"], space["dimension"]
    result["jacobian_rank"] = rank
    if dimension_only:
        result.update(type="integer", value=dimension)
    else:
        with cpu_stage("tangent_directions", dimension=dimension, equations=rank):
            equations, basis = _equations_and_basis(names, space)
        result.update(type=operation, coordinate_system="tangent", dimension=dimension,
                      equations=equations, basis=basis)
    return result
