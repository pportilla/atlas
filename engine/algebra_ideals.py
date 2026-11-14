"""Phase 1 ideal handlers for validated rational algebra requests."""

from . import ideal_operations as ideals
from . import native_polynomial as polynomial
from .algebra_execution import cpu_stage, stage_label
from .algebra_groebner import reduced_basis
from .polynomial import format_poly


def _basis(name, variables, generators, order):
    with stage_label(name):
        return reduced_basis(generators, variables, order=order)


def handle(operation, ring, operands):
    variables = tuple(map(polynomial.Symbol, ring["variables"]))
    generators = operands["ideal"]
    if operation == "ideal":
        names = ring["variables"]
        return {"type": "ideal", "ring": ring,
                "generators": [format_poly(polynomial.terms(variables, f), names) for f in generators]}
    options = dict(basis=_basis, stage=cpu_stage)
    if operation == "dimension":
        value = ideals.dimension(variables, generators, **options)
        return {"type": "integer", "ring": ring, "value": value}

    options["order"] = ring["order"]
    if operation == "eliminate":
        removed = tuple(map(polynomial.Symbol, operands["variables"]))
        result = ideals.eliminate(variables, generators, removed, **options)
    elif operation == "intersection":
        result = ideals.intersection(variables, generators, operands["other_ideal"], **options)
    elif operation == "quotient":
        result = ideals.quotient(variables, generators, operands["other_ideal"], **options)
    elif operation == "saturation":
        result = ideals.saturate(variables, generators, operands["polynomial"], **options)
    else:
        raise ValueError(f"Unknown ideal operation {operation!r}.")

    names = [v.name for v in result.variables]
    return {"type": "ideal", "ring": {"field": "QQ", "variables": names, "order": result.order},
            "generators": [format_poly(polynomial.terms(result.variables, f), names)
                           for f in result.generators]}
