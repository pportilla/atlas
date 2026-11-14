"""Ideal operations for validated rational algebra requests."""

import json

from . import ideal_operations as ideals
from . import native_polynomial as polynomial
from .algebra_execution import cpu_stage, current_execution, stage_label
from .algebra_groebner import reduced_basis
from .native_library import affine_request
from .polynomial import InputError, format_poly, sparse_row


def _basis(name, variables, generators, order):
    with stage_label(name):
        return reduced_basis(generators, variables, order=order)


def _primary_decomposition(ring, variables, generators):
    names = ring["variables"]
    request = ["primary_decomposition", " ".join(names), ring["order"]]
    request.extend(polynomial.row(variables, f) for f in generators)
    with cpu_stage("primary_decomposition", generators=len(generators)):
        rows = iter(affine_request(request))

    def generators_from_rows(count):
        return [format_poly(sparse_row(next(rows)), names) for _ in range(count)]

    components = []
    for _ in range(int(next(rows))):
        nq, np = map(int, next(rows).split())
        components.append({"primary": generators_from_rows(nq), "prime": generators_from_rows(np)})
    return {"type": "primary-decomposition", "ring": ring, "components": components}


def handle(operation, ring, operands):
    variables = tuple(map(polynomial.Symbol, ring["variables"]))
    generators = operands["ideal"]
    if operation in ("normalization", "localization", "ideal-power", "symbolic-power"):
        opcode = operation.replace("-", "_")
        request = [opcode, " ".join(ring["variables"]), ring["order"]]
        rows = generators
        if operation == "localization":
            prime = operands["prime"]
            query = "numerator" in operands or "denominator" in operands
            request.append(f"{len(generators)} {len(prime)} {int(query)}")
            rows = [*generators, *prime]
            if query:
                rows.extend(operands.get(key, 1) for key in ("numerator", "denominator"))
        elif "exponent" in operands:
            request.append(operands["exponent"])
        with cpu_stage(opcode, generators=len(generators)):
            reply, = affine_request([*request, *(polynomial.row(variables, f) for f in rows)])
        result = json.loads(reply)
        if "error" in result:
            raise InputError(result["error"]["message"])
        return result
    if operation == "primary-decomposition":
        return _primary_decomposition(ring, variables, generators)
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
    elif operation == "radical":
        scope = current_execution()
        op = "ideal_radical_gpu" if scope and scope.replay.compute == "gpu" else "ideal_radical"
        request = [op, " ".join(ring["variables"]), ring["order"]]
        with cpu_stage("radical", generators=len(generators)):
            rows = affine_request([*request, *(polynomial.row(variables, g) for g in generators)])
        return {"type": "ideal", "ring": ring, "generators": [format_poly(sparse_row(row), ring["variables"])
                                                             for row in rows]}
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
