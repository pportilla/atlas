"""Local standard bases and normal forms over QQ at rational affine points."""

from . import native_polynomial as np
from .algebra_execution import cpu_stage, current_execution, stage_label
from .local_kernel import local_polynomials, local_request
from .point_germs import _nonzero_at_point, _translate
from .polynomial import InputError, format_poly
from .rational_point import coordinate_fraction, normalize_point


def _standard_basis(variables, generators, order):
    run = current_execution()
    operation = "local_basis_gpu" if run and run.replay.compute == "gpu" else "local_basis"
    with stage_label("local_standard_basis"), cpu_stage("local_standard_basis", generators=len(generators)):
        return local_polynomials(operation, variables, order, generators)


def _center_query(variables, shifts, query, basis, order):
    if not any(shifts):
        return query
    reply = local_polynomials("local_query_jet", variables, order, (*basis, query), point=shifts)
    return reply[0] if reply else _translate(variables, shifts, (query,))[0]


def _normal_form(variables, basis, query, order):
    run = current_execution()
    if run and run.replay.compute == "gpu" and basis:
        from .gpu_normal_form import normal_form
        heads = local_request("local_associate_heads", variables, order, basis)
        if heads:
            with stage_label("local_normal_form"):
                remainder = normal_form(variables, heads, np.row(variables, query))
            if remainder is not None:
                return np.S.One, np.from_row(variables, remainder), "standard"
    reduction, unit, remainder = local_request("local_normal_form", variables, order, (*basis, query))
    return np.from_row(variables, unit), np.from_row(variables, remainder), reduction


def handle(operation, ring, operands, *, known_basis=False):
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Local standard bases require QQ and lex or grevlex order.")
    names = ring["variables"]
    variables = tuple(map(np.Symbol, names))
    point = normalize_point(operands.get("point", [0] * len(names)), len(names))
    shifts = tuple(map(coordinate_fraction, point))
    normal = operation == "local-normal-form"
    if not normal and operation != "local-standard-basis":
        raise InputError(f"Unknown local standard basis operation {operation!r}.")
    result = {"type": operation, "ring": ring, "scope": "point", "point": point}
    generators = tuple(g for f in operands["ideal"] if (g := np.coerce(f)))
    query = np.coerce(operands["polynomial"]) if normal else None
    projected = None
    with cpu_stage("inspect_point_ideal", generators=len(generators)):
        if normal and (not query or not generators):
            result.update(reduction="standard", unit="1",
                          remainder=format_poly(np.terms(variables, query), names))
            return result
        if any(shifts) and generators and all(len(g._terms) == 1 for g in generators):
            projected = local_polynomials("local_project", variables, ring["order"], generators, point=shifts)
        if any(g.is_number for g in generators) or projected == (np.S.One,) or (
                projected is None and any(shifts) and
                any(_nonzero_at_point(variables, shifts, g) for g in generators)):
            if normal:
                result.update(reduction="standard", unit="1", remainder="0")
            else:
                result["generators"] = ["1"]
            return result
    if projected is not None:
        if normal:
            with cpu_stage("local_normal_form", divisors=len(projected)):
                _, remainder, reduction = _normal_form(variables, projected, query, ring["order"])
            result.update(unit="1", reduction=reduction, remainder=format_poly(np.terms(variables, remainder), names))
        else:
            result["generators"] = [format_poly(np.terms(variables, f), names) for f in projected]
        return result
    with cpu_stage("center_at_point", variables=len(names)):
        centered = _translate(variables, shifts, generators)
    if normal and known_basis:
        # Only private worker metadata can identify rows already certified at
        # this same point and order. Wire generator arrays require a fresh basis.
        with cpu_stage("restore_local_basis", generators=len(centered)):
            basis = centered
    else:
        basis = _standard_basis(variables, centered, ring["order"])
    if normal:
        if basis == (np.S.One,):
            result.update(reduction="standard", unit="1", remainder="0")
            return result
        with cpu_stage("center_query_at_point", variables=len(names)):
            query = _center_query(variables, shifts, query, basis, ring["order"])
        with cpu_stage("local_normal_form", divisors=len(basis)):
            unit, remainder, reduction = _normal_form(variables, basis, query, ring["order"])
        outputs = (unit, remainder)
    else:
        outputs = basis
    with cpu_stage("restore_point_coordinates", polynomials=len(outputs)):
        restored = _translate(variables, tuple(-a for a in shifts), outputs)
        texts = [format_poly(np.terms(variables, f), names) for f in restored]
    if normal:
        unit_text, remainder_text = texts
        result.update(unit=unit_text, remainder=remainder_text, reduction=reduction)
    else:
        result["generators"] = texts
    return result
