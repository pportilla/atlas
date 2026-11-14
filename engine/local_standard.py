"""Local standard bases and normal forms over QQ at rational affine points."""

from time import perf_counter_ns

from . import native_polynomial as np
from .algebra_execution import cpu_stage, current_execution, record_stage, stage_label
from .local_kernel import local_polynomials, local_request
from .point_germs import _nonzero_at_point, _translate
from .polynomial import InputError, format_poly
from .rational_point import coordinate_fraction, normalize_point


def _standard_basis(variables, generators, order):
    run = current_execution()
    operation = "local_basis_gpu" if run and run.replay.compute == "gpu" else "local_basis"
    with stage_label("local_standard_basis"), cpu_stage("local_standard_basis", generators=len(generators)):
        return local_polynomials(operation, variables, order, generators)


def _restore_point(variables, shifts, rows, order):
    if not any(shifts):
        return rows
    return local_polynomials("local_translate", variables, order, rows, point=tuple(-a for a in shifts))


def _center_query(variables, shifts, query, basis, order):
    if not any(shifts):
        return query
    reply = local_polynomials("local_query_jet", variables, order, (*basis, query), point=shifts)
    return reply[0] if reply else _translate(variables, shifts, (query,))[0]


def _homogeneous_remainder(variables, centered, order, query, shifts):
    if len(centered) < 2 or all(len(row._terms) <= 1 for row in centered):
        return None
    run = current_execution()
    operation = "local_homogeneous_normal_form"
    if run and run.replay.compute == "gpu":
        from .gpu_normal_form import _eligible
        # A nonmonomial input can have a monomial reduced basis. Let the
        # device dispatcher inspect it whenever a witness batch is possible.
        # Translation can enlarge the term count of a shifted query.
        if any(shifts) or _eligible(len(variables), len(query._terms), 1):
            return None
        operation += "_gpu"
    with stage_label("local_standard_basis"):
        try:
            reply = local_request(operation, variables, order, (*centered, query), point=shifts)
        except ArithmeticError as error:
            if str(error) == "Unknown local operation.":  # Older installed kernel.
                return None
            raise
    if not reply:
        return None
    for timing in reply[:-1]:
        name, elapsed = timing.split()
        label = {"homogeneous_native_standard_basis": "local_standard_basis",
                 "homogeneous_native_normal_form": "local_normal_form"}.get(name, name)
        record_stage(name, "cpu", label=label, elapsed_ns=int(elapsed))
    return np.from_row(variables, reply[-1])


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


def _sparse_point_result(variables, shifts, generators, query, order, known_basis):
    normal = query is not None
    kernel = "local_point_normal_form" if normal else "local_point_basis"
    label = "local_normal_form" if normal else "local_standard_basis"
    run = current_execution()
    if run and run.replay.compute == "gpu":
        kernel += "_gpu"
    rows = (*generators, query) if normal else generators
    with stage_label(label):
        started = perf_counter_ns()
        reply = local_request(kernel, variables, order, rows, point=shifts,
                              known_basis=known_basis if normal else None)
        if not reply:
            # The native dispatcher selected ordinary centering for this size.
            return None
        record_stage(kernel.removesuffix("_gpu"), "cpu", generators=len(generators),
                     elapsed_ns=perf_counter_ns() - started)
    names = [v.name for v in variables]
    def polynomial(row):
        return format_poly(np.terms(variables, np.from_row(variables, row)), names)
    if normal:
        reduction, unit, remainder = reply
        return dict(reduction=reduction, unit=polynomial(unit), remainder=polynomial(remainder))
    return dict(generators=[polynomial(row) for row in reply])


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
    if any(shifts) and generators:
        sparse = _sparse_point_result(variables, shifts, generators, query, ring["order"], known_basis)
        if sparse is not None:
            result.update(sparse)
            return result
    with cpu_stage("center_at_point", variables=len(names)):
        centered = _translate(variables, shifts, generators)
    if not normal:
        basis = _standard_basis(variables, centered, ring["order"])
        outputs = basis
    else:
        direct = None if known_basis else _homogeneous_remainder(variables, centered, ring["order"], query, shifts)
        if direct is not None:
            with cpu_stage("restore_point_coordinates", polynomials=1):
                remainder, = _restore_point(variables, shifts, (direct,), ring["order"])
                result.update(unit="1", reduction="standard", remainder=format_poly(np.terms(variables, remainder), names))
            return result
        if known_basis:
            # Only private worker metadata can identify rows already certified at this same point and order.
            with cpu_stage("restore_local_basis", generators=len(centered)):
                basis = centered
        else:
            basis = _standard_basis(variables, centered, ring["order"])
        if basis == (np.S.One,):
            result.update(reduction="standard", unit="1", remainder="0")
            return result
        with cpu_stage("center_query_at_point", variables=len(names)):
            query = _center_query(variables, shifts, query, basis, ring["order"])
        with cpu_stage("local_normal_form", divisors=len(basis)):
            unit, remainder, reduction = _normal_form(variables, basis, query, ring["order"])
        outputs = (unit, remainder)
    with cpu_stage("restore_point_coordinates", polynomials=len(outputs)):
        restored = _restore_point(variables, shifts, outputs, ring["order"])
        texts = [format_poly(np.terms(variables, f), names) for f in restored]
    if normal:
        unit_text, remainder_text = texts
        result.update(unit=unit_text, remainder=remainder_text, reduction=reduction)
    else:
        result["generators"] = texts
    return result
