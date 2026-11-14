"""Rational quotient algebras globally and at rational affine points.

The local order takes the terms of lowest total degree first, breaking ties
by the requested order on the coordinates. A homogeneous lift realizes this
order with an ordinary global Gröbner computation (Lazard homogenization).
The standard monomials give homogeneous bases of the successive pieces of
the maximal-ideal filtration. In finite length these lift to a basis of the
local algebra. A surviving pure power in every degree implies infinite length.
"""

from . import native_polynomial as np
from .algebra_execution import cpu_stage, current_execution, stage_label
from .algebra_groebner import reduced_basis
from .local_kernel import local_request
from .polynomial import InputError, format_poly, integer_from_decimal, integer_text, rational_text
from .quotient_monomials import decode_row, quotient_monomials
from .rational_point import coordinate_fraction, normalize_point


def _basis_heads(variables, generators, order, stage, *, local=False):
    from .polynomial import sparse_row

    with cpu_stage(stage, generators=len(generators)):
        rows = local_request("local_heads" if local else "local_global_heads", variables, order, generators)
        return [next(iter(sparse_row(row))) for row in rows]


def _initial_monomials(variables, generators, order, local):
    """Ask the kernel for the initial ideal in the specified coordinates."""
    if local:
        from .local_standard import _standard_basis
        generators = _standard_basis(variables, generators, order)
    elif len(generators) > 1 and any(len(f._terms) > 1 for f in generators):
        with stage_label("quotient_basis"):
            generators = reduced_basis(generators, variables, order=order).exprs
    return _basis_heads(variables, generators, order, "quotient_initial_monomials", local=local)


def _basis_text(basis, names, shifts):
    if not any(shifts):
        return [format_poly({e: 1}, names) for e in basis]
    # Return the centered monomials in the original ring. Keeping products
    # factored also avoids a binomial expansion for every basis element.
    axes = [name if not a else name + ("-" if a > 0 else "+") + rational_text(abs(a))
            for name, a in zip(names, shifts)]
    result = []
    for exponents in basis:
        factors = []
        grouped = sum(exponents) > 1
        for axis, shift, power in zip(axes, shifts, exponents):
            if power:
                factor = f"({axis})" if shift and grouped else axis
                factors.append(factor if power == 1 else factor + "^" + integer_text(power))
        result.append("*".join(factors) or "1")
    return result


def _point_quotient(operation, ring, operands, variables):
    names = ring["variables"]
    number = operation in ("milnor-number", "tjurina-number")
    point = normalize_point(operands["point"], len(names)) if "point" in operands else None
    shifts = tuple(map(coordinate_fraction, point)) if point is not None else (0,) * len(names)
    result = {"ring": ring, "scope": "origin" if point is None else "point"}
    if point is not None:
        result["point"] = point
    if operation == "local-quotient-basis":
        rows = tuple(map(np.coerce, operands["ideal"]))
        result["type"] = "quotient-basis"
    else:
        if not variables:
            raise InputError("A hypersurface germ requires at least one coordinate.")
        rows = (np.coerce(operands["polynomial"]),)
        result["type"] = "integer" if number else operation
    label = operation.replace("-", "_")
    kernel = label if label.startswith("local_") else "local_" + label
    execution = current_execution()
    if execution and execution.replay.compute == "gpu":
        kernel += "_gpu"
    try:
        with stage_label(label), cpu_stage(label, variables=len(names), generators=len(rows)):
            length, derivative_count, *rows_out = local_request(kernel, variables, ring["order"], rows, point=shifts)
    except ArithmeticError as error:
        if str(error) in ("The hypersurface must pass through the supplied point.",
                          "The hypersurface must pass through the origin."):
            raise InputError(str(error)) from error
        raise
    dimension = None if length == "infinite" else integer_from_decimal(length)
    result["finite"] = dimension is not None
    if number:
        result["value"] = dimension
    else:
        count = int(derivative_count)
        if operation == "local-jacobian-algebra":
            result["generators"] = [format_poly(np.terms(variables, np.from_row(variables, row)), names)
                                    for row in rows_out[:count]]
        result["dimension"] = dimension
        result["basis"] = None if dimension is None else _basis_text(
            map(decode_row, rows_out[count:]), names, shifts)
    return result


def handle(operation, ring, operands, *, known_basis=False):
    """Compute QQ quotient lengths and bases globally or at a rational point."""
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Quotient algebras require QQ and lex or grevlex order.")
    names = ring["variables"]
    variables = tuple(map(np.Symbol, names))
    if operation in ("local-quotient-basis", "local-jacobian-algebra", "milnor-number", "tjurina-number"):
        return _point_quotient(operation, ring, operands, variables)
    if operation not in ("multiplicity", "quotient-basis"):
        raise InputError(f"Unknown quotient algebra operation {operation!r}.")
    if "point" in operands:
        raise InputError("A global quotient calculation does not take a point.")
    number = operation == "multiplicity"
    generators = tuple(map(np.coerce, operands["ideal"]))
    if number and known_basis:
        heads = _basis_heads(variables, generators, ring["order"], "restore_global_basis")
    else:
        heads = _initial_monomials(variables, generators, ring["order"], False)
    with cpu_stage("quotient_length" if number else "standard_monomials", generators=len(heads)):
        dimension, basis = quotient_monomials(len(variables), heads, include_basis=not number)
        result = {"ring": ring, "scope": "global", "finite": dimension is not None}
        if number:
            result.update(type="integer", value=dimension)
        else:
            result.update(type="quotient-basis", dimension=dimension,
                          basis=None if basis is None else _basis_text(basis, names, (0,) * len(names)))
    return result
