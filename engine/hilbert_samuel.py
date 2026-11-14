"""Exact graded and cumulative lengths of the maximal-ideal filtration over QQ."""

from . import native_polynomial as np
from .algebra_execution import cpu_stage
from .local_algebra import _initial_monomials
from .point_germs import _large_translation, centered_ideal
from .polynomial import InputError
from .native_library import affine_request
from .quotient_monomials import decode_row, encode_row
from .rational_point import coordinate_fraction, normalize_point


def monomial_hilbert_series(width, heads):
    header, *rows = affine_request(["monomial_series", str(width), *map(encode_row, heads)])
    dimension, multiplicity = decode_row(header)
    return {"denominator_power": width, "dimension": dimension, "multiplicity": multiplicity,
            "numerator": [{"degree": degree, "coefficient": value} for degree, value in map(decode_row, rows)]}


def _invariants(width, heads, degrees):
    degrees = tuple(degrees)
    header, *rows = affine_request(["monomial_function", str(width), encode_row(degrees), *map(encode_row, heads)])
    dimension, multiplicity = decode_row(header)
    return dimension, multiplicity, dict(zip(degrees, map(decode_row, rows)))


def handle(operation, ring, operands):
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Hilbert–Samuel calculations require QQ and lex or grevlex order.")
    if operation not in ("hilbert-samuel-multiplicity", "hilbert-samuel-function"):
        raise InputError(f"Unknown Hilbert–Samuel operation {operation!r}.")
    function = operation == "hilbert-samuel-function"
    degrees = operands.get("degrees", list(range(6))) if function else []
    if not isinstance(degrees, list) or any(type(n) is not int or n < 0 for n in degrees):
        raise InputError("Degrees must be a list of nonnegative integers.")

    variables = tuple(map(np.Symbol, ring["variables"]))
    generators = tuple(np.coerce(f) for f in operands["ideal"] if f)
    heads = None
    result = {"ring": ring, "scope": "origin"}
    if "point" in operands:
        point = normalize_point(operands["point"], len(variables))
        result.update(scope="point", point=point)
        shifts = tuple(map(coordinate_fraction, point))
        if any(shifts):
            with cpu_stage("center_at_point", variables=len(variables), generators=len(generators)):
                if len(generators) == 1 and _large_translation(variables, shifts, generators[0]):
                    # Initial ideals of principal germs have one generator.
                    # Its total degree alone determines their Hilbert series.
                    from .point_order import order_at_rational_point
                    order = order_at_rational_point(variables, shifts, generators[0])
                    heads = [tuple(order if i == 0 else 0 for i in range(len(variables)))]
                else:
                    generators = centered_ideal(variables, shifts, generators)

    if heads is None:
        heads = _initial_monomials(variables, generators, ring["order"], local=True)
    with cpu_stage("hilbert_samuel_staircase", generators=len(heads), degrees=len(degrees)):
        dimension, multiplicity, values = _invariants(len(variables), heads, degrees)
    result["dimension"] = dimension
    if function:
        result.update(type=operation, degrees=degrees, values=[values[n][0] for n in degrees],
                      graded_values=[values[n][1] for n in degrees], multiplicity=multiplicity)
    else:
        result.update(type="integer", value=multiplicity)
    return result
