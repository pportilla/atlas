"""Transport a checked, embedded number field for polynomial arithmetic."""
from functools import lru_cache

from .kernel_json import kernel_json
from .marked_field import program
from .point_geometry import encode_terms
from .polynomial import InputError, format_poly, integer_from_decimal, parse_expression


@lru_cache(maxsize=256)
def _description(generator, embedding, relation, degree, index):
    request = {"program": program(embedding)}
    if relation is not None:
        request["relation"] = encode_terms(parse_expression(relation, [generator]))
    if degree is not None:
        request["degree"] = degree
    if index is not None:
        request["root_index"] = index
    checked = kernel_json("atlas_arithmetic_field", "description", **request)
    coefficients = checked["coefficients"].split()
    polynomial = format_poly({(j,): integer_from_decimal(c) for j, c in enumerate(coefficients)}, [generator])
    return polynomial, checked["degree"], checked["root_index"]


def checked_field(field, coordinates):
    from .algebra import _fields, _names
    _fields(field, "field", ("generator", "embedding"), ("minimal_polynomial", "degree", "root_index"))
    generator = field["generator"]
    _names([*coordinates, generator])
    if not isinstance(field["embedding"], str):
        raise InputError("The marked embedding must be an exact algebraic literal.")
    if "minimal_polynomial" in field and not isinstance(field["minimal_polynomial"], str):
        raise InputError("The minimal polynomial must be a polynomial string in the field generator.")
    for key in ("degree", "root_index"):
        if key in field and (type(field[key]) is not int or field[key] < 1):
            raise InputError("The field degree and root index must be positive integers.")
    relation, degree, index = _description(generator, field["embedding"], field.get("minimal_polynomial"),
                                          field.get("degree"), field.get("root_index"))
    return {"generator": generator, "embedding": field["embedding"], "minimal_polynomial": relation,
            "degree": degree, "root_index": index}
