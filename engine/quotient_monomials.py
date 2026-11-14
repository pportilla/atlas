"""Transport exponent rows to the native monomial quotient kernel."""

from operator import index

from .native_library import affine_request
from .polynomial import integer_from_decimal, integer_text


def encode_row(row):
    return " ".join(integer_text(index(value)) for value in row)


def decode_row(line):
    return tuple(map(integer_from_decimal, line.split()))


def degree_bounds(width, terms):
    result, = affine_request(["monomial_degree_bounds", str(width), *map(encode_row, terms)])
    return decode_row(result)


def quotient_monomials(width, leading, *, include_basis):
    """Return the finite vector space dimension and requested standard monomials."""
    width = index(width)
    if width < 0:
        raise ValueError("A ring cannot have negative coordinate count.")
    operation = "monomial_basis" if include_basis else "monomial_length"
    rows = []
    for powers in leading:
        row = tuple(map(index, powers))
        if len(row) != width or any(exponent < 0 for exponent in row):
            raise ValueError("Expected one nonnegative exponent per coordinate.")
        rows.append(encode_row(row))
    length, *basis = affine_request([operation, str(width), *rows])
    if length == "infinite":
        return None, None
    return integer_from_decimal(length), [decode_row(row) for row in basis] if include_basis else None
