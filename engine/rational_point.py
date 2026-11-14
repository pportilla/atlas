"""Exact rational point coordinates for public algebra requests."""

from fractions import Fraction
import re

from .polynomial import InputError, integer_from_decimal, rational_text


_COORDINATE = re.compile(r"([+-]?[0-9]+)(?:/([+-]?[0-9]+))?")


def coordinate_fraction(value):
    """Read one JSON integer or signed integer/fraction string as a rational."""
    if type(value) is int:
        return Fraction(value)
    match = _COORDINATE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise InputError("Each point coordinate must be an integer or a rational string such as '-3/2'.")
    denominator = integer_from_decimal(match[2] or "1")
    if denominator == 0:
        raise InputError("A point coordinate cannot have a zero denominator.")
    return Fraction(integer_from_decimal(match[1]), denominator)


def normalize_point(point, dimension):
    """Return canonical rational strings in the order of the ring's variables."""
    if not isinstance(point, list) or len(point) != dimension:
        raise InputError(f"The point must be an array of {dimension} rational coordinates.")
    return [rational_text(coordinate_fraction(value)) for value in point]
