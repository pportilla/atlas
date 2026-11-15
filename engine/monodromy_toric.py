"""Exact Euler characteristics of open total-transform components over zero."""
import ctypes
from operator import index

from .polynomial import integer_from_decimal, integer_text


def _volume(points, dimension):
    """Normalized Newton polytope volume in the full orbit lattice."""
    from .groebner import _kernel
    points = tuple(dict.fromkeys(tuple(map(index, p)) for p in points))
    if any(len(p) != dimension for p in points):
        raise ValueError("Each lattice point must have the stated dimension.")
    library = _kernel()
    if library is None or not hasattr(library, "atlas_lattice_volume"):
        raise RuntimeError("Build the Atlas lattice-volume kernel with make.")
    operation = library.atlas_lattice_volume
    operation.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    operation.restype = ctypes.c_void_p
    rows = [str(len(points))] if dimension == 0 else [" ".join(map(integer_text, p)) for p in points]
    payload = ("\n".join([str(dimension), *rows]) + "\n").encode()
    result = operation(payload, len(payload))
    if not result:
        raise ArithmeticError("The kernel could not compute the lattice volume.")
    try:
        value = integer_from_decimal(ctypes.string_at(result).decode("ascii"))
    finally:
        library.atlas_groebner_release(result)
    return value
