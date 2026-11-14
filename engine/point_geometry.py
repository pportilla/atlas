"""Transport sparse point-local operations to the exact kernel."""

import ctypes
import json

from .native_library import _kernel
from .polynomial import InputError, integer_from_decimal, integer_text, rational_text
from .rational_point import coordinate_fraction


def request(operation, **operands):
    library = _kernel()
    if library is None or not hasattr(library, "atlas_point_geometry"):
        raise RuntimeError("Build the Atlas point geometry kernel with make.")
    function = library.atlas_point_geometry
    function.argtypes = [ctypes.c_char_p]
    function.restype = ctypes.c_char_p
    payload = json.dumps(dict(operation=operation, **operands), separators=(",", ":"))
    response = json.loads(function(payload.encode("ascii")))
    if "error" in response:
        raise InputError(response["error"])
    return response["result"]


def encode_terms(terms):
    return [[rational_text(c), [integer_text(k) for k in powers]] for powers, c in terms.items() if c]


def decode_terms(data):
    return {tuple(map(integer_from_decimal, powers)): coordinate_fraction(c) for c, powers in data}


def point_operation(operation, point, terms, **options):
    return request(operation, point=list(map(rational_text, point)), terms=encode_terms(terms), **options)
