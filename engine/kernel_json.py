"""Call a JSON operation on the exact native kernel."""

import ctypes
import json

from .native_library import _kernel
from .polynomial import InputError


def kernel_json(symbol, operation, **operands):
    library = _kernel()
    if library is None or not hasattr(library, symbol):
        raise RuntimeError("Build the Atlas computational kernel with make.")
    function = getattr(library, symbol)
    function.argtypes = [ctypes.c_char_p]
    function.restype = ctypes.c_char_p
    payload = json.dumps(dict(operation=operation, **operands), separators=(",", ":"))
    response = json.loads(function(payload.encode("ascii")))
    if "error" in response:
        raise InputError(response["error"])
    return response["result"]
