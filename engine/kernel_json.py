"""Call a JSON operation on the exact native kernel."""

import ctypes
import json
import os

from .native_library import _kernel
from .polynomial import InputError


def _json_reply(symbol, payload):
    library = _kernel()
    if library is None or not hasattr(library, symbol):
        raise RuntimeError("Build the Atlas computational kernel with make.")
    function = getattr(library, symbol)
    function.argtypes = [ctypes.c_char_p]
    function.restype = ctypes.c_char_p
    directory = os.environ.get("ATLAS_RECORD_REQUESTS")
    if directory:
        from .request_recording import PendingRequest
        trace = PendingRequest(directory, "json", payload, symbol=symbol)
    reply = function(payload.encode("ascii"))
    if directory:
        trace.finish(reply.decode("utf-8"))
    return reply


def kernel_json(symbol, operation, **operands):
    payload = json.dumps(dict(operation=operation, **operands), separators=(",", ":"))
    response = json.loads(_json_reply(symbol, payload))
    if "error" in response:
        raise InputError(response["error"])
    return response["result"]
