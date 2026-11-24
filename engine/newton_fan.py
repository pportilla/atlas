"""Exact common unimodular refinement of the Newton normal fan."""
import ctypes
from functools import cache
import json
import sys
from pathlib import Path
import subprocess

from .binomial_kernel import _integer
from .polynomial import LimitError, integer_text

_BINARY = Path(__file__).resolve().parents[1] / "native" / "build" / "newton_fan"


@cache
def _small_fan():
    name = "libsmall_newton_fan" + (".dylib" if sys.platform == "darwin" else ".so")
    try:
        library = ctypes.CDLL(str(_BINARY.with_name(name)))
    except OSError:
        return None
    operation = library.atlas_small_newton_fan
    operation.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    operation.restype = ctypes.c_char_p
    return operation


# Seconds for one fan construction when the caller has no shorter deadline.
FAN_SECONDS = 35


def build_fan(exponents, *, exhaustive=False, seconds=None):
    """Return primitive rays, full-dimensional smooth cones and boundary data.

    The first n rays are the coordinate vectors. New boundary rays can modify
    positive-dimensional coordinate strata away from the origin. The output
    describes the ambient toric map and does not assert face nondegeneracy.
    A caller with a deadline gives the `seconds` it has left: the
    construction stops with a LimitError after the shorter of the two times.
    """
    fan = _request(exponents, exhaustive, seconds=seconds)
    if exhaustive:
        fan["rays"] = [[_integer(v) if isinstance(v, str) else v for v in ray] for ray in fan["rays"]]
    return fan


def trial_supports(exponents):
    """Bounded native proposals for regular monomial modifications."""
    choices = _request(exponents, True, operation="trials")
    return [tuple(tuple(_integer(v) if isinstance(v, str) else v for v in e) for e in support)
            for support in choices] if isinstance(choices, list) else []


def bounded_fan(support, *, trials=False):
    """Return None when the fixed-work native implementation declines the request."""
    # Larger coordinates and all cases outside the fixed-work native helper
    # keep their isolated process timeout, including exhaustive inputs.
    if (1 <= len(support) <= 3 and isinstance(support[0], (tuple, list))
            and 2 <= (n := len(support[0])) <= 4
            and all(isinstance(row, (tuple, list)) and len(row) == n
                    and all(type(x) is int and 0 <= x <= 3 for x in row) for row in support)):
        small = _small_fan()
        if small is not None:
            payload = " ".join(str(x) for row in support for x in row)
            answer = small(f"{n} {len(support)} {payload}".encode("ascii"), trials)
            if answer is not None:
                return json.loads(answer)
    return None


def _request(exponents, exhaustive, *, operation=None, seconds=None):
    if not _BINARY.is_file():
        raise RuntimeError("The Newton fan engine is not built. Run make.")
    support = list(exponents)
    result = bounded_fan(support, trials=operation == "trials")
    if result is not None:
        return result
    if exhaustive:
        if any(type(e) is not int or e < 0 for row in support for e in row):
            raise LimitError("Expected nonnegative integer exponents.")
        # JSONcpp numeric values are machine-sized. Send exact decimal strings
        # to the GMP reader, including on interpreters with a digit limit.
        support = [[integer_text(e) for e in row] for row in support]
    request = {"exponents": support, "exhaustive": exhaustive}
    if operation:
        request["operation"] = operation
    try:
        from . import native_process
        timeout = FAN_SECONDS if seconds is None else max(1.0, min(FAN_SECONDS, seconds))
        proc = native_process.run("newton_fan", input=json.dumps(request), timeout=timeout)
    except subprocess.TimeoutExpired:
        raise LimitError("Newton fan construction exceeded its time budget.") from None
    try:
        fan = json.loads(proc.stdout)
    except ValueError:
        raise LimitError("Newton fan construction stopped before producing a full fan.") from None
    if proc.returncode or isinstance(fan, dict) and "error" in fan:
        raise LimitError(fan.get("error", "Newton fan construction failed."))
    return fan
