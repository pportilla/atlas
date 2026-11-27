"""Exact common unimodular refinement of the Newton normal fan."""
import ctypes
from functools import cache
import json
import os
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


# Seconds a caller that has other candidates to try gives one fan.
TRIAL_SECONDS = 35
# Memory one chart of a fan takes through assembly, verification and the
# saved record: a fixed part and a part for each stratum of its cone. The
# solver measured 30 to 32 kB for a chart in five variables (32 strata) and
# 34 kB in six (64 strata). These constants give 40 kB and 48 kB.
_CHART_BYTES = 32 << 10
_STRATUM_BYTES = 256


def available_memory():
    """Bytes the system reports as available, or None when it does not say."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) << 10
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def _chart_bytes(dimension, terms):
    return _CHART_BYTES + (_STRATUM_BYTES << dimension) + 64 * terms


def chart_cone_limit(dimension, terms):
    """The number of cones whose charts fit in half the available memory, or None when it is unknown."""
    available = available_memory()
    return None if available is None else available // (2 * _chart_bytes(dimension, terms))


def require_chart_memory(fan, terms):
    """Raise LimitError when the charts of the fan cannot fit in half the available memory.

    The fan itself has no fixed size limit. One chart per cone is assembled,
    verified and saved, so this is the size at which the atlas would exhaust
    the machine.
    """
    dimension = len(fan["rays"][0])
    cones = len(fan["cones"])
    limit = chart_cone_limit(dimension, terms)
    if limit is not None and cones > limit:
        need = cones * _chart_bytes(dimension, terms)
        raise LimitError(f"The Newton atlas has {cones} charts and needs about {(need >> 20) + 1} MB of memory; "
                         f"{available_memory() >> 20} MB are available.")


def build_fan(exponents, *, exhaustive=False, seconds=TRIAL_SECONDS, chart_terms=None):
    """Return primitive rays, full-dimensional smooth cones and boundary data.

    The first n rays are the coordinate vectors. New boundary rays can modify
    positive-dimensional coordinate strata away from the origin. The output
    describes the ambient toric map and does not assert face nondegeneracy.
    The construction may take `seconds`, or as long as it needs with None.
    Its size is bounded by the available memory. With `chart_terms`, the
    number of terms of the polynomial whose charts the caller will build on
    every cone, the construction stops with a LimitError as soon as the fan
    has more cones than require_chart_memory accepts.
    """
    support = list(exponents)
    limit = None if chart_terms is None or not support else chart_cone_limit(len(support[0]), chart_terms)
    fan = _request(support, exhaustive, seconds=seconds, cone_limit=limit)
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


def _request(exponents, exhaustive, *, operation=None, seconds=TRIAL_SECONDS, cone_limit=None):
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
    if cone_limit is not None:
        request["cone_limit"] = cone_limit
    try:
        from . import native_process
        proc = native_process.run("newton_fan", input=json.dumps(request), timeout=seconds)
    except subprocess.TimeoutExpired:
        raise LimitError("Newton fan construction exceeded its time budget.") from None
    try:
        fan = json.loads(proc.stdout)
    except ValueError:
        raise LimitError("Newton fan construction stopped before producing a full fan.") from None
    if proc.returncode or isinstance(fan, dict) and "error" in fan:
        raise LimitError(fan.get("error", "Newton fan construction failed."))
    return fan
