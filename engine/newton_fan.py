"""Exact common unimodular refinement of the Newton normal fan."""
import json
from pathlib import Path
import subprocess

from .binomial_kernel import _integer
from .polynomial import LimitError, integer_text

_BINARY = Path(__file__).resolve().parents[1] / "native" / "build" / "newton_fan"


def build_fan(exponents, *, exhaustive=False):
    """Return primitive rays, full-dimensional smooth cones and boundary data.

    The first n rays are the coordinate vectors. New boundary rays can modify
    positive-dimensional coordinate strata away from the origin. The output
    describes the ambient toric map and does not assert face nondegeneracy.
    """
    if not _BINARY.is_file():
        raise RuntimeError("The Newton fan engine is not built. Run make.")
    support = list(exponents)
    if exhaustive:
        if any(type(e) is not int or e < 0 for row in support for e in row):
            raise LimitError("Expected nonnegative integer exponents.")
        # JSONcpp numeric values are machine-sized. Send exact decimal strings
        # to the GMP reader, including on interpreters with a digit limit.
        support = [[integer_text(e) for e in row] for row in support]
    request = {"exponents": support, "exhaustive": exhaustive}
    try:
        proc = subprocess.run([str(_BINARY)], input=json.dumps(request),
                              capture_output=True, text=True, timeout=35)
    except subprocess.TimeoutExpired:
        raise LimitError("Newton fan construction exceeded its time budget.") from None
    try:
        fan = json.loads(proc.stdout)
    except ValueError:
        raise LimitError("Newton fan construction stopped before producing a full fan.") from None
    if proc.returncode or "error" in fan:
        raise LimitError(fan.get("error", "Newton fan construction failed."))
    if exhaustive:
        fan["rays"] = [[_integer(v) if isinstance(v, str) else v for v in ray] for ray in fan["rays"]]
    return fan
