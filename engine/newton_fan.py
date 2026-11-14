"""Exact common unimodular refinement of the Newton normal fan."""
import json
from pathlib import Path
import subprocess

from .polynomial import LimitError

_BINARY = Path(__file__).resolve().parents[1] / "native" / "build" / "newton_fan"


def build_fan(exponents):
    """Return primitive rays, full-dimensional smooth cones and boundary data.

    The first n rays are the coordinate vectors. New boundary rays can modify
    positive-dimensional coordinate strata away from the origin. The output
    describes the ambient toric map and does not assert face nondegeneracy.
    """
    if not _BINARY.is_file():
        raise RuntimeError("The Newton fan engine is not built. Run make.")
    try:
        proc = subprocess.run([str(_BINARY)], input=json.dumps({"exponents": list(exponents)}),
                              capture_output=True, text=True, timeout=35)
    except subprocess.TimeoutExpired:
        raise LimitError("Newton fan construction exceeded its time budget.") from None
    try:
        fan = json.loads(proc.stdout)
    except ValueError:
        raise LimitError("Newton fan construction stopped before producing a full fan.") from None
    if proc.returncode or "error" in fan:
        raise LimitError(fan.get("error", "Newton fan construction failed."))
    return fan
