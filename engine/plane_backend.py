"""Run exact point blowups of a plane curve over the algebraic closure of Q."""
from functools import lru_cache
import json
import re
import os
from pathlib import Path
import subprocess
import sys

from .polynomial import InputError, LimitError, NonReducedError, parse_expression, rational_text


_WORKER = Path(__file__).resolve().parents[1] / "native" / "build" / "plane"


def resolve_plane(poly, variables, mode, *, exhaustive=False):
    if not _WORKER.is_file():
        raise RuntimeError("Native plane resolver is not built. Run make.")
    from .quotient_monomials import degree_bounds
    if degree_bounds(len(variables), poly)[1] > sys.maxsize:
        raise LimitError("Plane input degree exceeds a signed native word.")
    payload = {"terms":[[*e,rational_text(c)] for e,c in poly.items()],
               "variables":variables, "mode":mode, "exhaustive":exhaustive}
    proc = subprocess.run([str(_WORKER)], input=json.dumps(payload), capture_output=True,
                          text=True, timeout=None if exhaustive else 50,
                          env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
    if proc.returncode:
        raise LimitError(f"Native plane worker stopped with status {proc.returncode}.")
    try:
        result = json.loads(proc.stdout)
    except ValueError:
        raise RuntimeError("Native plane worker did not return JSON.") from None
    if "error" in result:
        message = result["error"]
        if result.get("kind")=="input":
            if "nonreduced" in message:
                raise NonReducedError(message)
            raise InputError(message)
        raise LimitError(message)
    return result

@lru_cache(maxsize=256)
def ordered_roots(polynomial):
    """Distinct exact roots of a record's rational polynomial, in (real, imaginary) order."""
    terms = parse_expression(polynomial, ("T",))
    rows = [[power, str(c)] for (power,), c in terms.items()]
    done = subprocess.run([str(_WORKER)], input=json.dumps({"ordered_roots": rows, "exhaustive": True}),
                          capture_output=True, text=True, check=True)
    roots = json.loads(done.stdout)
    if isinstance(roots, dict):
        raise ValueError(roots.get("error", "The plane worker does not support ordered chart roots."))
    return tuple(roots)


@lru_cache(maxsize=256)
def coefficient_field(roots, rectangles=None, *, exhaustive=False):
    """Common field for FLINT's exact root identifiers, including the embedding."""
    parsed = []
    for text in roots:
        match = re.fullmatch(r"PolynomialRootIndexed\(List\(([-\d, ]+)\), (\d+)\)", text)
        if match is None:
            raise ValueError("Invalid algebraic coefficient in a plane chart.")
        parsed.append({"coefficients": match.group(1).split(", "), "index": int(match.group(2))})
    request = {"field_roots": parsed, "exhaustive": exhaustive}
    if rectangles is not None:
        request["root_rectangles"] = rectangles
    done = subprocess.run([str(_WORKER)], input=json.dumps(request),
                          capture_output=True, text=True, check=True)
    field = json.loads(done.stdout)
    if "error" in field:
        raise LimitError(field["error"])
    return field
