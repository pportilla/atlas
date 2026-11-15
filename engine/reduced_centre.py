"""Transport the bounded reduced-singular-locus proof before loading graph assembly."""
import subprocess
from time import monotonic

from .polynomial import encode_sparse_row


def smooth_reduced_graph(poly, names, mode, *, seconds=0.25):
    """One bounded, replay-aware proof of the graph over the reduced singular locus."""
    from .native_library import _kernel
    if (not hasattr(_kernel(), "atlas_smooth_reduced_centre_abi")
            or not 3 <= len(names) <= 8 or len(poly) > 128 or seconds <= 0
            or any(c.numerator.bit_length() > 256 or c.denominator.bit_length() > 256 or
                   any(k > 32 for k in e) for e, c in poly.items())):
        return None
    start = monotonic()
    if len(set(names)) != len(names):
        raise ValueError("Expected distinct polynomial coordinates.")
    if any(len(e) != len(names) for e, c in poly.items() if c):
        raise ValueError("Invalid polynomial exponent vector.")
    equation = encode_sparse_row(poly)
    from . import native_process
    try:
        done = native_process.run("graph_verify", input=f"smooth_reduced_graph {len(names)} {int(mode == 'embedded')}\n{equation}\n",
                                  timeout=min(seconds, 0.1))
        if done.returncode or not done.stdout:
            return None
        header, *rows = done.stdout.splitlines()
        kind, size = header.split()
        size = int(size)
        if kind not in ("graph", "centre") or not 2 <= size <= len(names) or len(rows) <= size:
            return None
        centre, proof = rows[:size], rows[size:]
        exceptional = None
        if kind == "graph":
            tag, count = proof[0].split()
            count = int(count)
            if mode != "embedded" or tag != "exceptional" or not 1 <= count < len(proof) - 1:
                return None
            exceptional, proof = proof[1:count + 1], proof[count + 1:]
        verdict = proof[0].split()
        if len(verdict) != 2 or verdict[0] != "pass" or not verdict[1].isdigit():
            return None
        at = 1
        for _ in range(size):
            ambient, strict = map(int, proof[at].split())
            if ambient < 0 or strict < 0:
                return None
            at += 1 + ambient + strict
        if at != len(proof):
            return None
        # Keep record assembly and replay state in the shared graph producer.
        remaining = seconds - (monotonic() - start)
        from .general_resolution import _reduced_graph_record, _run_attempt
        if exceptional is not None:
            state, record = _reduced_graph_record(names, mode, centre, proof, exceptional=exceptional)
        else:
            info = {"operation": "reduced_graph_record", "variables": list(names), "mode": mode,
                    "centre": centre, "proof": proof}
            state, record = _run_attempt(info, remaining, spare=False)
    except (OSError, ValueError, IndexError, ArithmeticError, subprocess.TimeoutExpired):
        return None
    return record if state == "certified" else None
