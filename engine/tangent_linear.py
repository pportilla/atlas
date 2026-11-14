"""Exact rational row spaces, optionally selected by a modular GPU rank witness."""

import ctypes
from time import perf_counter_ns

from .point_geometry import request
from .algebra_execution import cpu_stage, current_execution, record_fallback, record_stage


def _opencl_rows(buffer, rows, cols, prime, index):
    from .native_backend import _gpu_library

    plugin = _gpu_library()
    if plugin is None or not hasattr(plugin, "resolution_gpu_row_basis"):
        raise RuntimeError("Rebuild the OpenCL rank plugin for rectangular Jacobians.")
    function = plugin.resolution_gpu_row_basis
    function.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int,
                         ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    identifiers = (ctypes.c_int * min(rows, cols))()
    rank = function(buffer, rows, cols, prime, index, identifiers)
    if not 0 <= rank <= min(rows, cols):
        raise RuntimeError("OpenCL row reduction unavailable on the selected device.")
    return identifiers[:rank]


def _device_row_basis(rows, width):
    run = current_execution()
    if not run or run.replay.compute == "cpu" or not rows:
        return None
    scope = run.replay
    if scope.compute == "auto" and min(len(rows), width) < 256:
        return None
    if width > 512 or len(rows) > 16384:
        record_fallback("The Jacobian exceeds the rank plugin's matrix dimensions.")
        return None
    choices = ("triton", "opencl") if scope.backend == "auto" else (scope.backend,)
    prime = 1_000_000_007
    residues = request("residues", width=width, rows=rows, prime=prime)
    buffer = (ctypes.c_uint32 * len(residues))(*residues)
    for backend in choices:
        started = perf_counter_ns()
        device = None
        try:
            if backend == "opencl":
                ids = _opencl_rows(buffer, len(rows), width, prime, scope.index)
            elif backend == "triton":
                from .gpu_devices import triton_device
                device = triton_device(scope.index)
                if device is None:
                    raise RuntimeError("Triton device unavailable")
                from .gpu_rank import independent_rows
                ids = independent_rows(buffer, len(rows), width, prime, device)
            else:
                return None
            if len(ids) != len(set(ids)) or any(not 0 <= i < len(rows) for i in ids):
                raise ValueError("Invalid device row indices")
        except Exception as exc:
            record_fallback(f"{backend} Jacobian rank: {exc}")
            continue
        record_stage("jacobian_row_selection", backend, device=scope.record_device(backend, device),
                     rows=len(rows), columns=width, candidate_rank=len(ids), prime=prime,
                     elapsed_ns=perf_counter_ns()-started)
        return ids
    return None


def linear_constraints(rows, width, *, dimension_only=False):
    """Return the QQ rank and, when requested, reduced row echelon equations."""
    witness = _device_row_basis(rows, width)
    result = None
    if witness is not None:
        with cpu_stage("certify_jacobian_row_space", candidate_rank=len(witness)):
            result = request("row_space", width=width, rows=rows, witness=witness, rank_only=dimension_only)
        if result is None:
            record_fallback("The modular pivot rows did not span the exact Jacobian. Recomputed over QQ.")
    with cpu_stage("rational_jacobian_echelon", rows=len(rows), columns=width):
        if result is None:
            result = request("row_space", width=width, rows=rows, rank_only=dimension_only)
    return result
