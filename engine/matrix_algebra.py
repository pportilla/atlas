"""Validate rational matrices and dispatch their exact and modular row operations."""
from fractions import Fraction
from time import perf_counter_ns

from .algebra_execution import cpu_stage, current_execution, record_fallback, record_stage
from .point_geometry import request
from .polynomial import InputError, integer_text, rational_text
from .rational_point import coordinate_fraction


def _entry(value):
    if type(value) is int:
        return integer_text(value)
    if isinstance(value, Fraction):
        return rational_text(value)
    if isinstance(value, str) and value.removeprefix("-").isascii() and value.removeprefix("-").isdigit():
        return value
    return rational_text(coordinate_fraction(value))


def matrix_operands(rows, *, rhs=None, columns=None):
    if not isinstance(rows, (list, tuple)) or not all(isinstance(row, (list, tuple)) for row in rows):
        raise InputError("Pass the matrix as a list of rational rows.")
    if columns is None:
        columns = len(rows[0]) if rows else 0
    if type(columns) is not int or not 0 <= columns <= 2**31-1:
        raise InputError("The column count must be a nonnegative integer below 2^31.")
    if any(len(row) != columns for row in rows):
        raise InputError("All matrix rows must have the specified column count.")
    operands = {"rows": [[_entry(v) for v in row] for row in rows], "columns": columns}
    if rhs is not None:
        if not isinstance(rhs, (list, tuple)) or len(rhs) != len(rows):
            raise InputError("The right-hand side needs one entry per matrix row.")
        operands["rhs"] = [_entry(v) for v in rhs]
    return operands


def _determinant(rows, width):
    run = current_execution()
    if not run or run.replay.compute != "gpu" or run.replay.backend not in ("auto", "triton"):
        return None
    if not 1 <= width <= 6 or width != len(rows):
        return None
    from .gpu_devices import torch_device_context, triton_device
    device = triton_device(run.replay.index)
    if device is None:
        return None
    plan = request("matrix_determinant_plan", rows=rows, width=width)
    if plan is None:
        return None
    from .triton_buffers import to_device, to_host
    from .triton_ir import launch_kernel
    import torch
    started = perf_counter_ns()
    selected = device["torch_device"]
    try:
        with torch_device_context(selected):
            inputs = to_device(torch.tensor(plan["entries"],dtype=torch.int64), selected)
            output = torch.empty(1,device=selected,dtype=torch.int64)
            launch_kernel("determinant_small", (width,), (1,), inputs, output)
            candidate = str(to_host(output).item())
    except Exception as exc:
        record_fallback(f"Triton determinant: {exc}")
        return None
    return candidate, device, perf_counter_ns()-started


def handle(operation, ring, operands):
    from .tangent_linear import _device_row_basis

    rows, width, rhs = operands["rows"], operands["columns"], operands.get("rhs")
    determinant = _determinant(rows, width)
    augmented = rhs is not None
    if augmented:
        rows = [[*row, b] for row, b in zip(rows, rhs)]
    witness = _device_row_basis(rows, width + augmented, label="matrix_row_selection") if rows and rows[0] else None
    args = dict(rows=rows, width=width, augmented=augmented)
    if determinant is not None:
        args["det_candidate"] = determinant[0]
    with cpu_stage("rational_matrix_analysis", rows=len(rows), columns=width):
        result = request("matrix_analysis", **args, witness=witness)
        if result is None:
            record_fallback("The modular rows did not span the rational matrix. Recomputed the row space over QQ.")
            result = request("matrix_analysis", **args)
    verified = result.pop("determinant_device_verified", False)
    if determinant is not None:
        if verified:
            _,device,elapsed = determinant
            record_stage("matrix_determinant", "triton", device=current_execution().replay.record_device("triton",device),
                         dimension=width, host_verified=True, elapsed_ns=elapsed)
        else:
            record_fallback("The device determinant failed exact modular verification. Recomputed over QQ.")
    return dict(type="matrix-analysis", ring=ring, **result)


def matrix_analysis(rows, *, rhs=None, columns=None, compute="auto"):
    """Return the exact rank, determinant, pivots, row reduction, image, kernel and optional solution over QQ.

    >>> M = matrix_analysis([[1, 2], [2, 4]], rhs=[3, 6])
    >>> print(M.rank, M.solution.status)  # 1 underdetermined

    Docs: examples/algebra-api.md#exact-rational-linear-algebra
    """
    from .algebra import dispatch
    from .session_algebra import _typed
    from .session_polynomial import QQ
    operands = {"rows": rows}
    if rhs is not None:
        operands["rhs"] = rhs
    if columns is not None:
        operands["columns"] = columns
    record = dispatch({"operation": "matrix-analysis", "ring": {"variables": []},
                       "compute": compute, "operands": operands})
    return _typed(record, QQ)
