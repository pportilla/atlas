"""Local Krull dimension and smoothness at a rational point of an affine QQ scheme."""

from functools import lru_cache
from time import perf_counter_ns

from . import native_polynomial as np
from .algebra_execution import cpu_stage, current_execution, stage_label
from .local_dimension_input import cpu_invariants, measured_cpu_invariants
from .native_library import affine_request
from .point_geometry import encode_terms
from .polynomial import InputError
from .rational_point import normalize_point
from .tangent_linear import point_jacobian_rank


@lru_cache(maxsize=128)
def _coordinates(names):
    return tuple(map(np.Symbol, names))


def _inputs(operation, ring, operands, canonical=False):
    if operation not in ("local-dimension", "local-smoothness"):
        raise InputError(f"Unknown local dimension operation {operation!r}.")
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Local dimension requires QQ and lex or grevlex order.")
    variables = _coordinates(tuple(ring["variables"]))
    result = {"ring": ring, "scope": "origin"}
    if "point" in operands:
        point = operands["point"] if canonical else normalize_point(operands["point"], len(variables))
        result.update(scope="point", point=point)
    else:
        point = ("0",) * len(variables)
    generators = tuple(filter(None, operands["ideal"]))
    return result, variables, generators, point


def _result(result, smoothness, invariants):
    dimension, embedding, rank, smooth = invariants
    result["local_dimension"] = dimension
    if smoothness:
        result.update(type="boolean", value=bool(smooth), embedding_dimension=embedding, jacobian_rank=rank)
    else:
        result.update(type="integer", value=dimension)
    return result


def calculate_cpu(operation, ring, operands, *, canonical=False, started=None):
    """Compute with an isolated native CPU scope and return the worker's execution record."""
    entered = perf_counter_ns()
    if started is None:
        started = entered
    result, variables, generators, point = _inputs(operation, ring, operands, canonical)
    smoothness = operation == "local-smoothness"
    invariants, counts = measured_cpu_invariants(variables, generators, point, ring["order"] == "lex", smoothness)
    finished = perf_counter_ns()
    stages = []
    if counts is not None and counts[2]:
        stages.append(dict(operation="groebner_trace", backend="cpu", batches=counts[2], pairs=counts[3]))
    if counts is not None and counts[0]:
        stages.append(dict(operation="groebner_replay", backend="cpu", batches=counts[0], primes=counts[1]))
    if counts is not None and (counts[4] or counts[5]):
        stages.append(dict(operation="groebner_certification", backend="cpu", bases=counts[4],
                           attempts=counts[5], lift_ms=counts[6]/1e6, verify_ms=counts[7]/1e6))
    stages.append(dict(operation="local_dimension", backend="cpu", generators=len(generators),
                       elapsed_ns=finished - entered))
    execution = {"requested": "cpu", "backend": "cpu", "device": "CPU", "stages": stages,
                 "fallback_reasons": [], "elapsed_ms": (finished-started)/1e6}
    return _result(result, smoothness, invariants), execution


def handle(operation, ring, operands):
    result, variables, generators, point = _inputs(operation, ring, operands)
    run = current_execution()
    compute = run.replay.compute if run else "cpu"
    smoothness = operation == "local-smoothness"
    if compute == "cpu":
        with cpu_stage("local_dimension", generators=len(generators)):
            values = cpu_invariants(variables, generators, point, ring["order"] == "lex", smoothness)
        return _result(result, smoothness, values)

    width = len(variables)
    rank = -1
    if smoothness and (compute == "gpu" or min(width, len(generators)) >= 256):
        rank = point_jacobian_rank(point, [encode_terms(np.terms(variables, f)) for f in generators])

    kernel = operation.replace("-", "_") + ("_gpu" if compute == "gpu" else "")
    with stage_label("local_dimension"), cpu_stage("local_dimension", generators=len(generators)):
        try:
            line, = affine_request([kernel, " ".join(ring["variables"]), ring["order"], " ".join(point),
                                    str(rank), *(np.row(variables, f) for f in generators)])
        except ArithmeticError as exc:
            if str(exc) == "The point must lie on the scheme.":
                raise InputError(str(exc)) from exc
            raise
    return _result(result, smoothness, map(int, line.split()))
