"""One rational ideal arithmetic protocol for the Atlas CLI, HTTP and sessions."""
import re
from subprocess import TimeoutExpired

from .algebra_replay import make_replay
from .polynomial import InputError, integer_from_decimal, integer_text, parse_expression
from .rational_point import normalize_point


LOCAL_QUOTIENT_OPERANDS = {
    "local-quotient-basis": ("ideal",),
    "local-jacobian-algebra": ("polynomial",),
    "milnor-number": ("polynomial",),
    "tjurina-number": ("polynomial",),
}
LOCAL_STANDARD_OPERANDS = {
    "local-standard-basis": ("ideal",),
    "local-normal-form": ("ideal", "polynomial"),
}
HILBERT_SAMUEL_OPERANDS = {
    "hilbert-samuel-function": ("ideal",),
    "hilbert-samuel-multiplicity": ("ideal",),
}
POINT_OPERATIONS = (LOCAL_QUOTIENT_OPERANDS.keys() | LOCAL_STANDARD_OPERANDS.keys()
                    | HILBERT_SAMUEL_OPERANDS.keys())
OPERATION_OPERANDS = {
    "gcd": ("polynomial", "other_polynomial"),
    "resultant": ("polynomial", "other_polynomial", "variables"),
    "factor": ("polynomial",),
    "ideal": ("ideal",),
    "groebner": ("ideal",),
    "normal-form": ("ideal", "polynomial"),
    "membership": ("ideal", "polynomial"),
    "eliminate": ("ideal", "variables"),
    "intersection": ("ideal", "other_ideal"),
    "quotient": ("ideal", "other_ideal"),
    "saturation": ("ideal", "polynomial"),
    "dimension": ("ideal",),
    "quotient-basis": ("ideal",),
    **LOCAL_QUOTIENT_OPERANDS,
    **LOCAL_STANDARD_OPERANDS,
    **HILBERT_SAMUEL_OPERANDS,
}
BASIS_OPERATIONS = ("groebner", "normal-form", "membership")


def error_record(code, message):
    return {"status": "error", "error": {"code": code, "message": message}}


def _fields(value, label, required, optional=()):
    if not isinstance(value, dict):
        raise InputError(f"{label} must be an object.")
    for key in required:
        if key not in value:
            raise InputError(f"Missing {label} field: {key}.")
    for key in value:
        if key not in required and key not in optional:
            raise InputError(f"Unknown {label} field: {key!r}.")


def _names(names):
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", name)
        for name in names
    ):
        raise InputError("Use an array of variable names of up to 24 letters, digits or underscores, starting with a letter.")
    if len(set(names)) != len(names):
        raise InputError("Choose distinct variable names.")
    return names


def _degrees(values):
    if not isinstance(values, list):
        raise InputError("degrees must be an array of nonnegative integers.")
    degrees = []
    for value in values:
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
            value = integer_from_decimal(value)
        if type(value) is not int or value < 0:
            raise InputError("Each degree must be a nonnegative integer or an unsigned decimal string.")
        degrees.append(value)
    return degrees


def _request(data):
    _fields(data, "request", ("operation", "ring", "operands"), ("compute",))
    operation = data["operation"]
    if not isinstance(operation, str) or operation not in OPERATION_OPERANDS:
        raise InputError("Unknown algebra operation. Choose " + ", ".join(OPERATION_OPERANDS) + ".")
    compute = data.get("compute", "auto")
    if compute not in ("cpu", "auto", "gpu"):
        raise InputError("Choose cpu, auto or gpu for compute.")

    ring = data["ring"]
    _fields(ring, "ring", ("variables",), ("field", "order"))
    names = _names(ring["variables"])
    if ring.get("field", "QQ") != "QQ":
        raise InputError("The supported coefficient field is QQ.")
    order = ring.get("order", "grevlex")
    if order not in ("lex", "grevlex"):
        raise InputError("Unsupported monomial order. Choose lex or grevlex.")

    operands = data["operands"]
    optional = ("point",) if operation in POINT_OPERATIONS else ()
    if operation == "hilbert-samuel-function":
        optional += ("degrees",)
    _fields(operands, "operands", OPERATION_OPERANDS[operation], optional)
    if "degrees" in operands:
        operands = dict(operands, degrees=_degrees(operands["degrees"]))
    if "point" in operands:
        operands = dict(operands, point=normalize_point(operands["point"], len(names)))
    elif operation in LOCAL_STANDARD_OPERANDS:
        # These two operations have always returned an explicit origin point.
        operands = dict(operands, point=["0"] * len(names))
    for key in ("ideal", "other_ideal"):
        if key in operands and (not isinstance(operands[key], list)
                                or not all(isinstance(f, str) for f in operands[key])):
            raise InputError(f"{key} must be an array of polynomial strings.")
    for key in ("polynomial", "other_polynomial"):
        if key in operands and not isinstance(operands[key], str):
            raise InputError(f"{key} must be a polynomial string.")
    if "variables" in operands:
        removed = _names(operands["variables"])
        if operation == "resultant":
            if len(removed) != 1 or removed[0] not in names:
                raise InputError("Choose exactly one declared variable to eliminate for the resultant.")
        elif not removed or any(v not in names for v in removed):
            raise InputError("Choose one or more declared variables to eliminate.")

    return {"operation": operation, "ring": {"field": "QQ", "variables": names, "order": order},
            "operands": operands, "compute": compute}


def dispatch(data, *, known_basis=False):
    """Compute a structured success record, raising for invalid or unfinished work.

    known_basis is private worker metadata supplied only for a basis previously
    returned to a session or retained by the HTTP server. Public generator
    arrays always require a fresh basis calculation.
    """
    from . import native_polynomial as native
    from .algebra_execution import algebra_execution

    request = _request(data)
    operation, ring = request["operation"], request["ring"]
    names = ring["variables"]
    variables = tuple(native.Symbol(name) for name in names)
    operands = request["operands"].copy()

    def parse(text):
        try:
            terms = parse_expression(text, names)
        except UnicodeEncodeError as exc:
            raise InputError("A polynomial string must contain valid Unicode.") from exc
        return native.from_terms(variables, terms)

    with algebra_execution(request["compute"]) as execution:
        for key in ("ideal", "other_ideal"):
            if key in operands:
                operands[key] = [parse(text) for text in operands[key]]
        for key in ("polynomial", "other_polynomial"):
            if key in operands:
                operands[key] = parse(operands[key])
        result_operands = operands
        handler_options = {}
        if operation in BASIS_OPERATIONS:
            from . import algebra_groebner as handler
            if known_basis:
                basis = handler.basis_from_reduced(operands["ideal"], variables, order=ring["order"])
                result_operands = dict(operands, ideal=basis)
        elif operation in ("gcd", "factor", "resultant"):
            from . import algebra_polynomial as handler
        elif operation == "quotient-basis" or operation in LOCAL_QUOTIENT_OPERANDS:
            from . import local_algebra as handler
        elif operation in LOCAL_STANDARD_OPERANDS:
            from . import local_standard as handler
            if known_basis and operation == "local-normal-form":
                handler_options["known_basis"] = True
        elif operation in HILBERT_SAMUEL_OPERANDS:
            from . import hilbert_samuel as handler
        else:
            from . import algebra_ideals as handler
        result = handler.handle(operation, ring, result_operands, **handler_options)
        observed = execution.report()
    return {"status": "ok", "operation": operation, "result": result,
            "execution": observed, "replay": make_replay(request, operands)}


def calculate_algebra(request, *, known_basis=False, timeout=None, owner=None):
    """Run one operation in an isolated worker and return (HTTP status, record)."""
    from .worker_process import run_worker

    args = ["engine.algebra_worker"]
    if known_basis:
        args.append("--known-basis")
    try:
        # Send integers as decimal strings across the worker boundary, where
        # the host's JSON encoder may enforce a decimal digit limit.
        operands = request.get("operands") if isinstance(request, dict) else None
        if isinstance(operands, dict):
            wire_operands = operands.copy()
            for key in ("point", "degrees"):
                if isinstance(operands.get(key), list):
                    wire_operands[key] = [integer_text(v) if type(v) is int else v for v in operands[key]]
            request = dict(request, operands=wire_operands)
        return run_worker(args, request, timeout, owner=owner)
    except (TypeError, ValueError, RecursionError):
        return 400, error_record("invalid_request", "Send a JSON algebra request.")
    except TimeoutExpired:
        return 422, error_record("computation_limit", "Algebra worker reached its time limit.")
    except (OSError, RuntimeError):
        return 503, error_record("worker_unavailable", "Algebra worker stopped before returning an exact result.")
