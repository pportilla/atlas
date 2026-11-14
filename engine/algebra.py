"""One rational ideal arithmetic protocol for the Atlas CLI, HTTP and sessions."""
import re
from subprocess import TimeoutExpired

from .algebra_replay import make_replay
from .polynomial import InputError, integer_text, parse_expression
from .rational_point import normalize_point


QUOTIENT_ALGEBRA_OPERANDS = {
    "quotient-basis": ("ideal",),
    "local-quotient-basis": ("ideal",),
    "local-jacobian-algebra": ("polynomial",),
    "milnor-number": ("polynomial",),
    "tjurina-number": ("polynomial",),
}
POINT_OPERATIONS = {
    "local-standard-basis": ("ideal",),
    "local-normal-form": ("ideal", "polynomial"),
}
OPERATION_OPERANDS = {
    "gcd": ("polynomial", "other_polynomial"),
    "ideal": ("ideal",),
    "groebner": ("ideal",),
    "normal-form": ("ideal", "polynomial"),
    "membership": ("ideal", "polynomial"),
    "eliminate": ("ideal", "variables"),
    "intersection": ("ideal", "other_ideal"),
    "quotient": ("ideal", "other_ideal"),
    "saturation": ("ideal", "polynomial"),
    "dimension": ("ideal",),
    **QUOTIENT_ALGEBRA_OPERANDS,
    **POINT_OPERATIONS,
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
    point_operation = operation in POINT_OPERATIONS
    _fields(operands, "operands", OPERATION_OPERANDS[operation], ("point",) if point_operation else ())
    if point_operation:
        point = normalize_point(operands["point"], len(names)) if "point" in operands else ["0"] * len(names)
        operands = dict(operands, point=point)
    for key in ("ideal", "other_ideal"):
        if key in operands and (not isinstance(operands[key], list)
                                or not all(isinstance(f, str) for f in operands[key])):
            raise InputError(f"{key} must be an array of polynomial strings.")
    for key in ("polynomial", "other_polynomial"):
        if key in operands and not isinstance(operands[key], str):
            raise InputError(f"{key} must be a polynomial string.")
    if "variables" in operands:
        removed = _names(operands["variables"])
        if not removed or any(v not in names for v in removed):
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
        elif operation == "gcd":
            from . import algebra_polynomial as handler
        elif operation in QUOTIENT_ALGEBRA_OPERANDS:
            from . import local_algebra as handler
        elif operation in POINT_OPERATIONS:
            from . import local_standard as handler
            if known_basis and operation == "local-normal-form":
                handler_options["known_basis"] = True
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
        # Encode point integers in their equivalent string form before the
        # worker transport calls json.dumps under the host's decimal digit limit.
        operands = request.get("operands") if isinstance(request, dict) else None
        if isinstance(operands, dict) and isinstance(operands.get("point"), list):
            point = [integer_text(v) if type(v) is int else v for v in operands["point"]]
            request = dict(request, operands=dict(operands, point=point))
        return run_worker(args, request, timeout, owner=owner)
    except (TypeError, ValueError, RecursionError):
        return 400, error_record("invalid_request", "Send a JSON algebra request.")
    except TimeoutExpired:
        return 422, error_record("computation_limit", "Algebra worker reached its time limit.")
    except (OSError, RuntimeError):
        return 503, error_record("worker_unavailable", "Algebra worker stopped before returning an exact result.")
