"""Transport a single rational algebra calculation across the worker boundary."""
import json
import sys
from subprocess import TimeoutExpired

from .algebra import dispatch, error_record
from .native_library import Unavailable
from .polynomial import InputError, LimitError, parse_expression


def _literal(request):
    """Return sparse terms in the caller's generator order, including a scalar ring."""
    from . import native_polynomial as native

    names = request["variables"]
    generators = tuple(map(native.Symbol, names))
    terms = parse_expression(request["source"], names, coefficient_modulus=request.get("coefficient_modulus"))
    value = native.from_terms(generators, terms)
    return {"terms": native.row(generators, value)}


def main():
    try:
        # Exact results can be sent back as operands, including rational
        # coefficients longer than Python's decimal literal guard.
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(0)
        request = json.load(sys.stdin)
        if sys.argv[1:] == ["--polynomial"]:
            record = _literal(request)
        else:
            record = dispatch(request, known_basis=sys.argv[1:] == ["--known-basis"])
        status = 200
    except (InputError, UnicodeEncodeError) as exc:
        status, record = 400, error_record("invalid_request", str(exc))
    except (LimitError, TimeoutExpired, MemoryError, RecursionError) as exc:
        status, record = 422, error_record("computation_limit", str(exc) or "Computation budget exceeded.")
    except Unavailable as exc:
        status, record = 503, error_record("engine_unavailable", str(exc))
    except (ArithmeticError, RuntimeError) as exc:
        status, record = 422, error_record("computation_failed", str(exc))
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        status, record = 500, error_record("internal_error", "Algebra worker could not return an exact result.")
    print(json.dumps([status, record], separators=(",", ":")))


if __name__ == "__main__":
    main()
