"""One isolated worker per local API request or CLI invocation."""
import json
import sys
from subprocess import TimeoutExpired
from .resolve import resolve
from .polynomial import InputError, LimitError


def _response(read_request):
    """Use the same resolution and exception mapping on the API and CLI."""
    try:
        request = read_request()
        if isinstance(request,dict) and request.get("search","exhaustive")=="exhaustive" and hasattr(sys,"set_int_max_str_digits"):
            # Generated number-field certificates may need larger exact integers.
            sys.set_int_max_str_digits(0)
        data = resolve(request)
        status = 200
    except InputError as exc:
        data, status = {'error':str(exc)}, 400
    except (LimitError,TimeoutExpired) as exc:
        data, status = {'error':str(exc) or 'Computation budget exceeded.', 'code':'computation_limit'}, 422
    except (MemoryError,RecursionError) as exc:
        data, status = {'error':str(exc) or 'Computation budget exceeded.'}, 422
    except RuntimeError as exc:
        data, status = {'error':str(exc)}, 503
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        data, status = {'error':'Internal engine error; no resolution was returned.'}, 500
    return status, data


def evaluate(request):
    return _response(lambda: request)


def main():
    print(json.dumps(_response(lambda: json.load(sys.stdin)), separators=(',',':')))


if __name__=='__main__':
    main()
