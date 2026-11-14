"""One isolated worker per local API request or CLI invocation."""
import json
import sys
from subprocess import TimeoutExpired
from .resolve import resolve
from .polynomial import InputError, LimitError


def main():
    try:
        request = json.load(sys.stdin)
        if isinstance(request,dict) and request.get("search","exhaustive")=="exhaustive" and hasattr(sys,"set_int_max_str_digits"):
            # The bounded input grammar validates user expressions independently.
            # Generated number-field certificates may need larger exact integers.
            sys.set_int_max_str_digits(0)
        data = resolve(request)
        status = 200
    except InputError as exc:
        data, status = {'error':str(exc)}, 400
    except (LimitError,TimeoutExpired,MemoryError,RecursionError) as exc:
        data, status = {'error':str(exc) or 'Computation budget exceeded.'}, 422
    except RuntimeError as exc:
        data, status = {'error':str(exc)}, 503
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        data, status = {'error':'Internal engine error; no resolution was returned.'}, 500
    print(json.dumps([status, data],separators=(',',':')))


if __name__=='__main__':
    main()
