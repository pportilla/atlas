"""Mathematical operations and execution shared by the Atlas prompt and scripts."""
import sys
from fractions import Fraction
from math import isfinite
from pathlib import Path

from .exact_json import result_json
from .polynomial import rational_text
from .session_algebra import _coordinate_space, algebra_bindings
from .session_expressions import Evaluator, SessionError
from .session_maps import hom
from .session_monodromy import check_resolution, unavailable_check
from .session_polynomial import QQ, Polynomial, _CoordinateSpace, coordinates as coordinates, session_declarations
from .session_quotient import QuotientElement, QuotientIdeal, QuotientRing, quotient_ring
from .session_rings import ring_bindings
from .session_values import Divisor, DualComplex, Resolution


def json_value(value):
    if isinstance(value, (DualComplex, Divisor)):
        return json_value(value._record)
    if isinstance(value, (QuotientElement, QuotientIdeal, QuotientRing)):
        return value.result
    if isinstance(value, Polynomial):
        return str(value)
    if isinstance(value, Fraction):
        return rational_text(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON record fields must have string names.")
        return {key: json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON numbers must be finite.")
    return value


def print_json(**fields):
    """Emit one JSON record, with canonical polynomials and exact rational strings."""
    print(result_json(json_value(fields)), flush=True)


def _resolution_input(polynomial, variables, ring, declare, active):
    if isinstance(polynomial, Polynomial):
        if variables is not None or ring is not None:
            raise ValueError("A polynomial already carries its coordinates.")
        ring = polynomial.parent
    elif isinstance(polynomial, str):
        if ring is None and variables is None:
            ring = active
        if (ring is None) == (variables is None):
            raise ValueError('For a polynomial string, supply ring=f.parent or variables=["x", "y"], '
                             'or choose a default ring with use_ring(f.parent).')
        if variables is not None:
            if isinstance(variables, (tuple, list)) and variables and all(isinstance(v, str) for v in variables):
                variables = declare(*variables)
            ring = _coordinate_space(variables)
    else:
        raise ValueError("resolve expects a polynomial or a polynomial string with its ring.")
    if not isinstance(ring, _CoordinateSpace) or ring.coeffs is not QQ or ring.has_relations or not ring.generators:
        raise ValueError("resolve expects an equation in affine polynomial coordinates over QQ.")
    return str(polynomial), list(ring.names), ring


def new_session(calculate):
    """Register session operations here so both entry points return the same values."""
    declare, ring = session_declarations()
    session = Evaluator({}, reserved=("help", "quit", "exit"))
    rings = session.rings

    def resolve(polynomial, *, variables=None, ring=None, mode="embedded", search="exhaustive",
                compute=None, monodromy=False):
        expression, names, parent = _resolution_input(polynomial, variables, ring, declare, rings.active)
        if type(monodromy) is not bool:
            raise ValueError("monodromy must be true or false.")
        options = {"monodromy": True} if monodromy else {}
        status, record = calculate(expression, names, mode=mode, search=search, compute=compute, **options)
        if status != 200:
            raise ValueError(record["error"])
        return Resolution(record, parent, source_polynomial=polynomial if isinstance(polynomial, Polynomial) else None,
                          _from_solver=True)

    def monodromy_check(polynomial, *, variables=None, ring=None, search="quick", compute=None, check_isolation=False):
        if type(check_isolation) is not bool:
            raise ValueError("check_isolation must be true or false.")
        if isinstance(polynomial, Resolution):
            if variables is not None or ring is not None:
                raise ValueError("A resolution already carries its coordinates.")
            resolution = polynomial
        else:
            expression, names, parent = _resolution_input(polynomial, variables, ring, declare, rings.active)
            status, record = calculate(expression, names, mode="embedded", search=search,
                                       compute=compute, monodromy=True)
            if status == 400:
                raise ValueError(record["error"])
            if status != 200:
                return unavailable_check(record["error"])
            resolution = Resolution(record, parent, source_polynomial=polynomial if isinstance(polynomial, Polynomial) else None,
                                    _from_solver=True)
        return check_resolution(resolution, check_isolation=check_isolation, compute=compute)

    def dual_complex(resolution):
        # Older scripts use this function's raw replay dictionary.
        from .resolution_complex import dual_complex as replay
        if not isinstance(resolution, Resolution):
            raise ValueError("Pass an Atlas resolution to recompute its dual complex.")
        return replay(resolution._record)

    def recompute_dual_complex(resolution):
        return DualComplex(resolution, dual_complex(resolution))

    # The ring functions replace polynomial and ideal with readers that use the active ring.
    session.builtins.update({"QQ": QQ, "coordinates": declare, "ring": ring, "resolve": resolve,
                             "recompute_dual_complex": recompute_dual_complex, "dual_complex": dual_complex,
                             "monodromy_check": monodromy_check, "len": len, "print": print, "print_json": print_json,
                             "quotient_ring": quotient_ring, "hom": hom, **algebra_bindings(), **ring_bindings(rings)})
    return session


def run_script(filename, calculate):
    """Execute a file in a fresh session. Only explicit output goes to stdout."""
    name = "<stdin>" if filename == "-" else filename
    try:
        source = sys.stdin.read() if filename == "-" else Path(filename).read_text(encoding="utf-8")
        for _ in new_session(calculate).run(source):
            pass
    except SessionError as error:
        print(f"{name}:{error.line}: {error}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError) as error:
        print(f"{name}: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Script interrupted.", file=sys.stderr)
        return 130
    return 0
