"""Mathematical operations and execution shared by the Atlas prompt and scripts."""
import sys
from fractions import Fraction
from functools import wraps
from math import isfinite
from pathlib import Path

from .exact_json import result_json
from .matrix_algebra import matrix_analysis
from .modular_batch import modular_arithmetic
from .polynomial import rational_text
from .session_algebra import _coordinate_space, algebra_bindings
from .session_algebra import polynomial_arithmetic as calculate_polynomial
from .session_expressions import Evaluator, SessionError
from .session_maps import hom
from .session_monodromy import check_resolution, inconclusive_check
from .session_polynomial import QQ, Polynomial, _CoordinateSpace, coordinates as coordinates, session_declarations
from .session_quotient import QuotientElement, QuotientIdeal, QuotientRing, quotient_ring
from .session_rings import ring_bindings
from .session_values import Divisor, DualComplex, Resolution
from .zoo_search import zoo_search


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

    def resolve(polynomial, *, variables=None, ring=None, mode=None, search="exhaustive",
                compute=None, monodromy=False, method="auto"):
        expression, names, parent = _resolution_input(polynomial, variables, ring, declare, rings.active)
        if type(monodromy) is not bool:
            raise ValueError("monodromy must be true or false.")
        options = {"monodromy": True} if monodromy else {}
        if method != "auto":
            options["method"] = method
        if mode is None:
            # Jung's method resolves the surface itself; every other method defaults to embedded.
            mode = "intrinsic" if method == "jung" else "embedded"
        status, record = calculate(expression, names, mode=mode, search=search, compute=compute, **options)
        if status != 200:
            raise ValueError(record["error"])
        return Resolution(record, parent, source_polynomial=polynomial if isinstance(polynomial, Polynomial) else None,
                          _from_solver=True)

    def monodromy_check(polynomial, *, variables=None, ring=None, search="exhaustive", compute=None, check_isolation=False):
        if type(check_isolation) is not bool:
            raise ValueError("check_isolation must be true or false.")
        if search not in ("quick", "exhaustive"):
            raise ValueError("Search must be quick or exhaustive.")
        if compute not in (None, "cpu", "auto", "gpu"):
            raise ValueError("Compute must be cpu, auto, or gpu.")
        zetas = None
        if isinstance(polynomial, Resolution):
            if variables is not None or ring is not None:
                raise ValueError("A resolution already carries its coordinates.")
            resolution = polynomial
        else:
            expression, names, parent = _resolution_input(polynomial, variables, ring, declare, rings.active)
            status, record = calculate(expression, names, mode="embedded", search=search,
                                       compute=compute, monodromy=True, zetas_only=True)
            if status == 422 and record.get("code") == "computation_limit" and search == "quick":
                return inconclusive_check(record["error"])
            if status == 400:
                raise ValueError(record["error"])
            if status != 200:
                raise RuntimeError(record["error"])
            resolution = Resolution(record, parent, source_polynomial=polynomial if isinstance(polynomial, Polynomial) else None,
                                    _from_solver=True)
            # A bounded solve already attempted both zetas. Display its outcome
            # without launching a second search after it reached its deadline.
            if search == "quick":
                zetas = record.get("topological_zeta", {}), record.get("monodromy_zeta", {})
        return check_resolution(resolution, check_isolation=check_isolation, compute=compute, search=search, zetas=zetas)

    def jung_resolution(polynomial, *, variables=None, ring=None):
        """Return the resolution graph and invariants of a surface singularity at the origin from Jung's method.

        >>> x, y, z = coordinates("x", "y", "z")
        >>> J = jung_resolution(x^2 + y^3 + z^5)
        >>> print(len(J.curves), J.K2, J.euler_characteristic, J.geometric_genus)  # 8 0 9 0

        Docs: docs/thematic/jung-method.md#resolve-a-surface-with-jungs-method
        """
        expression, names, parent = _resolution_input(polynomial, variables, ring, declare, rings.active)
        from .jung import JungResolution, jung_record
        return JungResolution(jung_record(expression, names), parent)

    def dual_complex(resolution):
        # Older scripts use this function's raw replay dictionary.
        from .resolution_complex import dual_complex as replay
        if not isinstance(resolution, Resolution):
            raise ValueError("Pass an Atlas resolution to recompute its dual complex.")
        return replay(resolution._record)

    def recompute_dual_complex(resolution):
        return DualComplex(resolution, dual_complex(resolution))

    def with_scalar_default(function, names):
        @wraps(function)
        def calculate(*args, **options):
            if rings.active is not None and options.get("variables") is None:
                inputs = (*args[:len(names)], *(options[key] for key in names if key in options))
                if inputs and all(type(value) is int or isinstance(value, Fraction) for value in inputs):
                    options["variables"] = rings.active.coordinates
            return function(*args, **options)
        return calculate

    @wraps(calculate_polynomial)
    def polynomial_arithmetic(operation, polynomial, other=None, *, exponent=None, variable=None, images=None,
                              order=None, variables=None, compute="auto"):
        inputs = [polynomial, other, variable]
        if isinstance(images, (list, tuple)):
            inputs.extend(images)
        if variables is None and rings.active is not None and not any(
                isinstance(item, Polynomial) and item.parent is not QQ for item in inputs):
            variables = rings.active.coordinates
        return calculate_polynomial(operation, polynomial, other, exponent=exponent, variable=variable,
                                    images=images, order=order, variables=variables, compute=compute)

    # The ring functions replace polynomial and ideal with readers that use the active ring.
    algebra = algebra_bindings()
    for name, inputs in (("absolute_factor", ("polynomial",)), ("squarefree_factor", ("polynomial",)),
                         ("polynomial_division", ("dividend", "divisor")), ("cancel_fraction", ("numerator", "denominator")),
                         ("extended_gcd", ("left", "right")), ("hensel_factors", ("polynomial", "left", "right"))):
        algebra[name] = with_scalar_default(algebra[name], inputs)
    session.builtins.update({"QQ": QQ, "coordinates": declare, "ring": ring, "resolve": resolve,
                             "matrix_analysis": matrix_analysis, "modular_arithmetic": modular_arithmetic,
                             "recompute_dual_complex": recompute_dual_complex, "dual_complex": dual_complex,
                             "monodromy_check": monodromy_check, "zoo_search": zoo_search,
                             "jung_resolution": jung_resolution,
                             "len": len, "print": print, "print_json": print_json,
                             "quotient_ring": quotient_ring, "hom": hom, **algebra, **ring_bindings(rings),
                             "polynomial_arithmetic": polynomial_arithmetic})
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
