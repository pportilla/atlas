"""Mathematical operations and execution shared by the Atlas prompt and scripts."""
import sys
from fractions import Fraction
from math import isfinite
from pathlib import Path

from .exact_json import result_json
from .polynomial import rational_text
from .session_algebra import algebra_bindings
from .session_expressions import Evaluator, SessionError
from .session_monodromy import check_resolution, unavailable_check
from .session_polynomial import QQ, Polynomial, coordinates
from .session_values import Resolution


HELP = """Define coordinates and keep the results of calculations:

  x, y = coordinates("x", "y")       # optional: coeffs=QQ
  f = y^2 - x^3
  R = resolve(f)
  R.divisors
  E = R.divisor("E3")
  C = R.chart("B3.u")
  u, v = C.coordinates
  s = polynomial("v-1", variables=C.coordinates)
  s == C.strict_transform                # True
  total = C.pullback(f)
  total.multiplicity(E.equation_in(C))   # 6

  M = monodromy_check(f, search="quick")
  M.status                               # verified, unresolved, unavailable or isolated_candidate
  M.poles                                # actual poles and their monodromy witnesses
  M.candidate_poles                      # pole_order is zero for cancelled values
  print_json(equation=f, candidates=M.candidate_poles, poles=M.poles)

  D = factor((-3/2)*(x+y)^2)
  D.unit                                 # -3/2
  D.factors[0].polynomial                # x+y
  D.factors[0].multiplicity              # 2

  I = ideal(x^2*(x-1), y)
  quotient_basis(I).dimension            # 3, including the point (1, 0)
  local_quotient_basis(I).dimension      # 2 at the origin
  B = local_standard_basis(I, point=(1, 0))
  h = local_normal_form(x-1, B)          # uses the point saved in B
  h.remainder == 0                       # x-1 belongs to I locally at (1, 0)

For a local normal form h of g, h.unit equals 1 at the chosen point, and
h.unit*g agrees with h.remainder modulo the localized ideal. When h.reduction
is "standard", every remainder term is reduced. When it is "weak", a nonzero
remainder has a reduced local leading term.

  A = local_jacobian_algebra(f)
  A.generators                           # partial derivatives in coordinate order
  A.basis                                # monomials in the original coordinates
  milnor_number(f).value                 # 2 at the origin
  tjurina_number(f).value                # 2 at the origin

Local quotient bases, Jacobian algebras, Milnor numbers and Tjurina numbers
default to the origin. Pass rational coordinates in ring order to select a point.

  g = y^2 - (x-1/2)^3
  milnor_number(g, point=(1/2, 0)).value # 2 at the translated cusp

For an infinite-dimensional quotient, .finite is False and both .basis and
.dimension are None. For an infinite Milnor or Tjurina number, .value is None.

Inspect R.charts and R.branches, or use print(...) for explicit output.
Run the same expressions in a file with ./atlas run FILE.atlas."""


def _json_value(value):
    if isinstance(value, Polynomial):
        return str(value)
    if isinstance(value, Fraction):
        return rational_text(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON record fields must have string names.")
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON numbers must be finite.")
    return value


def print_json(**fields):
    """Emit one JSON record, with canonical polynomials and exact rational strings."""
    print(result_json(_json_value(fields)), flush=True)


def _resolution_input(polynomial, variables):
    if isinstance(polynomial, Polynomial):
        if variables is not None:
            raise ValueError("A polynomial already carries its coordinates.")
        if polynomial.coeffs is not QQ or polynomial._space.has_relations:
            raise ValueError("resolve expects an equation in affine polynomial coordinates over QQ.")
        return str(polynomial), list(polynomial.variables), polynomial._space
    if isinstance(polynomial, str):
        if not isinstance(variables, (list, tuple)):
            raise ValueError('For a polynomial string, supply variables=["x", "y"].')
        return polynomial, list(variables), None
    raise ValueError("resolve expects a polynomial or a polynomial string with variables.")


def new_session(calculate):
    """Register session operations here so both entry points return the same values."""
    def resolve(polynomial, *, variables=None, mode="embedded", search="exhaustive",
                compute=None, monodromy=False):
        expression, names, source_space = _resolution_input(polynomial, variables)
        if type(monodromy) is not bool:
            raise ValueError("monodromy must be true or false.")
        options = {"monodromy": True} if monodromy else {}
        status, record = calculate(expression, names, mode=mode, search=search, compute=compute, **options)
        if status != 200:
            raise ValueError(record["error"])
        return Resolution(record, source_space, _from_solver=True)

    def monodromy_check(polynomial, *, variables=None, search="quick", compute=None, check_isolation=False):
        if type(check_isolation) is not bool:
            raise ValueError("check_isolation must be true or false.")
        if isinstance(polynomial, Resolution):
            if variables is not None:
                raise ValueError("A resolution already carries its coordinates.")
            resolution = polynomial
        else:
            expression, names, source_space = _resolution_input(polynomial, variables)
            status, record = calculate(expression, names, mode="embedded", search=search,
                                       compute=compute, monodromy=True)
            if status == 400:
                raise ValueError(record["error"])
            if status != 200:
                return unavailable_check(record["error"])
            resolution = Resolution(record, source_space, _from_solver=True)
        return check_resolution(resolution, check_isolation=check_isolation, compute=compute)

    return Evaluator({"QQ": QQ, "coordinates": coordinates, "resolve": resolve,
                      "monodromy_check": monodromy_check, "len": len, "print": print, "print_json": print_json,
                      **algebra_bindings()}, reserved=("help", "quit", "exit"))


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
