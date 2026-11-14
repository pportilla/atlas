"""Mathematical operations and execution shared by the Atlas prompt and scripts."""
import sys
from pathlib import Path

from .session_algebra import algebra_bindings
from .session_expressions import Evaluator, SessionError
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

For an infinite-dimensional quotient, .finite is False and both .basis and
.dimension are None. For an infinite Milnor or Tjurina number, .value is None.

Inspect R.charts and R.branches, or use print(...) for explicit output.
Run the same expressions in a file with ./atlas run FILE.atlas."""


def new_session(calculate):
    """Register session operations here so both entry points return the same values."""
    def resolve(polynomial, *, variables=None, mode="embedded", search="exhaustive", compute=None):
        source_space = None
        if isinstance(polynomial, Polynomial):
            if variables is not None:
                raise ValueError("A polynomial already carries its coordinates.")
            if polynomial.coeffs is not QQ or polynomial._space.has_relations:
                raise ValueError("resolve expects an equation in affine polynomial coordinates over QQ.")
            source_space = polynomial._space
            variables = list(polynomial.variables)
            polynomial = str(polynomial)
        elif isinstance(polynomial, str):
            if not isinstance(variables, (list, tuple)):
                raise ValueError('For a polynomial string, supply variables=["x", "y"].')
            variables = list(variables)
        else:
            raise ValueError("resolve expects a polynomial or a polynomial string with variables.")
        status, record = calculate(polynomial, variables, mode=mode, search=search, compute=compute)
        if status != 200:
            raise ValueError(record["error"])
        return Resolution(record, source_space)

    return Evaluator({"QQ": QQ, "coordinates": coordinates, "resolve": resolve,
                      "len": len, "print": print, **algebra_bindings()}, reserved=("help", "quit", "exit"))


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
