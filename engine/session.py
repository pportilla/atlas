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
from .session_polynomial import QQ, Polynomial, coordinates as coordinates, session_coordinates
from .session_quotient import QuotientElement, QuotientRing, quotient_ring
from .session_values import Divisor, DualComplex, Resolution


HELP = """Define coordinates and keep the results of calculations:

  x, y = coordinates("x", "y")       # optional: coeffs=QQ
  f = y^2 - x^3
  g = polynomial("x^2+y", ring=f.parent)
  R = resolve(f)
  R.polynomial == f                      # True in the original coordinates
  D = R.dual_complex                     # the recorded exceptional complex
  D.vertices                             # [E1, E2, E3]
  R.branches[0].meets                    # [E3]
  G = recompute_dual_complex(R)          # independently rebuild incidence from the charts
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

  q = resultant(y^2-x, y-2, y)
  x1, = q.coordinates                    # x in the surviving ring QQ[x]
  q == 4-x1                              # True

  S = hilbert_series(ideal(x^2, y^3))    # (1-t^2)*(1-t^3)/(1-t)^2
  S.numerator[1].degree                  # 2
  S.dimension, S.multiplicity            # (0, 6)

  I = ideal(x^2*(x-1), y)
  J = radical(I)                         # the ideal of the two reduced points
  A = quotient_ring(ideal(x*y - 1))
  u, v = A.gens                         # residue classes of x and y
  u.inverse() == v                      # True in A
  (u^3 / u).representative              # x^2 in the ambient coordinates
  K = ideal_power(I, 2)                  # ordinary square
  S = symbolic_power(J, 2)               # symbolic square of the radical ideal
  D = primary_decomposition(I)           # each component has a primary ideal and its prime
  Q, P = D.components[0].primary, D.components[0].prime
  L = localize(ideal(x*y), ideal(x), numerator=x, denominator=y)
  N = normalization(ideal(y^2-x^3))     # normalize the affine cusp
  C = N.components[0]
  C.relations, C.images                 # presentation and images of x and y
  C.fractions                           # normalized coordinates as fractions on C.prime
  membership(x*(x-1), J).value           # True
  e = multiplicity(I)                    # total dimension of QQ[x,y]/I over QQ
  e.value, e.finite                      # (3, True)
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
  ae_codimension(x, y^2).value           # 0 for the stable fold map

  H = hilbert_samuel_function(ideal(f), degrees=[0, 2, 5])
  H.values                               # cumulative lengths 1, 5, 11
  H.graded_values                        # graded piece dimensions 1, 2, 2
  hilbert_samuel_multiplicity(ideal(f)).value  # 2 at the cusp

Local quotient bases, Jacobian algebras, Milnor numbers and Tjurina numbers
default to the origin. Pass rational coordinates in ring order to select a point.

  g = y^2 - (x-1/2)^3
  milnor_number(g, point=(1/2, 0)).value # 2 at the translated cusp

  B = local_standard_basis(ideal(g), point=(1/2, 0))
  C = tangent_cone(B)
  dx, dy = C.coordinates                 # displacements from (1/2, 0)
  C.generators[0] == dy^2                # True, the tangent cone is a double line
  S = tangent_cone_hilbert_series(C)     # graded at the cone vertex
  for term in S.numerator: print(term.degree, term.coefficient)
  S.denominator_power                    # 2, giving (1-t^2)/(1-t)^2

Both tangent cone functions accept an ideal with point= or a local standard
basis with its saved point. The loop prints each degree and its signed
coefficient in ascending degree order, here 0 1 followed by 2 -1.

  embedding_dimension(B).value           # 2 at the cusp, using B's saved point
  local_dimension(B).value               # 1 at the same point
  local_smoothness(B).value              # False: local dimension 1, embedding dimension 2
  V = zariski_tangent_space(ideal(y-(x-1/2)^2), point=(1/2, 0))
  dx, dy = V.coordinates
  V.equations[0] == dy                   # True, the parabola's tangent is dy = 0
  V.basis                                 # one direction (1, 0)

Both functions also accept C to compute at the tangent cone's vertex.

For an infinite-dimensional quotient, quotient_basis(I) has .finite = False
and .basis = .dimension = None. The multiplicity(I) result has .finite = False
and .value = None. For an infinite Milnor or Tjurina number, .value is None.

Type help R.charts for chart lookup, help R.dual_complex for exceptional
intersections, and help R.branches for branch attachments. Use print(...) for script output.
Run the same expressions in a file with ./atlas run FILE.atlas."""


def json_value(value):
    if isinstance(value, (DualComplex, Divisor)):
        return json_value(value._record)
    if isinstance(value, (QuotientElement, QuotientRing)):
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


def _resolution_input(polynomial, variables):
    if isinstance(polynomial, Polynomial):
        if variables is not None:
            raise ValueError("A polynomial already carries its coordinates.")
        if polynomial.coeffs is not QQ or polynomial._space.has_relations:
            raise ValueError("resolve expects an equation in affine polynomial coordinates over QQ.")
        return str(polynomial), list(polynomial.variables), polynomial
    if isinstance(polynomial, str):
        if not isinstance(variables, (list, tuple)):
            raise ValueError('For a polynomial string, supply variables=["x", "y"].')
        return polynomial, list(variables), None
    raise ValueError("resolve expects a polynomial or a polynomial string with variables.")


def new_session(calculate):
    """Register session operations here so both entry points return the same values."""
    def resolve(polynomial, *, variables=None, mode="embedded", search="exhaustive",
                compute=None, monodromy=False):
        expression, names, source_polynomial = _resolution_input(polynomial, variables)
        if type(monodromy) is not bool:
            raise ValueError("monodromy must be true or false.")
        options = {"monodromy": True} if monodromy else {}
        status, record = calculate(expression, names, mode=mode, search=search, compute=compute, **options)
        if status != 200:
            raise ValueError(record["error"])
        return Resolution(record, source_polynomial=source_polynomial, _from_solver=True)

    def monodromy_check(polynomial, *, variables=None, search="quick", compute=None, check_isolation=False):
        if type(check_isolation) is not bool:
            raise ValueError("check_isolation must be true or false.")
        if isinstance(polynomial, Resolution):
            if variables is not None:
                raise ValueError("A resolution already carries its coordinates.")
            resolution = polynomial
        else:
            expression, names, source_polynomial = _resolution_input(polynomial, variables)
            status, record = calculate(expression, names, mode="embedded", search=search,
                                       compute=compute, monodromy=True)
            if status == 400:
                raise ValueError(record["error"])
            if status != 200:
                return unavailable_check(record["error"])
            resolution = Resolution(record, source_polynomial=source_polynomial, _from_solver=True)
        return check_resolution(resolution, check_isolation=check_isolation, compute=compute)

    def dual_complex(resolution):
        # Older scripts use this function's raw replay dictionary.
        from .resolution_complex import dual_complex as replay
        if not isinstance(resolution, Resolution):
            raise ValueError("Pass an Atlas resolution to recompute its dual complex.")
        return replay(resolution._record)

    def recompute_dual_complex(resolution):
        return DualComplex(resolution, dual_complex(resolution))

    return Evaluator({"QQ": QQ, "coordinates": session_coordinates(), "resolve": resolve,
                      "recompute_dual_complex": recompute_dual_complex, "dual_complex": dual_complex,
                      "monodromy_check": monodromy_check, "len": len, "print": print, "print_json": print_json,
                      "quotient_ring": quotient_ring, **algebra_bindings()}, reserved=("help", "quit", "exit"))


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
