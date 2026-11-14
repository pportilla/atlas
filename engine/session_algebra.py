"""Exact ideal arithmetic shared by the Atlas prompt and scripts."""

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import json
from typing import NamedTuple
from weakref import WeakValueDictionary

from . import native_polynomial as native
from .exact_json import result_json
from .native_library import affine_request
from .polynomial import LimitError, integer_text, parse_expression, rational_text, sparse_row
from .rational_point import normalize_point
from .session_polynomial import QQ, Polynomial, _CoordinateSpace


class _RingValue:
    _fields = {"variables", "coordinates", "coeffs", "order", "ring"}

    @property
    def variables(self):
        return self._space.names

    @property
    def coordinates(self):
        return self._space.coordinates

    @property
    def coeffs(self):
        return QQ

    @property
    def ring(self):
        return {"field": "QQ", "variables": list(self.variables), "order": self.order}

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"This algebra value has no Atlas field {name!r}.")


@dataclass(frozen=True, eq=False, repr=False)
class Ideal(_RingValue):
    _space: _CoordinateSpace
    generators: tuple[Polynomial, ...]
    order: str
    _fields = _RingValue._fields | {"generators"}
    _label = "ideal"

    def __repr__(self):
        ring = f"QQ[{', '.join(self.variables)}]" if self.variables else "QQ"
        generators = ", ".join(map(str, self.generators)) or "0"
        return f"{self._label}({generators}) over {ring} ({self.order})"


class _Result(_RingValue):
    _fields = _RingValue._fields | {"record", "json", "operation", "execution", "type", "result", "status"}

    @property
    def record(self):
        return deepcopy(self._record)

    @property
    def json(self):
        return result_json(self._record)

    @property
    def operation(self):
        return self._record["operation"]

    @property
    def execution(self):
        return deepcopy(self._record["execution"])

    @property
    def type(self):
        return self._record["result"]["type"]

    @property
    def result(self):
        return deepcopy(self._record["result"])

    @property
    def status(self):
        return self._record["status"]


@dataclass(frozen=True, eq=False, repr=False)
class _IdealResult(_Result, Ideal):
    _record: dict
    _fields = _Result._fields | Ideal._fields


class _Basis(_IdealResult):
    """Only the dispatcher can return a verified, reduced basis to the session."""

    _label = "Groebner basis"


@dataclass(frozen=True, eq=False, repr=False)
class _PointResult(_Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    _fields = _Result._fields | {"scope", "point"}

    @property
    def scope(self):
        return self._record["result"]["scope"]

    @property
    def point(self):
        return list(self._record["result"]["point"])


@dataclass(frozen=True, eq=False, repr=False)
class _LocalStandardBasis(_PointResult):
    generators: tuple[Polynomial, ...]
    _fields = _PointResult._fields | {"generators"}

    def __repr__(self):
        return f"Local standard basis at [{', '.join(self.point)}]: {self.generators}"


@dataclass(frozen=True, eq=False, repr=False)
class _LocalNormalForm(_PointResult):
    remainder: Polynomial
    unit: Polynomial
    _fields = _PointResult._fields | {"remainder", "unit", "reduction"}

    @property
    def reduction(self):
        return self._record["result"]["reduction"]

    def __repr__(self):
        return f"Local normal form at [{', '.join(self.point)}] (remainder={self.remainder}, unit={self.unit})"


@dataclass(frozen=True, eq=False, repr=False)
class _OrderedPolynomial(Polynomial):
    order: str

    def session_attribute(self, name):
        if name == "order":
            return self.order
        return super().session_attribute(name)


@dataclass(frozen=True, eq=False, repr=False)
class _PolynomialResult(_Result, _OrderedPolynomial):
    _record: dict
    _fields = _Result._fields | {"polynomial"}

    def session_attribute(self, name):
        if name in self._fields:
            return super().session_attribute(name)
        return Polynomial.session_attribute(self, name)

    @property
    def polynomial(self):
        return self


class _Factor(NamedTuple):
    polynomial: Polynomial
    multiplicity: int

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"A factor has no Atlas field {name!r}.")


@dataclass(frozen=True, eq=False, repr=False)
class _Factorization(_Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    unit: Fraction
    factors: tuple[_Factor, ...]
    _fields = _Result._fields | {"unit", "factors"}

    def __repr__(self):
        terms = [rational_text(self.unit)]
        for polynomial, multiplicity in self.factors:
            power = "" if multiplicity == 1 else "^" + integer_text(multiplicity)
            terms.append(f"({polynomial}){power}")
        return "*".join(terms)


@dataclass(frozen=True, eq=False, repr=False)
class _ScalarResult(_Result):
    _space: _CoordinateSpace
    order: str
    value: int | bool | None
    _record: dict
    _fields = _Result._fields | {"value"}

    def __repr__(self):
        return repr(self.value)

    def __bool__(self):
        return bool(self.value)

    def __eq__(self, other):
        if isinstance(other, _ScalarResult):
            return self.value == other.value
        return self.value == other


class _QuotientData:
    _fields = {"scope", "finite"}

    @property
    def scope(self):
        return self._record["result"]["scope"]

    @property
    def finite(self):
        return self._record["result"]["finite"]


class _LocalNumber(_QuotientData, _ScalarResult):
    _fields = _ScalarResult._fields | _QuotientData._fields

    def __repr__(self):
        return integer_text(self.value) if self.finite else "infinite"


@dataclass(frozen=True, eq=False, repr=False)
class _QuotientBasis(_QuotientData, _Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    basis: tuple[Polynomial, ...] | None
    _fields = _Result._fields | _QuotientData._fields | {"basis", "dimension"}

    @property
    def dimension(self):
        return self._record["result"]["dimension"]

    def __repr__(self):
        dimension = self.dimension if self.finite else "infinite"
        return f"{self.type}({self.scope}, dimension={dimension})"


@dataclass(frozen=True, eq=False, repr=False)
class _JacobianAlgebra(_QuotientBasis):
    generators: tuple[Polynomial, ...]
    _fields = _QuotientBasis._fields | {"generators"}


def _polynomial(value, space):
    if isinstance(value, Polynomial):
        if value._space is not space:
            raise ValueError("These operands belong to different coordinate spaces.")
        return value
    if type(value) is int or isinstance(value, Fraction):
        return Polynomial(space, native.coerce(value))
    raise TypeError("Expected a polynomial or an exact rational number.")


def _unpack(values):
    if len(values) == 1 and isinstance(values[0], (tuple, list)):
        return tuple(values[0])
    return values


_scalar_space = _CoordinateSpace(())
# Keep repeated eliminations from the same declaration in the same surviving
# space while users hold a result. Independent coordinate declarations stay distinct.
_subspaces = WeakValueDictionary()


def _result_space(space, names):
    if not names:
        return space if not space.generators else _scalar_space
    if tuple(names) == space.names:
        return space
    by_name = dict(zip(space.names, space.generators))
    survivors = tuple(by_name[name] for name in names)
    projected = _subspaces.get(survivors)
    if projected is None:
        projected = _CoordinateSpace(survivors)
        _subspaces[survivors] = projected
    return projected


def _coordinate_space(variables):
    variables = _unpack((variables,))
    if not all(isinstance(v, Polynomial) for v in variables):
        raise TypeError("Pass coordinate generators, for example variables=C.coordinates.")
    space = variables[0]._space if variables else _scalar_space
    if (any(v._space is not space for v in variables) or
            tuple(v._value for v in variables) != space.generators):
        raise ValueError("Use all coordinates in their declared order from one coordinate space.")
    return space


def polynomial(source, *, variables):
    """Parse a polynomial in the coordinate space owned by the supplied generators."""
    from .session_pullback import chart_source
    from .worker_process import run_worker

    if not isinstance(source, str) or not source.strip():
        raise ValueError("polynomial expects a nonempty expression string.")
    space = _coordinate_space(variables)
    generators = space.polynomial_generators
    source = chart_source(source, space)
    names = [g.name for g in generators]
    # Short, flat sums of monomials cannot expand products of sums. The
    # bounded parser also caps powers and intermediate coefficient sizes.
    if len(source) <= 128 and "(" not in source and ")" not in source:
        try:
            terms = parse_expression(source, names, bounded=True)
        except LimitError:
            pass  # A quick-parser limit selects the unrestricted worker.
        else:
            return Polynomial(space, native.from_terms(generators, terms))

    request = {"source": source, "variables": names}
    status, record = run_worker(["engine.algebra_worker", "--polynomial"], request, None)
    if status != 200:
        raise ValueError(record["error"]["message"])
    return Polynomial(space, native.from_row(generators, record["terms"]))


def _polynomial_inputs(values, order, variables, default_space=None):
    # Coordinate expressions carry their space; computed results also carry
    # the order used for their calculation.
    if order is None:
        orders = {f.order for f in values if isinstance(f, _OrderedPolynomial)}
        if len(orders) > 1:
            raise ValueError("Choose an explicit monomial order for these polynomials.")
        order = next(iter(orders), "grevlex")
    if order not in ("lex", "grevlex"):
        raise ValueError("Unsupported monomial order. Choose lex or grevlex.")
    if variables is None:
        space = next((f._space for f in values if isinstance(f, Polynomial)), default_space)
        if space is None:
            raise ValueError("For a constant or empty ideal, supply variables=(x, y) or variables=() over QQ.")
    else:
        space = _coordinate_space(variables)
    if space.coeffs is not QQ:
        raise ValueError("Algebra operations require polynomial coordinates over QQ.")
    if space.has_relations:
        raise ValueError("Algebra operations require polynomial coordinates with no quotient-ring relations.")
    return space, tuple(_polynomial(f, space) for f in values), order


def ideal(*generators, order=None, variables=None):
    """Declare an ideal. For scalar-only generators specify variables=(x, y)."""
    space, generators, order = _polynomial_inputs(_unpack(generators), order, variables)
    return Ideal(space, generators, order)


def _typed(record, space):
    result = record["result"]
    space = _result_space(space, result["ring"]["variables"])
    order = result["ring"]["order"]

    def decode(text):
        # Kernel output is already canonical. Decode its exact coefficients as
        # arithmetic results, which can exceed the session's input-literal limit.
        if not space.generators:
            return native.coerce(sparse_row(text).get((), 0))
        row, = affine_request(["decode", " ".join(space.names), text])
        return native.from_row(space.generators, row)

    def polynomials(expressions):
        return tuple(Polynomial(space, decode(text)) for text in expressions)

    kind = result["type"]
    if kind in ("basis", "ideal"):
        cls = _Basis if kind == "basis" else _IdealResult
        return cls(space, polynomials(result["generators"]), order, record)
    if kind == "local-standard-basis":
        return _LocalStandardBasis(space, order, record, polynomials(result["generators"]))
    if kind == "local-normal-form":
        remainder, unit = polynomials((result["remainder"], result["unit"]))
        return _LocalNormalForm(space, order, record, remainder, unit)
    if kind in ("quotient-basis", "local-jacobian-algebra"):
        basis = None if result["basis"] is None else polynomials(result["basis"])
        if kind == "local-jacobian-algebra":
            return _JacobianAlgebra(space, order, record, basis, polynomials(result["generators"]))
        return _QuotientBasis(space, order, record, basis)
    if kind == "polynomial":
        return _PolynomialResult(space, decode(result["polynomial"]), order, record)
    if kind == "factorization":
        unit = sparse_row(result["unit"])[()]
        factors = tuple(
            _Factor(_OrderedPolynomial(space, decode(entry["polynomial"]), order), entry["multiplicity"])
            for entry in result["factors"]
        )
        return _Factorization(space, order, record, unit, factors)
    if kind in ("boolean", "integer"):
        cls = _LocalNumber if "finite" in result else _ScalarResult
        return cls(space, order, result["value"], record)
    raise ValueError(f"Unknown algebra result type {kind!r}.")


def _evaluate(request, *, known_basis=False):
    from .algebra import calculate_algebra

    status, record = calculate_algebra(request, known_basis=known_basis)
    if status != 200:
        raise ValueError(record["error"]["message"])
    return record


def algebra_request(source):
    """Replay an algebra JSON request in a fresh coordinate space."""
    if not isinstance(source, str):
        raise TypeError("algebra_request expects a JSON request string.")
    request = json.loads(source)
    record = _evaluate(request)
    names = record["result"]["ring"]["variables"]
    space = _CoordinateSpace.named(names) if names else _scalar_space
    return _typed(record, space)


def _require_ideal(value):
    if not isinstance(value, Ideal):
        raise TypeError("Pass an ideal or a Gröbner basis.")
    return value


def _calculate(operation, left, compute, **operands):
    local_basis = isinstance(left, _LocalStandardBasis) and operation == "local-normal-form"
    if not local_basis:
        left = _require_ideal(left)
    if "other_ideal" in operands:
        right = _require_ideal(operands["other_ideal"])
        if right._space is not left._space:
            raise ValueError("These ideals belong to different coordinate spaces.")
        if right.order != left.order:
            raise ValueError("These ideals have different monomial orders.")
        operands["other_ideal"] = [str(f) for f in right.generators]
    if "polynomial" in operands:
        operands["polynomial"] = str(_polynomial(operands["polynomial"], left._space))
    operands["ideal"] = [str(f) for f in left.generators]
    request = {"operation": operation, "ring": left.ring, "operands": operands, "compute": compute}
    known_basis = local_basis or (isinstance(left, _Basis) and operation in ("groebner", "normal-form", "membership"))
    return _typed(_evaluate(request, known_basis=known_basis), left._space)


def groebner(ideal, *, compute="auto"):
    return _calculate("groebner", ideal, compute)


def normal_form(polynomial, ideal, *, compute="auto"):
    return _calculate("normal-form", ideal, compute, polynomial=polynomial)


def _local_point(point, ideal):
    if point is None:
        return ["0"] * len(ideal.variables)
    if not isinstance(point, (list, tuple)):
        raise TypeError("Pass the point as a list or tuple in coordinate order.")
    entries = [rational_text(value) if isinstance(value, Fraction) else value for value in point]
    return normalize_point(entries, len(ideal.variables))


def local_standard_basis(ideal, *, point=None, compute="auto"):
    ideal = _require_ideal(ideal)
    return _calculate("local-standard-basis", ideal, compute, point=_local_point(point, ideal))


def local_normal_form(polynomial, ideal, *, point=None, compute="auto"):
    if isinstance(ideal, _LocalStandardBasis):
        saved_point = ideal.point
        if point is not None and _local_point(point, ideal) != saved_point:
            raise ValueError("The supplied point differs from the saved local standard basis point.")
        point = saved_point
    else:
        ideal = _require_ideal(ideal)
        point = _local_point(point, ideal)
    return _calculate("local-normal-form", ideal, compute, polynomial=polynomial, point=point)


def membership(polynomial, ideal, *, compute="auto"):
    return _calculate("membership", ideal, compute, polynomial=polynomial)


def _calculate_polynomials(operation, order, variables, compute, **operands):
    space, values, order = _polynomial_inputs(operands.values(), order, variables, _scalar_space)
    request = {"operation": operation, "ring": {"field": "QQ", "variables": list(space.names), "order": order},
               "operands": {key: str(f) for key, f in zip(operands, values)}, "compute": compute}
    return _typed(_evaluate(request), space)


def gcd(left, right, *, order=None, variables=None, compute="auto"):
    """Monic polynomial GCD in the common coordinate space, or QQ for scalars."""
    return _calculate_polynomials("gcd", order, variables, compute, polynomial=left, other_polynomial=right)


def factor(polynomial, *, order=None, variables=None, compute="auto"):
    """Rational unit and monic irreducible factors in the polynomial's coordinate space."""
    return _calculate_polynomials("factor", order, variables, compute, polynomial=polynomial)


def local_jacobian_algebra(polynomial, *, order=None, variables=None, compute="auto"):
    return _calculate_polynomials("local-jacobian-algebra", order, variables, compute, polynomial=polynomial)


def milnor_number(polynomial, *, order=None, variables=None, compute="auto"):
    return _calculate_polynomials("milnor-number", order, variables, compute, polynomial=polynomial)


def tjurina_number(polynomial, *, order=None, variables=None, compute="auto"):
    return _calculate_polynomials("tjurina-number", order, variables, compute, polynomial=polynomial)


def eliminate(ideal, *variables, compute="auto"):
    _require_ideal(ideal)
    names = []
    for variable in _unpack(variables):
        if isinstance(variable, Polynomial):
            coordinate = _polynomial(variable, ideal._space)._value
            if not coordinate.is_Symbol:
                raise ValueError("Elimination requires coordinate variables.")
            variable = coordinate.name
        names.append(variable)
    return _calculate("eliminate", ideal, compute, variables=names)


def intersection(ideal, other_ideal, *, compute="auto"):
    return _calculate("intersection", ideal, compute, other_ideal=other_ideal)


def quotient(ideal, other_ideal, *, compute="auto"):
    return _calculate("quotient", ideal, compute, other_ideal=other_ideal)


def saturation(ideal, polynomial, *, compute="auto"):
    return _calculate("saturation", ideal, compute, polynomial=polynomial)


def dimension(ideal, *, compute="auto"):
    return _calculate("dimension", ideal, compute)


def quotient_basis(ideal, *, compute="auto"):
    return _calculate("quotient-basis", ideal, compute)


def local_quotient_basis(ideal, *, compute="auto"):
    return _calculate("local-quotient-basis", ideal, compute)


def algebra_bindings():
    return {"polynomial": polynomial, "ideal": ideal, "groebner": groebner, "gcd": gcd, "factor": factor,
            "normal_form": normal_form, "membership": membership,
            "local_standard_basis": local_standard_basis, "local_normal_form": local_normal_form,
            "eliminate": eliminate, "intersection": intersection, "quotient": quotient,
            "saturation": saturation, "saturate": saturation, "dimension": dimension,
            "quotient_basis": quotient_basis, "local_quotient_basis": local_quotient_basis,
            "local_jacobian_algebra": local_jacobian_algebra, "milnor_number": milnor_number,
            "tjurina_number": tjurina_number, "algebra_request": algebra_request}
