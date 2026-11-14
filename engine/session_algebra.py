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
from .rational_point import coordinate_fraction, normalize_point
from .series_display import format_hilbert_series
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


class _PrimaryComponent(NamedTuple):
    primary: Ideal
    prime: Ideal

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"A primary component has no Atlas field {name!r}.")


@dataclass(frozen=True, eq=False, repr=False)
class _PrimaryDecomposition(_Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    components: tuple[_PrimaryComponent, ...]
    _fields = _Result._fields | {"components"}

    def __repr__(self):
        if not self.components:
            return "The unit ideal has no primary components."
        return "\n".join(f"Primary component {i}: {q}\nAssociated prime: {p}"
                         for i, (q, p) in enumerate(self.components, start=1))


class _LocalElement(NamedTuple):
    numerator: Polynomial
    denominator: Polynomial
    member: bool
    unit: bool

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"A local fraction has no Atlas field {name!r}.")


@dataclass(frozen=True, eq=False, repr=False)
class _Localization(_Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    prime: Ideal
    contraction: Ideal
    element: _LocalElement | None
    _fields = _Result._fields | {"prime", "contraction", "element", "fraction"}

    def fraction(self, f, s=1, *, compute=None):
        """Test f/s for membership in the localized ideal and invertibility in the local ring."""
        mode = self.execution["requested"] if compute is None else compute
        return localize(self.contraction, self.prime, numerator=f, denominator=s, compute=mode).element

    def __repr__(self):
        from .algebra_cli import localization_lines
        return "\n".join(localization_lines(self._record["result"]))


class _NormalizationFraction(NamedTuple):
    numerator: Polynomial
    denominator: Polynomial

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"A normalization fraction has no Atlas field {name!r}.")


@dataclass(frozen=True, eq=False)
class _NormalizedComponent(_RingValue):
    _space: _CoordinateSpace
    order: str
    prime: Ideal
    relations: Ideal
    images: tuple[Polynomial, ...]
    fractions: tuple[_NormalizationFraction, ...]
    _fields = _RingValue._fields | {"prime", "relations", "images", "fractions"}


@dataclass(frozen=True, eq=False, repr=False)
class _Normalization(_Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    components: tuple[_NormalizedComponent, ...]
    _fields = _Result._fields | {"components", "input_reduced"}

    @property
    def input_reduced(self):
        return self._record["result"]["input_reduced"]

    def __repr__(self):
        from .algebra_cli import normalization_lines
        return "\n".join(normalization_lines(self._record["result"]))


class _AtPoint:
    @property
    def scope(self):
        return self._record["result"]["scope"]

    @property
    def point(self):
        result = self._record["result"]
        if result["scope"] == "origin":
            return ["0"] * len(self.variables)
        return list(result["point"])


@dataclass(frozen=True, eq=False, repr=False)
class _PointResult(_AtPoint, _Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    _fields = _Result._fields | {"scope", "point"}


@dataclass(frozen=True, eq=False, repr=False)
class _LocalStandardBasis(_PointResult):
    generators: tuple[Polynomial, ...]
    _fields = _PointResult._fields | {"generators"}

    def __repr__(self):
        return f"Local standard basis at [{', '.join(self.point)}]: {self.generators}"


class _TangentResult(_PointResult):
    _fields = _PointResult._fields | {"coordinate_system", "dimension"}

    @property
    def coordinate_system(self):
        return self._record["result"]["coordinate_system"]

    @property
    def dimension(self):
        return self._record["result"]["dimension"]


@dataclass(frozen=True, eq=False, repr=False)
class _TangentCone(_TangentResult):
    generators: tuple[Polynomial, ...]
    _fields = _TangentResult._fields | {"generators", "multiplicity"}

    @property
    def multiplicity(self):
        return self._record["result"]["multiplicity"]

    def __repr__(self):
        return f"Tangent cone at [{', '.join(self.point)}] with generators {self.generators}"


class _TangentDimensions:
    @property
    def ambient_dimension(self):
        return self._record["result"]["ambient_dimension"]

    @property
    def jacobian_rank(self):
        return self._record["result"]["jacobian_rank"]


@dataclass(frozen=True, eq=False, repr=False)
class _ZariskiTangentSpace(_TangentDimensions, _TangentResult):
    equations: tuple[Polynomial, ...]
    basis: tuple[tuple[Fraction, ...], ...]
    _fields = _TangentResult._fields | {"ambient_dimension", "jacobian_rank", "equations", "basis"}

    def __repr__(self):
        return f"Zariski tangent space at [{', '.join(self.point)}] of dimension {self.dimension}"


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
    _fields = Polynomial._fields | {"order"}


@dataclass(frozen=True, eq=False, repr=False)
class _PolynomialResult(_Result, _OrderedPolynomial):
    _record: dict
    _fields = _Result._fields | _OrderedPolynomial._fields | {"polynomial"}

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


class _QuotientNumber(_QuotientData, _ScalarResult):
    _fields = _ScalarResult._fields | _QuotientData._fields
    _infinite_display = "infinite colength"

    def __repr__(self):
        return integer_text(self.value) if self.finite else self._infinite_display


class _LocalNumber(_AtPoint, _QuotientNumber):
    _fields = _QuotientNumber._fields | {"point"}
    _infinite_display = "infinite"


class _AeCodimension(_LocalNumber):
    _fields = _LocalNumber._fields | {"method", "lower_bound", "jet_degree", "contact_order", "reason"}

    @property
    def method(self):
        return self._record["result"]["method"]

    @property
    def lower_bound(self):
        return self._record["result"].get("lower_bound")

    @property
    def jet_degree(self):
        return self._record["result"].get("jet_degree")

    @property
    def contact_order(self):
        return self._record["result"].get("contact_order")

    @property
    def reason(self):
        return self._record["result"].get("reason")

    def __repr__(self):
        return "inconclusive" if self.finite is None else super().__repr__()


@dataclass(frozen=True, eq=False, repr=False)
class _HilbertSeries(_Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    _fields = _Result._fields | {"scope", "numerator", "denominator_power", "dimension", "multiplicity"}

    @property
    def scope(self):
        return self._record["result"]["scope"]

    @property
    def numerator(self):
        return deepcopy(self._record["result"]["numerator"])

    @property
    def denominator_power(self):
        return self._record["result"]["denominator_power"]

    @property
    def dimension(self):
        return self._record["result"]["dimension"]

    @property
    def multiplicity(self):
        return self._record["result"]["multiplicity"]

    def __repr__(self):
        return format_hilbert_series(self._record["result"]["numerator"], self.denominator_power)


class _TangentConeHilbertSeries(_AtPoint, _HilbertSeries):
    _fields = _HilbertSeries._fields | {"point"}


class _HilbertSamuelMultiplicity(_AtPoint, _ScalarResult):
    _fields = _ScalarResult._fields | {"point", "scope", "dimension"}

    @property
    def dimension(self):
        return self._record["result"]["dimension"]

    def __repr__(self):
        return integer_text(self.value)


class _EmbeddingDimension(_TangentDimensions, _AtPoint, _ScalarResult):
    _fields = _ScalarResult._fields | {"point", "scope", "ambient_dimension", "jacobian_rank"}

class _LocalDimension(_AtPoint, _ScalarResult):
    _fields = _ScalarResult._fields | {"point", "scope", "local_dimension"}

    @property
    def local_dimension(self):
        return self._record["result"]["local_dimension"]


class _LocalSmoothness(_LocalDimension):
    _fields = _LocalDimension._fields | {"embedding_dimension", "jacobian_rank"}

    @property
    def embedding_dimension(self):
        return self._record["result"]["embedding_dimension"]

    @property
    def jacobian_rank(self):
        return self._record["result"]["jacobian_rank"]


class _HilbertSamuelFunction(_PointResult):
    _fields = _PointResult._fields | {"degrees", "values", "graded_values", "dimension", "multiplicity"}

    @property
    def degrees(self):
        return list(self._record["result"]["degrees"])

    @property
    def values(self):
        return list(self._record["result"]["values"])

    @property
    def graded_values(self):
        return list(self._record["result"]["graded_values"])

    @property
    def dimension(self):
        return self._record["result"]["dimension"]

    @property
    def multiplicity(self):
        return self._record["result"]["multiplicity"]

    def __repr__(self):
        return (f"Hilbert–Samuel function ({self.scope}, dimension={self.dimension}, "
                f"multiplicity={integer_text(self.multiplicity)}, values={result_json(self.values)})")


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


class _LocalQuotientBasis(_AtPoint, _QuotientBasis):
    _fields = _QuotientBasis._fields | {"point"}


@dataclass(frozen=True, eq=False, repr=False)
class _JacobianAlgebra(_LocalQuotientBasis):
    generators: tuple[Polynomial, ...]
    _fields = _LocalQuotientBasis._fields | {"generators"}


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


def _typed(record, space, *, quotient_parent=None):
    result = record["result"]
    kind = result["type"]
    names = result["ring"]["variables"]
    # Tangent equations live in fresh displacement coordinates, even at the origin.
    space = (_CoordinateSpace.named(names) if kind in ("tangent-cone", "zariski-tangent-space")
             else _result_space(space, names))
    order = result["ring"]["order"]

    def decode(text, coordinate_space=space):
        # Kernel output is already canonical. Decode its exact coefficients as
        # arithmetic results, which can exceed the session's input-literal limit.
        if not coordinate_space.generators:
            return native.coerce(sparse_row(text).get((), 0))
        row, = affine_request(["decode", " ".join(coordinate_space.names), text])
        return native.from_row(coordinate_space.generators, row)

    def polynomials(expressions, coordinate_space=space):
        return tuple(Polynomial(coordinate_space, decode(text, coordinate_space)) for text in expressions)

    if kind in ("quotient-ring", "quotient-element"):
        from .session_quotient import QuotientElement, QuotientRing, quotient_ring
        if kind == "quotient-ring":
            relations = Ideal(space, polynomials(result["relations"]), order)
            return QuotientRing(space, order, record, relations)
        if quotient_parent is None:
            # Replayed one-shot operations also provide a usable parent, with
            # its own declaration record and the replay's fresh coordinates.
            relations = Ideal(space, polynomials(result["relations"]), order)
            quotient_parent = quotient_ring(relations, compute=record["execution"]["requested"])
        representative = _OrderedPolynomial(space, decode(result["representative"]), order)
        return QuotientElement(quotient_parent, representative, record)
    if kind == "localization":
        element = result.get("element")
        if element is not None:
            numerator, denominator = polynomials((element["numerator"], element["denominator"]))
            element = _LocalElement(numerator, denominator, element["member"], element["unit"])
        return _Localization(space, order, record, Ideal(space, polynomials(result["prime"]), order),
                             Ideal(space, polynomials(result["generators"]), order), element)
    if kind == "normalization":
        components = []
        for component in result["components"]:
            target_names = component["ring"]["variables"]
            target = _CoordinateSpace.named(target_names) if target_names else _scalar_space
            target_order = component["ring"]["order"]
            fractions = tuple(_NormalizationFraction(*polynomials((f["numerator"], f["denominator"])))
                              for f in component["fractions"])
            components.append(_NormalizedComponent(
                target, target_order, Ideal(space, polynomials(component["prime"]), order),
                Ideal(target, polynomials(component["relations"], target), target_order),
                polynomials(component["images"], target), fractions))
        return _Normalization(space, order, record, tuple(components))
    if kind in ("basis", "ideal"):
        cls = _Basis if kind == "basis" else _IdealResult
        return cls(space, polynomials(result["generators"]), order, record)
    if kind == "primary-decomposition":
        components = tuple(_PrimaryComponent(Ideal(space, polynomials(component["primary"]), order),
                                             Ideal(space, polynomials(component["prime"]), order))
                           for component in result["components"])
        return _PrimaryDecomposition(space, order, record, components)
    if kind == "local-standard-basis":
        return _LocalStandardBasis(space, order, record, polynomials(result["generators"]))
    if kind == "tangent-cone":
        return _TangentCone(space, order, record, polynomials(result["generators"]))
    if kind == "zariski-tangent-space":
        directions = tuple(tuple(map(coordinate_fraction, vector)) for vector in result["basis"])
        return _ZariskiTangentSpace(space, order, record, polynomials(result["equations"]), directions)
    if kind == "tangent-cone-hilbert-series":
        return _TangentConeHilbertSeries(space, order, record)
    if kind == "local-normal-form":
        remainder, unit = polynomials((result["remainder"], result["unit"]))
        return _LocalNormalForm(space, order, record, remainder, unit)
    if kind == "hilbert-series":
        return _HilbertSeries(space, order, record)
    if kind == "hilbert-samuel-function":
        return _HilbertSamuelFunction(space, order, record)
    if kind in ("quotient-basis", "local-jacobian-algebra"):
        basis = None if result["basis"] is None else polynomials(result["basis"])
        if kind == "local-jacobian-algebra":
            return _JacobianAlgebra(space, order, record, basis, polynomials(result["generators"]))
        cls = _QuotientBasis if result["scope"] == "global" else _LocalQuotientBasis
        return cls(space, order, record, basis)
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
        if record["operation"] == "ae-codimension":
            return _AeCodimension(space, order, result["value"], record)
        if record["operation"] == "hilbert-samuel-multiplicity":
            return _HilbertSamuelMultiplicity(space, order, result["value"], record)
        if record["operation"] == "embedding-dimension":
            return _EmbeddingDimension(space, order, result["value"], record)
        if record["operation"] == "local-dimension":
            return _LocalDimension(space, order, result["value"], record)
        if record["operation"] == "local-smoothness":
            return _LocalSmoothness(space, order, result["value"], record)
        cls = _ScalarResult
        if "finite" in result:
            cls = _QuotientNumber if result["scope"] == "global" else _LocalNumber
        return cls(space, order, result["value"], record)
    raise ValueError(f"Unknown algebra result type {kind!r}.")


def _evaluate(request, *, known_basis=False, typed_map=False):
    from .algebra import calculate_algebra

    status, record = calculate_algebra(request, known_basis=known_basis, _typed_map=typed_map)
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
    accepts_cone = operation in ("tangent-cone-hilbert-series", "zariski-tangent-space", "embedding-dimension",
                                 "local-dimension", "local-smoothness")
    local_basis = (isinstance(left, _LocalStandardBasis)
                   and (accepts_cone or operation in ("local-normal-form", "local-quotient-basis",
                                                      "hilbert-samuel-function", "hilbert-samuel-multiplicity",
                                                      "tangent-cone")))
    cone = isinstance(left, _TangentCone) and accepts_cone
    if not (local_basis or cone):
        left = _require_ideal(left)
    point = operands.pop("point", None)
    if point is not None:
        operands["point"] = _local_point(point, len(left.variables))
        if cone and any(coordinate != "0" for coordinate in operands["point"]):
            raise ValueError("The point of a saved tangent cone must be its vertex (all tangent coordinates zero).")
    if local_basis:
        saved_point = left.point
        if operands.setdefault("point", saved_point) != saved_point:
            raise ValueError("The supplied point differs from the saved local standard basis point.")
    for key in ("other_ideal", "prime"):
        if key not in operands:
            continue
        right = _require_ideal(operands[key])
        if right._space is not left._space:
            raise ValueError("These ideals belong to different coordinate spaces.")
        if right.order != left.order:
            raise ValueError("These ideals have different monomial orders.")
        operands[key] = [str(f) for f in right.generators]
    for key in ("polynomial", "numerator", "denominator"):
        if key in operands:
            operands[key] = str(_polynomial(operands[key], left._space))
    operands["ideal"] = [str(f) for f in left.generators]
    request = {"operation": operation, "ring": left.ring, "operands": operands, "compute": compute}
    known_basis = ((local_basis and operation == "local-normal-form") or
                   (isinstance(left, _Basis) and operation in ("groebner", "normal-form", "membership", "multiplicity")))
    return _typed(_evaluate(request, known_basis=known_basis), left._space)


def groebner(ideal, *, compute="auto"):
    return _calculate("groebner", ideal, compute)


def normal_form(polynomial, ideal, *, compute="auto"):
    return _calculate("normal-form", ideal, compute, polynomial=polynomial)


def _local_point(point, dimension):
    if not isinstance(point, (list, tuple)):
        raise TypeError("Pass the point as a list or tuple in coordinate order.")
    entries = [rational_text(value) if isinstance(value, Fraction) else value for value in point]
    return normalize_point(entries, dimension)


def local_standard_basis(ideal, *, point=None, compute="auto"):
    return _calculate("local-standard-basis", ideal, compute, point=point)


def local_normal_form(polynomial, ideal, *, point=None, compute="auto"):
    return _calculate("local-normal-form", ideal, compute, polynomial=polynomial, point=point)


def membership(polynomial, ideal, *, compute="auto"):
    return _calculate("membership", ideal, compute, polynomial=polynomial)


def _elimination_name(variable, space):
    if isinstance(variable, Polynomial):
        coordinate = _polynomial(variable, space)._value
        if not coordinate.is_Symbol:
            raise ValueError("Elimination requires a coordinate variable.")
        variable = coordinate.name
    if not isinstance(variable, str) or variable not in space.names:
        raise ValueError("Choose a declared coordinate variable to eliminate.")
    return variable


def _calculate_polynomials(operation, order, variables, compute, *, point=None, eliminate_variable=None, **operands):
    default_space = eliminate_variable._space if isinstance(eliminate_variable, Polynomial) else _scalar_space
    space, values, order = _polynomial_inputs(operands.values(), order, variables, default_space)
    inputs = {key: str(f) for key, f in zip(operands, values)}
    if operation == "resultant":
        inputs["variables"] = [_elimination_name(eliminate_variable, space)]
    if point is not None:
        inputs["point"] = _local_point(point, len(space.names))
    request = {"operation": operation, "ring": {"field": "QQ", "variables": list(space.names), "order": order},
               "operands": inputs, "compute": compute}
    return _typed(_evaluate(request), space)


def gcd(left, right, *, order=None, variables=None, compute="auto"):
    """Monic polynomial GCD in the common coordinate space, or QQ for scalars."""
    return _calculate_polynomials("gcd", order, variables, compute, polynomial=left, other_polynomial=right)


def resultant(left, right, variable, *, order=None, variables=None, compute="auto"):
    """Resultant with respect to one coordinate, in the ring of the surviving coordinates."""
    return _calculate_polynomials("resultant", order, variables, compute, polynomial=left, other_polynomial=right,
                                  eliminate_variable=variable)


def factor(polynomial, *, order=None, variables=None, compute="auto"):
    """Rational unit and monic irreducible factors in the polynomial's coordinate space."""
    return _calculate_polynomials("factor", order, variables, compute, polynomial=polynomial)


def local_jacobian_algebra(polynomial, *, point=None, order=None, variables=None, compute="auto"):
    return _calculate_polynomials("local-jacobian-algebra", order, variables, compute, polynomial=polynomial, point=point)


def milnor_number(polynomial, *, point=None, order=None, variables=None, compute="auto"):
    return _calculate_polynomials("milnor-number", order, variables, compute, polynomial=polynomial, point=point)


def ae_codimension(*components, order=None, variables=None, compute="auto"):
    """Extended left-right codimension of a map germ, in target coordinate order."""
    space, values, order = _polynomial_inputs(_unpack(components), order, variables, _scalar_space)
    request = {"operation": "ae-codimension", "ring": {"field": "QQ", "variables": list(space.names), "order": order},
               "operands": {"components": [str(f) for f in values]}, "compute": compute}
    # The source literals have already been parsed. Their exact products can
    # have more decimal digits than the caller permits in a new input literal.
    return _typed(_evaluate(request, typed_map=True), space)


def tjurina_number(polynomial, *, point=None, order=None, variables=None, compute="auto"):
    return _calculate_polynomials("tjurina-number", order, variables, compute, polynomial=polynomial, point=point)


def eliminate(ideal, *variables, compute="auto"):
    _require_ideal(ideal)
    names = [_elimination_name(v, ideal._space) for v in _unpack(variables)]
    return _calculate("eliminate", ideal, compute, variables=names)


def intersection(ideal, other_ideal, *, compute="auto"):
    return _calculate("intersection", ideal, compute, other_ideal=other_ideal)


def quotient(ideal, other_ideal, *, compute="auto"):
    return _calculate("quotient", ideal, compute, other_ideal=other_ideal)


def saturation(ideal, polynomial, *, compute="auto"):
    return _calculate("saturation", ideal, compute, polynomial=polynomial)


def ideal_power(ideal, n, *, compute="auto"):
    """Return the ordinary n-th power of this ideal in its original polynomial ring."""
    return _calculate("ideal-power", ideal, compute, exponent=n)


def symbolic_power(ideal, n, *, compute="auto"):
    """Return the n-th symbolic power of a prime or radical ideal over QQ."""
    return _calculate("symbolic-power", ideal, compute, exponent=n)


def radical(ideal, *, compute="auto"):
    """The radical ideal in the input polynomial ring over QQ."""
    return _calculate("radical", ideal, compute)


def primary_decomposition(ideal, *, compute="auto"):
    """Return the primary ideals and their associated primes in the input ring over QQ."""
    return _calculate("primary-decomposition", ideal, compute)


def localize(ideal, prime=None, *, numerator=None, denominator=None, compute="auto"):
    """Localize at a prime, optionally extending an ideal and testing a fraction."""
    if prime is None:
        prime = _require_ideal(ideal)
        ideal = Ideal(prime._space, (), prime.order)
    operands = {key: value for key, value in (("numerator", numerator), ("denominator", denominator))
                if value is not None}
    return _calculate("localization", ideal, compute, prime=prime, **operands)


def normalization(ideal, *, compute="auto"):
    """Present the normalization of the reduced QQ quotient, with its component maps."""
    return _calculate("normalization", ideal, compute)


def dimension(ideal, *, compute="auto"):
    return _calculate("dimension", ideal, compute)


def quotient_basis(ideal, *, compute="auto"):
    return _calculate("quotient-basis", ideal, compute)


def multiplicity(ideal, *, compute="auto"):
    """Total dimension over QQ of a finite-colength quotient by a global ideal."""
    return _calculate("multiplicity", ideal, compute)


def hilbert_series(ideal, *, compute="auto"):
    """Hilbert series of a quotient by a homogeneous ideal with the standard grading."""
    return _calculate("hilbert-series", ideal, compute)


def local_quotient_basis(ideal, *, point=None, compute="auto"):
    return _calculate("local-quotient-basis", ideal, compute, point=point)


_DEFAULT_DEGREES = object()


def hilbert_samuel_function(ideal, *, degrees=_DEFAULT_DEGREES, point=None, compute="auto"):
    operands = {}
    if degrees is not _DEFAULT_DEGREES:
        if not isinstance(degrees, (list, tuple)):
            raise TypeError("Pass degrees as a list or tuple of nonnegative integers.")
        operands["degrees"] = list(degrees)
    return _calculate("hilbert-samuel-function", ideal, compute, point=point, **operands)


def hilbert_samuel_multiplicity(ideal, *, point=None, compute="auto"):
    return _calculate("hilbert-samuel-multiplicity", ideal, compute, point=point)


def tangent_cone(ideal, *, point=None, compute="auto"):
    """Homogeneous tangent equations in a new ring of displacements at the point."""
    return _calculate("tangent-cone", ideal, compute, point=point)


def tangent_cone_hilbert_series(ideal, *, point=None, compute="auto"):
    """Graded Hilbert series at the chosen point, or at the vertex of a saved cone."""
    return _calculate("tangent-cone-hilbert-series", ideal, compute, point=point)


def zariski_tangent_space(ideal, *, point=None, compute="auto"):
    """Tangent equations in fresh displacement coordinates and a basis of rational directions."""
    return _calculate("zariski-tangent-space", ideal, compute, point=point)


def embedding_dimension(ideal, *, point=None, compute="auto"):
    """Dimension of the Zariski tangent space at the chosen point."""
    return _calculate("embedding-dimension", ideal, compute, point=point)


def local_dimension(ideal, *, point=None, compute="auto"):
    """Krull dimension of the local ring at a rational point of the scheme."""
    return _calculate("local-dimension", ideal, compute, point=point)


def local_smoothness(ideal, *, point=None, compute="auto"):
    """Return whether the scheme is smooth at the point, with its local and embedding dimensions."""
    return _calculate("local-smoothness", ideal, compute, point=point)


def algebra_bindings():
    return {"polynomial": polynomial, "ideal": ideal, "groebner": groebner, "gcd": gcd, "factor": factor,
            "resultant": resultant,
            "normal_form": normal_form, "membership": membership,
            "local_standard_basis": local_standard_basis, "local_normal_form": local_normal_form,
            "eliminate": eliminate, "intersection": intersection, "quotient": quotient,
            "saturation": saturation, "saturate": saturation, "radical": radical, "dimension": dimension,
            "ideal_power": ideal_power, "symbolic_power": symbolic_power,
            "primary_decomposition": primary_decomposition, "normalization": normalization,
            "localize": localize, "localization": localize,
            "quotient_basis": quotient_basis, "local_quotient_basis": local_quotient_basis,
            "multiplicity": multiplicity, "hilbert_series": hilbert_series,
            "hilbert_samuel_function": hilbert_samuel_function,
            "hilbert_samuel_multiplicity": hilbert_samuel_multiplicity,
            "tangent_cone": tangent_cone, "tangent_cone_hilbert_series": tangent_cone_hilbert_series,
            "zariski_tangent_space": zariski_tangent_space, "embedding_dimension": embedding_dimension,
            "local_dimension": local_dimension, "local_smoothness": local_smoothness,
            "local_jacobian_algebra": local_jacobian_algebra, "milnor_number": milnor_number,
            "ae_codimension": ae_codimension,
            "tjurina_number": tjurina_number, "algebra_request": algebra_request}
