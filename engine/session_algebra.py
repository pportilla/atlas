"""Exact ideal arithmetic shared by the Atlas prompt and scripts."""

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import json
from time import perf_counter_ns
from typing import TYPE_CHECKING, NamedTuple
from weakref import WeakValueDictionary

from . import native_polynomial as native
from .algebra import QUOTIENT_IDEAL_OPERATIONS
from .exact_json import result_json
from .native_library import affine_request, current_replay_scope
from .polynomial import LimitError, format_poly, integer_text, parse_expression, rational_text, sparse_row
from .rational_point import coordinate_fraction, normalize_point
from .series_display import format_hilbert_series
from .session_polynomial import QQ, Polynomial, _Coordinates, _CoordinateSpace
from .session_ideal_native import ideal_constructor, values as ideal_values

if TYPE_CHECKING:
    from .session_maps import RingHomomorphism


class _RingValue:
    _fields = {"variables", "coordinates", "coeffs", "order", "ring", "ambient"}

    @property
    def ambient(self):
        return self._space

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


_unprepared = object()


@dataclass(frozen=True, eq=False, repr=False)
class Ideal(_RingValue):
    _space: _CoordinateSpace
    generators: tuple[Polynomial, ...]
    order: str
    _fields = _RingValue._fields | {"generators", "parent", "quotient_ring"}
    _label = "ideal"

    def __init__(self, _space, generators, order, prepared=_unprepared):
        state = self.__dict__
        state["_space"] = _space
        state["generators"] = generators
        state["order"] = order
        if prepared is not _unprepared:
            state["_native_ideal"] = prepared

    @property
    def parent(self):
        return self._space

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # An immutable ideal owns its quotient parent even inside copied records.
        return self

    @cached_property
    def quotient_ring(self):
        """The default quotient parent for this particular ideal."""
        from .session_quotient import quotient_ring
        return quotient_ring(self)

    @cached_property
    def _native_ideal(self):
        return _assemble_ideal(self._space, self.generators)

    def __add__(self, other):
        return _binary_ideals("ideal-sum", self, other, "auto", infix=True)

    def __mul__(self, other):
        return _binary_ideals("ideal-product", self, other, "auto", infix=True)

    def __pow__(self, exponent):
        return ideal_power(self, exponent)

    def _generator_text(self, names):
        return map(str, self.generators)

    def __repr__(self):
        names = self.variables
        ring = f"QQ[{', '.join(names)}]" if names else "QQ"
        generators = ", ".join(self._generator_text(names)) or "0"
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


class _NativeIdealResult(_IdealResult):
    """A computed ideal. Convert its native generators to session values only when requested."""

    status = "ok"
    type = "ideal"

    def __init__(self, parent, order, handle, operation, left, right, compute, elapsed):
        # Capture independent native operand handles for replay. Keeping the
        # session results here would retain the entire history of a product.
        state = self.__dict__
        state["_space"] = parent
        state["order"] = order
        state["_native_ideal"] = handle
        state["_history"] = operation, left, right, compute, elapsed

    @property
    def operation(self):
        return self._history[0]

    @cached_property
    def generators(self):
        return tuple(Polynomial(self._space, value) for value in ideal_values(self._native_ideal, self._space.generators))

    def _generator_text(self, names):
        # Polynomial.__str__ prints factors in name order. Native expressions
        # already follow this convention when the ring has that order.
        if tuple(sorted(names)) == names:
            return self._native_ideal.expressions(names)
        return super()._generator_text(names)

    @property
    def result(self):
        ring = self.ring
        return {"type": "ideal", "ring": ring, "generators": self._native_ideal.expressions(ring["variables"])}

    @property
    def execution(self):
        operation, left, _, compute, elapsed = self._history
        stage = {"operation": operation.replace("-", "_"), "backend": "cpu",
                 "generators": left.size(), "elapsed_ns": elapsed}
        return {"requested": compute, "backend": "cpu", "device": "CPU", "stages": [stage],
                "fallback_reasons": [], "elapsed_ms": round(elapsed / 1e6, 3)}

    @cached_property
    def _record(self):
        from .algebra_replay import make_replay

        operation, left, right, compute, _ = self._history
        coordinates = self._space.generators
        prepared = {"ideal": ideal_values(left, coordinates)}
        operands = {"ideal": list(map(str, prepared["ideal"]))}
        if operation == "ideal-power":
            operands["exponent"] = prepared["exponent"] = str(right)
        else:
            prepared["other_ideal"] = ideal_values(right, coordinates)
            operands["other_ideal"] = list(map(str, prepared["other_ideal"]))
        result = self.result
        request = {"operation": operation, "ring": result["ring"], "operands": operands, "compute": compute}
        return {"status": "ok", "operation": operation, "result": result,
                "execution": self.execution, "replay": make_replay(request, prepared, generator_text=operands)}


class _Basis(_IdealResult):
    """Only the dispatcher can return a verified, reduced basis to the session."""

    _label = "Groebner basis"


@dataclass(frozen=True, eq=False, repr=False)
class _EliminatedIdeal(_IdealResult):
    inclusion: "RingHomomorphism"
    _fields = _IdealResult._fields | {"inclusion"}


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
    _fields = _RingValue._fields | {"prime", "relations", "images", "fractions", "map"}

    @cached_property
    def map(self):
        from .session_maps import hom
        domain = self.prime.quotient_ring
        codomain = self.relations.quotient_ring
        return hom(domain, codomain, images=tuple(codomain(f) for f in self.images))


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


class _NativeGCDResult(_PolynomialResult):
    """A computed GCD with its native value and the operands needed for replay."""

    status = "ok"
    type = "polynomial"
    operation = "gcd"

    def __init__(self, parent, order, handle, left, right, compute, elapsed):
        self.__dict__.update(_space=parent, order=order, _native_polynomial=handle,
                             _operands=(left, right), _compute=compute, _elapsed=elapsed)

    def __bool__(self):
        return not self._space._native_ring.polynomial_is_zero(self._native_polynomial)

    def _expression(self, handle):
        row = self._space._native_ring.polynomial_row(handle)
        return native.from_row(self._space.generators, row.decode("ascii"))

    @cached_property
    def _value(self):
        return self._expression(self._native_polynomial)

    @property
    def result(self):
        text = format_poly(native.terms(self._space.generators, self._value), self.variables)
        return {"type": self.type, "ring": self.ring, "polynomial": text}

    @property
    def execution(self):
        stage = {"operation": "polynomial_gcd", "backend": "cpu", "algorithm": "flint",
                 "order": self.order, "elapsed_ns": self._elapsed}
        return {"requested": self._compute, "backend": "cpu", "device": "CPU", "stages": [stage],
                "fallback_reasons": [], "elapsed_ms": round(self._elapsed / 1e6, 3)}

    @cached_property
    def _record(self):
        from .algebra_replay import make_replay

        prepared = {key: self._expression(handle) for key, handle in
                    zip(("polynomial", "other_polynomial"), self._operands)}
        request = {"operation": self.operation, "ring": self.ring, "compute": self._compute,
                   "operands": {key: str(poly) for key, poly in prepared.items()}}
        return {"status": self.status, "operation": self.operation, "result": self.result,
                "execution": self.execution, "replay": make_replay(request, prepared)}


@dataclass(frozen=True, eq=False, repr=False)
class _Resultant(_PolynomialResult):
    inclusion: "RingHomomorphism"
    _fields = _PolynomialResult._fields | {"inclusion"}


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


class _QuotientMembership(_ScalarResult):
    _fields = _ScalarResult._fields | {"parent"}

    @property
    def parent(self):
        return self._space

    @property
    def variables(self):
        return self.parent.variables

    @property
    def ring(self):
        return self.parent.ring


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
    _fields = _LocalNumber._fields | {"method", "lower_bound", "jet_degree", "contact_order", "last_weight", "reason"}

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
    def last_weight(self):
        return self._record["result"].get("last_weight")

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
        if value._space is QQ and space is not QQ:
            return Polynomial(space, value._value)
        if value._space is not space:
            raise ValueError("These operands belong to different coordinate spaces.")
        return value
    if type(value) is int or isinstance(value, Fraction):
        return Polynomial(space, native.coerce(value))
    from .session_quotient import QuotientElement
    if isinstance(value, QuotientElement):
        raise TypeError("This operation needs an ambient polynomial. Use the class's .representative "
                        "when you intend to work in its ambient ring.")
    raise TypeError("Expected a polynomial or an exact rational number.")


def _unpack(values):
    if len(values) == 1 and isinstance(values[0], (tuple, list)):
        return tuple(values[0])
    return values


# Keep repeated eliminations from the same declaration in the same surviving
# space while users hold a result. Independent coordinate declarations stay distinct.
_subspaces = WeakValueDictionary()


def _result_space(space, names):
    if not names:
        return space if not space.generators else QQ
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
    from .session_quotient import QuotientElement
    if isinstance(variables, _Coordinates):
        return variables.parent
    variables = _unpack((variables,))
    if not variables:
        return QQ
    if not isinstance(variables[0], (Polynomial, QuotientElement)):
        raise TypeError("Pass coordinate generators, for example variables=C.coordinates.")
    parent = variables[0].parent
    if len(variables) != len(parent.variables) or any(
        actual is not coordinate for actual, coordinate in zip(variables, parent.coordinates)
    ):
        raise ValueError("Use all coordinates in their declared order from one coordinate space, or supply its ring directly.")
    return parent


def _declared_space(ring, variables):
    from .session_quotient import QuotientRing
    if (ring is None) == (variables is None):
        raise ValueError("Supply exactly one of ring=f.parent or variables=(x, y).")
    if ring is not None and not isinstance(ring, (_CoordinateSpace, QuotientRing)):
        raise TypeError("Pass a live parent ring, for example ring=f.parent.")
    return ring if ring is not None else _coordinate_space(variables)


def polynomial(source, *, ring=None, variables=None):
    """Evaluate a polynomial expression in the specified parent, reducing it in a quotient ring."""
    from .session_pullback import chart_source
    from .session_quotient import QuotientRing
    from .worker_process import run_worker

    if not isinstance(source, str) or not source.strip():
        raise ValueError("polynomial expects a nonempty expression string.")
    space = _declared_space(ring, variables)
    if isinstance(space, QuotientRing):
        return space(polynomial(source, ring=space.ambient))
    generators = space.polynomial_generators
    source = chart_source(source, space)
    names = [g.name for g in generators]
    modulus = str(space.coeffs.minimal_polynomial) if space.coeffs is not QQ and "/" in source else None
    # Short, flat sums of monomials cannot expand products of sums. The
    # bounded parser also caps powers and intermediate coefficient sizes.
    if modulus is None and len(source) <= 128 and "(" not in source and ")" not in source:
        try:
            row = parse_expression(source, names, bounded=True, as_row=True)
        except LimitError:
            pass  # A quick-parser limit selects the unrestricted worker.
        else:
            return Polynomial(space, native.from_kernel_row(generators, row))

    request = {"source": source, "variables": names}
    if modulus is not None:
        request["coefficient_modulus"] = modulus
    status, record = run_worker(["engine.algebra_worker", "--polynomial"], request, None)
    if status != 200:
        raise ValueError(record["error"]["message"])
    return Polynomial(space, native.from_kernel_row(generators, record["terms"]))


def _polynomial_inputs(values, order, variables, default_space=None, *, ring=None):
    from .session_quotient import QuotientElement, QuotientRing
    if variables is None and ring is None:
        space = default_space
        for f in values:
            if isinstance(f, Polynomial):
                if f._space is not QQ:
                    space = f._space
                    break
                if space is None:
                    space = QQ
        if space is None:
            if any(isinstance(f, QuotientElement) for f in values):
                raise TypeError("Ideal generators must lie in an ambient polynomial ring. "
                                "Use .representative to select an ambient polynomial explicitly.")
            raise ValueError("For a constant or empty ideal, supply ring=QQ, ring=f.parent, or variables=(x, y).")
    else:
        space = _declared_space(ring, variables)
    if isinstance(space, QuotientRing):
        raise TypeError("This operation requires a polynomial ring. Use K.ambient for the ambient ring of a quotient K.")
    if space.coeffs is not QQ:
        raise ValueError("Algebra operations require polynomial coordinates over QQ.")
    if space.has_relations:
        raise ValueError("Algebra operations require polynomial coordinates with no quotient-ring relations.")
    # The common declaration already holds immutable polynomials in their exact
    # live parent. The general path also handles scalar embeddings and order-bearing results.
    if isinstance(values, tuple) and all(type(f) is Polynomial and f._space is space for f in values):
        polynomials = values
        if order is None:
            order = "grevlex"
    else:
        polynomials = tuple(_polynomial(f, space) for f in values)
        if order is None:
            orders = {f.order for f in polynomials if isinstance(f, _OrderedPolynomial)}
            if len(orders) > 1:
                raise ValueError("Choose an explicit monomial order for these polynomials.")
            order = next(iter(orders), "grevlex")
    if order not in ("lex", "grevlex"):
        raise ValueError("Unsupported monomial order. Choose lex or grevlex.")
    return space, polynomials, order


def _assemble_ideal(space, generators):
    ring = space._native_ring
    if ring is None:
        return None
    return ring.ideal(tuple(f._native_polynomial for f in generators))


def ideal(*generators, order=None, variables=None, ring=None):
    """Declare an ideal in a polynomial or quotient ring, using ring= to select its parent explicitly."""
    from .session_quotient import QuotientElement, QuotientIdeal, QuotientRing
    generators = _unpack(generators)
    explicit_parent = variables is not None or ring is not None
    if explicit_parent:
        ring = _declared_space(ring, variables)
    else:
        ring = next((f.parent for f in generators if isinstance(f, QuotientElement)), None)
    if isinstance(ring, QuotientRing):
        if order is not None and order != ring.order:
            raise ValueError(f"This quotient ring uses the monomial order {ring.order!r}.")
        return QuotientIdeal.declare(ring, generators, project=explicit_parent)
    space, generators, order = _polynomial_inputs(generators, order, None, ring=ring)
    return Ideal(space, generators, order, _assemble_ideal(space, generators))


# Native construction handles polynomial generators that share a parent, and
# falls back to this Python declaration for every other input.
declare_ideal = ideal
ideal = ideal_constructor(declare_ideal, Polynomial, Ideal)


def _decode_algebra_polynomial(text, space):
    # Native arithmetic output can exceed the session's input-literal limits.
    if not space.generators:
        return native.coerce(sparse_row(text).get((), 0))
    row, = affine_request(["decode", " ".join(space.names), text])
    return native.from_kernel_row(space.generators, row)


def _typed(record, space, *, quotient_parent=None, defining_ideal=None, replay_generators=None):
    result = record["result"]
    kind = result["type"]
    names = result["ring"]["variables"]
    original_space = space
    # Tangent equations live in fresh displacement coordinates, even at the origin.
    space = (_CoordinateSpace.named(names) if kind in ("tangent-cone", "zariski-tangent-space")
             else _result_space(space, names))
    order = result["ring"]["order"]

    def decode(text, coordinate_space=space):
        return _decode_algebra_polynomial(text, coordinate_space)

    def polynomials(expressions, coordinate_space=space):
        return tuple(Polynomial(coordinate_space, decode(text, coordinate_space)) for text in expressions)

    def inclusion():
        from .session_maps import hom
        # The surviving generators retain their identities in the original ring.
        return hom(space, original_space, images=[Polynomial(original_space, g) for g in space.generators])

    if "relations" in result["ring"] and record["operation"] != "quotient-ring":
        from .session_quotient import QuotientIdeal
        if quotient_parent is None:
            # Public replay creates a fresh owner. Preserve the caller's defining
            # equations even when the native operation normalized those equations.
            ambient = {key: value for key, value in result["ring"].items() if key != "relations"}
            normalized = result["ring"]["relations"]
            equations = normalized if replay_generators is None else replay_generators
            declaration = dict(operation="quotient-ring", ring=ambient, compute=record["execution"]["requested"],
                               operands=dict(action="declare", ideal=equations))
            receipt = _evaluate(declaration, known_basis=equations == normalized)
            quotient_parent = _typed(receipt, space, replay_generators=equations)
        if kind == "quotient-ideal":
            return QuotientIdeal(quotient_parent, order, record)
        if kind == "boolean":
            return _QuotientMembership(quotient_parent, order, result["value"], record)
    if kind in ("quotient-ring", "quotient-element"):
        from .session_quotient import QuotientElement, quotient_parent_from_record
        if quotient_parent is None:
            quotient_parent = quotient_parent_from_record(
                record, space, polynomials(result["relations"]), defining_ideal, replay_generators)
        if kind == "quotient-ring":
            return quotient_parent
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
            target = _CoordinateSpace.named(target_names)
            target_order = component["ring"]["order"]
            fractions = tuple(_NormalizationFraction(*polynomials((f["numerator"], f["denominator"])))
                              for f in component["fractions"])
            components.append(_NormalizedComponent(
                target, target_order, Ideal(space, polynomials(component["prime"]), order),
                Ideal(target, polynomials(component["relations"], target), target_order),
                polynomials(component["images"], target), fractions))
        return _Normalization(space, order, record, tuple(components))
    if record["operation"] == "eliminate":
        return _EliminatedIdeal(space, polynomials(result["generators"]), order, record, inclusion())
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
    if record["operation"] == "resultant":
        return _Resultant(space, decode(result["polynomial"]), order, record, inclusion())
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


def _evaluate_live_quotient_ideal(request):
    """Evaluate already-serialized live operands; the kernel may decline in-process work."""
    from .algebra import dispatch

    if request["compute"] in ("cpu", "auto") and current_replay_scope() is None:
        ring, operands = request["ring"], request["operands"]
        groups = (ring["variables"], ring["relations"], operands["ideal"], operands.get("other_ideal", ()),
                  (operands.get("polynomial", ""),))
        exponent = operands.get("exponent", 0)
        short_exponent = exponent.bit_length() <= 64 if type(exponent) is int else len(exponent) <= 32
        # Bound parsing before the native work gate. Every polynomial here is
        # already expanded by Atlas; a public request can contain arbitrary expressions.
        if short_exponent and all(len(group) <= 16 and all(len(row) <= 256 for row in group) for group in groups):
            result = dispatch(request, known_basis=True, try_bounded_quotient=True)
            if result is not None:
                return result
    return _evaluate(request, known_basis=True)


def algebra_request(source):
    """Replay an algebra JSON request in a fresh coordinate space."""
    if not isinstance(source, str):
        raise TypeError("algebra_request expects a JSON request string.")
    request = json.loads(source)
    record = _evaluate(request)
    names = request["ring"]["variables"]
    space = _CoordinateSpace.named(names)
    if record["operation"] == "quotient-ring" and "relations" in request["ring"]:
        from .session_quotient import replay_further_quotient
        return replay_further_quotient(request, record, space)
    generators = (request["operands"]["ideal"] if record["operation"] == "quotient-ring"
                  else request["ring"].get("relations"))
    return _typed(record, space, replay_generators=generators)


def _require_ideal(value, *, allow_quotient=False):
    if not isinstance(value, Ideal):
        raise TypeError("Pass an ideal or a Gröbner basis.")
    if not allow_quotient and not isinstance(value.parent, _CoordinateSpace):
        raise ValueError("This operation currently requires an ideal in a polynomial ring.")
    return value


def _calculate(operation, left, compute, **operands):
    from .session_quotient import QuotientIdeal
    if isinstance(left, QuotientIdeal) and operation in QUOTIENT_IDEAL_OPERATIONS:
        parent = left.parent
        if "other_ideal" in operands:
            # _binary_ideals checked both live parents before serializing.
            operands["other_ideal"] = operands["other_ideal"]._record["result"]["generators"]
        if "polynomial" in operands:
            operands["polynomial"] = str(parent._lift(operands["polynomial"]))
        operands["ideal"] = left._record["result"]["generators"]
        request = dict(operation=operation, ring=parent.ring, operands=operands, compute=compute)
        return _typed(_evaluate_live_quotient_ideal(request), parent.ambient, quotient_parent=parent)
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
            raise ValueError("These ideals belong to different parent rings (different coordinate spaces). "
                             "Map their generators into the same ring first.")
        if right.order != left.order and operation not in ("ideal-sum", "ideal-product"):
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
    entries = []
    for value in point:
        if isinstance(value, Polynomial):
            if value.parent is not QQ:
                raise ValueError("A polynomial used as a point coordinate must have parent QQ.")
            value = value._value.scalar
        entries.append(rational_text(value) if isinstance(value, Fraction) else value)
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
    default_space = eliminate_variable._space if isinstance(eliminate_variable, Polynomial) else QQ
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
    if compute in ("auto", "cpu") and current_replay_scope() is None:
        # Ordinary polynomials already carry their complete parent, and
        # native handles exist only for polynomial rings over QQ.
        if (variables is None and type(left) is Polynomial and type(right) is Polynomial
                and left._space is right._space and order in (None, "grevlex", "lex")):
            space = left._space
            order = order or "grevlex"
        else:
            space, (left, right), order = _polynomial_inputs((left, right), order, variables, QQ)
        calculate = getattr(space._native_ring, "polynomial_gcd", None)
        if calculate is not None:
            a, b = left._native_polynomial, right._native_polynomial
            started = perf_counter_ns()
            handle = calculate(a, b, order == "lex")
            if handle is not None:
                return _NativeGCDResult(space, order, handle, a, b, compute, perf_counter_ns() - started)
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
    """Compute the extended left-right codimension of a map germ at the origin.

    Pass the components in target order as separate arguments or as one list.
    They must belong to the same source coordinate space over QQ and vanish
    at the origin. The result measures deformations modulo infinitesimal changes
    of source and target coordinates, including constant vector fields.

    When r.finite is True, r.value is the exact codimension. False certifies
    infinity. None means the calculation was inconclusive. In that case,
    r.lower_bound and r.reason give the partial result. For a weighted
    calculation, r.last_weight gives the last weight checked.

    >>> t = coordinates("t", coeffs=QQ)
    >>> r = ae_codimension(t^2, t^3)
    >>> print(r.finite, r.value)  # True 1

    Docs: examples/local-algebra.md#compare-the-cusp-and-its-parametrization
    """
    space, values, order = _polynomial_inputs(_unpack(components), order, variables, QQ)
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


def _binary_ideals(operation, left, right, compute, *, infix=False):
    if not (isinstance(left, Ideal) and isinstance(right, Ideal)):
        if infix:
            name = "addition" if operation == "ideal-sum" else "multiplication"
            raise TypeError(f"Ideal {name} requires two ideals in the same parent ring.")
        raise TypeError("Pass an ideal or a Gröbner basis.")
    if left._space is not right._space:
        raise ValueError("These ideals belong to different parent rings (different coordinate spaces). "
                         "Map their generators into the same ring first.")
    if compute in ("auto", "cpu") and current_replay_scope() is None:
        started = perf_counter_ns()
        a, b = left._native_ideal, right._native_ideal
        if a is not None and b is not None:
            result = a.sum(b) if operation == "ideal-sum" else a.product(b)
            if result is not None:
                return _NativeIdealResult(left._space, left.order, result, operation, a, b, compute,
                                         perf_counter_ns() - started)
    return _calculate(operation, left, compute, other_ideal=right)


def ideal_sum(left, right, *, compute="auto"):
    """Return the ideal generated by both sets of generators, using the left ideal's monomial order.

    >>> x, y = coordinates("x", "y")
    >>> I, J = ideal(x^2, y), ideal(x, y^2)
    >>> S = ideal_sum(I, J)
    >>> membership(x, S).value  # True

    Docs: docs/sessions.md#add-multiply-and-raise-ideals-to-powers
    """
    return _binary_ideals("ideal-sum", left, right, compute)


def ideal_product(left, right, *, compute="auto"):
    """Return the ideal generated by all pairwise products, using the left ideal's monomial order.

    >>> x, y = coordinates("x", "y")
    >>> I, J = ideal(x^2, y), ideal(x, y^2)
    >>> P = ideal_product(I, J)
    >>> membership(x*y, P).value  # True

    Docs: docs/sessions.md#add-multiply-and-raise-ideals-to-powers
    """
    return _binary_ideals("ideal-product", left, right, compute)


def quotient(ideal, other_ideal, *, compute="auto"):
    return _calculate("quotient", ideal, compute, other_ideal=other_ideal)


def saturation(ideal, polynomial, *, compute="auto"):
    return _calculate("saturation", ideal, compute, polynomial=polynomial)


def ideal_power(ideal, n, *, compute="auto"):
    """Return the ordinary n-th power of this ideal in its parent ring."""
    if isinstance(n, Fraction) and n.denominator == 1:
        n = n.numerator
    if type(n) is not int and not isinstance(n, str):
        raise TypeError("The exponent must be a nonnegative integer or an unsigned decimal string.")
    if type(n) is int and 0 <= n <= 4 and compute in ("auto", "cpu") and current_replay_scope() is None:
        started = perf_counter_ns()
        source = _require_ideal(ideal, allow_quotient=True)._native_ideal
        if source is not None:
            result = source.power(n, ideal.order == "lex")
            if result is not None:
                return _NativeIdealResult(ideal._space, ideal.order, result, "ideal-power", source, n, compute,
                                         perf_counter_ns() - started)
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
            "ideal_sum": ideal_sum, "ideal_product": ideal_product, "ideal_power": ideal_power,
            "symbolic_power": symbolic_power,
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
