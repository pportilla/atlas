"""Declared quotient rings and their residue classes in an Atlas session."""

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

from .session_algebra import (Ideal, _IdealResult, _OrderedPolynomial, _Result, _decode_algebra_polynomial, _evaluate,
                              _evaluate_live_quotient_ideal, _polynomial, _require_ideal, _typed, polynomial)
from .session_polynomial import QQ, Polynomial, _Coordinates, _CoordinateSpace


@dataclass(frozen=True, eq=False, repr=False)
class QuotientRing(_Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    defining_ideal: Ideal
    relations: tuple[Polynomial, ...]
    _fields = _Result._fields | {"defining_ideal", "relations", "gens", "generators", "apply", "element",
                                "projection", "declaration_executions"}

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Copying a class or map preserves its live parent.
        return self

    @property
    def declaration_executions(self):
        """Most recent successful declaration in each requested compute mode."""
        first = self._record["execution"]
        return deepcopy({first["requested"]: first, **self.__dict__.get("_later_declarations", {})})

    def _remember_declaration(self, execution):
        self.__dict__.setdefault("_later_declarations", {})[execution["requested"]] = execution

    @cached_property
    def gens(self):
        images = _Coordinates(self, (self(f) for f in self.ambient.coordinates))
        # Concurrent first projections must expose the same coordinate objects.
        return self.__dict__.setdefault("gens", images)

    @cached_property
    def projection(self):
        from .session_maps import hom
        return hom(self.defining_ideal.parent, self, images=self.gens)

    @property
    def generators(self):
        return self.gens

    @property
    def coordinates(self):
        """The images of the ambient coordinates in this quotient, in declaration order."""
        return self.gens

    @property
    def ring(self):
        return dict(super().ring, relations=list(self._record["result"]["relations"]))

    def __repr__(self):
        ambient = f"QQ[{', '.join(self.variables)}]" if self.variables else "QQ"
        return f"{ambient}/({', '.join(map(str, self.relations)) or '0'}) ({self.order})"

    def _lift(self, value, *, project=False):
        if isinstance(value, QuotientElement):
            source = self
            while isinstance(source, QuotientRing):
                if value.parent is source:
                    return value.representative
                if not project:
                    break
                source = source.defining_ideal.parent
            raise ValueError("These elements belong to different quotient rings.")
        return _polynomial(value, self._space)

    def apply(self, action, *values, exponent=None, compute=None):
        """Compute a quotient operation with a structured, replayable result."""
        if len(values) > 2:
            raise TypeError("A quotient-ring operation accepts at most two polynomial operands.")
        operands = dict(ideal=self._record["result"]["relations"], action=action)
        for name, value in zip(("polynomial", "other_polynomial"), values):
            operands[name] = str(self._lift(value))
        if exponent is not None:
            operands["exponent"] = exponent
        mode = self._record["execution"]["requested"] if compute is None else compute
        request = dict(operation="quotient-ring", ring=super().ring, operands=operands, compute=mode)
        # These rows came from this ring's native declaration. Replay of a public
        # request always computes a basis before constructing a ring object.
        record = _evaluate(request, known_basis=True)
        if action == "declare":
            self._remember_declaration(record["execution"])
            return self
        return _typed(record, self._space, quotient_parent=self)

    def element(self, polynomial):
        if isinstance(polynomial, str):
            raise TypeError("Use polynomial(source, ring=K) to read a string in the quotient K, "
                            "or K(f) to project an ambient polynomial f.")
        if isinstance(polynomial, QuotientElement):
            if polynomial.parent is self:
                return polynomial
            polynomial = self._lift(polynomial, project=True)
        return self.apply("reduce", polynomial)

    __call__ = element


@dataclass(frozen=True, eq=False, repr=False)
class QuotientElement(_Result):
    parent: QuotientRing
    representative: Polynomial
    _record: dict
    _fields = _Result._fields | {"parent", "representative", "inverse", "is_unit"}

    @property
    def _space(self):
        return self.parent._space

    @property
    def order(self):
        return self.parent.order

    @property
    def coordinates(self):
        return self.parent.coordinates

    @property
    def ring(self):
        return self.parent.ring

    def inverse(self):
        """Return the multiplicative inverse in this quotient."""
        return self.parent.apply("inverse", self)

    def is_unit(self):
        """Return whether this class is invertible."""
        return self.parent.apply("unit", self).value

    def __str__(self):
        return self._record["result"]["representative"]

    def __repr__(self):
        # Distinguish residue classes from ambient polynomials in coordinate tuples.
        return f"[{self}]"

    def __bool__(self):
        return not self.parent.apply("equal", self, 0).value

    def __eq__(self, other):
        if not isinstance(other, (QuotientElement, Polynomial, Fraction)) and type(other) is not int:
            return NotImplemented
        if isinstance(other, Polynomial) and other.parent is not QQ:
            raise ValueError("Project the polynomial into the quotient ring before comparing, for example A(f) == a.")
        return self.parent.apply("equal", self, other).value

    def __neg__(self):
        return self.parent.apply("negate", self)

    def __pos__(self):
        return self

    def __add__(self, other):
        return self.parent.apply("add", self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.parent.apply("subtract", self, other)

    def __rsub__(self, other):
        return self.parent.apply("subtract", other, self)

    def __mul__(self, other):
        return self.parent.apply("multiply", self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.parent.apply("divide", self, other)

    def __rtruediv__(self, other):
        return self.parent.apply("divide", other, self)

    def __pow__(self, exponent):
        return self.parent.apply("power", self, exponent=exponent)


class _QuotientIdealGenerator(QuotientElement):
    """A residue class already reduced by the native operation that produced its ideal."""

    status = "ok"
    type = "quotient-element"
    operation = "quotient-ring"

    def __init__(self, parent, representative, text, compute):
        self.__dict__.update(parent=parent, representative=representative, _text=text, _compute=compute)

    def __str__(self):
        return self._text

    @property
    def result(self):
        return dict(self.parent.result, type=self.type, representative=self._text)

    @cached_property
    def _record(self):
        # A generator's own replay receipt needs a single-class projection.
        # Only compute it when explicitly asked for that receipt or its timing.
        return self.parent.apply("reduce", self.representative, compute=self._compute)._record


class QuotientIdeal(_IdealResult):
    """An ideal of a particular quotient. Its recorded generator rows are reduced in that quotient."""

    # Ambient polynomial-ideal handles do not carry quotient relations.
    _native_ideal = None

    def __init__(self, parent, order, record):
        self.__dict__.update(_space=parent, order=order, _record=record)

    @cached_property
    def generators(self):
        parent = self.parent
        compute = self._record["execution"]["requested"]
        values = []
        for row in self._record["result"]["generators"]:
            representative = _OrderedPolynomial(parent.ambient, _decode_algebra_polynomial(row, parent.ambient), self.order)
            values.append(_QuotientIdealGenerator(parent, representative, row, compute))
        return tuple(values)

    @classmethod
    def declare(cls, parent, generators, *, project):
        if not project and any(isinstance(f, Polynomial) and f.parent is not QQ for f in generators):
            raise ValueError("These ideal generators have different parent rings. "
                             "Use ring=K to project ambient polynomials into the quotient K.")
        representatives = tuple(parent._lift(f, project=project) for f in generators)
        request = dict(operation="ideal", ring=parent.ring,
                       operands={"ideal": [str(f) for f in representatives]},
                       compute=parent._record["execution"]["requested"])
        return _typed(_evaluate_live_quotient_ideal(request), parent.ambient, quotient_parent=parent)

    @property
    def variables(self):
        return self.parent.variables

    @property
    def ring(self):
        return self.parent.ring

    def __repr__(self):
        generators = ", ".join(f"[{row}]" for row in self._record["result"]["generators"]) or "0"
        return f"ideal({generators}) over {self.parent}"


def replay_further_quotient(request, record, space):
    """Restore the intermediate ring and its defining ideal from the two sets of generators."""
    ambient = record["result"]["ring"]
    compute = record["execution"]["requested"]
    relations = request["ring"]["relations"]
    declaration = dict(operation="quotient-ring", ring=ambient, compute=compute,
                       operands=dict(action="declare", ideal=relations))
    parent = _typed(_evaluate(declaration), space, replay_generators=relations)
    source = dict(operation="ideal", ring=parent.ring, compute=compute,
                  operands={"ideal": request["operands"]["ideal"]})
    defining = _typed(_evaluate(source, known_basis=True), space, quotient_parent=parent)
    result = _typed(record, space, defining_ideal=defining)
    defining.__dict__["quotient_ring"] = result
    return result


def quotient_parent_from_record(record, space, relations, defining_ideal, replay_generators):
    result = record["result"]
    order = result["ring"]["order"]
    if defining_ideal is not None:
        return QuotientRing(space, order, record, defining_ideal, relations)

    rows = result["relations"]
    written = rows if replay_generators is None else replay_generators
    decoded = dict(zip(rows, relations))
    for text in written:
        if text not in decoded:
            decoded[text] = polynomial(text, ring=space)
    source = Ideal(space, tuple(decoded[text] for text in written), order)
    if result["type"] == "quotient-element":
        # The native replay has already verified this reduced basis. Obtain the
        # parent's own declaration receipt without recomputing the basis.
        request = dict(operation="quotient-ring", ring=result["ring"], compute=record["execution"]["requested"],
                       operands=dict(action="declare", ideal=rows))
        record = _evaluate(request, known_basis=True)
        if written != rows:
            from .algebra_replay import make_replay
            operands = dict(action="declare", ideal=written)
            record["replay"] = make_replay(dict(request, operands=operands), operands)
    parent = QuotientRing(space, order, record, source, relations)
    source.__dict__["quotient_ring"] = parent
    return parent


def quotient_ring(ideal, *, compute="auto", fresh=False):
    """Return the default quotient for this ideal. Use fresh=True for an independent ring."""
    ideal = _require_ideal(ideal, allow_quotient=True)
    if type(fresh) is not bool:
        raise ValueError("fresh must be True or False.")
    existing = ideal.__dict__.get("quotient_ring")
    if existing is not None and not fresh and compute == "auto":
        return existing
    rows = ideal._record["result"]["generators"] if isinstance(ideal, QuotientIdeal) else [str(f) for f in ideal.generators]
    request = dict(operation="quotient-ring", ring=ideal.ring, compute=compute,
                   operands=dict(action="declare", ideal=rows))
    record = _evaluate_live_quotient_ideal(request) if isinstance(ideal, QuotientIdeal) else _evaluate(request)
    if fresh or existing is None:
        space = ideal.parent.ambient if isinstance(ideal, QuotientIdeal) else ideal.parent
        declared = _typed(record, space, defining_ideal=ideal)
        if fresh:
            return declared
        # Concurrent first declarations may finish together. The first parent
        # published on this ideal owns the coordinates used by every caller.
        existing = ideal.__dict__.setdefault("quotient_ring", declared)
        if existing is declared:
            return declared
    existing._remember_declaration(record["execution"])
    return existing
