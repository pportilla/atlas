"""Declared quotient rings and their residue classes in an Atlas session."""

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

from .session_algebra import Ideal, _Result, _evaluate, _polynomial, _require_ideal, _typed
from .session_polynomial import Polynomial, _CoordinateSpace


@dataclass(frozen=True, eq=False, repr=False)
class QuotientRing(_Result):
    _space: _CoordinateSpace
    order: str
    _record: dict
    defining_ideal: Ideal
    _fields = _Result._fields | {"defining_ideal", "relations", "gens", "generators", "apply", "element"}

    @property
    def relations(self):
        return self.defining_ideal.generators

    @cached_property
    def gens(self):
        return tuple(self(f) for f in self.coordinates)

    @property
    def generators(self):
        return self.gens

    def __repr__(self):
        ambient = f"QQ[{', '.join(self.variables)}]" if self.variables else "QQ"
        return f"{ambient}/({', '.join(map(str, self.relations)) or '0'}) ({self.order})"

    def _lift(self, value):
        if isinstance(value, QuotientElement):
            if value.parent is not self:
                raise ValueError("These elements belong to different quotient rings.")
            return value.representative
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
        request = dict(operation="quotient-ring", ring=self.ring, operands=operands, compute=mode)
        # These rows came from this ring's native declaration. Replay of a public
        # request always computes a basis before constructing a ring object.
        return _typed(_evaluate(request, known_basis=True), self._space, quotient_parent=self)

    def element(self, polynomial):
        if isinstance(polynomial, QuotientElement) and polynomial.parent is self:
            return polynomial
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

    def inverse(self):
        """Return the multiplicative inverse in this quotient."""
        return self.parent.apply("inverse", self)

    def is_unit(self):
        """Return whether this class is invertible."""
        return self.parent.apply("unit", self).value

    def __repr__(self):
        return self._record["result"]["representative"]

    def __bool__(self):
        return not self.parent.apply("equal", self, 0).value

    def __eq__(self, other):
        if not isinstance(other, (QuotientElement, Polynomial, Fraction)) and type(other) is not int:
            return NotImplemented
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


def quotient_ring(ideal, *, compute="auto"):
    """Declare the quotient of the ambient polynomial ring by an ideal over QQ."""
    ideal = _require_ideal(ideal)
    request = dict(operation="quotient-ring", ring=ideal.ring, compute=compute,
                   operands=dict(action="declare", ideal=[str(f) for f in ideal.generators]))
    return _typed(_evaluate(request), ideal._space)
