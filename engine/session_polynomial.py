"""Polynomials bound to an explicit coordinate space in Atlas sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property, lru_cache
import re

from . import native_polynomial as native
from .point_geometry import support_profile
from .session_ideal_native import native_ring


@dataclass(frozen=True, eq=False)
class _NumberField:
    """A coefficient field of a chart with its chosen geometric embedding."""

    parameter: native.Expr
    minimal_polynomial: native.Expr
    embedding: str

    @cached_property
    def degree(self):
        return support_profile(native.terms((self.parameter,), self.minimal_polynomial))[1]

    def __repr__(self):
        return f"QQ[{self.parameter}]/({self.minimal_polynomial}), {self.parameter} = {self.embedding}"

    @lru_cache(maxsize=512)
    def inverse(self, value):
        """Invert in QQ[theta]/(m), using the chart's own primitive parameter."""
        from .native_library import affine_request

        row, = affine_request(["parse_field_input", self.parameter.name, str(self.minimal_polynomial), f"1/({value})"])
        return native.from_row((self.parameter,), row)

    def reduce(self, value, variables):
        if self.parameter not in value.free_symbols:
            return value
        degrees, _, _ = support_profile(native.terms(variables, value))
        if degrees[-1] < self.degree:
            return value

        from .affine_kernel import _request, _ring, _sparse_expression

        ring = _ring(variables)
        _, row = _request(["wide_division", ring.header, "lex", ring.text(value), ring.text(self.minimal_polynomial)])
        return _sparse_expression(variables, row)


class _Coordinates(tuple):
    """The ordered coordinate objects of a live parent, including an empty tuple."""

    _fields = {"parent"}

    def __new__(cls, parent, coordinates):
        value = super().__new__(cls, coordinates)
        value._parent = parent
        return value

    @property
    def parent(self):
        return self._parent

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def session_attribute(self, name):
        if name == "parent":
            return self.parent
        raise AttributeError(f"A coordinate tuple has no Atlas field {name!r}.")


@dataclass(frozen=True, eq=False)
class _CoordinateSpace:
    generators: tuple[native.Expr, ...]
    has_relations: bool = False
    coeffs: _RationalField | _NumberField = field(default_factory=lambda: QQ)
    _fields = {"coordinates", "variables", "coeffs"}

    @classmethod
    def named(cls, names, *, coeffs=None, has_relations=False):
        if coeffs is None:
            coeffs = QQ
        if not names and coeffs is QQ and not has_relations:
            return QQ
        return cls(tuple(native.Dummy(name) for name in names), has_relations=has_relations, coeffs=coeffs)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Ring identity survives copies of surrounding session objects.
        return self

    @cached_property
    def _native_ring(self):
        # Native construction can release the GIL; concurrent first users must
        # all retain the same context for this particular live parent.
        if self.has_relations or self.coeffs is not QQ:
            return None
        ring = native_ring(len(self.generators))
        return self.__dict__.setdefault("_native_ring", ring)

    @property
    def polynomial_generators(self):
        if self.coeffs is QQ:
            return self.generators
        return (*self.generators, self.coeffs.parameter)

    @property
    def names(self):
        return tuple(generator.name for generator in self.generators)

    @property
    def variables(self):
        return self.names

    @cached_property
    def coordinates(self):
        # Share one coordinate object per generator, including concurrent first
        # access. A computed polynomial equal to a coordinate is a separate value.
        coordinates = _Coordinates(self, (Polynomial(self, generator) for generator in self.generators))
        return self.__dict__.setdefault("coordinates", coordinates)

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"A polynomial parent has no Atlas field {name!r}.")

    def __repr__(self):
        field = "QQ" if self.coeffs is QQ else f"QQ[{self.coeffs.parameter.name}]"
        ring = f"{field}[{', '.join(self.names)}]" if self.generators else field
        return ring + (" (quotient)" if self.has_relations else "")


class _RationalField(_CoordinateSpace):
    """The rational field is also the polynomial ring with no generators."""

    def __init__(self):
        super().__init__((), coeffs=self)


QQ = _RationalField()


@dataclass(frozen=True, eq=False)
class Polynomial:
    """A native value and its ambient coordinates, including unused variables."""

    _space: _CoordinateSpace
    _value: native.Expr
    _fields = {"variables", "coeffs", "multiplicity", "parent"}

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # The value and its native polynomial are immutable in this parent.
        return self

    def __post_init__(self):
        if self._space.has_relations:
            object.__setattr__(self, "_value", self._space.relation_basis.remainder(self._value))
        elif self._space.coeffs is not QQ:
            reduced = self._space.coeffs.reduce(self._value, self._space.polynomial_generators)
            object.__setattr__(self, "_value", reduced)
        else:
            parent = self._space
            ring = parent._native_ring
            handle = None if ring is None else ring.polynomial(native.generator_row(parent.generators, self._value))
            object.__setattr__(self, "_native_polynomial", handle)

    @property
    def variables(self):
        return self._space.names

    @property
    def coeffs(self):
        return self._space.coeffs

    @property
    def parent(self):
        return self._space

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"A polynomial has no Atlas field {name!r}.")

    def __str__(self):
        return str(self._value)

    __repr__ = __str__

    def __bool__(self):
        return bool(self._value)

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            if self._space is not other._space and self._space is not QQ and other._space is not QQ:
                return False
            return self._value == other._value
        if type(other) is int or isinstance(other, Fraction):
            return self._value == other
        return NotImplemented

    def _operand(self, other):
        """Return the common parent and the operand embedded in it."""
        if isinstance(other, Polynomial):
            if self._space is QQ:
                return other._space, other._value
            if self._space is not other._space and other._space is not QQ:
                raise ValueError("These polynomials belong to different coordinate spaces.")
            return self._space, other._value
        if type(other) is int or isinstance(other, Fraction):
            return self._space, native.coerce(other)
        raise TypeError("Polynomial arithmetic requires a polynomial or an exact rational number.")

    def multiplicity(self, factor):
        """Largest exponent n for which factor**n divides this polynomial."""
        space, divisor = self._operand(factor)
        if space.has_relations:
            raise ValueError("Computing multiplicity on a chart with ambient relations requires "
                             "divisibility in its quotient ring.")
        if not divisor.has(*space.generators):
            raise ValueError("Multiplicity requires a nonconstant polynomial factor.")
        if not self._value:
            raise ValueError("The zero polynomial has no finite factor multiplicity.")

        variables = space.polynomial_generators
        from .affine_kernel import _request, _ring
        from .polynomial import integer_from_decimal

        ring = _ring(variables)
        request = ["factor_multiplicity", ring.header, str(len(space.generators)),
                   ring.text(self._value), ring.text(divisor)]
        result, = _request(request)
        if result == "field_basis":
            from .wide_basis import wide_basis

            # With the coefficient parameter last, the rational lex basis is
            # the monic field divisor followed by its defining relation.
            basis = wide_basis((divisor, space.coeffs.minimal_polynomial), variables, "lex")
            if basis is None:
                raise ArithmeticError("The native coefficient-field division kernel is unavailable.")
            result, = _request([*request, *(ring.text(row) for row in basis.exprs)])
        return integer_from_decimal(result)

    def __pos__(self):
        return self

    def __neg__(self):
        return Polynomial(self._space, -self._value)

    def __add__(self, other):
        if not isinstance(other, (Polynomial, Fraction)) and type(other) is not int:
            return NotImplemented
        space, value = self._operand(other)
        return Polynomial(space, self._value + value)

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, (Polynomial, Fraction)) and type(other) is not int:
            return NotImplemented
        space, value = self._operand(other)
        return Polynomial(space, self._value - value)

    def __rsub__(self, other):
        space, value = self._operand(other)
        return Polynomial(space, value - self._value)

    def __mul__(self, other):
        if not isinstance(other, (Polynomial, Fraction)) and type(other) is not int:
            return NotImplemented
        space, value = self._operand(other)
        return Polynomial(space, self._value * value)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, (Polynomial, Fraction)) and type(other) is not int:
            return NotImplemented
        space, divisor = self._operand(other)
        if not divisor.is_number and space.coeffs is not QQ and divisor.free_symbols == {space.coeffs.parameter}:
            return Polynomial(space, self._value * space.coeffs.inverse(divisor))
        if not divisor.is_number or not divisor:
            raise ValueError("Division requires a nonzero constant in the coefficient field.")
        return Polynomial(space, self._value / divisor)

    def __rtruediv__(self, other):
        space, value = self._operand(other)
        return Polynomial(space, value) / self

    def __pow__(self, exponent):
        if not (type(exponent) is int or isinstance(exponent, Fraction)):
            raise ValueError("Use a nonnegative integer for the exponent.")
        return Polynomial(self._space, self._value ** exponent)


def _declare_ring(names, coeffs, make_parent):
    if coeffs is not QQ:
        raise ValueError("The supported coefficient field is QQ.")
    if not names or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", n)
                        for n in names):
        raise ValueError("Use coordinate names of up to 24 letters, digits, or underscores, starting with a letter.")
    if len(names) != len(set(names)):
        raise ValueError("Choose distinct coordinate names.")
    return make_parent(names, coeffs=coeffs)


def _declare_coordinates(names, coeffs, make_parent):
    generators = _declare_ring(names, coeffs, make_parent).coordinates
    return generators[0] if len(names) == 1 else generators


def coordinates(*names, coeffs=QQ):
    """Declare generators in an independent coordinate space outside an Atlas session."""
    return _declare_coordinates(names, coeffs, _CoordinateSpace.named)


def session_declarations():
    """Create the coordinates and ring functions for one prompt or script.

    Both functions share the session's rings, so ring("x", "y") is the parent
    of the generators that coordinates("x", "y") returns.
    """
    parents = {}

    def declared_parent(names, *, coeffs):
        key = (coeffs, names)
        parent = parents.get(key)
        if parent is None:
            # Publish the first completed construction. All callers of this
            # session must receive that same live ring for the same declaration.
            parent = parents.setdefault(key, _CoordinateSpace.named(names, coeffs=coeffs))
        return parent

    def parent_maker(fresh):
        if type(fresh) is not bool:
            raise ValueError("fresh must be True or False.")
        return _CoordinateSpace.named if fresh else declared_parent

    def coordinates(*names, coeffs=QQ, fresh=False):
        """Reuse the ring with these ordered coordinate names. Pass fresh=True for a separate ring."""
        return _declare_coordinates(names, coeffs, parent_maker(fresh))

    def ring(*names, coeffs=QQ, fresh=False):
        """Return the polynomial ring over QQ with these ordered coordinate names.

        >>> R = ring("x", "y")
        >>> x, y = R.coordinates
        >>> f = polynomial("x^2 + y", ring=R)
        >>> f.parent == R

        Docs: docs/sessions.md#choose-the-ring-of-a-polynomial
        """
        return _declare_ring(names, coeffs, parent_maker(fresh))

    return coordinates, ring
