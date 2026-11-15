"""Polynomials bound to an explicit coordinate space in Atlas sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache, cached_property, lru_cache
import re

from . import native_polynomial as native
from .point_geometry import support_profile


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

    @property
    def coordinates(self):
        return tuple(Polynomial(self, generator) for generator in self.generators)

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

    def __post_init__(self):
        if self._space.has_relations:
            object.__setattr__(self, "_value", self._space.relation_basis.remainder(self._value))
        elif self._space.coeffs is not QQ:
            reduced = self._space.coeffs.reduce(self._value, self._space.polynomial_generators)
            object.__setattr__(self, "_value", reduced)

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


def _declare_coordinates(names, coeffs, make_parent):
    if coeffs is not QQ:
        raise ValueError("The supported coefficient field is QQ.")
    if not names or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", n)
                        for n in names):
        raise ValueError("Use coordinate names of up to 24 letters, digits, or underscores, starting with a letter.")
    if len(names) != len(set(names)):
        raise ValueError("Choose distinct coordinate names.")

    generators = make_parent(names, coeffs=coeffs).coordinates
    return generators[0] if len(names) == 1 else generators


def coordinates(*names, coeffs=QQ):
    """Declare generators in an independent coordinate space outside an Atlas session."""
    return _declare_coordinates(names, coeffs, _CoordinateSpace.named)


def session_coordinates():
    """Create the coordinates function for one prompt or script."""
    declared_parent = cache(_CoordinateSpace.named)

    def coordinates(*names, coeffs=QQ, fresh=False):
        """Reuse the ring with these ordered coordinate names. Pass fresh=True for a separate ring."""
        if type(fresh) is not bool:
            raise ValueError("fresh must be True or False.")
        make_parent = _CoordinateSpace.named if fresh else declared_parent
        return _declare_coordinates(names, coeffs, make_parent)

    return coordinates
