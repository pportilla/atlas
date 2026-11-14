"""Polynomials bound to an explicit coordinate space in Atlas sessions."""

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import re

from . import native_polynomial as native


class _RationalField:
    def __repr__(self):
        return "QQ"


QQ = _RationalField()


@dataclass(frozen=True, eq=False)
class _NumberField:
    """A coefficient field of a chart with its chosen geometric embedding."""

    parameter: native.Expr
    minimal_polynomial: native.Expr
    embedding: str

    @cached_property
    def degree(self):
        return max(powers[0] for powers in native.terms((self.parameter,), self.minimal_polynomial))

    def __repr__(self):
        return f"QQ[{self.parameter}]/({self.minimal_polynomial}), {self.parameter} = {self.embedding}"

    def reduce(self, value, variables):
        if self.parameter not in value.free_symbols:
            return value
        if all(powers[-1] < self.degree for powers in native.terms(variables, value)):
            return value

        from .affine_kernel import _request, _ring, _sparse_expression

        ring = _ring(variables)
        _, row = _request(["wide_division", ring.header, "lex", ring.text(value), ring.text(self.minimal_polynomial)])
        return _sparse_expression(variables, row)


@dataclass(frozen=True, eq=False)
class _CoordinateSpace:
    generators: tuple[native.Expr, ...]
    has_relations: bool = False
    coeffs: _RationalField | _NumberField = QQ

    @classmethod
    def named(cls, names, *, coeffs=QQ, has_relations=False):
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
    def coordinates(self):
        return tuple(Polynomial(self, generator) for generator in self.generators)


@dataclass(frozen=True, eq=False)
class Polynomial:
    """A native value and its ambient coordinates, including unused variables."""

    _space: _CoordinateSpace
    _value: native.Expr

    def __post_init__(self):
        if self._space.coeffs is not QQ:
            reduced = self._space.coeffs.reduce(self._value, self._space.polynomial_generators)
            object.__setattr__(self, "_value", reduced)

    @property
    def variables(self):
        return self._space.names

    @property
    def coeffs(self):
        return self._space.coeffs

    def session_attribute(self, name):
        if name in ("variables", "coeffs", "multiplicity"):
            return getattr(self, name)
        raise AttributeError(f"A polynomial has no Atlas field {name!r}.")

    def __str__(self):
        return str(self._value)

    __repr__ = __str__

    def __bool__(self):
        return bool(self._value)

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return self._space is other._space and self._value == other._value
        if type(other) is int or isinstance(other, Fraction):
            return self._value == other
        return NotImplemented

    def _operand(self, other):
        if isinstance(other, Polynomial):
            if self._space is not other._space:
                raise ValueError("These polynomials belong to different coordinate spaces.")
            return other._value
        if type(other) is int or isinstance(other, Fraction):
            return native.coerce(other)
        raise TypeError("Polynomial arithmetic requires a polynomial or an exact rational number.")

    def multiplicity(self, factor):
        """Largest exponent n for which factor**n divides this polynomial."""
        divisor = self._operand(factor)
        if self._space.has_relations:
            raise ValueError("Computing multiplicity on a chart with ambient relations requires "
                             "divisibility in its quotient ring.")
        if not divisor.has(*self._space.generators):
            raise ValueError("Multiplicity requires a nonconstant polynomial factor.")
        if not self._value:
            raise ValueError("The zero polynomial has no finite factor multiplicity.")

        variables = self._space.polynomial_generators
        divisor_terms = native.terms(variables, divisor)
        axes = len(self._space.generators)
        monomials = {term[:axes] for term in divisor_terms}
        if len(monomials) == 1:
            # A monomial times a nonzero field scalar divides term by term.
            powers = next(iter(monomials))
            terms = native.terms(variables, self._value)
            return min(term[i] // power for i, power in enumerate(powers) if power
                       for term in terms)

        from .affine_kernel import _request, _ring, _sparse_expression

        ring = _ring(variables)
        divisors = [ring.text(divisor)]
        if self.coeffs is not QQ and divisor.has(self.coeffs.parameter):
            from .wide_basis import wide_basis

            # With the coefficient parameter last, the reduced rational lex
            # basis consists of the monic field factor followed by its modulus.
            basis = wide_basis((divisor, self.coeffs.minimal_polynomial), variables, "lex")
            if basis is None:
                raise ArithmeticError("The native coefficient-field division kernel is unavailable.")
            divisors = [ring.text(row) for row in basis.exprs]
        value = self._value
        order = 0
        while value.has(*self._space.generators):
            quotient, *_, remainder = _request(["wide_division", ring.header, "lex", ring.text(value), *divisors])
            if remainder:
                break
            value = _sparse_expression(variables, quotient)
            if self.coeffs is not QQ:
                value = self.coeffs.reduce(value, variables)
            order += 1
        return order

    def __pos__(self):
        return self

    def __neg__(self):
        return Polynomial(self._space, -self._value)

    def __add__(self, other):
        return Polynomial(self._space, self._value + self._operand(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Polynomial(self._space, self._value - self._operand(other))

    def __rsub__(self, other):
        return Polynomial(self._space, self._operand(other) - self._value)

    def __mul__(self, other):
        return Polynomial(self._space, self._value * self._operand(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        divisor = self._operand(other)
        if not divisor.is_number or not divisor:
            raise ValueError("Division requires a nonzero rational constant.")
        return Polynomial(self._space, self._value / divisor)

    def __rtruediv__(self, other):
        return Polynomial(self._space, self._operand(other)) / self

    def __pow__(self, exponent):
        if not (type(exponent) is int or isinstance(exponent, Fraction)):
            raise ValueError("Use a nonnegative integer for the exponent.")
        return Polynomial(self._space, self._value ** exponent)


def coordinates(*names, coeffs=QQ):
    """Declare generators with rational coefficients in a fresh coordinate space."""
    if coeffs is not QQ:
        raise ValueError("The supported coefficient field is QQ.")
    if not names or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", n)
                        for n in names):
        raise ValueError("Use coordinate names of up to 24 letters, digits, or underscores, starting with a letter.")
    if len(names) != len(set(names)):
        raise ValueError("Choose distinct coordinate names.")

    generators = _CoordinateSpace.named(names).coordinates
    return generators[0] if len(names) == 1 else generators
