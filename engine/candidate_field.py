"""The marked coefficient field on the exhaustive verifier's private wire.

A polynomial in K[x] is represented by a rational polynomial in (x, theta).
The native block-order operations reduce coefficients modulo the minimal
polynomial. The parameter has its own identity even if a user coordinate is
printed with the same name.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property, lru_cache

from . import native_polynomial as np
from .reference import QQ


@dataclass(frozen=True)
class NativeField:
    coefficients: tuple
    index: int
    parameter: np.Expr
    is_AlgebraicField = True
    dom = QQ

    @property
    def degree(self):
        return len(self.coefficients)-1

    @cached_property
    def exact_root(self):
        relation = np.from_terms((self.parameter,), {(i,): c for i,c in enumerate(reversed(self.coefficients))})
        return f"CRootOf({relation}, {self.index})"


@lru_cache(maxsize=128)
def indexed_field(coefficients, index):
    field = NativeField(tuple(map(Fraction, coefficients)), index, np.Dummy("coefficient_root"))
    return field, field.parameter


@dataclass(frozen=True)
class FieldPolynomial:
    value: np.Expr
    gens: tuple
    domain: NativeField

    def as_expr(self):
        return self.value

    def as_dict(self):
        theta = self.domain.parameter
        return {e: np.from_terms((theta,), {(j,): c for j,c in enumerate(reversed(coeffs))})
                for e,coeffs in field_power_rows(self).items()}

    def monoms(self):
        return tuple(field_power_rows(self))

    @cached_property
    def _degrees(self):
        from .point_geometry import support_profile
        powers = self.monoms()
        if not powers:
            return tuple(-1 for _ in self.gens), -1
        axes, total, _ = support_profile(powers)
        return axes, total

    def degree(self, variable):
        return self._degrees[0][self.gens.index(variable)]

    def total_degree(self):
        return self._degrees[1]


def field_power_rows(poly):
    groups = {}
    for e,c in np.terms((*poly.gens, poly.domain.parameter), poly.value).items():
        groups.setdefault(e[:-1], {})[e[-1]] = c
    return {e: tuple(coeffs.get(j, Fraction(0)) for j in range(max(coeffs), -1, -1))
            for e,coeffs in groups.items()}
