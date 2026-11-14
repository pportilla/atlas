"""Reusable exact rational bases for Atlas algebra operations.

The caller supplies ordered native coordinates and owns the execution scope.
Only the wire adapter converts the resulting polynomials to text.
"""
from dataclasses import dataclass

from . import groebner as gb
from . import native_polynomial as np
from .polynomial import format_poly
from .quotient_monomials import degree_bounds


def _operand(value, variables):
    if isinstance(value, bool):
        raise TypeError("Expected a polynomial or an exact rational number.")
    value = np.coerce(value)
    if not value.free_symbols.issubset(variables):
        raise ValueError("The polynomial uses a coordinate outside the basis ring.")
    return value


@dataclass(frozen=True, eq=False)
class ReducedBasis:
    """The canonical generators together with the divisor table for this ring."""
    gens: tuple
    order: str
    exprs: tuple
    _basis: object

    def remainder(self, polynomial):
        polynomial = _operand(polynomial, self.gens)
        if not self.gens:  # The only ideals in QQ are (0) and (1).
            return np.S.Zero if self.exprs else polynomial
        return self._basis.remainder(polynomial)

    def reduce(self, polynomial):
        return None, self.remainder(polynomial)

    def contains(self, polynomial):
        polynomial = _operand(polynomial, self.gens)
        if not self.gens:
            return bool(self.exprs) or not polynomial
        return self._basis.contains(polynomial)


class _CheckedRationalBasis(gb._Checked):
    def remainder(self, polynomial):
        answer = self._native.remainder(polynomial)
        if answer != self._reference.remainder(polynomial):
            raise AssertionError("The native normal form differs from SymPy's.")
        return answer


def _restore_native(generators, variables, order):
    """Load verified rows without forming new critical pairs in either representation."""
    support = (m for f in generators for m in np.terms(variables, f))
    if len(variables) > gb._MAX_VARIABLES or (any(generators) and degree_bounds(len(variables), support)[1] > gb._MAX_DEGREE):
        from .wide_basis import wide_basis_from_rows
        basis = wide_basis_from_rows([np.row(variables, f) for f in generators], variables, order)
        if basis is not None:
            gb.record_stage("basis_restore", "cpu", algorithm="wide", generators=len(generators))
    else:
        basis = gb._native(generators, variables, order, known=len(generators))
    if basis is None:
        raise gb.Unavailable("Rebuild the native kernel to load this rational basis.")
    return basis


def _make_basis(generators, variables, order, *, trusted):
    variables = tuple(variables)
    if (any(not isinstance(v, np.Expr) or not v.is_Symbol for v in variables)
            or len(set(variables)) != len(variables)):
        raise ValueError("Supply distinct polynomial coordinates in their declared order.")
    if order not in ("lex", "grevlex", "first"):
        raise ValueError("Unsupported monomial order.")
    generators = tuple(_operand(f, variables) for f in generators)
    if not variables:
        return ReducedBasis((), order, (np.S.One,) if any(generators) else (), None)
    if gb._MODE in ("sympy", "check"):
        from tests.groebner_oracle import _SymbolicBasis
    if gb._MODE == "sympy":
        basis = _SymbolicBasis(generators, variables, order, trusted)
    elif gb._MODE == "check":
        basis = (_restore_native(generators, variables, order) if trusted
                 else gb._native(generators, variables, order, known=0))
        if basis is None:
            raise gb.Unavailable("Rebuild the native kernel to check this rational basis.")
        basis = _CheckedRationalBasis(basis, _SymbolicBasis(generators, variables, order, trusted))
    elif trusted:
        basis = _restore_native(generators, variables, order)
    else:
        basis = gb.groebner(generators, variables, order=order)
    return ReducedBasis(variables, order, tuple(basis.exprs), basis)


def reduced_basis(generators, variables, *, order="grevlex"):
    """Compute once over QQ. The internal 'first' order is used for elimination."""
    return _make_basis(generators, variables, order, trusted=False)


def basis_from_reduced(generators, variables, *, order):
    """Restore certified divisors from a trusted Atlas result in the same ordered ring."""
    return _make_basis(generators, variables, order, trusted=True)


def normal_form(polynomial, basis):
    """The unique exact remainder in the coordinate space of the saved basis."""
    return basis.remainder(polynomial)


def ideal_membership(polynomial, basis):
    """Whether the exact remainder by the saved basis is zero."""
    return basis.contains(polynomial)


def handle(operation, ring, operands):
    """Evaluate a validated wire request, optionally reusing a prepared basis."""
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise ValueError("The basis handlers require QQ and lex or grevlex order.")
    variables = tuple(np.Symbol(name) for name in ring["variables"])
    basis = operands["ideal"]
    if isinstance(basis, ReducedBasis):
        if basis.gens != variables or basis.order != ring["order"]:
            raise ValueError("The saved basis belongs to a different ring or monomial order.")
    else:
        basis = reduced_basis(basis, variables, order=ring["order"])

    def display(f):
        return format_poly(np.terms(variables, f), ring["variables"])

    if operation == "groebner":
        return {"type": "basis", "ring": ring, "generators": [display(f) for f in basis.exprs]}
    if operation == "normal-form":
        remainder = normal_form(operands["polynomial"], basis)
        return {"type": "polynomial", "ring": ring, "polynomial": display(remainder)}
    if operation == "membership":
        return {"type": "boolean", "ring": ring, "value": ideal_membership(operands["polynomial"], basis)}
    raise ValueError(f"Unknown basis operation {operation!r}.")
