"""Reusable exact rational bases for Atlas algebra operations.

The caller supplies ordered native coordinates and owns the execution scope.
Only the wire adapter converts the resulting polynomials to text.
"""
from dataclasses import dataclass

from . import groebner as gb
from . import native_polynomial as np
from .ordered_ideal_kernel import scalar_ideal
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
            return np.from_row((), scalar_ideal("remainder", self.exprs, polynomial))
        if (self.order in ("lex", "grevlex") and len(self.exprs) == 1 and
                len(self.exprs[0]._terms) == 2 and len(polynomial._terms) >= 8):
            from .native_library import current_replay_scope
            scope = current_replay_scope()
            if scope is not None and scope.compute == "gpu" and scope.backend in ("auto", "triton"):
                from .gpu_factor_division import factor_division_hooks
                from .kernel_json import kernel_json
                with factor_division_hooks():
                    result = kernel_json("atlas_polynomial_division", "try-binomial-remainder", width=len(self.gens),
                                         order=self.order, left=np.row(self.gens, polynomial),
                                         right=np.row(self.gens, self.exprs[0]))
                if result is not None:
                    return np.from_row(self.gens, result["remainder"])
        return self._basis.remainder(polynomial)

    def reduce(self, polynomial):
        return None, self.remainder(polynomial)

    def contains(self, polynomial):
        polynomial = _operand(polynomial, self.gens)
        if not self.gens:
            return scalar_ideal("membership", self.exprs, polynomial)
        return self._basis.contains(polynomial)


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
        rows = scalar_ideal("basis", generators)
        return ReducedBasis((), order, tuple(np.from_row((), row) for row in rows), None)
    if trusted:
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
