"""Rational bases and division with arbitrary-precision exponent vectors."""
from .affine_kernel import _library
from .algebra_execution import record_stage
from .field_wide import WideBlockBasis, _ask, _expanded_row, wide_block_basis
from .reference import QQ


class WideBasis(WideBlockBasis):
    """Keep the packed QQ interface and omit quotients in the sparse ring."""

    def remainder(self, value):
        if hasattr(self.library, "atlas_wide_remainder_abi"):
            row = _expanded_row(self.gens, value)
            answer, = _ask(self.library, "remainder", len(self.gens), self.order, False, (row, *self.rows))
            result = self.expression(answer)
        else:
            result = super().remainder(value)
        record_stage("normal_form", "cpu", divisors=len(self.rows), algorithm="wide")
        return result

    def reduce(self, value):
        return None, self.remainder(value)

    def contains(self, value):
        return self.remainder(value) == 0


def wide_basis_from_rows(rows, gens, order):
    """Use already reduced rows in basis order for division beyond the packed range."""
    library = _library()
    if order in ("lex", "grevlex", "first") and hasattr(library, "atlas_wide_block"):
        return WideBasis(library, gens, QQ, None, order, rows)
    return None


def wide_basis(ideal, gens, order):
    if order == "first":
        return wide_block_basis(ideal, gens, order)
    library = _library()
    if order not in ("lex", "grevlex") or not gens or not hasattr(library, "atlas_wide_block"):
        return None
    # Intermediate leading exponents can exceed a machine word even when every
    # input exponent fits. The sparse block kernel uses fmpz throughout.
    rows = _ask(library, "basis", len(gens), order, False, (_expanded_row(gens, f) for f in ideal))
    return WideBasis(library, gens, QQ, None, order, rows)


def wide_colon(ideal, by, variables):
    """Use the same elimination and exact division as rational algebra requests."""
    from .ideal_operations import quotient
    if not hasattr(_library(), "atlas_wide_block"):
        return None
    return quotient(variables, ideal, by).generators
