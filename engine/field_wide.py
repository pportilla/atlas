"""Unpacked coefficient-block bases and ordered division with exact FLINT rationals."""
import ctypes
from functools import cached_property

from . import native_polynomial as np
from .chart_expressions import chart_expressions
from .reference import sp, QQ

from . import groebner as gb
from .affine_kernel import _request, _ring, _sparse_expression, _sparse_terms
from .field_polynomial import _line_expression, _line_polynomial, lift, modulus


def _expanded_row(variables, f):
    row = gb._lines.get((variables, f))
    if row is not None and "*" not in row:
        return row
    # Exact cancellation of a non-polynomial quotient is not polynomial input.
    if np.native_ring(variables):
        return np.row(variables, f)
    if any(p.is_Pow and p.exp.is_negative for p in sp.preorder_traversal(sp.sympify(f))):
        raise sp.PolynomialError("Expected a polynomial in the specified coordinates.")
    ring = _ring(variables)
    row, = _request(["function", ring.header, ring.text(f)])
    return row


def _ask(library, operation, n, order, field, rows):
    function = library.atlas_wide_block
    function.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    function.restype = ctypes.c_void_p
    header = f"{operation} {n} {gb._ORDERS[order]} {int(field)}\n"
    payload = (header + "".join(row + "\n" for row in rows)).encode("ascii")
    answer = function(payload, len(payload))
    if not answer:
        raise ArithmeticError(library.atlas_groebner_error().decode())
    try:
        return ctypes.string_at(answer).decode("ascii").splitlines()
    finally:
        library.atlas_groebner_release(answer)


class WideBlockBasis:
    def __init__(self, library, variables, domain, theta, order, rows):
        self.library, self.gens, self.domain, self.theta, self.order = library, variables, domain, theta, order
        self.rows = rows
        self.extended = variables if theta is None else (*variables, theta)
        self.relation = None if theta is None else _expanded_row(self.extended, modulus(domain, theta))
        self.proper = theta is not None and bool(rows) and _sparse_terms(rows[-1])[0][0][-1] > 0
        self.output_rows = rows[:-1] if self.proper else rows
        self.exprs = [self.expression(row) for row in self.output_rows]

    def __len__(self):
        return len(self.exprs)

    def __iter__(self):
        return iter(self.exprs)

    def __getitem__(self, index):
        return self.exprs[index]

    def expression(self, row):
        if self.theta is None:
            return _sparse_expression(self.gens, row)
        return _line_expression(row, self.gens, self.domain)

    @cached_property
    def polys(self):
        if self.theta is None:
            if np.native_ring(self.gens):
                return self.exprs
            return [sp.Poly.from_dict(dict(_sparse_terms(row)), *self.gens, domain=sp.QQ) for row in self.output_rows]
        return [_line_polynomial(row, self.gens, self.domain) for row in self.output_rows]

    def _divide(self, rows):
        return _ask(self.library, "divide", len(self.gens), self.order, self.theta is not None, rows)

    def _division(self, expr):
        f = expr if self.theta is None else lift(expr, self.gens, self.domain, self.theta)
        return self._divide([_expanded_row(self.extended, f), *self.rows])

    def remainder(self, expr):
        return self.expression(self._division(expr)[-1])

    def contains(self, expr):
        return not self._division(expr)[-1]

    def reduce(self, expr):
        *quotients, remainder = self._division(expr)
        if self.theta is not None:
            if self.proper:
                quotients.pop()  # coefficient relation, zero in K
            quotients = [self._divide((row, self.relation))[-1] if row else "" for row in quotients]
        if not remainder and not any(quotients):
            return [], chart_expressions(self.gens).S.Zero
        return [self.expression(row) for row in quotients], self.expression(remainder)


def wide_block_basis(ideal, variables, order, domain=QQ, theta=None):
    library = gb._kernel()
    if not hasattr(library, "atlas_wide_block") or order not in gb._ORDERS or not variables:
        return None
    if order == "grlex" and not hasattr(library, "atlas_grlex_abi"):
        return None
    variables = tuple(variables)
    if theta is None:
        extended, functions = variables, ideal
    else:
        extended = (*variables, theta)
        functions = [modulus(domain, theta), *(lift(f, variables, domain, theta) for f in ideal)]
    rows = _ask(library, "basis", len(variables), order, theta is not None,
                (_expanded_row(extended, f) for f in functions))
    return WideBlockBasis(library, variables, domain, theta, order, rows)
