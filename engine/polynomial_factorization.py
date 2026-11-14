"""Irreducible factorizations in ordered rational polynomial rings."""

from time import perf_counter_ns

from . import native_polynomial as np
from .affine_kernel import _request, _ring
from .algebra_execution import record_stage
from .polynomial import InputError, integer_from_decimal


def factor_polynomial(variables, polynomial, *, order="grevlex"):
    """Return the rational unit and distinct monic irreducibles with their multiplicities."""
    variables = tuple(variables)
    polynomial = np.coerce(polynomial)
    if order not in ("lex", "grevlex"):
        raise InputError("Factorization requires lex or grevlex order.")
    if (len(set(variables)) != len(variables) or
            any(not isinstance(v, np.Expr) or not v.is_Symbol for v in variables) or
            not polynomial.free_symbols.issubset(variables)):
        raise InputError("Supply a polynomial in the declared rational coordinates.")
    if not polynomial:
        raise InputError("The zero polynomial has no irreducible factorization.")
    # Even a constant in the empty coordinate space can be read in a
    # temporary one-variable kernel ring. Its result has no factors.
    kernel_vars = variables or (np.Symbol("v0"),)
    ring = _ring(kernel_vars)
    started = perf_counter_ns()
    algorithm, unit_row, *rows = _request([
        "ordered_factorization", ring.header, order, ring.text(polynomial),
    ])
    unit = np.from_row(kernel_vars, unit_row).scalar
    factors = [(np.from_row(variables, row), integer_from_decimal(exponent))
               for exponent, row in zip(rows[::2], rows[1::2])]
    factors.sort(key=lambda part: np.default_sort_key(part[0]))
    details = {} if algorithm == "rational_constant" else {"order": order}
    record_stage("polynomial_factorization", "cpu", algorithm=algorithm,
                 elapsed_ns=perf_counter_ns() - started, **details)
    return unit, tuple(factors)
