"""Irreducible factorizations in ordered rational polynomial rings."""

from math import gcd

from . import native_polynomial as np
from .affine_kernel import _request, _ring
from .algebra_execution import cpu_stage
from .polynomial import InputError, integer_from_decimal, sparse_row


def _leading_coefficient(terms, order):
    if order == "lex":
        return terms[max(terms)]
    head = max(terms, key=lambda e: (sum(e), tuple(-i for i in reversed(e))))
    return terms[head]


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
    if polynomial.is_number:
        with cpu_stage("polynomial_factorization", algorithm="rational_constant"):
            return polynomial.scalar, ()

    terms = np.terms(variables, polynomial)
    # Leading coefficients multiply in both supported orders. Once every
    # factor is monic, the unit is the leading coefficient of the input.
    unit = _leading_coefficient(terms, order)
    if len(terms) == 2 and gcd(*(a - b for a, b in zip(*terms))) == 1:
        with cpu_stage("polynomial_factorization", algorithm="primitive_binomial", order=order):
            # A primitive exponent difference is part of a lattice basis.
            # In Laurent coordinates the binomial is a unit times T-c.
            # Removing its common monomial leaves an irreducible in QQ[x].
            common = tuple(map(min, *terms))
            reduced = {tuple(a - b for a, b in zip(e, common)): c / unit
                       for e, c in terms.items()}
            factors = [(v, m) for v, m in zip(variables, common) if m]
            factors.append((np.from_terms(variables, reduced), 1))
    else:
        with cpu_stage("polynomial_factorization", algorithm="flint", order=order):
            ring = _ring(variables)
            rows = _request(["factor", ring.header, ring.text(polynomial)])
            factors = []
            for exponent, row in zip(rows[1::2], rows[2::2]):
                factor = sparse_row(row)
                lead = _leading_coefficient(factor, order)
                monic = np.from_terms(variables, {e: c / lead for e, c in factor.items()})
                factors.append((monic, integer_from_decimal(exponent)))
    factors.sort(key=lambda part: np.default_sort_key(part[0]))
    return unit, tuple(factors)
