"""Certify the maximal ideal from sparse linear terms at a rational point."""

from fractions import Fraction
from math import gcd, lcm

from . import native_polynomial as np


def _primitive_terms(variables, f):
    # Scaling a generator by a nonzero rational preserves the local ideal.
    # Remove scalar content so it cannot annihilate a row modulo a prime.
    terms = np.terms(variables, f)
    denominator = lcm(*(c.denominator for c in terms.values()))
    integers = [(e, c.numerator * (denominator // c.denominator)) for e, c in terms.items()]
    content = gcd(*(c for _, c in integers))
    return [(e, c // content) for e, c in integers]


def _linear_row(terms, point, prime):
    row = [0] * len(point)
    for powers, coefficient in terms:
        # At a zero coordinate only first powers can survive differentiation,
        # and the derivative must be taken in that coordinate.
        zero_degree = sum(e for a, e in zip(point, powers) if not a)
        if zero_degree > 1:
            continue
        value = coefficient
        for a, e in zip(point, powers):
            if a and e:
                if prime:
                    value = value * pow(a, e % (prime - 1), prime) % prime
                elif a == -1 and e % 2:
                    value = -value
        for i, (a, e) in enumerate(zip(point, powers)):
            if not e:
                continue
            if zero_degree:
                if not a:
                    row[i] += value
            else:
                # Scale each nonzero coordinate column by a. This removes
                # division from the logarithmic derivative e * value / a.
                row[i] += e * value
    return [x % prime for x in row] if prime else row


def _full_rank(rows, dimension, prime):
    pivots = {}
    for row in rows:
        for col, value in enumerate(row):
            if not value:
                continue
            if col not in pivots:
                scale = pow(value, -1, prime) if prime else Fraction(1, value)
                pivots[col] = [x * scale % prime if prime else x * scale for x in row]
                if len(pivots) == dimension:
                    return True
                break
            pivot = pivots[col]
            for j in range(col, dimension):
                entry = row[j] - value * pivot[j]
                row[j] = entry % prime if prime else entry
    return False


def maximal_ideal_from_linear_parts(variables, shifts, generators):
    """Prove full cotangent rank for generators known to vanish at shifts.

    Nakayama's lemma then gives the full maximal ideal in the local ring.
    A failed modular certificate simply leaves the usual basis computation
    in charge. At points with coordinates 0 or ±1 we can test rational rank
    exactly, even when enormous exponents are multiples of the trial primes.
    """
    if len(generators) < len(variables) or not variables:
        return False
    terms = [_primitive_terms(variables, f) for f in generators]
    if all(abs(a) in (0, 1) for a in shifts):
        rows = (_linear_row(row, shifts, None) for row in terms)
        return _full_rank(rows, len(variables), None)

    for prime in (1000000007, 1000000009, 998244353, 1004535809):
        if any(a.denominator % prime == 0 for a in shifts):
            continue
        point = [a.numerator * pow(a.denominator, -1, prime) % prime for a in shifts]
        rows = (_linear_row(row, point, prime) for row in terms)
        if _full_rank(rows, len(variables), prime):
            return True
    return False
