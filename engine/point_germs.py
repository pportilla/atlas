"""Sparse generators of a rational ideal in coordinates centered at a point."""

from math import lcm, prod

from . import native_polynomial as np
from .local_mora import _divides


def _translate(variables, shifts, polynomials):
    if not any(shifts):
        return polynomials
    mapping = {v: v + a for v, a in zip(variables, shifts) if a}
    return tuple(f.xreplace(mapping) for f in polynomials)


def nonzero_terms_at_point(shifts, terms):
    """Decide exact rational vanishing without expanding equal or reciprocal powers."""
    if not any(shifts):
        return bool(terms.get((0,) * len(shifts)))
    active = [(e, c) for e, c in terms.items() if c and all(not k or a for k, a in zip(e, shifts))]
    if len(active) < 2:
        return bool(active)

    # Use one positive base >= 1 for each pair of reciprocal magnitudes.
    # Signs and powers of +/-1 then belong entirely to the coefficient.
    indices = {}
    axes = []
    for i, a in enumerate(shifts):
        magnitude = abs(a)
        if magnitude and magnitude != 1:
            base = magnitude if magnitude > 1 else 1 / magnitude
            j = indices.setdefault(base, len(indices))
            axes.append((i, j, 1 if magnitude > 1 else -1))
    negative = [i for i, a in enumerate(shifts) if a < 0]
    grouped = {}
    for powers, c in active:
        exponents = [0] * len(indices)
        for i, j, direction in axes:
            exponents[j] += direction * powers[i]
        key = tuple(exponents)
        sign = -1 if sum(powers[i] for i in negative) % 2 else 1
        grouped[key] = grouped.get(key, 0) + sign * c
    active = [(e, c) for e, c in grouped.items() if c]
    if len(active) < 2:
        return bool(active)
    if len({c < 0 for _, c in active}) == 1:
        return True

    # Factor shared powers, including negative ones from reciprocal axes.
    # The remaining exponents are nonnegative for both modular and exact evaluation.
    minimum = tuple(map(min, zip(*(e for e, _ in active))))
    active = [(tuple(k - m for k, m in zip(e, minimum)), c) for e, c in active]
    if any(k.bit_length() > 16 for e, _ in active for k in e):
        # A nonzero residue certifies a nonzero rational evaluation.
        common = lcm(*(c.denominator for _, c in active))
        for prime in (1000000007, 1000000009, 998244353):
            if any(base.denominator % prime == 0 for base in indices):
                continue
            residues = [base.numerator * pow(base.denominator, -1, prime) % prime for base in indices]
            value = sum((common // c.denominator) * c.numerator *
                        prod(pow(v, k, prime) for v, k in zip(residues, e) if k) for e, c in active)
            if value % prime:
                return True
    return bool(sum(c * prod(base**k for base, k in zip(indices, e) if k) for e, c in active))


def _nonzero_at_point(variables, point, f):
    return nonzero_terms_at_point(point, np.terms(variables, f))


def _point_generator(variables, shifts, f):
    terms = np.terms(variables, f)
    common = tuple(map(min, zip(*terms)))
    powers = tuple(0 if a else e for a, e in zip(shifts, common))
    if len(terms) == 1:
        return powers, np.S.One
    if any(common):
        terms = {tuple(k - m for k, m in zip(e, common)): c for e, c in terms.items()}
    if nonzero_terms_at_point(shifts, terms):
        return powers, np.S.One
    return powers, np.from_terms(variables, terms) if any(common) else f


def _with_powers(variables, powers, f):
    if not any(powers):
        return f
    if f == 1:
        return np.from_terms(variables, {powers: 1})
    terms = {tuple(k + m for k, m in zip(e, powers)): c for e, c in np.terms(variables, f).items()}
    return np.from_terms(variables, terms)


def _large_translation(variables, shifts, f, *, min_terms=65537):
    # A shifted monomial expands into the product of its binomial term counts.
    return any(prod(k + 1 for k, a in zip(powers, shifts) if a) >= min_terms
               for powers in np.terms(variables, f))


def centered_ideal(variables, shifts, generators):
    """Return generators of the same local ideal after translating the point to zero.

    A power of a coordinate nonzero at the point is a unit. Coordinates zero
    there are unchanged by translation, so their common powers can be saved
    and reattached afterwards, even when their exponents are very large.
    """
    variables, shifts = tuple(variables), tuple(shifts)
    generators = tuple(f for f in generators if f)
    if any(f.is_number for f in generators):
        return (np.S.One,)

    # Defer translations until every generator has been inspected: a later
    # unit makes even a very large expansion of an earlier row unnecessary.
    rows = []
    monomials = set()
    for generator in generators:
        powers, f = _point_generator(variables, shifts, generator)
        if f == 1:
            if not any(powers):
                return (np.S.One,)
            if powers in monomials:
                continue
            monomials.add(powers)
        rows.append((powers, f))

    centered = []
    shifted = any(shifts)
    try_tangent_rank = shifted and len(generators) >= len(variables)
    for powers, f in rows:
        if f != 1 and monomials:
            # Discard multiples of another generator's proven monomial. Its
            # support is on unshifted axes, so the same deletion works before
            # translation. Account for our own saved powers in divisibility.
            original = np.terms(variables, f)
            divisors = [tuple(max(0, i - j) for i, j in zip(m, powers)) for m in monomials]
            terms = {e: c for e, c in original.items() if not any(_divides(m, e) for m in divisors)}
            if len(terms) < len(original):
                if not terms:
                    continue
                extra, f = _point_generator(variables, shifts, np.from_terms(variables, terms))
                powers = tuple(i + j for i, j in zip(powers, extra))
        if f != 1 and shifted:
            # Even moderate degrees can require an expensive homogenized basis.
            # Test from a shifted eighth power or a pair of shifted squares.
            if try_tangent_rank and _large_translation(variables, shifts, f, min_terms=9):
                from .point_tangent import maximal_ideal_from_linear_parts
                # Every original generator vanishes. Full rank of their
                # linear parts at this point gives the maximal ideal by Nakayama.
                if maximal_ideal_from_linear_parts(variables, shifts, generators):
                    return variables
                try_tangent_rank = False
            f, = _translate(variables, shifts, (f,))
        centered.append(_with_powers(variables, powers, f))
    return tuple(centered)
