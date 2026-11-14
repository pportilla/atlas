"""Euler integration on coordinate tori using the character lattice of the equations."""
from functools import lru_cache
from itertools import combinations
from math import prod

from .affine_kernel import _polynomial, frame_linear, sparse_terms
from .constructible_euler import euler_characteristic
from .groebner import groebner


def _basis(differences, dimension):
    """Triangular integral row basis, or None if the characters have smaller rank."""
    rows = sorted(set(row for row in differences if any(row)))
    rows = [list(row) for row in rows]
    for j in range(dimension):
        pivot = next((i for i in range(j, len(rows)) if rows[i][j]), None)
        if pivot is None:
            return None
        rows[j], rows[pivot] = rows[pivot], rows[j]
        for i in range(j + 1, len(rows)):
            while rows[i][j]:
                q = rows[j][j] // rows[i][j]
                rows[j], rows[i] = rows[i], [a - q*b for a, b in zip(rows[j], rows[i])]
        if rows[j][j] < 0:
            rows[j] = [-a for a in rows[j]]
        for i in range(j):
            q = (rows[i][j] + rows[j][j]//2) // rows[j][j]
            rows[i] = [a - q*b for a, b in zip(rows[i], rows[j])]
    return rows[:dimension]


def _differences(support):
    first = next(iter(support))
    return (tuple(v - w for v, w in zip(m, first)) for m in support)


def benefits_from_lattice(variables, ideals):
    """Select the projection when a divisor has full-rank, nonprimitive support."""
    for ideal in ideals:
        for f in ideal:
            support = [m for m, _ in sparse_terms(variables, f)]
            if len(support) < 3:
                continue
            differences = list(_differences(support))
            active = [i for i in range(len(variables)) if any(m[i] for m in differences)]
            if not 2 <= len(active) <= 5:
                continue
            matrix = _basis((tuple(m[i] for i in active) for m in differences), len(active))
            if matrix and any(matrix[j][j] > 1 for j in range(len(active))):
                return True
    return False


def _pullback_equation(variables, terms, matrix):
    # Subtracting the exponent of a monomial divides the equation by a torus unit.
    coefficients = {}
    for residual, c in zip(_differences(terms), terms.values()):
        powers = []
        for i, row in enumerate(matrix):
            q, rem = divmod(residual[i], row[i])
            if rem:
                raise ArithmeticError("Character outside the exponent lattice.")
            powers.append(q)
            residual = [p - q*b for p, b in zip(residual, row)]
        coefficients[tuple(powers)] = c
    minimum = tuple(map(min, zip(*coefficients)))
    return _polynomial({tuple(p - m for p, m in zip(power, minimum)): c
                        for power, c in coefficients.items()}, variables)


def _torus_sum(variables, equations):
    terms = [dict(sparse_terms(variables, f)) for f in equations]
    active = tuple(i for i in range(len(variables)) if any(m[i] for p in terms for m in p))
    # Unused affine coordinates contribute chi(A^1)=1. Avoid exponential
    # enumeration in high dimension; the ordinary projector is valid there.
    if len(active) > 5:
        return euler_characteristic(variables, equations)
    total = 0
    for size in range(len(active) + 1):
        for axes in combinations(active, size):
            zero = tuple(i for i in active if i not in axes)
            cuts = [{tuple(m[i] for i in axes): c for m, c in p.items() if all(m[i] == 0 for i in zero)}
                    for p in terms]
            if any(len(cut) == 1 for cut in cuts):
                continue  # a nonzero monomial never vanishes on this torus
            cuts = [cut for cut in cuts if cut]
            if not axes:
                total += 1
                continue
            matrix = _basis((m for cut in cuts for m in _differences(cut)), size)
            if matrix is None:
                continue  # each fibre has a positive-dimensional torus factor, of chi zero
            coords = tuple(variables[i] for i in axes)
            transformed = [_pullback_equation(coords, cut, matrix) for cut in cuts]
            degree = prod(row[i] for i, row in enumerate(matrix))
            # The monomial isogeny has this many points in each fibre.
            total += degree * euler_characteristic(coords, transformed, [(x,) for x in coords])
    return total


def strata(variables, base, ideals, removed):
    """Euler characteristics on exactly each subset of the divisors, off the removed loci."""
    @lru_cache(maxsize=512)
    def closed(ring, equations):
        generators = tuple(groebner(equations, ring).exprs)
        if generators == (1,):
            return 0
        for f in generators:
            for x in ring:
                linear = frame_linear(ring, f, x)
                if linear is not None and not linear[0].free_symbols:
                    rest = [g for g in generators if g != f]
                    if not rest:
                        return 1
                    value = -linear[1]/linear[0]
                    return closed(tuple(v for v in ring if v != x),
                                  tuple(g.xreplace({x: value}) for g in rest))
        return _torus_sum(ring, generators)

    indices = range(len(ideals))
    values = {}
    for size in range(len(ideals), 0, -1):
        for subset in combinations(indices, size):
            equations = tuple(dict.fromkeys((*base, *(g for i in subset for g in ideals[i]))))
            value = 0
            for k in range(len(removed) + 1):
                for excluded in combinations(removed, k):
                    cut = tuple(dict.fromkeys((*equations, *(g for ideal in excluded for g in ideal))))
                    value += (-1)**k * closed(variables, cut)
            # The closed intersection also contains the higher strata.
            value -= sum(chi for supers, chi in values.items() if set(subset).issubset(supers))
            if value:
                values[subset] = value
    return values
