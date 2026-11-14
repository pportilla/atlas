"""The lattice and ownership of charts in a regular affine fan over the orthant."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from math import gcd
from operator import index

from .native_library import Unavailable, _library, affine_request
from .polynomial import checked_native, format_poly, sparse_row


def split_cone(item, dimension):
    """The optional cone in the split base coordinates, in chart-coordinate order."""
    if "split_cone" not in item:
        return ()
    from .polynomial import InputError
    cone = item["split_cone"]
    if (not isinstance(cone, (tuple, list)) or len(cone) != dimension or not dimension or
            any(not isinstance(ray, (tuple, list)) or len(ray) != dimension or
                any(type(x) != int or x < 0 for x in ray) or not any(ray) for ray in cone)):
        raise InputError("A split cone needs one nonnegative integer ray per base coordinate.")
    return tuple(map(tuple, cone))


def _determinant(matrix):
    n = len(matrix)
    names = [f"x{j}" for j in range(n)]
    units = [tuple(int(i == j) for j in range(n)) for i in range(n)]

    def native():
        rows = [format_poly(dict(zip(units, row)), names) for row in matrix]
        value, = affine_request(["minor", " ".join(names), " ".join(map(str, range(n))), *rows])
        return sparse_row(value).get((0,) * n, 0)

    def reference():
        from sympy import Matrix
        return int(Matrix(matrix).det())

    return checked_native(native, reference)


def _complete_reference(cones, dimension):
    """The positive sum of regular cones has multiplicity one.

    Intersect with sum(x_i)=1. Cancellation of all interior oriented facets
    makes the sum of simplex indicators constant on the open standard
    simplex. Their volumes sum to its volume, so that constant is one.
    The closed cones therefore cover the orthant and have disjoint interiors.
    This also supplies the complete orbit compactifications of internal rays.
    """
    canonical = tuple(tuple(sorted(cone)) for cone in cones)
    if len(set(canonical)) != len(canonical):
        return False
    facets, volume = Counter(), Fraction(0)
    for cone in canonical:
        if (len(cone) != dimension or any(len(ray) != dimension for ray in cone) or
                any(type(x) is not int or x < 0 for ray in cone for x in ray) or
                any(not any(ray) for ray in cone)):
            return False
        determinant = _determinant(cone)
        if determinant not in (-1, 1):
            return False
        weight = Fraction(1)
        for ray in cone:
            weight /= sum(ray)
        volume += weight
        for i in range(dimension):
            facets[cone[:i] + cone[i+1:]] += int(determinant)*(-1)**i
    return volume == 1 and all(
        value == 0 or any(all(ray[j] == 0 for ray in face) for j in range(dimension))
        for face, value in facets.items())


def excluded_faces(cone, earlier):
    """The minimal coordinate products vanishing off all earlier overlaps."""
    cone = tuple(cone)
    excluded = {frozenset(i for i, ray in enumerate(cone) if ray not in other) for other in earlier}
    sets = sorted(excluded, key=lambda indices: (len(indices), sorted(indices)))
    minimal = []
    for indices in sets:
        if not any(previous <= indices for previous in minimal):
            minimal.append(indices)
    return minimal


@lru_cache(maxsize=128)
def fan_cover(cones, dimension):
    """Return the minimal excluded-coordinate products of every chart of a complete fan."""
    if any(len(cone) != dimension or any(len(ray) != dimension or
           any(type(x) is not int for x in ray) for ray in cone) for cone in cones):
        return None

    def reference():
        if not _complete_reference(cones, dimension):
            return None
        earlier, regions = [], []
        for cone in cones:
            regions.append(tuple(excluded_faces(cone, earlier)))
            earlier.append(frozenset(cone))
        return tuple(regions)

    def native():
        if not 1 <= dimension <= 8 or not hasattr(_library(), "atlas_fan_cover_abi"):
            raise Unavailable("Native fan certification is unavailable for this dimension or kernel.")
        rays = dict.fromkeys(ray for cone in cones for ray in cone)
        indices = {ray: i for i, ray in enumerate(rays)}
        header, *rows = affine_request(["fan_cover", f"{dimension} {len(rays)}",
                                        *(" ".join(map(str, ray)) for ray in rays),
                                        *(" ".join(str(indices[ray]) for ray in cone) for cone in cones)])
        if header == "invalid":
            return None
        if header != "complete" or len(rows) != len(cones):
            raise ArithmeticError("Invalid fan cover returned by the kernel.")
        def region(row):
            parts = [tuple(i for i in range(dimension) if mask & (1 << i)) for mask in map(int, row.split())]
            return tuple(map(frozenset, sorted(parts, key=lambda p: (len(p), p))))
        return tuple(region(row) for row in rows)

    return checked_native(native, reference)


def complete_fan(cones, dimension):
    return fan_cover(cones, dimension) is not None


def _reference_lattice_pivots(support):
    """Fraction-free echelon form of exponent differences. Return pivot columns."""
    origin = support[0]
    basis = {}
    for exponent in support[1:]:
        row = [v-u for u,v in zip(origin, exponent)]
        for pivot, previous in sorted(basis.items()):
            if row[pivot]:
                a, b = previous[pivot], row[pivot]
                row = [a*x-b*y for x,y in zip(row, previous)]
                content = gcd(*row)
                if content > 1:
                    row = [x//content for x in row]
        pivot = next((i for i,x in enumerate(row) if x), None)
        if pivot is not None:
            basis[pivot] = row
            if len(basis) == len(origin):
                break
    return sorted(basis)


def lattice_pivots(exponents):
    """Indices of the pivot columns in the support's affine difference space."""
    rows = tuple(tuple(map(index, row)) for row in exponents)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Expected nonempty Laurent support with a common vector length.")

    def native():
        result = affine_request(["lattice_pivots", str(len(rows[0])), *(" ".join(map(str,r)) for r in rows)])
        return list(map(int, result))

    return checked_native(native, lambda: _reference_lattice_pivots(rows))


def _lattice_images(rays, exponents, *, translate=False):
    rays, exponents = (tuple(tuple(map(index, row)) for row in rows) for rows in (rays, exponents))
    n = len(rays)
    if not n or any(len(r) != n for r in (*rays, *exponents)):
        raise ValueError("Expected a nonempty square matrix of rays and matching exponent vectors.")

    def native():
        op = "lattice_support" if translate else "lattice_characters"
        rows = affine_request([op, str(n), *(" ".join(map(str,r)) for r in (*rays, *exponents))])
        return tuple(tuple(map(int, row.split())) for row in rows)

    def reference():
        import sympy as sp
        matrix = sp.Matrix(rays)
        if abs(matrix.det()) != 1:
            raise ArithmeticError("Expected a unimodular matrix of rays.")
        inverse = matrix.inv()
        result = tuple(tuple(map(int, inverse*sp.Matrix(e))) for e in exponents)
        if translate and result:
            origin = tuple(map(min, zip(*result)))
            result = tuple(tuple(v-m for v,m in zip(row, origin)) for row in result)
        return result

    return checked_native(native, reference)


def lattice_characters(rays, exponents):
    """Recover each character a in the common lattice from its local exponents R*a."""
    return _lattice_images(rays, exponents)


def canonical_laurent_terms(rays, terms):
    """Express Laurent terms in the common lattice modulo monomial translation."""
    support = [(e,c) for e,c in terms.items() if c]
    images = _lattice_images(rays, (e for e,_ in support), translate=True)
    return {a:c for a, (_,c) in zip(images, support)}
