"""The lattice and ownership of charts in a regular affine fan over the orthant."""
from functools import lru_cache
from operator import index

from .native_library import Unavailable, _library, affine_request
from .polynomial import format_poly, integer_from_decimal, integer_text, sparse_row


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

    rows = [format_poly(dict(zip(units, row)), names) for row in matrix]
    value, = affine_request(["minor", " ".join(names), " ".join(map(str, range(n))), *rows])
    return sparse_row(value).get((0,) * n, 0)


def excluded_faces(cone, earlier):
    """The minimal coordinate products vanishing off all earlier overlaps."""
    from .binomial_kernel import run
    incidence = [tuple(int(ray in other) for ray in cone) for other in earlier]
    if not incidence:
        return []
    rows = run(f"excluded_faces {len(cone)} {len(incidence)}", *incidence)
    return [frozenset(map(int, row.split())) for row in rows]


@lru_cache(maxsize=128)
def fan_cover(cones, dimension):
    """Return the minimal excluded-coordinate products of every chart of a complete fan."""
    if any(len(cone) != dimension or any(len(ray) != dimension or
           any(type(x) is not int for x in ray) for ray in cone) for cone in cones):
        return None

    if not 1 <= dimension <= 8 or not hasattr(_library(), "atlas_fan_cover_abi"):
        raise Unavailable("Native fan certification is unavailable for this dimension or kernel.")
    rays = dict.fromkeys(ray for cone in cones for ray in cone)
    indices = {ray: i for i, ray in enumerate(rays)}
    header, *rows = affine_request(["fan_cover", f"{dimension} {len(rays)}",
                                    *(" ".join(map(integer_text, ray)) for ray in rays),
                                    *(" ".join(str(indices[ray]) for ray in cone) for cone in cones)])
    if header == "invalid":
        return None
    if header != "complete" or len(rows) != len(cones):
        raise ArithmeticError("Invalid fan cover returned by the kernel.")
    def region(row):
        parts = [tuple(i for i in range(dimension) if mask & (1 << i)) for mask in map(int, row.split())]
        return tuple(map(frozenset, sorted(parts, key=lambda p: (len(p), p))))
    return tuple(region(row) for row in rows)


def complete_fan(cones, dimension):
    return fan_cover(cones, dimension) is not None


def lattice_pivots(exponents):
    """Indices of the pivot columns in the support's affine difference space."""
    rows = tuple(tuple(map(index, row)) for row in exponents)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Expected nonempty Laurent support with a common vector length.")

    result = affine_request(["lattice_pivots", str(len(rows[0])), *(" ".join(map(integer_text,r)) for r in rows)])
    return list(map(int, result))


def _lattice_images(rays, exponents, *, translate=False):
    rays, exponents = (tuple(tuple(map(index, row)) for row in rows) for rows in (rays, exponents))
    n = len(rays)
    if not n or any(len(r) != n for r in (*rays, *exponents)):
        raise ValueError("Expected a nonempty square matrix of rays and matching exponent vectors.")

    op = "lattice_support" if translate else "lattice_characters"
    rows = affine_request([op, str(n), *(" ".join(map(integer_text,r)) for r in (*rays, *exponents))])
    return tuple(tuple(map(integer_from_decimal, row.split())) for row in rows)

def lattice_characters(rays, exponents):
    """Recover each character a in the common lattice from its local exponents R*a."""
    return _lattice_images(rays, exponents)


def canonical_laurent_terms(rays, terms):
    """Express Laurent terms in the common lattice modulo monomial translation."""
    support = [(e,c) for e,c in terms.items() if c]
    images = _lattice_images(rays, (e for e,_ in support), translate=True)
    return {a:c for a, (_,c) in zip(images, support)}
