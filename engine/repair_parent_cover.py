"""Certify the unrepaired parent cover from its maps and owned regions."""
from functools import lru_cache
from .binomial_kernel import run

from .affine_kernel import frame_locus, sparse_terms
from .blowup_ideal import radical_subset
from .fan_geometry import _determinant, fan_cover


def _cone(chart, names):
    """Group monomials of the original coordinate functions by their common coefficient column."""
    mapping = dict(chart.coordinates)
    if (chart.coefficient_field or chart.ambient or len(chart.variables) != len(names)
            or len(mapping) != len(chart.coordinates) or mapping.keys() != set(names)):
        return None
    rows = [dict(sparse_terms(chart.variables, mapping[name])) for name in names]
    exponents = set().union(*rows)
    if len(exponents) != len(names):
        return None
    columns = {}
    for e in exponents:
        c = tuple(row.get(e, 0) for row in rows)
        if c in columns:
            return None
        columns[c] = e
    linear = tuple(sorted(columns))
    return linear, tuple(zip(*(columns[c] for c in linear)))


@lru_cache(maxsize=128)
def _parent_cover(parents):
    if not parents:
        return False
    names = tuple(sorted(name for name, _ in parents[0].coordinates))
    if not 1 <= len(names) <= 8:
        return False
    data = [_cone(chart, names) for chart in parents]
    if any(item is None for item in data):
        return False
    linear = data[0][0]
    if any(columns != linear for columns, _ in data) or not _determinant(linear):
        return False
    cones = tuple(cone for _, cone in data)
    if fan_cover(cones, len(names)) is None:
        return False

    # On a regular chart every torus orbit is given by zero coordinates
    # with all the others units. The set of their rays identifies this
    # very same orbit in every cone containing the face. The invertible
    # common linear map sends it to the origin precisely when every base
    # monomial has a positive exponent at some zero coordinate.
    owners = {}
    for chart, cone in zip(parents, cones):
        coords = chart.variables
        maps = tuple(f for _, f in chart.coordinates)
        if not (radical_subset(coords, maps, chart.fibre) and radical_subset(coords, chart.fibre, maps)):
            return False
        for row in run(f"origin_orbits {len(names)}", *cone):
            face = tuple(map(int, row.split()))
            rays = frozenset(cone[i] for i in face)
            owners.setdefault(rays, 0)
            zeros = tuple(coords[i] for i in face)
            units = tuple(v for v in coords if v not in zeros)
            if any(not frame_locus(coords, zeros, (*units,g)) for g in chart.region):
                # A valid fan partition owns all of this orbit or none of it.
                if not frame_locus(coords, (*zeros,*chart.region), units):
                    return False
                continue
            owners[rays] += 1
    return all(count == 1 for count in owners.values())


def certified_parent_cover(parents):
    """Verify a complete regular parent fan with exactly one owner per origin orbit.

    Each parent coordinate map is a vector of rational polynomials with
    exactly n distinct monomials and n independent coefficient columns.
    Aligning these columns across charts recovers the common linear change
    of base and their exponent columns give the rays in its torus lattice.
    No position in the saved list or parent ID is used as a geometric fact.
    The centres in 'removed' are filled in for this pre-repair proof.
    """
    return _parent_cover(tuple(parents))
