"""Certify the unrepaired parent cover from its maps and owned regions."""
from functools import lru_cache
from .binomial_kernel import run

from .blowup_ideal import radical_subset
from .centre_weights import centre_rows
from .fan_geometry import _determinant, fan_cover
from .kernel_json import kernel_json
from .polynomial import integer_from_decimal, kernel_fraction


def _cone(chart, names):
    """Group monomials of the original coordinate functions by their common coefficient column."""
    mapping = dict(chart.coordinates)
    if (chart.coefficient_field or chart.ambient or len(chart.variables) != len(names)
            or len(mapping) != len(chart.coordinates) or mapping.keys() != set(names)):
        return None
    cone = kernel_json("atlas_repair_geometry", "parent_cone",
                       images=centre_rows(chart.variables, (mapping[name] for name in names)))
    if cone is None:
        return None
    linear = tuple(tuple(map(kernel_fraction, row)) for row in cone["linear"])
    rays = tuple(tuple(map(integer_from_decimal, ray)) for ray in cone["rays"])
    return linear, rays


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

    ray_ids, charts = {}, []
    for chart, cone in zip(parents, cones):
        coords = chart.variables
        maps = tuple(f for _, f in chart.coordinates)
        if not (radical_subset(coords, maps, chart.fibre) and radical_subset(coords, chart.fibre, maps)):
            return False
        charts.append(dict(rays=[ray_ids.setdefault(ray, len(ray_ids)) for ray in cone],
                           faces=[list(map(int, row.split())) for row in run(f"origin_orbits {len(names)}", *cone)],
                           region=centre_rows(coords, chart.region)))
    return kernel_json("atlas_repair_geometry", "parent_partition", charts=charts)


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
