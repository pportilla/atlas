"""Standard monomials of a rational polynomial quotient from an initial ideal."""

from math import prod
from operator import index


def _components(width, generators):
    """Group coordinates linked by the support of a mixed generator."""
    parent = list(range(width))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for row in generators:
        support = iter(i for i, exponent in enumerate(row) if exponent)
        first = root(next(support))
        for i in support:
            parent[root(i)] = first

    coordinates = {}
    for i in range(width):
        coordinates.setdefault(root(i), []).append(i)
    rows = {component: [] for component in coordinates}
    for row in generators:
        component = root(next(i for i, exponent in enumerate(row) if exponent))
        rows[component].append(tuple(row[i] for i in coordinates[component]))
    return [(tuple(group), rows[component]) for component, group in coordinates.items()]


def _standard_boxes(bounds, generators):
    """Partition the staircase into disjoint half-open boxes of exponents."""
    width = len(bounds)
    pending = [((0,) * width, bounds, 0)]
    while pending:
        lower, upper, start = pending.pop()
        for i in range(start, len(generators)):
            row = generators[i]
            if any(power >= end for power, end in zip(row, upper)):
                continue  # No multiple of this generator is in the box.
            # Exclude the upper orthant above row. Each surviving point has a
            # unique first coordinate at which its exponent is below row.
            corner = list(lower)
            for axis, power in enumerate(row):
                if corner[axis] < power:
                    end = upper[:axis] + (power,) + upper[axis + 1:]
                    pending.append((tuple(corner), end, i + 1))
                    corner[axis] = power
            break
        else:
            yield lower, upper


def _box_points(lower, upper):
    """Iterate with a mixed-radix counter, without materializing axis ranges."""
    point = list(lower)
    while True:
        yield tuple(point)
        for i in range(len(point) - 1, -1, -1):
            point[i] += 1
            if point[i] < upper[i]:
                break
            point[i] = lower[i]
        else:
            return


def _planar_length(bounds, generators):
    """Area of the staircase below sorted mixed generators, all inside bounds."""
    width, ceiling = bounds
    left = total = 0
    for x, y in generators:
        if y < ceiling:
            total += (x - left) * ceiling
            left, ceiling = x, y
    return total + (width - left) * ceiling


def quotient_monomials(width, leading, *, include_basis):
    """Return (vector space dimension, requested standard monomials).

    Infinite length gives (None, None). Over the coefficient field alone the
    basis of the zero ideal's quotient is [()]. A unit ideal has empty basis.
    """
    width = index(width)
    if width < 0:
        raise ValueError("A ring cannot have negative coordinate count.")
    bounds = [None] * width
    mixed = set()
    for powers in leading:
        row = tuple(map(index, powers))
        if len(row) != width or any(exponent < 0 for exponent in row):
            raise ValueError("Expected one nonnegative exponent per coordinate.")
        support = [i for i, exponent in enumerate(row) if exponent]
        if not support:
            return 0, [] if include_basis else None
        if len(support) == 1:
            i = support[0]
            if bounds[i] is None or row[i] < bounds[i]:
                bounds[i] = row[i]
        else:
            mixed.add(row)

    if any(bound is None for bound in bounds):
        # All pure powers of an unbounded variable survive.
        return None, None
    bounds = tuple(bounds)
    # Pure powers already exclude mixed generators outside their box.
    generators = sorted(row for row in mixed if all(e < b for e, b in zip(row, bounds)))

    if include_basis:
        # Split on actual generator thresholds so even a thin staircase in a
        # large bounding box visits only its surviving points.
        basis = [p for lower, upper in _standard_boxes(bounds, generators)
                 for p in _box_points(lower, upper)]
        # Stable sorting preserves descending coordinate order within each degree.
        basis.sort(reverse=True)
        basis.sort(key=sum)
        return len(basis), basis

    if not generators:
        return prod(bounds), None
    if width == 2:
        return _planar_length(bounds, generators), None
    # For disjoint sets of coordinates the quotient is a tensor product.
    # Multiply component lengths without traversing their Cartesian product.
    length = 1
    for coordinates, rows in _components(width, generators):
        component_bounds = tuple(bounds[i] for i in coordinates)
        if len(coordinates) == 2:
            # All exponents outside this component vanish, so projecting the
            # sorted generators preserves (x, y) order, smallest y first on ties.
            length *= _planar_length(component_bounds, rows)
        else:
            length *= sum(prod(end - start for start, end in zip(lower, upper))
                          for lower, upper in _standard_boxes(component_bounds, rows))
    return length, None
