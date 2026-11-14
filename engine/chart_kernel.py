"""Native crossings and Euler calculations on one chart, using sparse polynomial rows.

The ordinal indices in the answers refer to the supplied list of divisor
ideals. Both expression-based and serialized charts use this protocol.
"""
from .native_library import ideal_request


def crossings_rows(n, dimension, ambient, divisors, fibre, region):
    """Rank-defect loci, their divisor indices and whether they meet the owned region."""
    sizes = "".join(f" {len(ideal)}" for ideal in divisors)
    header = (f"crossings {n} {dimension} {len(ambient)} {len(divisors)}{sizes} "
              f"{len(fibre)} {len(region)}")
    rows = ideal_request(header, [*ambient, *(g for ideal in divisors for g in ideal), *fibre, *region])
    found, at = [], 0
    while at < len(rows) and rows[at] != "none":
        fields = rows[at].split()
        size = int(fields[1])
        count = int(fields[3 + size])
        found.append((tuple(map(int, fields[2:2 + size])), fields[2 + size] == "1",
                      rows[at + 1:at + 1 + count]))
        at += 1 + count
    return found


def strata_rows(n, base, ideals, removed):
    """Euler characteristics on V(base) outside removed loci, by exact component set."""
    sizes = "".join(f" {len(ideal)}" for ideal in ideals)
    removed_sizes = "".join(f" {len(ideal)}" for ideal in removed)
    header = f"strata {n} {len(base)} {len(ideals)}{sizes} {len(removed)}{removed_sizes}"
    rows = ideal_request(header, [*base, *(g for ideal in (*ideals, *removed) for g in ideal)])
    found = {}
    for row in rows:
        if row == "none":
            break
        fields = row.split()
        size = int(fields[1])
        found[tuple(map(int, fields[2:2 + size]))] = int(fields[2 + size])
    return found


def owned_strata(chart):
    """Compute Euler strata using equations restricted to the chart's owned fibre.

    A normal form has the same values as the original function on the closed
    base. If a base equation factors, partition its zero set by the first
    factor to vanish. Earlier factors are exclusions on later pieces.
    """
    from . import kernel
    from .affine_kernel import _sparse_expression, rational_factor_rows
    from .groebner import groebner

    variables = chart.variables
    base = tuple(dict.fromkeys((*chart.ambient, *chart.region, *chart.fibre)))
    ideals = [component[3] for component in chart.components]
    if not (chart.ambient or chart.region or chart.removed) or chart.torus:
        return kernel.strata(variables, base, ideals, chart.removed)

    basis = groebner(base, variables)
    generators = tuple(basis.exprs)
    if generators == (1,):
        return {}
    factors = ()
    if generators:
        shortest = min(generators, key=lambda g: len(str(g)))
        _, rows = rational_factor_rows(variables, shortest)
        if len(rows) > 1 or rows and rows[0][1] > 1:
            factors = tuple(_sparse_expression(variables, row) for row, _, _ in rows)
    result = {}
    removed = list(chart.removed)
    for factor in factors or (None,):
        part = groebner((*generators, factor), variables) if factor is not None else basis
        if tuple(part.exprs) == (1,):
            continue
        def restrict(ideals):
            return [tuple(dict.fromkeys(part.reduce(g)[1] for g in ideal)) for ideal in ideals]
        here = kernel.strata(variables, part.exprs, restrict(ideals), restrict(removed))
        if here is None:
            return None
        for labels, chi in here.items():
            result[labels] = result.get(labels, 0) + chi
        if factor is not None:
            removed.append((factor,))
    return {labels: chi for labels, chi in result.items() if chi}
