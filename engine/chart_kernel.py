"""Native crossings and Euler calculations on one chart, using sparse polynomial rows.

The ordinal indices in the answers refer to the supplied list of divisor
ideals. Both expression-based and serialized charts use this protocol.
"""
from .native_library import ideal_request
from .polynomial import integer_from_decimal, integer_text


def crossings_rows(n, dimension, ambient, divisors, fibre, region, *, bounded=False):
    """Rank-defect loci, their divisor indices and whether they meet the owned region."""
    sizes = "".join(f" {len(ideal)}" for ideal in divisors)
    operation = "bounded_crossings" if bounded else "crossings"
    header = (f"{operation} {n} {dimension} {len(ambient)} {len(divisors)}{sizes} "
              f"{len(fibre)} {len(region)}")
    rows = ideal_request(header, [*ambient, *(g for ideal in divisors for g in ideal), *fibre, *region])
    return crossing_loci(rows)


def crossing_loci(rows):
    if rows == ["unknown"]:
        return None
    found, at = [], 0
    while at < len(rows) and rows[at] != "none":
        fields = rows[at].split()
        size = int(fields[1])
        count = int(fields[3 + size])
        found.append((tuple(map(int, fields[2:2 + size])), fields[2 + size] == "1",
                      rows[at + 1:at + 1 + count]))
        at += 1 + count
    return found


def strata_rows(n, base, ideals, removed, *, monomial_only=False):
    """Euler characteristics by exact component set, or None for an undecided monomial-only request."""
    sizes = "".join(f" {len(ideal)}" for ideal in ideals)
    removed_sizes = "".join(f" {len(ideal)}" for ideal in removed)
    operation = "monomial_strata" if monomial_only else "strata"
    header = f"{operation} {n} {len(base)} {len(ideals)}{sizes} {len(removed)}{removed_sizes}"
    rows = ideal_request(header, [*base, *(g for ideal in (*ideals, *removed) for g in ideal)])
    if monomial_only and rows == ["undetermined"]:
        return None
    found = {}
    for row in rows:
        if row == "none":
            break
        fields = row.split()
        size = int(fields[1])
        found[tuple(map(int, fields[2:2 + size]))] = integer_from_decimal(fields[2 + size])
    return found


class InconsistentSheets(ArithmeticError):
    """A geometric count cannot be divided among the marked embeddings."""


def strata_totals(contributions):
    """Add disjoint stratum counts on the supplied coefficient-field sheets."""
    indices, labels, rows = {}, [], []
    for label, degree, chi, *orientation in contributions:
        sign, = orientation or (1,)
        if label not in indices:
            indices[label] = len(labels)
            labels.append(label)
        rows.append(f"{indices[label]} {integer_text(degree)} {integer_text(chi)} {sign}")
    if not rows:
        return {}
    totals = ideal_request("strata_totals", rows)
    if totals == ["inconsistent"]:
        raise InconsistentSheets("A geometric count is inconsistent across coefficient field embeddings.")
    result = {}
    for row in totals:
        key, value = row.split()
        result[labels[int(key)]] = integer_from_decimal(value)
    return result


def owned_strata(chart):
    """Compute exact component strata on the closed fibre of this chart's owned region."""
    from . import kernel
    base = tuple(dict.fromkeys((*chart.ambient, *chart.region, *chart.fibre)))
    return kernel.strata(chart.variables, base, [c[3] for c in chart.components], chart.removed)
