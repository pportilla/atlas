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
