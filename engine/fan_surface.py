"""Geometric boundary curves on a smooth surface in a regular threefold fan.

For an internal ray, its divisor is a toric surface over the origin. In a
maximal cone its intersection with the strict surface is a plane curve. A
coordinate factor is the orbit closure of the corresponding two-dimensional
cone. Every other geometrically irreducible factor has a dense open in the
ray's torus. Two such curves agree precisely when their Laurent equations in
the common character lattice differ by a unit. These two descriptions glue
the actual curves, including distinct branches in the same ambient divisor.

The Jacobian criterion on the strict surface verifies smooth curves,
transverse pairs and absence of triples on each owned region. Their finite
pair intersections are reduced, so their Euler characteristics count edges.
"""
from collections import Counter, defaultdict
from functools import lru_cache

from .atlas_torus import _normalized
from .chart_kernel import crossings_rows, strata_rows
from .fan_geometry import canonical_laurent_terms
from .native_library import affine_request, ideal_request
from .polynomial import format_poly, sparse_row


_AXES = ("x", "y", "z")
_UNITS = tuple(tuple(int(i == j) for j in range(3)) for i in range(3))


def _row(terms):
    return ";".join(f"{c} " + " ".join(map(str, e)) for e, c in terms)


@lru_cache(maxsize=4096)
def _factors(terms, index):
    restricted = {e: c for e, c in terms if not e[index]}
    if not restricted:
        return None  # an entire ambient divisor lies in the surface
    if len(restricted) == 1:
        powers = next(iter(restricted))
        return tuple((j, ((_UNITS[j], 1),)) for j, k in enumerate(powers) if k)
    _, *rows = affine_request(["geometric_factors", " ".join(_AXES), format_poly(restricted, _AXES)])
    factors = []
    for header, row in zip(rows[::2], rows[1::2]):
        _, absolute = map(int, header.split())
        factor = tuple(sorted(sparse_row(row).items()))
        coordinate = _UNITS.index(factor[0][0]) if len(factor) == 1 and factor[0][0] in _UNITS else None
        if coordinate is None and not absolute:
            count = ideal_request("plane_components 3 2 0", [_row(((_UNITS[index], 1),)), row])
            if count != ["1 1"]:
                return None
        factors.append((coordinate, factor))
    return tuple(factors)


def surface_complex(charts):
    groups, parents, counts = defaultdict(list), {}, Counter()
    for rays, boundary, terms, owned in charts:
        pieces = {}
        for i, label in boundary.items():
            factors = _factors(terms, i)
            if factors is None:
                return None
            for coordinate, factor in factors:
                if coordinate is None:
                    key = rays[i], _normalized(canonical_laurent_terms(rays, dict(factor)))
                else:
                    key = frozenset((rays[i], rays[coordinate]))
                if parents.setdefault(key, label) != label or key in pieces:
                    # A curve lying in two distinct exceptional divisors
                    # needs a common component label in the reduced boundary.
                    return None
                if key not in groups[label]:
                    groups[label].append(key)
                pieces[key] = (_row(((_UNITS[i], 1),)), _row(factor))
        strict = _row(terms)
        if crossings_rows(3, 2, [strict], list(pieces.values()), owned, ()):
            return None
        keys = list(pieces)
        for indices, value in strata_rows(3, [strict, *owned], list(pieces.values()), ()).items():
            if len(indices) == 2:
                counts[frozenset(keys[i] for i in indices)] += value
    names = {key: label if len(group) == 1 else f"{label}.{i}"
             for label, group in sorted(groups.items()) for i, key in enumerate(group, 1)}
    if len(set(names.values())) != len(names):
        return None
    branches_meet = any(len({parents[k] for k in pair}) == 1 for pair in counts)
    edges, numbering = [], Counter()
    parent_by_name = {name: parents[key] for key, name in names.items()}
    pairs = sorted((sorted(names[k] for k in pair), count) for pair, count in counts.items())
    for (u, v), count in pairs:
        p, q = (u, v) if branches_meet else sorted((parent_by_name[u], parent_by_name[v]))
        for _ in range(count):
            numbering[p, q] += 1
            edges.append(dict(id=f"C{p}-{q}.{numbering[p, q]}", dimension=1, vertices=[u, v], faces=[u, v]))
    return {"status": "complete", "vertices": [dict(id=name, dimension=1) for name in names.values()],
            "cells": edges, "convention": "Exceptional divisor on the smooth strict transform in the germ over the origin over Qbar."}
