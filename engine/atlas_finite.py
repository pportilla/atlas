"""Glue strata over a single finite set of points on an exceptional divisor.

Suppose a ratio of two original coordinates has a regular representative
t on every chart of a divisor D. The denominator is a nonzerodivisor in
each chart, so these representatives agree on overlaps by density. A lex
basis of a section can exhibit its coordinate algebra as Q[t,u_1,...,u_r]/
(p(t)), with the other coordinates monic linear graphs. For squarefree p
the section is one affine space per root, over the algebraic closure.

We require this presentation on every nonempty chart of every intersection
involving D, and ideal containment proves that the entire section maps to
the origin. The rational prime obtained by imposing one irreducible factor
of p contains a chart's ownership ideal precisely when each geometric
conjugate is generically owned there, unless a removed centre contains it.
The owned pieces partition the fibre, so exactly one owner for each root
proves one global component per root. Its faces carrying D have that same
value of t. Every other incident face must already be uniquely determined.
"""
from dataclasses import replace
from itertools import combinations, permutations

from .chart_expressions import chart_expressions

from .affine_kernel import _sparse_expression, rational_factor_rows, sparse_terms
from .blowup_ideal import ideal_subset, saturate
from .groebner import groebner
from .local_blowups import _is_strict, _own_failures


def _finite_base(chart, ideal, t, dimension):
    variables = tuple(v for v in chart.variables if v != t) + (t,)
    basis = groebner(ideal, variables, order="lex")
    if tuple(basis.exprs) == (1,):
        return chart_expressions(chart.variables).S.One
    polynomial = None
    pivots = set()
    for row in basis.exprs:
        head = sparse_terms(variables, row)[0][0]
        if not any(head[:-1]):
            if polynomial is not None or head[-1] < 2:
                return None
            polynomial = row
        else:
            # Lexicographic order makes this a monic graph in its head
            # coordinate with a polynomial in later coordinates as tail.
            if sum(head) != 1:
                return None
            pivots.add(head.index(1))
    return polynomial if len(variables)-1-len(pivots) == dimension else None


def _parameter(chart, numerator, denominator, boundary, dimension):
    mapping = dict(chart.coordinates)
    if numerator not in mapping or denominator not in mapping:
        return None
    a, b = mapping[numerator], mapping[denominator]
    for v in chart.variables:
        if not ideal_subset(chart.variables, (v*b-a,), chart.ambient):
            continue
        if not ideal_subset(chart.variables, saturate(chart.variables, chart.ambient, b), chart.ambient):
            return None
        p = _finite_base(chart, (*chart.ambient, *boundary), v, dimension-1)
        if p is not None and p != 1:
            return v, p
    return None


def _owners(chart, boundary, parameter, polynomial, factors, keys, dimension):
    """Count generic owners for each rational factor. Return None unless
    every section lies over the origin and has the expected presentation."""
    owners = {}
    for key in keys:
        if not key <= boundary.keys():
            continue
        ideal = (*chart.ambient, *(g for label in sorted(key) for g in boundary[label]))
        p = _finite_base(chart, ideal, parameter, dimension-len(key))
        if p == 1:
            continue
        if p != polynomial or not ideal_subset(chart.variables, chart.fibre, ideal):
            return None
        owners[key] = []
        for f in factors:
            prime = (*ideal, f)
            present = not any(ideal_subset(chart.variables, hole, prime) for hole in chart.removed)
            owners[key].append(present and ideal_subset(chart.variables, chart.region, prime))
    return owners


def _complex(counts, label, degree, dimension):
    names, vertices, cells = {}, [], []
    for key in sorted(counts, key=lambda ids: (len(ids), sorted(ids))):
        labels = sorted(key)
        for i in range(degree if label in key else 1):
            if len(key) == 1:
                name = labels[0] + (f".{i+1}" if label in key else "")
                vertices.append({"id": name, "dimension": dimension-1})
            else:
                def face(ids):
                    ids = frozenset(ids)
                    return names[ids, i if label in ids else 0]
                name = "C" + "-".join(labels) + f".{i+1}"
                cells.append({"id": name, "dimension": len(key)-1, "vertices": [face((v,)) for v in labels],
                              "faces": [face(f) for f in combinations(labels, len(key)-1)]})
            names[key, i] = name
    return {"status": "complete", "vertices": vertices, "cells": cells,
            "convention": "Ambient exceptional divisor in the germ over the origin over Qbar."}


def finite_complex(charts, dimension, counts):
    """A single split divisor with a regular coordinate ratio on all its opens."""
    unknown = [next(iter(key)) for key, value in counts.items() if len(key) == 1 and value is None]
    if len(unknown) != 1 or any(chart.coefficient_field for chart in charts):
        return None
    label, = unknown
    if any(n != 1 for key, n in counts.items() if label not in key):
        return None
    if any(key-{v} not in counts for key in counts if len(key) > 1 for v in key):
        return None
    keys = [key for key in counts if label in key]
    supporting = [(c, {k: ideal for k, _, _, ideal in c.components if not _is_strict(k)})
                  for c in charts if any(k == label for k, _, _, _ in c.components)]
    if not supporting or any(
        len(c.coordinates) != dimension or len(dict(c.coordinates)) != dimension or
        set(c.fibre) != set(dict(c.coordinates).values()) for c, _ in supporting
    ):
        return None
    for numerator, denominator in permutations(dict(supporting[0][0].coordinates), 2):
        first_t = None
        for chart, boundary in supporting:
            found = _parameter(chart, numerator, denominator, boundary[label], dimension)
            if found is None:
                break
            t, p = found
            if first_t is None:
                first_t, first_p = t, p
                degree = sparse_terms((t,), p)[0][0][0]
                parts = rational_factor_rows((t,), p)[1]
                if any(e != 1 for _, e, _ in parts) or any(counts[k] not in (None, degree) for k in keys):
                    break
                roots = [_sparse_expression((t,), row) for row, _, _ in parts]
                owners = {key: [0]*len(roots) for key in keys}
            swap = {first_t: t}
            if p != first_p.xreplace(swap):
                break
            local = _owners(chart, boundary, t, p, [f.xreplace(swap) for f in roots], keys, dimension)
            if local is None:
                break
            for key, row in local.items():
                owners[key] = [n+int(yes) for n, yes in zip(owners[key], row)]
        else:
            if all(n == 1 for row in owners.values() for n in row):
                for chart in charts:
                    boundary = tuple(row for row in chart.components if not _is_strict(row[0]))
                    if _own_failures(replace(chart, components=boundary), dimension, {})[0]:
                        return None
                return _complex(counts, label, degree, dimension)
    return None
