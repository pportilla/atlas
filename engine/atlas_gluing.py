"""Glue geometric intersection components on overlaps of rational affine charts.

On two affine opens of a separated resolution the diagonal is closed in
their product. Off the exceptional divisors it is cut out by equality of
the original coordinates, since the modification is an isomorphism there.
Its closure, obtained by saturation, therefore cuts out the entire overlap.
A single nonzero equation from each exceptional ideal suffices: inverting
their product removes every exceptional divisor and retains a dense open.
The diagonal of the overlap is irreducible whenever nonempty on one marked
embedding. For a chart with centres removed, intersect this closed diagonal
with the complement of all removed centres on both sides before using it.
An irreducible piece supported entirely in a deleted centre never appears.

For a fixed number field, adjoining one common parameter to every chart and
equating its two copies in the diagonal matches the chosen sheets. The
kernel can work on all embeddings together because the tests of nonempty
intersection and ideal containment commute with conjugacy. Local components
are used in this argument only when their ideals define a single geometric
prime on each sheet (a geometrically integral constant family or a verified
triangular graph over the field). The finite cover of several repair centres
does not select an embedding and therefore cannot use this identification.
For surface charts, a section M²=d of a coordinate flat with M a primitive
monomial and d a nonzero rational nonsquare supplies its own quadratic
splitting field. Sections in that square class then have two explicit
geometric primes whose identifications are tested on the same diagonal.

A smooth divisor intersection has disjoint geometric components. Hence two
local primes of that intersection belong to the same component precisely
when they meet on the diagonal, allowing a chain of overlaps. Gluing before
intersecting with the origin fibre retains the parts of a component that
join two visible charts. A face is the unique containing component of the
smaller intersection, tested by containment of ideals on a common chart.
"""
from collections import defaultdict
from dataclasses import replace
from functools import cache, lru_cache
from itertools import combinations
from math import gcd, isqrt

from . import native_polynomial as np
from .chart_expressions import chart_expressions
from .reference import sp

from .affine_kernel import _sparse_expression, decode, pullbacks, rational_factor_rows, sparse_terms
from .blowup_ideal import ideal_subset, saturate, smooth_locus
from .constructible_euler import euler_characteristic
from .geometric_components import geometric_components
from .groebner import groebner
from .local_blowups import _is_strict, _own_failures, _present


def _common_field(charts):
    """Present the chosen embeddings using one common parameter. Equality of
    that parameter on the overlap identifies the sheets, even when two
    charts have independently named conjugate roots of the same polynomial."""
    fields = list(dict.fromkeys(c.coefficient_field for c in charts if c.coefficient_field))
    if not fields:
        return charts
    api = chart_expressions(charts[0].variables)
    theta = api.Symbol("atlas_theta")
    used = {v for c in charts for v in c.variables}
    while theta in used:
        theta = api.Symbol(theta.name + "_")
    identities = [(minimal.xreplace({v: theta}), degree, embedding) for v, minimal, degree, embedding in fields]
    if len(set(identities)) == 1:
        relation, degree, embedding = identities[0]
        values = dict.fromkeys(fields, theta)
    else:
        from .plane_backend import coefficient_field
        if not all(field[3].startswith("PolynomialRootIndexed(") for field in fields):
            return None
        common = coefficient_field(tuple(field[3] for field in fields))
        def read(source):
            return decode((api.Symbol("T"),), source).xreplace({api.Symbol("T"): theta})
        values = {field: read(word) for field, word in zip(fields, common["values"])}
        relation, degree, embedding = read(common["minimal_polynomial"]), common["degree"], common["exact"]
    out = []
    for c in charts:
        old = c.coefficient_field
        variables = tuple(v for v in c.variables if not old or v != old[0]) + (theta,)
        mapping = tuple(values[old] if old and v == old[0] else v for v in c.variables)
        def move(equations):
            if not old:
                return equations
            return pullbacks(variables, mapping, c.variables, equations, modulus=relation)
        ambient = tuple(g for g in c.ambient if not old or g != old[1])
        out.append(replace(c, variables=variables, ambient=(*move(ambient), relation),
                           coordinates=tuple(zip((name for name, _ in c.coordinates),
                                                 move(tuple(g for _, g in c.coordinates)))),
                           components=tuple((name, n, nu, move(ideal)) for name, n, nu, ideal in c.components),
                           fibre=move(c.fibre), region=move(c.region), removed=tuple(move(z) for z in c.removed),
                           coefficient_field=(theta, relation, degree, embedding)))
    return out


@lru_cache(maxsize=256)
def _quadratic_character(variables, equations):
    """On a coordinate flat, recognize M²=d with M a primitive monomial.

    The nonzero constant d makes every coordinate occurring in M a unit.
    Completing its primitive exponent vector to an integral torus basis
    proves that each of M=±sqrt(d) is geometrically integral. The unused
    affine coordinates are free, and the two pieces are disjoint.
    """
    fixed = tuple(v for v in variables if v in equations)
    rest = tuple(dict.fromkeys(f for f in equations if f not in fixed and f != 0))
    if not fixed or len(rest) != 1:
        return None
    free = tuple(v for v in variables if v not in fixed)
    if not free:
        return None
    api = chart_expressions(variables)
    mapping = tuple(api.S.Zero if v in fixed else v for v in variables)
    f, = pullbacks(free, mapping, variables, rest)
    terms = sparse_terms(free, f)
    if len(terms) != 2:
        return None
    (powers, leading), (constant, trailing) = terms
    if any(constant) or gcd(*powers) != 2:
        return None
    monomial = api.prod(v**(i//2) for v, i in zip(free, powers))
    return fixed, monomial, api.Rational(-trailing/leading)


def _rational_sqrt(value):
    if value < 0:
        return None
    a, b = isqrt(value.p), isqrt(value.q)
    rational = np.Rational if isinstance(value, np.Expr) else sp.Rational
    return rational(a, b) if a*a == value.p and b*b == value.q else None


def _character_field(charts):
    """One marked embedding of a quadratic splitting field for these opens.

    Conjugating the chosen square root exchanges both local branches on
    every chart, so either root gives the same incidence up to relabelling.
    Restrict to affine-space charts here; existing marked covers retain
    their own declared embedding.
    """
    if any(c.ambient or c.coefficient_field for c in charts):
        return charts, None
    for c in charts:
        for label, _, _, strict in c.components:
            if not _is_strict(label):
                continue
            for name, _, _, boundary in c.components:
                if _is_strict(name):
                    continue
                section = _quadratic_character(c.variables, (*strict, *boundary))
                if section is None or _rational_sqrt(section[2]) is not None:
                    continue
                d = section[2]
                api = chart_expressions(charts[0].variables)
                parameter = api.Symbol("atlas_theta")
                used = {v for t in charts for v in t.variables}
                while parameter in used:
                    parameter = api.Symbol(parameter.name + "_")
                relation = parameter**2 - d
                exact = f"PolynomialRootIndexed(List({-d.p}, 0, {d.q}), 1)"
                field = (parameter, relation, 2, exact)
                return [replace(t, variables=(*t.variables, parameter), ambient=(relation,),
                                coefficient_field=field) for t in charts], d
    return charts, None


def _geometric_primes(chart, equations, dimension, character=None):
    field = chart.coefficient_field
    if not field:
        return geometric_components(chart.variables, equations, dimension)
    parameter, relation, _, _ = field
    coordinates = tuple(v for v in chart.variables if v != parameter)
    whole = tuple(g for g in equations if g != relation)
    if all(parameter not in g.free_symbols for g in whole):
        if character is not None:
            section = _quadratic_character(coordinates, whole)
            if section is not None:
                fixed, monomial, d = section
                scale = _rational_sqrt(d/character)
                if scale is not None and len(coordinates) - len(fixed) - 1 == dimension:
                    return [(*fixed, monomial + sign*scale*parameter) for sign in (-1, 1)]
        # Base change preserves geometric irreducibility. These families
        # of primes restrict to one prime per marked embedding.
        return geometric_components(coordinates, whole, dimension)
    # Over a marked field a triangular monic graph over free coordinates
    # is geometrically integral on each sheet. A lexicographic basis over
    # Q containing the relation verifies the presentation simultaneously.
    basis = groebner(equations, (*coordinates, parameter), order="lex")
    if tuple(basis.exprs) == (1,):
        return []
    pivots, remaining = set(), set(coordinates)
    for equation in basis.exprs:
        first = max(e for e, _ in sparse_terms((*coordinates, parameter), equation))
        if first[-1] and not any(first[:-1]):
            continue
        if sum(first) != 1 or first[-1]:
            return None
        pivot = coordinates[first.index(1)]
        # Lex and a unit leading coordinate imply a monic equation
        # pivot = polynomial in the later coordinates alone.
        if pivot in pivots:
            return None
        pivots.add(pivot)
        remaining.remove(pivot)
    return [whole] if len(remaining) == dimension else None


def _curve_primes(chart, equations, character):
    """Prime pieces of the reduced curve, including branches that meet.

    For an equation h=prod f_i^e_i in a basis of I, V(I) is the union
    of V(I+(f_i)). Decompose each of those ideals geometrically; every
    returned prime is then a proved curve on the selected embedding.
    """
    primes = _geometric_primes(chart, equations, 1, character)
    if primes is not None:
        return primes
    basis = groebner(equations, chart.variables)
    for equation in basis.exprs:
        if chart.coefficient_field and equation.free_symbols <= {chart.coefficient_field[0]}:
            continue
        _, factors = rational_factor_rows(chart.variables, equation)
        if len(factors) < 2 and not any(m > 1 for _, m, _ in factors):
            continue
        parts = []
        for row, _, _ in factors:
            factor = _sparse_expression(chart.variables, row)
            piece = _geometric_primes(chart, (*equations, factor), 1, character)
            if piece is None:
                break
            parts.extend(piece)
        else:
            return list(dict.fromkeys(parts))
    return None


def _surface_graph(charts, boundaries, overlap, character):
    """On a smooth surface, vertices are the geometric prime curves and an
    edge is a transverse intersection point. Curves meeting in a single
    point are distinct: across two charts we glue primes only if the source
    prime has a nonempty open in the overlap and its *entire* trace on the
    diagonal is contained in the target prime. This is equality of the two
    irreducible curves on that open, as both have dimension one. Counting
    points on owned chart regions then counts each edge exactly once.
    """
    pieces = []
    sections = []
    for i, chart in enumerate(charts):
        for strict, _, _, hypersurface in chart.components:
            if not _is_strict(strict):
                continue
            carrier = (*chart.ambient, *hypersurface)
            owned = (*chart.fibre, *chart.region)
            smooth = smooth_locus(chart.variables, carrier, 2)
            if not smooth["smooth"] and _present(chart.variables, (*smooth["bad_ideal"], *owned), chart.removed):
                return None
            indices = []
            for label, divisor in boundaries[i].items():
                primes = _curve_primes(chart, (*carrier, *divisor), character)
                if primes is None:
                    return None
                for prime in primes:
                    ideal = tuple(dict.fromkeys((*chart.ambient, *prime)))
                    if any(ideal_subset(chart.variables, hole, ideal) for hole in chart.removed):
                        continue
                    indices.append(len(pieces))
                    pieces.append((i, strict, label, ideal))
            rows = tuple((str(k), 1, 1, pieces[k][3]) for k in indices)
            restricted = replace(chart, ambient=carrier, components=rows, fibre=owned, region=())
            if _own_failures(restricted, 2, {})[0]:
                return None
            sections.append((i, indices))

    parents = list(range(len(pieces)))

    def root(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k

    for p, (i, strict, label, prime) in enumerate(pieces):
        for q in range(p):
            j, other_strict, other_label, other = pieces[q]
            if i == j or strict != other_strict or label != other_label or root(p) == root(q):
                continue
            coords, swap_a, swap_b, graph, holes = overlap(j, i)
            source = (*graph, *(g.xreplace(swap_a) for g in other))
            if not _present(coords, source, holes):
                continue
            # A nonzero target generator anywhere on the open disproves
            # equality. Excluding its zero ideal tests this condition on the
            # actual punctured overlap, independently of its affine closure.
            if all(not _present(coords, source, (*holes, (g.xreplace(swap_b),))) for g in prime):
                parents[root(p)] = root(q)
    groups = defaultdict(list)
    for k, (_, _, label, _) in enumerate(pieces):
        if root(k) not in groups[label]:
            groups[label].append(root(k))
    visible = set()
    for k, (i, _, _, ideal) in enumerate(pieces):
        c = charts[i]
        if _present(c.variables, (*ideal, *c.fibre, *c.region), c.removed):
            visible.add(root(k))
    names = {}
    for label in sorted(groups):
        roots = [k for k in groups[label] if k in visible]
        for j, k in enumerate(roots, 1):
            names[k] = label if len(roots) == 1 else f"{label}.{j}"

    counts = defaultdict(int)
    for i, indices in sections:
        chart = charts[i]
        for p, q in combinations(indices, 2):
            a, b = root(p), root(q)
            if a == b:
                return None  # an irreducible self-intersection needs normalization
            if a not in visible or b not in visible:
                continue
            at = (*pieces[p][3], *pieces[q][3], *chart.fibre, *chart.region)
            # On an SNC surface the intersection is a finite set. Its
            # compactly supported Euler characteristic counts its points.
            count = chart.one_sheet(euler_characteristic(chart.variables, at, chart.removed))
            if count:
                counts[tuple(sorted((names[a], names[b])))] += count
    edges = [{"id": f"C{a}-{b}.{j}", "dimension": 1, "vertices": [a, b], "faces": [a, b]}
             for (a, b), n in sorted(counts.items()) for j in range(1, n+1)]
    return {"status": "complete", "vertices": [{"id": name, "dimension": 1} for name in names.values()],
            "cells": edges, "convention": "Exceptional divisor on the smooth strict transform in the germ over the origin over Qbar."}


def glued_complex(charts, dimension, intrinsic=False, *, counts=None):
    """Return the full complex when local geometric primes can be identified.

    On charts over marked fields, use the diagonal of the common coefficient
    field and local primes whose restriction to a single embedding is proved
    geometrically integral. Off deleted centres both the existence of an
    overlap and the final fibre test use the actual open sets. A marked field
    is distinct from a finite cover representing several geometric points:
    the latter fails the generic-injectivity test on implicit charts.
    """
    if any(not c.ambient and len(c.variables) != dimension for c in charts):
        return None
    if counts is not None and not intrinsic:
        from .atlas_finite import finite_complex
        finite = finite_complex(charts, dimension, counts)
        if finite is not None:
            return finite
    charts = _common_field(charts)
    if charts is None:
        return None
    character = None
    if intrinsic and dimension == 3:
        charts, character = _character_field(charts)
    for c in charts:
        extra = tuple(g for g in c.ambient if not c.coefficient_field or g != c.coefficient_field[1])
        if extra and not {g for _, g in c.coordinates}.issubset(c.variables):
            return None
    split_curves = False
    # Gluing labelled primes assumes each boundary piece is already smooth
    # on the owned fibre. A singular union carrying one label may have an
    # internal incidence; splitting it requires a different presentation.
    for c in charts:
        boundary = tuple(row for row in c.components if not _is_strict(row[0]))
        carriers = [(*c.ambient, *ideal) for label, _, _, ideal in c.components if _is_strict(label)] \
            if intrinsic else [c.ambient]
        for ambient in carriers:
            owned = replace(c, ambient=ambient, components=boundary,
                            fibre=tuple(dict.fromkeys((*c.fibre, *c.region))), region=())
            if _own_failures(owned, dimension-int(intrinsic), {})[0]:
                if not intrinsic or dimension != 3:
                    return None
                split_curves = True
    boundaries = [{k: ideal for k, _, _, ideal in c.components if not _is_strict(k)} for c in charts]
    # Generate each key only in charts where all those labels occur.
    keys = set()
    for boundary in boundaries:
        for size in range(1, min(len(boundary), dimension - int(intrinsic)) + 1):
            keys.update(tuple(sorted(key)) for key in combinations(boundary, size))

    @cache
    def overlap(i, j):
        a, b = charts[i], charts[j]
        # All functions on each side are renamed simultaneously. The two
        # coordinate blocks span a fresh product ring, so positional names
        # can also reuse cached expressions for successive chart products.
        api = chart_expressions(a.variables)
        va = api.symbols(f"atlasLeft0:{len(a.variables)}")
        vb = api.symbols(f"atlasRight0:{len(b.variables)}")
        rename_a, rename_b = dict(zip(a.variables, va)), dict(zip(b.variables, vb))
        images_a, images_b = dict(a.coordinates), dict(b.coordinates)
        if images_a.keys() != images_b.keys():
            raise ValueError("The chart maps use different base coordinates.")
        equations = [f.xreplace(rename_a) for f in a.ambient] + [f.xreplace(rename_b) for f in b.ambient]
        equations.extend(g.xreplace(rename_a) - images_b[name].xreplace(rename_b) for name, g in images_a.items())
        if a.coefficient_field:
            theta = a.coefficient_field[0]
            equations.append(rename_a[theta] - rename_b[theta])
        units = []
        for index, chart, rename in ((i, a, rename_a), (j, b, rename_b)):
            for ideal in boundaries[index].values():
                g = next((f for f in ideal if not ideal_subset(chart.variables, (f,), chart.ambient)), None)
                if g is None:
                    raise ValueError("The exceptional ideal vanishes on its entire chart.")
                units.append(g.xreplace(rename))
        coords = (*va, *vb)
        removed = tuple(tuple(g.xreplace(swap) for g in z)
                        for chart, swap in ((a, rename_a), (b, rename_b)) for z in chart.removed)
        return coords, rename_a, rename_b, saturate(coords, equations, api.Mul(*units)), removed

    for i, chart in enumerate(charts):
        if any(not chart.coefficient_field or g != chart.coefficient_field[1] for g in chart.ambient):
            coords, left, right, diagonal, _ = overlap(i, i)
            if not ideal_subset(coords, [left[x] - right[x] for x in chart.variables], diagonal):
                return None

    if split_curves:
        return _surface_graph(charts, boundaries, overlap, character)

    # A complete face map needs primes for every stratum. Discover an
    # unrepresented geometric component before saturating large chart
    # products for any of the already known strata.
    pieces_by_key = {}
    for key in sorted(keys, key=lambda ids: (len(ids), ids)):
        local = []
        for i, (chart, boundary) in enumerate(zip(charts, boundaries)):
            if any(label not in boundary for label in key):
                continue
            carrier = (*chart.ambient, *(g for label in key for g in boundary[label]))
            sections = [(label, ideal) for label, _, _, ideal in chart.components if _is_strict(label)] \
                if intrinsic else [(None, ())]
            for strict, ideal in sections:
                equations = (*carrier, *ideal)
                if chart.removed and not _present(chart.variables, equations, chart.removed):
                    continue
                primes = _geometric_primes(chart, equations, dimension-len(key)-int(intrinsic), character)
                if primes is None:
                    return None
                local.extend((i, strict, (*chart.ambient, *prime)) for prime in primes
                             if not any(ideal_subset(chart.variables, hole, (*chart.ambient, *prime))
                                        for hole in chart.removed))
        pieces_by_key[key] = local

    groups = {}
    vertices, cells = [], []
    for key, local in pieces_by_key.items():
        parent = list(range(len(local)))

        def root(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        for p, (i, strict, prime) in enumerate(local):
            for q in range(p):
                j, other_strict, other = local[q]
                if i == j or strict != other_strict or root(p) == root(q):
                    continue
                coords, swap_a, swap_b, graph, removed = overlap(j, i)
                intersection = (*graph, *(g.xreplace(swap_a) for g in other),
                                *(g.xreplace(swap_b) for g in prime))
                if _present(coords, intersection, removed):
                    parent[root(p)] = root(q)
        marked = defaultdict(dict)
        for p, (i, strict, prime) in enumerate(local):
            marked[root(p)][(i, strict)] = prime
        visible = [group for group in marked.values() if any(
            _present(charts[i].variables, (*prime, *charts[i].fibre, *charts[i].region), charts[i].removed)
            for (i, _), prime in group.items())]
        groups[key] = []
        for k, group in enumerate(visible, 1):
            if len(key) == 1:
                name = key[0] if len(visible) == 1 else f"{key[0]}.{k}"
                vertices.append({"id": name, "dimension": dimension - 1 - int(intrinsic)})
            else:
                local_key, prime = next(iter(group.items()))
                v = charts[local_key[0]].variables

                def contains(smaller):
                    hits = [name for name, pieces in groups[smaller]
                            if local_key in pieces and ideal_subset(v, pieces[local_key], prime)]
                    if len(hits) != 1:
                        raise ArithmeticError("A component has no unique containing face.")
                    return hits[0]

                ids = [contains((label,)) for label in key]
                faces = [contains(face) for face in combinations(key, len(key)-1)]
                name = "C" + "-".join(key) + f".{k}"
                cells.append({"id": name, "dimension": len(key)-1, "vertices": ids, "faces": faces})
            groups[key].append((name, group))
    return {"status": "complete", "vertices": vertices, "cells": cells,
            "convention": ("Exceptional divisor on the smooth strict transform" if intrinsic else
                           "Ambient exceptional divisor") + " in the germ over the origin over Qbar."}
