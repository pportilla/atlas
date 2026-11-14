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
prime on each sheet. For a finite cover of several repair centres,
first separate the marked sheets and verify the projection to each centre.
For affine charts, a section M²=d of a coordinate flat with M a primitive
monomial and d a nonzero rational nonsquare supplies its own quadratic
splitting field. Sections in that square class then have two explicit
geometric primes whose identifications are tested on the same diagonal.

For an intrinsic boundary, first split each restricted divisor into
geometric primes and test the reduced branches for simple normal crossings
on the owned fibre. Primes on two charts agree precisely when they have the
same generic point on the diagonal. Testing containment on a nonempty open
distinguishes primes that meet only in a lower-dimensional intersection. Gluing before
intersecting with the origin fibre retains the parts of a component that
join two visible charts. A face is the unique containing component of the
smaller intersection, tested by containment of ideals on a common chart.
"""
from collections import defaultdict
from dataclasses import replace
from functools import cache, lru_cache
from itertools import combinations, product

from . import native_polynomial as np
from .chart_expressions import chart_expressions
from .reference import sp

from .affine_kernel import _sparse_expression, decode, pullbacks, rational_factor_rows, sparse_terms
from .atlas_finite import _extend_chart, _field_mark, marked_primes, splitting_charts
from .blowup_ideal import ideal_basis, ideal_subset, radical_subset, saturate, smooth_locus
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
    if len(fields) == 1 and all(c.coefficient_field == fields[0] for c in charts):
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
        exact = tuple(_field_mark(field) for field in fields)
        if any(value is None for value in exact):
            return None
        common = coefficient_field(exact)
        def read(source):
            return decode((api.Symbol("T"),), source).xreplace({api.Symbol("T"): theta})
        values = {field: read(word) for field, word in zip(fields, common["values"])}
        relation, degree, embedding = read(common["minimal_polynomial"]), common["degree"], common["exact"]
    field = theta, relation, degree, embedding
    return [_extend_chart(c, field, values.get(c.coefficient_field)) for c in charts]


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
    if any(constant):
        return None
    from .binomial_kernel import primitive_character
    degree, direction = primitive_character(powers)
    if degree != 2:
        return None
    monomial = api.prod(v**i for v, i in zip(free, direction))
    return fixed, monomial, api.Rational(-trailing/leading)


@lru_cache(maxsize=256)
def _rational_sqrt(value):
    from .native_library import field_request
    root = field_request("atlas_field_coefficient", ["q 0", f"q {value} pow 1/2"])
    if root is None:
        raise ArithmeticError("The exact algebraic coefficient kernel is unavailable.")
    if root == "outside":
        return None
    rational = np.Rational if isinstance(value, np.Expr) else sp.Rational
    return rational(root.strip() or "0")


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


def _rational_primes(variables, equations, dimension, *, reduced_support=False):
    """A free affine factor preserves geometric primes and adds its dimension."""
    active = set().union(*(getattr(g, "free_symbols", ()) for g in equations))
    coordinates = tuple(v for v in variables if v in active)
    if not coordinates:
        return [] if any(equations) else ([()] if dimension == len(variables) else None)
    return geometric_components(coordinates, equations, dimension-len(variables)+len(coordinates),
                                reduced_support=reduced_support)


def _geometric_primes(chart, equations, dimension, character=None, *, reduced_support=False):
    field = chart.coefficient_field
    if not field:
        return _rational_primes(chart.variables, equations, dimension, reduced_support=reduced_support)
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
        direct = _rational_primes(coordinates, whole, dimension, reduced_support=reduced_support)
        if direct is not None:
            return direct
    return marked_primes(chart, equations, dimension, branches=reduced_support)



def _branch_primes(chart, equations, dimension, character):
    """Prime components of an equidimensional reduced support, including components that meet.

    For a basis element h = prod f_i^e_i, V(I) is the union of V(I+(f_i)).
    Every recursive refinement strictly enlarges the ideal. Returned ideals
    define geometric primes of the requested dimension on the marked sheet.
    """
    if chart.coefficient_field:
        return _geometric_primes(chart, equations, dimension, character, reduced_support=True)

    @cache
    def decompose(basis):
        primes = _geometric_primes(chart, basis, dimension, character, reduced_support=True)
        if primes is not None:
            return primes
        for equation in basis:
            _, factors = rational_factor_rows(chart.variables, equation)
            if len(factors) < 2 and not any(m > 1 for _, m, _ in factors):
                continue
            refinements = [tuple(ideal_basis(chart.variables, (*basis, _sparse_expression(chart.variables, row))))
                           for row, _, _ in factors]
            if basis in refinements:
                continue
            components = []
            for refinement in dict.fromkeys(refinements):
                primes = decompose(refinement)
                if primes is None:
                    break
                components.extend(primes)
            else:
                return list(dict.fromkeys(components))
        return None

    return decompose(tuple(ideal_basis(chart.variables, equations)))


def _prime_groups(charts, pieces, overlap, transition):
    """Glue irreducible primes by equality on a nonempty open of the diagonal."""
    parents = list(range(len(pieces)))

    @cache
    def basis_of(index, prime):
        return groebner(prime,charts[index].variables)

    def root(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = parents[k]
        return k

    @cache
    def image_coordinates(index, prime, mapping):
        # In this elimination order the standard representatives for marked
        # constants are the powers of theta alone. A normal form depending
        # on a geometric variable therefore proves nonconstancy over K.
        chart = charts[index]
        theta = chart.coefficient_field[:1]
        coords = tuple(v for v in chart.variables if v not in theta) + theta
        basis = groebner(prime,coords,order="lex")
        return tuple(basis.reduce(f)[1] for f in mapping)

    def different_images(i, prime, j, other):
        a,b = charts[i],charts[j]
        if not (a.repair_projection and b.repair_projection and a.repair_projection[0] == b.repair_projection[0]):
            return False
        left = image_coordinates(i,prime,a.repair_projection[1])
        right = image_coordinates(j,other,b.repair_projection[1])
        theta = a.coefficient_field[0] if a.coefficient_field else None
        for f,g in zip(left,right):
            # A generic value in the common constant field is a marked
            # scalar. A different constant, or a nonconstant function, proves
            # the Zariski closures of the two images unequal.
            if (f.free_symbols <= {theta} or g.free_symbols <= {theta}) and f != g:
                return True
        return False

    for p, ((i, strict), prime) in enumerate(pieces):
        distinct = set()
        for q in range(p):
            (j, other_strict), other = pieces[q]
            target = root(q)
            if i == j or strict != other_strict or root(p) == target or target in distinct:
                continue
            if different_images(i,prime,j,other):
                distinct.add(target)
                continue
            direct = transition(j, i)
            if direct is not None:
                transfer, holes = direct
                basis = basis_of(j,other)
                if not (all(any(not basis.contains(g) for g in h) for h in holes)
                        and all(transfer(g,basis) == 0 for g in prime)):
                    continue
            reverse = transition(i,j) if direct is not None else None
            if direct is not None and reverse is not None:
                # A regular birational map may contract a divisor. Both
                # directions must be regular on the actual identified open.
                # Pull the inverse's denominator ideals and the target holes
                # back before comparing the two generic points.
                _, target_holes = reverse
                # A finite intersection of nonempty opens of one geometric
                # prime is nonempty. Each removed ideal must have a generator
                # nonzero there. Membership is checked in its coordinate ring.
                if all(any(transfer(g,basis) != 0 for g in h) for h in target_holes):
                    parents[root(p)] = root(q)
                continue
            coords, swap_a, swap_b, graph, holes = overlap(j, i)
            source = (*graph, *(g.xreplace(swap_a) for g in other))
            if _present(coords, source, holes) and all(
                    not _present(coords, source, (*holes, (g.xreplace(swap_b),))) for g in prime):
                parents[root(p)] = target
            else:
                # The saturated diagonal covers the entire chart overlap.
                # Two nonempty chart restrictions of one irreducible prime
                # share its generic point. An empty restriction here, or a
                # point outside the target prime, rules out its whole class.
                distinct.add(target)
    groups = defaultdict(dict)
    for p, (section, prime) in enumerate(pieces):
        group = groups[root(p)]
        if section in group:
            # A geometric prime restricts to an irreducible subset of each open.
            return None
        group[section] = prime
    return [group for group in groups.values() if any(
        _present(charts[i].variables, (*prime, *charts[i].fibre, *charts[i].region), charts[i].removed)
        for (i, _), prime in group.items())]


class _NeedsSplittingField(Exception):
    """Carry the exact failed section into a proposed base change."""


def _intrinsic_complex(charts, boundaries, overlap, transition, dimension, character):
    """Compute the complex of the reduced boundary on a smooth strict transform.

    First identify the individual prime divisors, including meeting branches
    of one exceptional label, and certify that these geometric branches are
    SNC on the owned fibre. For each higher intersection glue its primes by
    equality on a dense open and take its faces by ideal containment. For a
    zero-dimensional intersection, fixing one prime of each incident face
    partitions its points. Euler characteristic on each owned piece therefore
    counts the top cells with precisely that boundary, including parallel cells.
    """
    local_labels = defaultdict(set)
    for i, chart in enumerate(charts):
        owned = (*chart.fibre, *chart.region)
        for strict, _, _, hypersurface in chart.components:
            if not _is_strict(strict):
                continue
            section = (i, strict)
            carrier = (*chart.ambient, *hypersurface)
            smooth = smooth_locus(chart.variables, carrier, dimension)
            if not smooth["smooth"] and _present(chart.variables, (*smooth["bad_ideal"], *owned), chart.removed):
                return None
            branch_ideals = {}
            for label, divisor in boundaries[i].items():
                equations = (*carrier, *divisor)
                if chart.removed and not _present(chart.variables, equations, chart.removed):
                    continue
                primes = _branch_primes(chart, equations, dimension-1, character)
                if primes is None:
                    raise _NeedsSplittingField(i, equations, dimension-1)
                for prime in primes:
                    ideal = tuple(dict.fromkeys((*chart.ambient, *prime)))
                    if any(ideal_subset(chart.variables, hole, ideal) for hole in chart.removed):
                        continue
                    ideal = tuple(ideal_basis(chart.variables, ideal))
                    local_labels[section, ideal].add(label)
                    branch_ideals.setdefault(ideal, len(branch_ideals))
            # I(carrier) is already among the Jacobian equations. Replacing
            # each boundary generator by its remainder modulo this same ideal
            # preserves the intersection ideal and its conormal span there.
            reducer = groebner(carrier, chart.variables)
            rows = []
            for ideal, index in branch_ideals.items():
                residues = tuple(dict.fromkeys(r for f in ideal if f not in carrier
                                               if (r := reducer.reduce(f)[1]) != 0))
                rows.append((str(index), 1, 1, residues))
            restricted = replace(chart, ambient=carrier, components=tuple(rows), fibre=owned, region=())
            if _own_failures(restricted, dimension, {})[0]:
                return None

    local_vertices = defaultdict(dict)
    faces = defaultdict(list)
    vertices, cells = [], []
    # Two different ambient divisors can restrict to exactly the same
    # prime on the hypersurface. Neither local names nor different ambient
    # labels distinguish geometric vertices. On each chart a reduced basis
    # removes duplicates. Across charts generic equality decides every match.
    groups = _prime_groups(charts, list(local_labels), overlap, transition)
    if groups is None:
        return None
    by_labels = defaultdict(list)
    for group in groups:
        labels = tuple(sorted(set().union(*(local_labels[item] for item in group.items()))))
        by_labels[labels].append(group)
    reserved = {label for labels in by_labels for label in labels}
    used = set()
    for labels, components in sorted(by_labels.items()):
        for j, group in enumerate(components, 1):
            base_name = "&".join(labels)
            name = base_name if len(components) == 1 else f"{base_name}.{j}"
            single = len(components) == len(labels) == 1
            while name in used or (name in reserved and not single):
                name += "_"
            used.add(name)
            vertex = {"id": name, "dimension": dimension-1}
            if len(labels) > 1:
                vertex["ambient_components"] = list(labels)
            vertices.append(vertex)
            for section, prime in group.items():
                local_vertices[section][name] = prime
                faces[section, (name,)].append((name, prime))

    # Each chart contributes only combinations of branches present there.
    for size in range(2, min(dimension, max(map(len, local_vertices.values()), default=0))+1):
        intersections = defaultdict(list)
        for section, divisors in local_vertices.items():
            for key in combinations(sorted(divisors), size):
                ideal = tuple(dict.fromkeys(g for name in key for g in divisors[name]))
                intersections[key].append((section, ideal))
        for key, sections in sorted(intersections.items()):
            subkeys = tuple(combinations(key, size-1))
            index = 0

            def add_cell(boundary):
                nonlocal index
                index += 1
                name = "C" + "-".join(key) + f".{index}"
                cells.append({"id": name, "dimension": size-1, "vertices": list(key), "faces": list(boundary)})
                return name

            if size == dimension:
                counts = defaultdict(int)
                for section, equations in sections:
                    chart = charts[section[0]]
                    owned = (*equations, *chart.fibre, *chart.region)
                    choices = [faces[section, subkey] for subkey in subkeys]
                    expected = chart.one_sheet(euler_characteristic(chart.variables, owned, chart.removed))
                    if expected == 0:
                        continue
                    accounted = 0
                    for incident in product(*choices):
                        # The restrictions for singleton faces already occur
                        # among the defining equations of a two-fold intersection.
                        extra = tuple(dict.fromkeys(g for _, face in incident for g in face if g not in equations))
                        count = (chart.one_sheet(euler_characteristic(chart.variables, (*owned, *extra), chart.removed))
                                 if extra else expected)
                        if count:
                            counts[tuple(name for name, _ in incident)] += count
                            accounted += count
                    if expected != accounted:
                        return None
                for boundary, count in sorted(counts.items()):
                    for _ in range(count):
                        add_cell(boundary)
                continue

            pieces = []
            for section, equations in sections:
                chart = charts[section[0]]
                if not _present(chart.variables, equations, chart.removed):
                    continue
                primes = _branch_primes(chart, equations, dimension-size, character)
                if primes is None:
                    raise _NeedsSplittingField(section[0], equations, dimension-size)
                for prime in primes:
                    ideal = tuple(dict.fromkeys((*chart.ambient, *prime)))
                    if not any(ideal_subset(chart.variables, hole, ideal) for hole in chart.removed):
                        pieces.append((section, ideal))
            groups = _prime_groups(charts, pieces, overlap, transition)
            if groups is None:
                return None
            for group in groups:
                boundary = []
                for subkey in subkeys:
                    incident = None
                    for section, prime in group.items():
                        matches = [name for name, face in faces[section, subkey]
                                   if ideal_subset(charts[section[0]].variables, face, prime)]
                        if len(matches) != 1 or incident is not None and matches[0] != incident:
                            return None
                        incident = matches[0]
                    boundary.append(incident)
                name = add_cell(boundary)
                for section, prime in group.items():
                    faces[section, key].append((name, prime))

    return {"status": "complete", "vertices": vertices, "cells": cells,
            "convention": "Exceptional divisor on the smooth strict transform in the germ over the origin over Qbar."}


def glued_complex(charts, dimension, intrinsic=False, *, counts=None, _descended=False):
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
    if intrinsic and not _descended:
        from .repair_replay import restore_repair_maps
        charts = restore_repair_maps(charts)
        if charts is None:
            return None
    if intrinsic and not _descended and any(c.repair_projection for c in charts):
        from .repair_gluing import repair_charts
        charts = repair_charts(charts)
        if charts is None:
            return None
        _descended = True
    charts = _common_field(charts)
    if charts is None:
        return None
    character = None
    if intrinsic and dimension >= 3:
        charts, character = _character_field(charts)
    if not intrinsic or dimension < 3:
        for c in charts:
            boundary = tuple(row for row in c.components if not _is_strict(row[0]))
            carriers = [(*c.ambient, *ideal) for label, _, _, ideal in c.components if _is_strict(label)] \
                if intrinsic else [c.ambient]
            for ambient in carriers:
                owned = replace(c, ambient=ambient, components=boundary,
                                fibre=tuple(dict.fromkeys((*c.fibre, *c.region))), region=())
                if _own_failures(owned, dimension-int(intrinsic), {})[0]:
                    return None
    boundaries = [{k: ideal for k, _, _, ideal in c.components if not _is_strict(k)} for c in charts]
    parent_equations = None
    if _descended:
        from .repair_gluing import parent_transitions
        parent_equations = parent_transitions(charts)
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
        transition = (parent_equations(a,b) if parent_equations and a.repair_projection and b.repair_projection
                      and a.repair_projection[0] != b.repair_projection[0] else None)
        if transition is not None:
            equations.extend(h.xreplace(rename_b)*d.xreplace(rename_a)-n.xreplace(rename_a)
                             for n,h,d in transition)
        elif (_descended and a.repair_projection and b.repair_projection and
                a.repair_projection[0] == b.repair_projection[0]):
            # Both factor through the same verified affine parent chart.
            # Equality there defines the diagonal away from the centres.
            equations.extend(g.xreplace(rename_a)-h.xreplace(rename_b)
                             for g,h in zip(a.repair_projection[1],b.repair_projection[1]))
        else:
            equations.extend(g.xreplace(rename_a) - images_b[name].xreplace(rename_b) for name, g in images_a.items())
        if a.coefficient_field:
            theta = a.coefficient_field[0]
            equations.append(rename_a[theta] - rename_b[theta])
        units = ([d.xreplace(rename_a) for _,_,d in transition] if transition else [])
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
        mapping = tuple(g for _, g in chart.coordinates)
        link = chart.repair_projection if _descended else ()
        if link:
            axes = tuple(v for v in chart.variables if not chart.coefficient_field or v != chart.coefficient_field[0])
            # Ordinary parents carry their own identity in the link table.
            # Their generic injectivity requires the map to the original base.
            if link[1] != axes:
                mapping = link[1]
        images = set(mapping)
        if chart.coefficient_field:
            images.add(chart.coefficient_field[0])
        if images.issuperset(chart.variables):
            continue
        if link:
            from .repair_gluing import _birational_parameters
            if _birational_parameters(chart, mapping) is not None:
                continue
            if link[1] == axes:
                # The common-parent overlap is the identity for this entry.
                # Using it as a fallback would leave the original map unchecked.
                return None
        coords, left, right, diagonal, _ = overlap(i, i)
        differences = tuple(left[x] - right[x] for x in chart.variables)
        if not radical_subset(coords, differences, diagonal):
            return None
        # The open used for saturation must be dense on each chart component.
        # Pull the closed self-overlap back along the actual diagonal. This
        # also rules out vacuously identifying sheets with an empty graph.
        identify = {v: x for swap in (left, right) for x, v in swap.items()}
        if not radical_subset(chart.variables, tuple(g.xreplace(identify) for g in diagonal), chart.ambient):
            return None

    parents_by_link = {c.repair_projection[0]: c for c in charts if _descended and c.repair_projection and
                       c.repair_projection[1] == tuple(v for v in c.variables
                                                      if not c.coefficient_field or v != c.coefficient_field[0])}

    @cache
    def transition(i, j):
        if not _descended:
            return None
        a,b = charts[i],charts[j]
        if not a.repair_projection or not b.repair_projection:
            return None
        pa,pb = a.repair_projection[0],b.repair_projection[0]
        target_parent = None
        if pa != pb:
            if pa not in parents_by_link or pb not in parents_by_link:
                return None
            target_parent = parents_by_link[pb]
        from .repair_gluing import transition_on_field_sheet
        return transition_on_field_sheet(a,b,a.repair_projection[1],b.repair_projection[1],target_parent=target_parent)

    if intrinsic and dimension >= 3:
        try:
            return _intrinsic_complex(charts, boundaries, overlap, transition, dimension-1, character)
        except _NeedsSplittingField as failed:
            enlarged = splitting_charts(charts, failed.args)
            if enlarged is None:
                return None
            return glued_complex(enlarged, dimension, intrinsic=True, _descended=_descended)

    # A complete face map needs primes for every stratum. Discover an
    # unrepresented geometric component before saturating large chart
    # products for any of the already known strata.
    # Generate only keys supported on a chart.
    keys = {tuple(sorted(key)) for boundary in boundaries
            for size in range(1, min(len(boundary), dimension - int(intrinsic)) + 1)
            for key in combinations(boundary, size)}
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
