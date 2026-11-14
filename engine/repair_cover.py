"""Verify the owned affine cover of the projective blowups over a deleted centre.

A point in the image of each repair chart does not establish coverage of
the exceptional projective fibre. Rebuild the local blowups on the saved
parent equation and verify the supplied chart partition over that fibre.
"""
from collections import Counter, defaultdict
from dataclasses import replace

from .affine_kernel import pullbacks
from .blowup_ideal import ideal_basis, radical_subset
from .chart_expressions import chart_expressions
from .groebner import groebner
from .local_blowups import GiveUp, _covers, _dimension, _is_strict, _present, acampo_over


def chart_signature(variables, ambient, coordinates, components):
    """Exact embedded chart equations in a fixed order of named coordinates."""
    variables = tuple(sorted(variables, key=lambda v: v.name))
    quotient = groebner(ambient, variables)
    images = tuple(sorted((name, quotient.reduce(f)[1]) for name, f in coordinates))
    divisors = tuple(sorted((name, n, ideal_basis(variables, (*ambient, *ideal)))
                            for name, n, ideal in components))
    return variables, tuple(quotient.exprs), images, divisors


def projected_patch(index, parent, chart, labels=None):
    """Compose a local blowup chart with its parent and retain its owned fibre."""
    local = dict(chart.coordinates)
    mapping = tuple(local[v.name] for v in parent.variables)
    images = pullbacks(chart.variables, mapping, parent.variables, (f for _, f in parent.coordinates))
    labels = labels or {}
    return replace(chart, coordinates=tuple(zip(dict(parent.coordinates), images)), fibre=images,
                   region=tuple(dict.fromkeys((*chart.region, *chart.fibre))),
                   components=tuple((labels.get(label, label), n, nu, ideal)
                                    for label, n, nu, ideal in chart.components),
                   repair_projection=(index, mapping))


def _owned_signature(chart, variables):
    # Variable slots express the proposed isomorphism of affine presentations.
    # The reduced ideal verifies the ambient isomorphism. Equality of the
    # parent coordinates modulo that ideal then verifies the isomorphism over
    # the parent. A purported relabelling has no evidential value by itself.
    rename = dict(zip(chart.variables, variables))
    def ideal(polys):
        return tuple(f.xreplace(rename) for f in polys)
    ambient = ideal(chart.ambient)
    quotient = groebner(ambient, variables)
    mapping = tuple(quotient.reduce(f)[1] for f in ideal(chart.repair_projection[1]))
    owned = ideal_basis(variables, (*ambient, *ideal(chart.fibre), *ideal(chart.region)))
    holes = tuple(ideal_basis(variables, (*owned, *ideal(hole))) for hole in chart.removed)
    components = Counter((label if _is_strict(label) else None, n, nu,
                          ideal_basis(variables, (*ambient, *ideal(divisor))))
                         for label, n, nu, divisor in chart.components)
    return (tuple(quotient.exprs), mapping, owned, frozenset(h for h in holes if h != (1,)),
            frozenset(components.items()))


def _lifted_centre(parent, centre, children, count):
    """Recover the graph of the archive's finite separating parameter.

    Project the owned fibre of a repair chart to (parent, parameter). Its
    image is closed once finite. Prove that its projection is precisely the
    deleted centre, with one lift per point and per parameter value.
    """
    from .atlas import _field_relation
    from .constructible_euler import euler_characteristic
    api = chart_expressions(parent.variables)
    coords = parent.variables
    targets = tuple(api.Dummy("repair_axis") for _ in coords)
    restore = dict(zip(targets, coords))
    for c in children:
        if len(c.ambient) != 1 or len(c.ambient[0].free_symbols) != 1:
            continue
        t, = c.ambient[0].free_symbols
        minimal = c.ambient[0]
        if t in coords or not _field_relation(t, minimal, count):
            continue
        ring = (*[v for v in c.variables if v != t], *targets, t)
        graph = (*c.ambient, *c.fibre, *c.region, *(v-g for v,g in zip(targets,c.repair_projection[1])))
        for unit in _covers(c.variables, c.removed):
            inverse = api.Dummy("repair_open")
            basis = groebner((*graph, 1-inverse*unit), (inverse,*ring), order="lex")
            image = tuple(g.xreplace(restore) for g in basis.exprs if g.free_symbols <= {*targets,t})
            if image == (1,) or _dimension((*coords,t), image) != 0:
                continue
            projected = groebner(image, (t,*coords), order="lex")
            closed = tuple(g for g in projected.exprs if t not in g.free_symbols)
            if (radical_subset(coords, closed, centre) and radical_subset(coords, centre, closed)
                    and euler_characteristic((*coords,t), image) == count):
                fibre = tuple(g for g in image if g.free_symbols != {t})
                return (*coords,t), (minimal,), fibre
    return None


def _local_pivots(index, parent, centre, children):
    """Replay the smooth local blowups, with all standard projective opens."""
    from .constructible_euler import euler_characteristic
    coords = parent.variables
    if _dimension(coords, centre) != 0:
        return None
    count = euler_characteristic(coords, centre)
    if count == 1:
        space, relation, fibre = coords, (), centre
    else:
        lift = _lifted_centre(parent, centre, children, count)
        if lift is None:
            return None
        space, relation, fibre = lift
    try:
        _, _, charts, _ = acampo_over(space, parent.components, fibre, ambient=relation,
                                      dimension=len(coords), count_strata=False)
    except GiveUp:
        return None
    return [projected_patch(index, parent, c) for c in charts]


def _common_parameter(chart, parameter):
    """Align an unmarked finite parameter for comparing all of its conjugates."""
    if not chart.ambient:
        return chart
    from .atlas import _field_relation
    from .atlas_finite import _extend_chart, _integer_polynomial
    from .marked_field import flint_literal
    if len(chart.ambient) != 1:
        return None
    relation, = chart.ambient
    if len(relation.free_symbols) != 1:
        return None
    t, = relation.free_symbols
    coeffs = tuple(_integer_polynomial(t, relation))
    if len(coeffs) != 1:
        return None
    degree = len(coeffs[0])-1
    if not _field_relation(t, relation, degree):
        return None
    mark = flint_literal(coeffs[0], 1)
    field = parameter, relation.xreplace({t: parameter}), degree, mark
    return _extend_chart(replace(chart, coefficient_field=(t,relation,degree,mark)), field, parameter)


def _uncovered(chart, opens):
    """Does this owned locus contain a point outside a union of locally closed subsets?"""
    def uncovered(i, equations, holes):
        if not _present(chart.variables, equations, holes):
            return False
        if i == len(opens):
            return True
        closed, excluded = opens[i]
        # The complement of V(I) minus union V(J) is D(I) union the V(J).
        if closed and uncovered(i+1, equations, (*holes, closed)):
            return True
        return any(uncovered(i+1, (*equations,*hole), holes) for hole in excluded)

    return uncovered(0, (*chart.ambient,*chart.fibre,*chart.region), chart.removed)


def _divisors_agree(chart, other, move, excluded):
    """Match the divisors on the common open, including the strict transform."""
    variables = chart.variables
    def visible(components, transport):
        groups = defaultdict(list)
        for label, n, nu, generators in components:
            ideal = tuple(transport(g) for g in generators)
            if _present(variables, (*chart.ambient,*ideal), excluded):
                groups[(label if _is_strict(label) else None, n, nu)].append(ideal)
        return groups

    local = visible(chart.components, lambda g: g)
    pulled = visible(other.components, move)
    if local.keys() != pulled.keys():
        return False

    def contains(ideal, polynomials):
        return all(not _present(variables, (*chart.ambient,*ideal), (*excluded,(g,))) for g in polynomials)

    for key, ideals in local.items():
        matches = pulled[key]
        if len(matches) != len(ideals):
            return False
        for ideal in ideals:
            for j, other_ideal in enumerate(matches):
                if contains(ideal, other_ideal) and contains(other_ideal, ideal):
                    matches.pop(j)
                    break
            else:
                return False
    return True


def _cover_by_transitions(reference, supplied):
    """Check coverage and unique ownership for different affine presentations.

    Identify the two locally owned fibres only where the birational maps in
    both directions are regular. Over each projective pivot, the inverse
    images of the archive's owners must be a disjoint cover. Conversely
    every archived point must occur on the projective model. This permits
    different affine coordinates and refinements of the original partition.
    """
    from .repair_gluing import transition_on_field_sheet
    parameter = chart_expressions(reference[0].variables).Dummy("cover_parameter")
    sides = [[_common_parameter(c, parameter) for c in family] for family in (reference,supplied)]
    if any(c is None for side in sides for c in side):
        return False
    reference, supplied = sides
    in_reference = [[] for _ in reference]
    in_supplied = [[] for _ in supplied]

    def owned_trace(target, direct, reverse):
        move, poles = direct
        _, other_poles = reverse
        return (tuple(move(g) for g in (*target.fibre,*target.region)),
                (*poles, *(tuple(move(g) for g in hole) for hole in other_poles)))

    for i, a in enumerate(reference):
        for j, b in enumerate(supplied):
            if a.coefficient_field != b.coefficient_field:
                continue
            forward = transition_on_field_sheet(a,b,a.repair_projection[1],b.repair_projection[1])
            reverse = transition_on_field_sheet(b,a,b.repair_projection[1],a.repair_projection[1])
            if forward is None or reverse is None:
                return False
            closed, holes = owned_trace(b, forward, reverse)
            base = (*a.ambient,*a.region,*a.fibre,*closed)
            excluded = (*a.removed,*holes)
            if not _present(a.variables, base, excluded):
                continue
            if not _divisors_agree(a,b,forward[0],excluded):
                return False
            if any(_present(a.variables, (*base,*other), (*excluded,*removed))
                   for other,removed in in_reference[i]):
                return False
            in_reference[i].append((closed, holes))
            in_supplied[j].append(owned_trace(a, reverse, forward))
    return (not any(_uncovered(c, opens) for c, opens in zip(reference, in_reference))
            and not any(_uncovered(c, opens) for c, opens in zip(supplied, in_supplied)))


def verify_repair_cover(charts):
    """Each supplied repair has exactly one owner for each local blowup patch."""
    links = defaultdict(list)
    for c in charts:
        if c.repair_projection:
            links[c.repair_projection[0]].append(c)
    for index, children in links.items():
        if not 0 <= index < len(charts):
            return False
        parent = charts[index]
        if parent.ambient or parent.coefficient_field or parent.repair_projection or not parent.removed:
            return False
        api = chart_expressions(parent.variables)
        axes = {}
        def signature(c):
            n = len(c.variables)
            if n not in axes:
                axes[n] = tuple(api.Dummy("pivot_axis") for _ in range(n))
            return _owned_signature(c, axes[n])
        reference = []
        for centre in parent.removed:
            pivots = _local_pivots(index, parent, centre, children)
            if pivots is None:
                return False
            reference.extend(pivots)
        recorded = Counter(signature(c) for c in children)
        expected = Counter(signature(c) for c in reference)
        if recorded == expected:
            continue
        # Exact copies cannot make up a missing nonempty pivot and cannot
        # own its points twice. More general presentations need transition
        # tests on their actual locally closed regions.
        if recorded <= expected or expected <= recorded or not _cover_by_transitions(reference, children):
            return False
    return bool(links)
