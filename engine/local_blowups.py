"""Blow up until a divisor has normal crossings over a closed set, and read off its strata there.

The setting is a smooth variety W = V(ambient) in affine space, a divisor
sum_j N_j D_j on it given by ideals (exceptional divisors with normal
crossings, and the components of a reduced hypersurface X, the strict
transform), and a closed set V(fibre). While normal crossings fail at some
point over the fibre, a centre is chosen and blown up.

The complete rule has two phases. While X is singular at one of those
points, the centre is the locus where the invariant of Bierstone and Milman
(engine/centres.py) is largest among them. The invariant is constant on
that locus, which is smooth with normal crossings with the exceptional
divisors, and it decreases over it; every sequence of such blowups is
finite (Theorem 1.14 of Bierstone and Milman, Desingularization algorithms
I, Moscow Math. J. 3 (2003)). Once X is smooth at all the points, in
dimension three, the centres are those of the procedure for a smooth
surface (engine/surface.py), also finite, which works with the curves that
the exceptional divisors cut on X. Both depend only on the geometry, so the
charts of every stage glue.

Every blowup is in graph form (engine/blowup_ideal.py), so the old
coordinates stay available and the fibre needs no rewriting. Each component
carries the order N of the function along it and the number nu, one more
than the order of the pulled-back volume form. The new exceptional divisor E
of a centre of codimension c gets N = sum_j N_j ord_E(D_j) and
nu = c + sum_j (nu_j - 1) ord_E(D_j), where the order of a divisor along E is
the least order of the generators of its ideal.
Each centre is checked to be smooth, and normal crossings are certified by
the Jacobian criterion in every final chart: an answer does not depend on
the choice of the centres being right. The procedure stops without an
answer when its time is over.

On each blowup the chart of the i-th generator of the centre keeps the points
where the earlier generators' ratios vanish. These pieces partition the
space, which gives Euler characteristics by addition.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import count, islice
from time import monotonic
from .resolution_progress import report

from .chart_expressions import chart_expressions
from .binomial_kernel import exceptional_data, separating_forms

from .blowup_ideal import (blowup_charts, empty_locus, ideal_basis, ideal_subset, radical_subset, saturate,
                           smooth_locus)
from .affine_kernel import frame_function, frame_linear, pullbacks
from .centre_weights import (centre_rows, centre_search, component_ideal_product, cover_products, level_weights,
                             maximum_indices, order_fields, ordered_parts, surface_maxima)
from .centres import GiveUp, maximum
from .groebner import colon, groebner
from .incidence import dual_cells, stratum_dimension


@dataclass(frozen=True)
class Chart:
    variables: tuple
    ambient: tuple
    components: tuple   # (label, N, nu, ideal)
    fibre: tuple
    region: tuple       # the part of the chart that is counted
    coordinates: tuple  # (name, expression) for the coordinates of the first chart
    history: tuple      # (stage, index of the pivot) of the blowups leading here
    removed: tuple = () # ideals of centres blown up in this chart: their points are no longer its own
    coefficient_field: tuple = ()  # (parameter, minimal polynomial, degree, chosen embedding)
    torus: tuple = ()  # regular cone rays and common linear map to the input coordinates
    point_blowup: tuple = ()  # parent chart index and rational centre of an ordered plane blowup pair
    split_cone: tuple = ()  # chart rays in the common split base coordinates
    repair_projection: tuple = ()  # parent chart index and its coordinate images on an unmarked repair cover
    # Optional exact root coordinates from the last field extension. The
    # native section solver checks them against the chart's current modulus.
    section_root_images: tuple = field(default=(), compare=False, repr=False)
    coordinate_translation: tuple = ()  # coordinate, polynomial offset and optional construction method
    arrangement_chart: tuple = ()  # projective pivot and optional intersection-line patch

    def one_sheet(self, count):
        """A field parameter describes one fixed embedding. Over Q the kernel
        sees its conjugates, each with the same geometric count."""
        if not self.coefficient_field or count is None:
            return count
        from .chart_kernel import InconsistentSheets, strata_totals
        try:
            return strata_totals([(None, self.coefficient_field[2], count)]).get(None, 0)
        except InconsistentSheets as exc:
            raise GiveUp(str(exc)) from exc


def _covers(variables, removed):
    """Functions whose nonvanishing sets cover the complement of the removed
    centres: the products of one generator of each ideal."""
    return cover_products(variables, removed)


def _present(variables, ideal, removed=()):
    """Whether V(ideal) has a point outside the removed centres."""
    if not removed:
        return not empty_locus(variables, ideal)
    from .kernel import present_outside
    result = present_outside(variables, ideal, removed)
    if result is None:
        result = level_weights("present_outside", (), width=len(variables), ideal=centre_rows(variables, ideal),
                               holes=[centre_rows(variables, z) for z in removed])
    return result


def _dimension(variables, ideal):
    """The dimension of V(ideal), from the leading monomials of a Groebner basis."""
    from . import kernel
    value = kernel.dimension(variables, ideal)
    if value is None:
        raise GiveUp("No native dimension certificate is available for this chart.")
    return value


def _generators(variables, ambient, ideal):
    """Few generators of the ideal modulo the ambient one."""
    if not ideal:
        return ()
    if len(ideal) == 1:
        remainder = groebner(ambient, variables).reduce(ideal[0])[1] if ambient else ideal[0]
        return (remainder,) if remainder else ()
    basis = groebner((*ambient, *ideal), variables).exprs
    selected = level_weights("few_generators", (), width=len(variables), **order_fields(variables, basis),
                             ambient=centre_rows(variables, ambient), basis=centre_rows(variables, basis))
    return tuple(basis[i] for i in selected)


def _expanded(variables, value):
    if getattr(value, "is_Atom", False) or getattr(value, "is_number", False):
        return value
    return frame_function(variables, value)


def _solved(variables, polynomial, variable):
    """Return the solution of a linear chart equation with scalar leading coefficient."""
    if polynomial == variable:
        return chart_expressions(variables).sympify(0)
    if variable not in getattr(polynomial, "free_symbols", ()):
        return None
    pair = frame_linear(variables, polynomial, variable)
    if pair is not None and pair[0].is_number:
        return _expanded(variables, -pair[1]/pair[0])
    return None


def _eliminate(chart):
    """An equation a*v+b=0 with a nonzero scalar exhibits the chart as a
    polynomial graph. Substituting v=-b/a in its defining ideals and every
    carried function is an isomorphism. Zero map coordinates remain in
    place; zero generators of ideals are immaterial."""
    while True:
        found = None
        for g in chart.ambient:
            for v in chart.variables:
                if chart.coefficient_field and v == chart.coefficient_field[0]:
                    continue
                value = _solved(chart.variables, g, v)
                if value is not None:
                    found = v, value
                    break
            if found:
                break
        if not found:
            return chart
        v, value = found
        variables = tuple(w for w in chart.variables if w != v)
        mapping = tuple(value if w == v else w for w in chart.variables)
        ideals = [chart.ambient, *(row[3] for row in chart.components), chart.fibre, chart.region, *chart.removed]
        functions = [*(g for ideal in ideals for g in ideal), *(g for _, g in chart.coordinates)]
        moved = iter(pullbacks(variables, mapping, chart.variables, functions)) if variables else iter(
            chart_expressions(chart.variables).sympify(g).xreplace({v: value}) for g in functions)

        def move(ideal):
            return tuple(dict.fromkeys(g for g in islice(moved, len(ideal)) if g != 0))

        ambient = move(chart.ambient)
        components = tuple((label, n, nu, move(ideal)) for label, n, nu, ideal in chart.components)
        fibre, region = move(chart.fibre), move(chart.region)
        removed = tuple(move(z) for z in chart.removed)
        chart = replace(chart, variables=variables, ambient=ambient, components=components, fibre=fibre, region=region,
                        removed=removed, coordinates=tuple((name, next(moved)) for name, _ in chart.coordinates))


def _failures(chart, dimension):
    """The sets of components whose intersection is not smooth of the expected
    dimension somewhere over the fibre: {labels: ideal of that locus}, and
    those among them seen inside the chart's own region (see
    `_failures_stepwise`). The kernel answers in one call."""
    _, (found,) = _checked_charts([chart], dimension, presence=False)
    return found


def _failures_stepwise(chart, dimension):
    """The arbitrary-width route of `_failures`.

    The chart W is smooth, of codimension c in its affine space. At a point
    of k of the divisors, their intersection is smooth of codimension k in W
    exactly when the Jacobian matrix of the equations of W and of the k
    divisors has rank c + k there (its dimension is at least the expected
    one). The matrix is only needed at the points over the fibre: its
    entries are reduced modulo their ideal.

    The packed and arbitrary-width kernels both compute the rank-defect loci.
    """
    from . import kernel
    labels = [component[0] for component in chart.components]
    divisors = [component[3] for component in chart.components]
    found = kernel.crossings(chart.variables, dimension, chart.ambient, divisors, chart.fibre, chart.region)
    if found is None:
        found = kernel.wide_crossings(chart.variables, dimension, chart.ambient, divisors, chart.fibre, chart.region)
    over = {tuple(labels[i] for i in positions): gens for positions, _, gens in found}
    inside = {tuple(labels[i] for i in positions) for positions, flag, _ in found if flag}
    return over, inside


def _order(variables, ambient, pivot, ideal, limit=64):
    """The order of an ideal along the divisor pivot = 0 of V(ambient): the
    least k such that some generator is not a multiple of pivot^(k+1) there.
    (The divisor is Cartier on a smooth variety, so its powers are its
    symbolic powers.)"""
    rows = centre_rows(variables, (*ambient, pivot, *ideal))
    order = level_weights("cartier_order", (), width=len(variables), ambient=rows[:len(ambient)],
                          pivot=rows[len(ambient)], ideal=rows[len(ambient)+1:], limit=limit)
    if order is None:
        raise GiveUp("A divisor vanishes on a whole chart, or to an order beyond the limit.")
    return order


class Uneven(GiveUp):
    """The parts of a centre are not alike: a divisor has different orders along them."""


def _point_factor_rows(variables, ideal, parameter):
    """Rational factors of the coordinate eliminant, with their multiplicities."""
    from .affine_kernel import rational_factor_rows
    for p in reversed(groebner(tuple(ideal), variables, order="lex").exprs):
        if p.free_symbols <= {parameter}:
            return rational_factor_rows((parameter,), p)[1]
    raise GiveUp("A centre expected to be finite is not.")


def _point_factors(variables, ideal, parameter):
    """Distinct factors and their degrees for splitting a finite centre."""
    from .affine_kernel import _sparse_expression, _sparse_terms
    return [(_sparse_expression((parameter,), row), _sparse_terms(row)[0][0][0])
            for row, _, _ in _point_factor_rows(variables, ideal, parameter)]


def _point_components(variables, ideal):
    """Orbits of a reduced finite centre over Q, via a separating linear form."""
    from . import kernel
    from .affine_kernel import pullbacks
    # V(0) as the only stratum returns chi of the whole closed fibre. For
    # a finite complex variety this counts geometric points with no weights.
    native = kernel.strata(variables, ideal, ((),))
    if native is None:
        from .constructible_euler import euler_characteristic
        count = euler_characteristic(variables, ideal)
    else:
        count = native.get((0,), 0)
    t = chart_expressions(variables).Dummy("form")
    # On d distinct points the number of integer a for which
    # sum a^i*x_i collides is at most (n-1)*binomial(d,2).
    from .chart_kernel import strata_totals
    for form in separating_forms(variables, count, first=1):
        try:
            factors = _point_factors((*variables, t), (*ideal, t - form), t)
            if strata_totals((None, 1, degree) for _, degree in factors).get(None, 0) != count:
                continue
            equations = pullbacks(variables, (form,), (t,), [f for f, _ in factors])
        except ArithmeticError as exc:
            raise GiveUp("The native kernel could not separate this finite centre.") from exc
        parts = [(degree, ideal_basis(variables, (*ideal, eq))) for (_, degree), eq in zip(factors, equations)]
        indices = ordered_parts(variables, [p for _, p in parts], [d for d, _ in parts])
        answer = [parts[i][1] for i in indices]
        return answer
    raise GiveUp("The points of a centre were not separated.")


def _linear(chart, centre, codimension):
    """Generators of a centre that are linear in the coordinates, when the
    chart is an affine space over parameters (the variables of its ambient
    equations) and the centre is cut out by coordinate = polynomial in the
    parameters, for as many coordinates as its codimension. Returns pairs
    (coordinate, polynomial), or None."""
    from .affine_kernel import _sparse_expression
    selected = level_weights("linear_centre", (), width=len(chart.variables), codimension=codimension,
                             ambient=centre_rows(chart.variables, chart.ambient), centre=centre_rows(chart.variables, centre))
    if selected is None:
        return None
    return [(chart.variables[i], _sparse_expression(chart.variables, row)) for i, row in selected]


def _child_variables(chart, linear, stage):
    """For each chart of the blowup of a linear centre: its variables, the new
    coordinate w and the ratios (None at the pivot). They take the places of
    the coordinates of the centre."""
    used = {v.name for v in chart.variables}

    def fresh(name):
        while name in used:
            name += "_"
        used.add(name)
        return chart_expressions(chart.variables).Symbol(name)

    position = {v: i for i, (v, _) in enumerate(linear)}
    out = []
    for j, (pivot, shift) in enumerate(linear):
        w = pivot if shift == 0 else fresh(f"{pivot.name}_{stage}")
        ratios = [None if i == j else fresh(f"r{j + 1}_{i + 1}") for i in range(len(linear))]
        variables = tuple(w if v == pivot else ratios[position[v]] if v in position else v for v in chart.variables)
        out.append((variables, w, ratios))
    return out


def _blow_up_linear(chart, linear, label, stage, codimension):
    """The blowup of a centre x_i = phi_i(parameters), written out: in the
    chart of the j-th coordinate, x_j = phi_j + w and x_i = phi_i + y_i*w for
    the others. A function becomes a polynomial in the new coordinates; its
    coefficients are reduced modulo the equations of the parameters, and its
    order along the new divisor w = 0 is then the least exponent of w.

    The kernel computes this (native/ideal.cpp, op_blowup); the arbitrary-width chart kernel
    answers when it cannot."""
    from . import kernel
    names = _child_variables(chart, linear, stage)
    native = kernel.blowup_linear(
        chart.variables, chart.ambient, [chart.variables.index(v) for v, _ in linear], [shift for _, shift in linear],
        [component[3] for component in chart.components], chart.fibre, chart.region,
        [*(expr for _, expr in chart.coordinates), *(g for z in chart.removed for g in z)])
    if native is None:
        native = _wide_linear_charts(chart, linear, names)
    children = []
    for j, ((variables, w, ratios), raw) in enumerate(zip(names, native)):
        components, weights = [], []
        for (name, n, nu, _), (order, here, rows) in zip(chart.components, raw["divisors"]):
            if here:
                components.append((name, n, nu, kernel.exprs(variables, rows)))
            weights.append((n, nu, order))
        components.append((label, *exceptional_data(codimension, weights), (w,)))
        region = (*kernel.exprs(variables, raw["region"]), *ratios[:j])
        moved = kernel.exprs(variables, raw["functions"])
        removed, at = [], len(chart.coordinates)
        for z in chart.removed:
            removed.append(tuple(g for g in moved[at:at + len(z)] if g != 0))
            at += len(z)
        children.append(Chart(variables, chart.ambient, tuple(components), kernel.exprs(variables, raw["fibre"]), region,
                              tuple((name, expr) for (name, _), expr in zip(chart.coordinates, moved)),
                              (*chart.history, (stage, j)), tuple(removed), chart.coefficient_field))
    return children


def _wide_linear_charts(chart, linear, names):
    """Transport an arbitrary-width linear chart, including any coefficient relation."""
    from .kernel import exprs
    from .polynomial import integer_from_decimal
    axes = chart.variables
    def residual_generators(task):
        variables = names[task["pivot"]][0]
        moved = exprs(variables,task["moved"])
        power = exprs(variables,[task["power"]])
        after = colon((*chart.ambient,*moved),power,variables)
        if after is None:
            raise GiveUp("An ideal quotient is not available.")
        return centre_rows(variables,_generators(variables,chart.ambient,after))
    charts = centre_search("linear_blowup", residual_generators, width=len(axes),
                           axes=[axes.index(v) for v,_ in linear], shifts=centre_rows(axes,[s for _,s in linear]),
                           ambient=centre_rows(axes,chart.ambient), divisors=[centre_rows(axes,c[3]) for c in chart.components],
                           fibre=centre_rows(axes,chart.fibre), region=centre_rows(axes,chart.region),
                           functions=centre_rows(axes,[*(f for _,f in chart.coordinates), *(g for h in chart.removed for g in h)]))
    if isinstance(charts,dict):
        raise GiveUp("A divisor contains a chart of a blowup.")
    for chart_data in charts:
        chart_data["divisors"] = [(integer_from_decimal(m),p,r) for m,p,r in chart_data["divisors"]]
    return charts


def _strict_ideal(variables, ambient, pivot, generators):
    """Order and residual ideal of a pulled-back divisor along the new pivot.

    In the regular chart ring let g_i = p^a_i h_i, with h_i the principal
    residual ideals returned by the kernel. Their common order is m=min a_i.
    Since p is a nonzerodivisor, J:(p^m) = sum_i p^(a_i-m) (h_i). Vanishing
    on an entire component of the pivot after this division still signals
    nonuniform order; the caller checks that before accepting the centre.
    """
    from . import kernel
    nonzero = tuple(g for g in generators if not ideal_subset(variables, (g,), ambient))
    if ambient and len(nonzero) == 1:
        principal = kernel.graph_strict_part(variables, ambient, pivot, nonzero[0])
        if principal is not None:
            return principal
    native = kernel.strict_parts(variables, ambient, ((pivot,),), nonzero) if nonzero else None
    if native:
        if len(native) == 1:
            (order,), after = native[0]
        else:
            from .binomial_kernel import _integer, run
            row, = run(f"residual_orders {len(native)}", tuple(orders[0] for orders, _ in native))
            order, *exponents = map(_integer, row.split())
            residual = [_expanded(variables, g*pivot**power) for power, (_, ideal) in zip(exponents, native)
                        for g in ideal]
            after = ideal_basis(variables, (*ambient, *residual))
        return order, after
    # Fallback for rings outside the kernel.
    order = _order(variables, ambient, pivot, generators, limit=64)
    after = colon((*ambient, *generators), (pivot**order,), variables)
    if after is None:
        raise GiveUp("The native kernel could not compute the residual divisor ideal.")
    return order, after


def _graph_divisor_present(child, after, parent, before, parent_points):
    from .kernel import graph_centre_incidence, graph_parent_slice
    equations = (*child.ambient, *after)
    if parent_points:
        if graph_centre_incidence(child.variables, equations, parent_points):
            return True
        if graph_parent_slice(parent.variables, parent.ambient, before, child.pivot, parent_points):
            return True
    return not empty_locus(child.variables, equations)


def _blow_up(chart, centre, label, stage, dimension, codimension, deadline=None):
    from .kernel import (centre_power_transforms, certify_exceptional_orders, graph_centre_points,
                         residual_dimension_below)
    linear = _linear(chart, centre, codimension)
    if linear:
        return _blow_up_linear(chart, linear, label, stage, codimension)
    gens = _generators(chart.variables, chart.ambient, centre)
    parent_points = ()
    if codimension == dimension and not chart.coefficient_field:
        parent_points = graph_centre_points(chart.variables, (*centre, *chart.ambient))
    charts = blowup_charts(chart.variables, chart.ambient, None, gens)
    shared = {name: centre_power_transforms(chart.variables, chart.ambient, gens, before[0], charts)
              for name, _, _, before in chart.components if len(before) == 1}
    children = []
    for child in charts:
        if deadline is not None and monotonic() > deadline:
            raise GiveUp("The time allowed for the blowups is over.")
        if empty_locus(child.variables, child.ambient):
            continue
        pivot = (child.pivot,)
        base = (*child.ambient, child.pivot)
        candidates = {name: (value[0], (value[1][child.pivot_index],))
                      for name, value in shared.items() if value is not None}
        certified = {}
        if candidates:
            orders = certify_exceptional_orders(child.variables, base, tuple(q for _, (q,) in candidates.values()),
                                                stratum_dimension(dimension, 1))
            if orders is not None:
                certified = {name: value for (name, value), ready in zip(candidates.items(), orders) if ready}
        components = []
        weights = []
        for old, n, nu, before in chart.components:
            # The transform of a divisor is its pullback with the new divisor
            # removed as often as it occurs: an ideal quotient by that power of
            # the pivot. The order must be the same on every component of the
            # new divisor: what is left may vanish on none of them.
            if old in certified:
                order, after = certified[old]
            else:
                order, after = _strict_ideal(child.variables, child.ambient, child.pivot, before)
                proper = residual_dimension_below(child.variables, base, after, stratum_dimension(dimension, 1))
                if proper is None:
                    raise GiveUp("No native certificate for the residual divisor dimension is available.")
                if not proper:
                    raise Uneven("A divisor has different orders along parts of one exceptional divisor.")
            if _graph_divisor_present(child, after, chart, before, parent_points):
                components.append((old, n, nu, after))
            weights.append((n, nu, order))
        components.append((label, *exceptional_data(codimension, weights), pivot))
        region = (*chart.region, *(r for j, r in enumerate(child.ratios) if j < child.pivot_index))
        # Few generators for every divisor: the transforms come with many redundant ones.
        components = tuple((name, n, nu, _generators(child.variables, child.ambient, ideal))
                           for name, n, nu, ideal in components)
        children.append(_eliminate(Chart(child.variables, child.ambient, components, chart.fibre, region,
                                         chart.coordinates, (*chart.history, (stage, child.pivot_index)),
                                         chart.removed, chart.coefficient_field)))
    return children


def _is_strict(label):
    return label == "strict" or (label[:1] == "S" and label[1:].isdigit())


def _hypersurface(chart):
    """Generators of the ideal of the strict transform: the product of the ideals of its components."""
    ideals = [ideal for label, _, _, ideal in chart.components if _is_strict(label)]
    gens = component_ideal_product(chart.variables, ideals)
    return _generators(chart.variables, chart.ambient, gens)


def _radical_points(variables, ideal):
    """Seidenberg's radical of a finite ideal, using the squarefree coordinate eliminants."""
    from .affine_kernel import _sparse_expression
    from .kernel import finite_radical
    answer = finite_radical(variables, ideal)
    if answer is not None:
        return answer
    try:
        extra = []
        for v in variables:
            order = (*(w for w in variables if w != v), v)
            factors = _point_factor_rows(order, ideal, v)
            # A squarefree eliminant is already in the input ideal.
            if all(multiplicity == 1 for _, multiplicity, _ in factors):
                continue
            distinct = (_sparse_expression((v,), row) for row, _, _ in factors)
            extra.append(frame_function((v,), chart_expressions((v,)).prod(distinct)))
        answer = ideal_basis(variables, (*ideal, *extra))
    except ArithmeticError as exc:
        raise GiveUp("The native kernel could not reduce this finite centre.") from exc
    return answer


def _reduced(chart, locus):
    """The reduced structure of a locus of dimension at most one, with its dimension."""
    ideal = ideal_basis(chart.variables, (*chart.ambient, *locus))
    dimension = _dimension(chart.variables, ideal)
    if dimension == 0:
        return _radical_points(chart.variables, ideal), 0
    if dimension == 1:
        return ideal, 1
    raise GiveUp("Normal crossings fail along a surface.")


class _Direct:
    """A short rule, kept as a last resort. Among the sets of components whose
    intersection is not smooth of the expected dimension somewhere over the
    fibre, take one with the most components (then the first in the order
    of the components), and let Z be the reduced locus over the fibre where
    it fails. If some point of Z is not a smooth point of a curve, the
    centre is the set of those points; otherwise it is the curve Z. The
    rule depends only on the geometry, so the charts glue. It reaches normal
    crossings in few blowups when it does, but it need not terminate: it is
    given few stages."""
    name, stages, retains = "direct", 6, False

    def choose(self, current, failures, dimension, deadline):
        labels = [component[0] for chart in current for component in chart.components]
        chosen = tuple(level_weights("direct_stratum", (), labels=labels,
                                     subsets=[group for _, inside in failures for group in inside]))
        loci = [_reduced(chart, over[chosen]) if chosen in over else None for chart, (over, _) in zip(current, failures)]
        special = []
        for chart, locus in zip(current, loci):
            if locus is None:
                special.append(None)
            elif locus[1] == 0:
                special.append(locus[0])
            else:
                bad = smooth_locus(chart.variables, locus[0], 1)["bad_ideal"]
                special.append(None if bad == (1,) else _reduced(chart, bad)[0])
        points = level_weights("direct_locus", (), parts=[s is not None for s in special])
        centres = [marked if points else (locus[0] if locus else None) for locus, marked in zip(loci, special)]
        return centres, {"components": list(chosen)}

    def created(self, label, detail):
        pass


class _Canonical:
    """The complete rule. While the strict transform X is singular at a
    point where normal crossings fail, the centre is the locus where the
    invariant of Bierstone and Milman is largest among those points
    (engine/centres.py). Once X is smooth at all of them, in dimension
    three, the procedure for a smooth surface takes over (engine/surface.py):
    it keeps X smooth, so the first phase does not come back. Where the
    surface has no plane model the invariant is used to the end, starting
    again from the divisors present. Each phase is finite."""
    # With `retains` a chart that is blown up stays, for its points off the
    # centre, and the charts of the blowup own only the new divisor: what is
    # far from a centre keeps the equations it had.
    name, stages, retains = "complete", 150, True

    def __init__(self, components, *, shared_curves=True):
        self.shared_curves = shared_curves
        self._loci, self._steps = {}, {}
        # For each exceptional divisor, the invariant of the centre that created it
        # (None for those given at the start), and the divisors in order of creation.
        self.records = {label: None for label, *_ in components if not _is_strict(label)}
        self.births = list(self.records)
        self.mode = "order"

    def _phase(self, current, failures, dimension):
        """Classify the strict transforms and select the active charts in one native call."""
        charts = []
        for chart, (over, _) in zip(current, failures):
            if not over:
                charts.append(None)
                continue
            variables = chart.variables
            charts.append(dict(width=len(variables), ambient=centre_rows(variables, chart.ambient),
                               strict=[dict(index=i, ideal=centre_rows(variables, ideal))
                                       for i, (label, _, _, ideal) in enumerate(chart.components) if _is_strict(label)],
                               strict_labels=[[_is_strict(label) for label in key] for key in over],
                               failures=[centre_rows(variables, ideal) for ideal in over.values()]))
        return level_weights("canonical_phase", (), phase=self.mode, dimension=dimension, charts=charts)

    def _locus(self, chart, bad, dimension, deadline):
        """The largest value of the invariant at the points of the chart where
        normal crossings fail, and the locus where the invariant has that value.

        The locus found may pass through other points over the fibre. If the
        invariant is larger at one of them, the blowup would not be admissible
        there: those points join the set over which the maximum is taken."""
        # A chart that has not changed gives the same answer: the records of its divisors are fixed.
        if id(chart) in self._loci:
            return self._loci[id(chart)][1]
        exceptional = [(label, ideal) for label, _, _, ideal in chart.components if not _is_strict(label)]
        strict = _hypersurface(chart)
        working = list(bad)
        for _ in range(8):
            item = maximum(chart.variables, chart.ambient, dimension, strict, exceptional, working,
                           self.records, self.births, deadline, away=chart.removed)
            if item is None:
                break
            on_fibre = (*item[2], *chart.fibre, *chart.region)
            if any(radical_subset(chart.variables, w, (*chart.ambient, *on_fibre)) for w in working):
                break  # the points of the locus over the fibre are among those considered
            check = maximum(chart.variables, chart.ambient, dimension, strict, exceptional, [on_fibre],
                            self.records, self.births, deadline, away=chart.removed)
            if check is None or 0 in maximum_indices((item,check)):
                break
            working.append(on_fibre)
        else:
            raise GiveUp("The locus of the invariant keeps meeting points of larger invariant.")
        self._loci[id(chart)] = chart, item
        return item

    def _on_surface(self, current, failures, surfaces, deadline=None):
        from .surface import step
        found = []
        for chart, (over, _), surface in zip(current, failures, surfaces):
            if not over:
                found.append(None)
                continue
            if surface is None:
                raise GiveUp("The strict transform is no longer a smooth surface at the points considered.")
            exceptional = [(label, ideal) for label, _, _, ideal in chart.components if not _is_strict(label)]
            if id(chart) not in self._steps:
                self._steps[id(chart)] = chart, step(chart.variables, chart.ambient, surface, exceptional,
                                                     list(over.values()), deadline, away=chart.removed)
            found.append(self._steps[id(chart)][1])
        selected = surface_maxima(found)
        if not selected:
            raise GiveUp("Normal crossings fail where the procedure for a surface has nothing to do.")
        key, kind, _ = found[selected[0]]
        if self.shared_curves and kind == "points":
            from .surface import shared_boundary_curve
            shared = shared_boundary_curve(current, surfaces, key)
            if shared is not None:
                return shared, {"rule": "shared boundary curve"}
        centres = [None] * len(current)
        for i in selected:
            _, kind, locus = found[i]
            centres[i] = _radical_points(current[i].variables, locus) if kind == "points" else locus
        return centres, {"rule": "surface"}

    def choose(self, current, failures, dimension, deadline):
        phase = self._phase(current, failures, dimension)
        self.mode = phase["phase"]
        if self.mode == "surface":
            from .surface import NoModel
            try:
                surfaces = [None if index is None else chart.components[index][3]
                            for chart, index in zip(current, phase["surfaces"])]
                return self._on_surface(current, failures, surfaces, deadline)
            except NoModel:
                # The invariant from here on, with the divisors present as the given ones.
                self.mode = "invariant"
                self._loci = {}
                labels = dict.fromkeys(label for chart in current for label, *_ in chart.components
                                       if not _is_strict(label))
                self.records, self.births = dict.fromkeys(labels), list(labels)
        found = [None] * len(current)
        for i in phase["charts"]:
            found[i] = self._locus(current[i], list(failures[i][0].values()), dimension, deadline)
        selected = maximum_indices(found, birth_width=len(self.births))
        if not selected:
            raise GiveUp("Normal crossings fail away from the strict transform.")
        centres = [None] * len(current)
        for i in selected:
            centres[i] = found[i][2]
        value = found[selected[0]][0]
        return centres, {"invariant": ["oo" if entry == float("inf") else str(entry) for entry in value],
                         "value": value, "rule": "invariant"}

    def created(self, label, detail):
        if "value" in detail:
            self.records[label] = detail["value"]
        else:
            self.records[label] = None
        self.births.append(label)


def _radical(chart, ideal):
    """The radical of an ideal on the chart, from the native decomposition kernel."""
    from .native_library import affine_request
    from .native_polynomial import from_terms, row
    from .polynomial import sparse_row
    names = " ".join(v.name for v in chart.variables)
    rows = affine_request(["ideal_radical", names, "grevlex", *(row(chart.variables, g) for g in ideal)])
    return tuple(ideal_basis(chart.variables, [from_terms(chart.variables, sparse_row(r)) for r in rows]))


class _Order(_Canonical):
    """The maximal order rule. While the strict transform X is singular at a
    point where normal crossings fail, the centre is the locus near the fibre
    where the order of X is largest among those points: the first entry of
    the invariant (native/centre_maximum.cpp with order_only), which needs no
    hypersurface of maximal contact. The locus is taken in the whole chart,
    so a curve or surface of singular points that leaves the fibre is blown
    up whole. A singular union may have smooth parts distinguished by an
    exceptional divisor. Otherwise descend to its singular locus. A smooth
    centre meeting the boundary improperly is treated first along that
    defect. Blowing up a smooth centre inside the
    locus of maximal order does not raise the order. The rule depends only
    on the geometry, so the charts glue; it need not terminate and is given
    few stages. Once X is smooth at the points, the complete rule continues,
    with the divisors present as given ones."""
    name, stages, retains = "order", 60, True

    # Stages in a row at the same order with strictly more charts selected
    # each time: the rule is not converging, and the next rule gets the time.
    RUNAWAY = 6

    def __init__(self, components, *, boundary_centres=True):
        super().__init__(components)
        self.boundary_centres = boundary_centres
        self.order_phase = True
        self._trend = (None, 0, 0)  # (order, charts selected, stages in a row)

    def _order_locus(self, chart, bad, dimension, deadline):
        key = ("order", id(chart))
        if key not in self._loci:
            exceptional = [(label, ideal) for label, _, _, ideal in chart.components if not _is_strict(label)]
            item = maximum(chart.variables, chart.ambient, dimension, _hypersurface(chart), exceptional, list(bad),
                           self.records, self.births, deadline, away=chart.removed, order_only=True)
            self._loci[key] = chart, item
        return self._loci[key][1]

    @staticmethod
    def _owned_parts(current, loci):
        """The reduced components of the loci that meet an owned point over
        the fibre, of the largest dimension among all charts. Components far
        from the fibre play no part, and a chart that does not own a
        component's points does not blow it up."""
        kept = {}
        for i, locus in loci.items():
            chart = current[i]
            parts = []
            for part in _prime_parts(chart, _radical(chart, (*chart.ambient, *locus)), None):
                if _present(chart.variables, (*chart.ambient, *chart.fibre, *chart.region, *part), chart.removed):
                    parts.append((_dimension(chart.variables, (*chart.ambient, *part)), part))
            kept[i] = parts
        sizes = [size for parts in kept.values() for size, _ in parts]
        centres = [None] * len(current)
        if not sizes:
            return centres
        # The native comparison of invariant rows selects the largest dimension.
        top = sizes[maximum_indices([((Fraction(size),), ()) for size in sizes])[0]]
        for i, parts in kept.items():
            chart = current[i]
            for size, part in parts:
                if size == top:
                    centres[i] = part if centres[i] is None else _union(chart.variables, centres[i], part)
        return centres

    @staticmethod
    def _singular_centres(current, centres):
        singular = []
        for chart, centre in zip(current, centres):
            if centre is None:
                singular.append(None)
                continue
            ideal = (*chart.ambient, *centre)
            size = _dimension(chart.variables, ideal)
            if size < 0:
                raise GiveUp("A centre is not smooth.")
            proof = smooth_locus(chart.variables, ideal, size)
            smooth = proof["smooth"] or chart.removed and not _present(
                chart.variables, (*chart.ambient, *proof["bad_ideal"]), chart.removed)
            singular.append(None if smooth else proof["bad_ideal"])
        return singular

    @staticmethod
    def _boundary_defects(current, centres):
        from .kernel import exprs
        from .kernel_json import kernel_json
        defects = {}
        for i, (chart, centre) in enumerate(zip(current, centres)):
            if centre is None:
                continue
            variables = chart.variables
            divisors = [ideal for label, _, _, ideal in chart.components if not _is_strict(label)]
            if not divisors:
                continue
            ideal = (*chart.ambient, *centre)
            rows = kernel_json("atlas_graph_wide", "centre_boundary_defect", width=len(variables),
                               dimension=str(_dimension(variables, ideal)), ideal=centre_rows(variables, ideal),
                               divisors=[centre_rows(variables, divisor) for divisor in divisors])
            if rows is not None:
                defect = exprs(variables, rows)
                if _present(variables, (*defect, *chart.fibre, *chart.region), chart.removed):
                    defects[i] = defect
        return defects

    def _smooth_centres(self, current, centres):
        from .native_library import Unavailable
        used_boundary = False
        for depth in range(4):
            singular = self._singular_centres(current, centres)
            if self.boundary_centres and depth == 0 and any(bad is not None for bad in singular):
                try:
                    parts = _apart(current, centres)
                    parts_singular = self._singular_centres(current, parts)
                except (GiveUp, Unavailable):
                    # Distinguishing a branch is optional. A divisor need
                    # only be locally principal on this chart.
                    pass
                else:
                    if all(bad is None for bad in parts_singular):
                        centres, singular, used_boundary = parts, parts_singular, True
            if all(bad is None for bad in singular):
                defects = self._boundary_defects(current, centres) if self.boundary_centres else {}
                if not defects:
                    return centres, used_boundary
                centres = self._owned_parts(current, defects)
                used_boundary = True
                continue
            # A chart where the locus is smooth contains none of its singular points.
            following = []
            for chart, centre, bad in zip(current, centres, singular):
                if centre is None or bad is None:
                    following.append(None)
                    continue
                locus = _radical(chart, (*chart.ambient, *centre, *bad))
                following.append(None if locus == (1,) else locus)
            centres = following
            if all(centre is None for centre in centres):
                break
        raise GiveUp("The singular locus of the maximal order locus did not become smooth.")

    def choose(self, current, failures, dimension, deadline):
        if self.order_phase:
            found = [None] * len(current)
            for i, (over, _) in enumerate(failures):
                if over:
                    found[i] = self._order_locus(current[i], list(over.values()), dimension, deadline)
            selected = maximum_indices(found, birth_width=0)
            # A maximal order above one: the strict transform is singular at some point.
            if selected and maximum_indices([found[selected[0]][:2], ((Fraction(1),), ())]) == [0]:
                value = found[selected[0]][0]
                order, count, streak = self._trend
                growing = order == value and len(selected) > count
                self._trend = (value, len(selected), streak + 1 if growing else 0)
                if self._trend[2] >= self.RUNAWAY:
                    raise GiveUp("The maximal order rule selects more charts at the same order at every stage.")
                centres = self._owned_parts(current, {i: found[i][2] for i in selected})
                centres, used_boundary = self._smooth_centres(current, centres)
                return centres, {"invariant": [str(entry) for entry in value], "value": value,
                                 "rule": "maximal order with boundary" if used_boundary else "maximal order"}
            # X is smooth at every point where normal crossings fail.
            self.order_phase = False
            self._loci = {}
            labels = dict.fromkeys(label for chart in current for label, *_ in chart.components
                                   if not _is_strict(label))
            self.records, self.births = dict.fromkeys(labels), list(labels)
        return super().choose(current, failures, dimension, deadline)


def _native_stage(operation, **fields):
    """One call of native/canonical_blowup.cpp; None when a packed operation
    is out of range and the arbitrary-width route has to answer."""
    from .centres import native_failure
    from .kernel_json import kernel_json
    answer = kernel_json("atlas_canonical_blowup", operation, **fields)
    if isinstance(answer, dict) and "declined" in answer:
        return None
    if isinstance(answer, dict) and "raise" in answer:
        raise native_failure(answer, Uneven=Uneven)
    return answer


def _packed_present():
    """Whether emptiness may use the packed open-locus test, as in blowup_ideal.empty_locus."""
    from .algebra_execution import current_execution
    from .native_library import current_replay_scope
    scope = current_replay_scope()
    return current_execution() is None and (scope is None or scope.compute == "cpu")


def _native_charts(charts):
    """Whether the native stage can take these charts: their coordinates are native polynomial symbols."""
    from .native_polynomial import native_ring
    return all(native_ring(chart.variables) for chart in charts)


def _chart_fields(chart, centre=None, *, coordinates=True, **flags):
    from .native_polynomial import row
    from .polynomial import integer_text
    variables = chart.variables

    def rows(ideal):
        return [row(variables, g) for g in ideal]

    return dict(names=[v.name for v in variables], identities=[v._coords[0].identity for v in variables],
                theta=variables.index(chart.coefficient_field[0]) if chart.coefficient_field else -1,
                ambient=rows(chart.ambient), fibre=rows(chart.fibre), region=rows(chart.region),
                components=[[label, integer_text(n), integer_text(nu), rows(ideal)]
                            for label, n, nu, ideal in chart.components],
                coordinates=rows(f for _, f in chart.coordinates) if coordinates else [],
                removed=[rows(z) for z in chart.removed], centre=None if centre is None else rows(centre), **flags)


def _chart_from_fields(parent, item, stage):
    from .kernel import exprs
    from .polynomial import integer_from_decimal
    api = chart_expressions(parent.variables)
    variables = tuple(parent.variables[source] if isinstance(source, int) else api.Symbol(source)
                      for source in item["variables"])

    def ideal(rows):
        return exprs(variables, rows)

    components = tuple((label, integer_from_decimal(n), integer_from_decimal(nu), ideal(rows))
                       for label, n, nu, rows in item["components"])
    coordinates = tuple((name, f) for (name, _), f in zip(parent.coordinates, ideal(item["coordinates"])))
    return Chart(variables, ideal(item["ambient"]), components, ideal(item["fibre"]), ideal(item["region"]),
                 coordinates, (*parent.history, (stage, item["pivot"])), tuple(ideal(z) for z in item["removed"]),
                 parent.coefficient_field)


def _blown_up(current, chosen, label, stage, dimension, size, deadline, retains=False, parts=None):
    """The charts after the blowup of the centre given in each chart (None where it does not meet the chart).

    With `retains` the chart stays for its points off the centre, and each
    chart of the blowup owns its part of the new divisor only. `parts` then
    lists, for each chart, the prime parts of its centre: the kernel blows
    them up one by one when they meet only inside the centres removed from
    the chart, where their union is singular (native/canonical_blowup.cpp).

    The kernel computes the whole stage (native/canonical_blowup.cpp); the
    step-by-step route below answers for rings beyond its packed range.
    Either way the parts of a centre in different charts must give the new
    divisor the same numerical data."""
    if _native_charts(chart for chart, centre in zip(current, chosen) if centre is not None):
        from .native_polynomial import row
        pieces = parts if parts is not None else [None] * len(current)
        answer = _native_stage("stage", charts=[
            dict(centre=None) if centre is None else
            _chart_fields(chart, centre) if not own else
            _chart_fields(chart, centre, parts=[[row(chart.variables, g) for g in part] for part in own])
            for chart, centre, own in zip(current, chosen, pieces)],
                               label=label, stage=stage, dimension=dimension, size=size, retains=bool(retains),
                               seconds=None if deadline is None else deadline - monotonic(),
                               packed_present=_packed_present())
        if answer is not None:
            following = []
            for item in answer["charts"]:
                if "keep" in item:
                    following.append(current[item["keep"]])
                elif "retain" in item:
                    chart = current[item["retain"]]
                    older = tuple(chart.removed[i] for i in item["holes"])
                    following.append(replace(chart, removed=(*older, tuple(chosen[item["retain"]]))))
                else:
                    following.append(_chart_from_fields(current[item["parent"]], item, stage))
            return following
    if parts is not None:
        raise Uneven("The parts of this centre can only be blown up one by one by the native stage.")
    following = _blown_up_stepwise(current, chosen, label, stage, dimension, size, deadline, retains)
    values = [[str(n), str(nu)] for chart in following for name, n, nu, _ in chart.components if name == label]
    if values and level_weights("common_value", (), values=values) is None:
        raise Uneven("The parts of a centre give the new divisor different numerical data.")
    return following


def _blown_up_stepwise(current, chosen, label, stage, dimension, size, deadline, retains=False):
    """The arbitrary-width route of `_blown_up`: each native operation called in turn."""
    following = []
    codimension = stratum_dimension(dimension, size)
    for chart, centre in zip(current, chosen):
        if centre is None:
            following.append(chart)
            continue
        if not retains:
            following.extend(_blow_up(chart, centre, label, stage, dimension, codimension, deadline))
            continue
        # The old charts partition the fibre. A point of the new divisor can
        # be owned by this chart only if its image is an owned centre point.
        if _present(chart.variables, (*chart.ambient, *chart.fibre, *chart.region, *centre), chart.removed):
            for child in _blow_up(chart, centre, label, stage, dimension, codimension, deadline):
                new = next((ideal for name, _, _, ideal in child.components if name == label), None)
                if new is not None:
                    following.append(replace(child, region=tuple(dict.fromkeys((*child.region, *new)))))
        centre = tuple(centre)
        older = ()
        if chart.removed:
            kept = level_weights("retained_centres", (), width=len(chart.variables),
                                 ambient=centre_rows(chart.variables, chart.ambient),
                                 centre=centre_rows(chart.variables, centre),
                                 holes=[centre_rows(chart.variables, hole) for hole in chart.removed])
            older = tuple(chart.removed[i] for i in kept)
        following.append(replace(chart, removed=(*older, centre)))
    return following


def _alike(current, chosen, single=False):
    """Of a finite centre, the points lying on the largest set of divisors
    (the sets are compared by size, then by their labels): this does not
    depend on the chart. The divisors, defined over Q, have the same orders
    at conjugate points. An individual orbit can be selected by its local
    index when only one chart contains any of the eligible points."""
    groups = [[] if centre is None else
              _point_components(chart.variables, ideal_basis(chart.variables, (*chart.ambient, *centre)))
              for chart, centre in zip(current, chosen)]
    signatures = [[[name for name, _, _, ideal in chart.components
                    if not empty_locus(chart.variables, (*chart.ambient, *part, *ideal))]
                   for part in group] for chart, group in zip(current, groups)]
    selected = level_weights("point_signatures", (), charts=signatures, single=single)
    if selected is None:
        raise Uneven("The divisor labels do not distinguish a compatible point centre across charts.")
    out = []
    for chart, group, indices in zip(current, groups, selected):
        parts = [group[i] for i in indices]
        if not parts:
            out.append(None)
            continue
        centre = parts[0]
        for other in parts[1:]:
            centre = _union(chart.variables, centre, other)
        out.append(centre)
    return out


def _apart(current, chosen):
    """Of a centre made of curves, those contained in the first divisor (in
    the order of the labels) that contains some of them and not the others."""
    from .centres import _principal
    labels = level_weights("label_order", (), labels=[name for chart in current for name, *_ in chart.components
                                                    if not _is_strict(name)])
    def owned(chart, part):
        return part is not None and _present(chart.variables, (*chart.ambient, *chart.fibre, *chart.region, *part),
                                             chart.removed)

    for label in labels:
        inside, rests, outside = [], [], False
        for chart, centre in zip(current, chosen):
            ideal = next((i for name, _, _, i in chart.components if name == label), None)
            if centre is None:
                inside.append(None)
                rests.append(None)
                continue
            whole = tuple(ideal_basis(chart.variables, (*chart.ambient, *centre)))
            if ideal is None:
                inside.append(None)
                rests.append(whole)
                outside = True
                continue
            rest = tuple(ideal_basis(chart.variables,
                                     saturate(chart.variables, whole, _principal(chart.variables, chart.ambient, ideal))))
            if rest == (1,):
                inside.append(whole)
                rests.append(None)
                continue
            outside = True
            part = colon(whole, rest, chart.variables)
            if part is None:
                raise GiveUp("An ideal quotient is not available.")
            part = tuple(ideal_basis(chart.variables, part))
            inside.append(None if part == (1,) else part)
            rests.append(rest)
        side = level_weights("split_curve_owned", (), outside=outside, parts=[part is not None for part in inside],
                             inside_owned=[owned(chart, part) for chart, part in zip(current, inside)],
                             outside_owned=[owned(chart, rest) for chart, rest in zip(current, rests)])
        if side == "inside":
            return inside
        if side == "outside":
            return rests
    raise Uneven("A divisor has different orders along curves that no divisor separates.")


def _prime_parts(chart, centre, size):
    """The prime components over Q of a smooth centre on the chart."""
    ideal = ideal_basis(chart.variables, (*chart.ambient, *centre))
    if ideal == (1,):
        return []
    if size == 0:
        return _point_components(chart.variables, ideal)
    from .native_library import affine_request
    from .native_polynomial import from_terms, row
    from .polynomial import sparse_row
    names = " ".join(v.name for v in chart.variables)
    rows = iter(affine_request(["primary_decomposition", names, "grevlex",
                                *(row(chart.variables, g) for g in ideal)]))
    parts = []
    for _ in range(int(next(rows))):
        primary, prime = map(int, next(rows).split())
        for _ in range(primary):
            next(rows)
        generators = [from_terms(chart.variables, sparse_row(next(rows))) for _ in range(prime)]
        parts.append(tuple(ideal_basis(chart.variables, (*chart.ambient, *generators))))
    return parts


def _by_numerical_data(current, chosen, label, stage, dimension, size, deadline, retains):
    """Of a centre whose parts give the new divisor different numerical data,
    the prime parts with the largest data (native/centre_weights.cpp,
    "largest_divisor_data"). The others are blown up at later stages."""
    parts, data = [], []
    for i, (chart, centre) in enumerate(zip(current, chosen)):
        pieces = [] if centre is None else _prime_parts(chart, centre, size)
        found = []
        for piece in pieces:
            trial = [None] * len(current)
            trial[i] = piece
            try:
                following = _blown_up(current, trial, label, stage, dimension, size, deadline, retains)
            except Uneven:
                found.append(None)
                continue
            found.append(sorted({(str(n), str(nu)) for child in following
                                 for name, n, nu, _ in child.components if name == label}))
        parts.append(pieces)
        data.append(found)
    selected = level_weights("largest_divisor_data", (), charts=[[None if values is None else
                                                                   [list(pair) for pair in values]
                                                                   for values in found] for found in data])
    if not selected or not any(selected):
        raise Uneven("No prime part of the centre gives its new divisor one set of numerical data.")
    kept = [[pieces[k] for k in indices] for pieces, indices in zip(parts, selected)]
    return _united(current, kept), kept


def _united(current, kept):
    """In each chart the union of the listed parts of a centre, or None."""
    out = []
    for chart, pieces in zip(current, kept):
        centre = None
        for piece in pieces:
            centre = piece if centre is None else _union(chart.variables, centre, piece)
        out.append(centre)
    return out


def _union(variables, first, second):
    from .centres import _intersection
    return tuple(ideal_basis(variables, _intersection(variables, first, second)))


def _own_failures(chart, dimension, known):
    """The failures of normal crossings at the points that the chart owns:
    those of its region, off the centres removed from it. In the form of
    `_failures`; `known` keeps the answer for a chart that does not change."""
    if id(chart) not in known:
        _checked_charts([chart], dimension, known, presence=False)
    return known[id(chart)][1]


def _owned(chart, over, inside, owned):
    """Keep the failures seen in the chart's own part, in the order of `inside`."""
    own = {}
    for labels in inside:
        here, ideal = owned(labels)
        if here:
            own[labels] = ideal
    return own


def _checked_charts(current, dimension, known=None, *, presence=True, crossings=True, seen=None):
    """The charts that still own a point over the fibre (all of them without
    `presence`) and, with `crossings`, their failures of normal crossings:
    those of `_own_failures`, kept in `known`, or with `known` None those of
    `_failures`. `seen` keeps whether a chart that does not change owns such
    a point. The kernel answers for all the new charts in one call
    (native/canonical_blowup.cpp, "failures")."""
    from .kernel import exprs
    own = known is not None
    seen = {} if seen is None else seen
    check = [presence and id(chart) not in seen for chart in current]
    asked = [crossings and not (own and id(chart) in known) for chart in current]
    sent = [i for i, chart in enumerate(current)
            if check[i] or (asked[i] and (not presence or seen[id(chart)][1]))]
    answers = {}
    if sent:
        if not _native_charts(current[i] for i in sent):
            return _checked_stepwise(current, dimension, known, presence, crossings)
        found = _native_stage("failures", dimension=dimension, packed_present=_packed_present(),
                              charts=[_chart_fields(current[i], coordinates=False, check=check[i], crossings=asked[i],
                                                    own=own) for i in sent])
        if found is None:
            return _checked_stepwise(current, dimension, known, presence, crossings)
        answers = dict(zip(sent, found))
    kept, failures = [], []
    for i, chart in enumerate(current):
        answer = answers.get(i, {})
        if check[i]:
            seen[id(chart)] = chart, answer["present"]
        if presence and not seen[id(chart)][1]:
            continue
        kept.append(chart)
        if not crossings:
            continue
        if asked[i]:
            labels = [component[0] for component in chart.components]
            over = {tuple(labels[i] for i in positions): exprs(chart.variables, rows)
                    for positions, _, rows in answer["loci"]}
            inside = {tuple(labels[i] for i in positions) for positions, flag, _ in answer["loci"] if flag}
            if not own:
                failures.append((over, inside))
                continue
            owned = {tuple(names): (here, tuple(exprs(chart.variables, rows))) for names, here, rows in answer["own"]}
            mine = _owned(chart, over, inside, owned.__getitem__)
            known[id(chart)] = chart, (mine, set(mine))
        failures.append(known[id(chart)][1])
    return kept, failures


def _checked_stepwise(current, dimension, known, presence, crossings):
    """The arbitrary-width route of `_checked_charts`."""
    if presence:
        current = [c for c in current if _present(c.variables, (*c.ambient, *c.fibre, *c.region), c.removed)]
    if not crossings:
        return current, []
    failures = []
    for chart in current:
        if known is None:
            failures.append(_failures_stepwise(chart, dimension))
            continue
        if id(chart) not in known:
            over, inside = _failures_stepwise(chart, dimension)

            def owned(labels):
                ideal = tuple(ideal_basis(chart.variables, (*over[labels], *chart.region)))
                return _present(chart.variables, ideal, chart.removed), ideal

            mine = _owned(chart, over, inside, owned)
            known[id(chart)] = chart, (mine, set(mine))
        failures.append(known[id(chart)][1])
    return current, failures


def _smooth_strict(current, dimension, failures, owned):
    """The origin solver carries the whole reduced strict hypersurface as one divisor."""
    if failures and (owned or all(not c.removed for c in current)):
        # Each singleton already has its native Jacobian rank-defect locus.
        return level_weights("smooth_strict", (), failures=[labels for _, inside in failures for labels in inside])
    from .repeated_tangent import fibre_has_crossings
    return all(fibre_has_crossings(replace(c, components=tuple(row for row in c.components if row[0] == "strict")),
                                   dimension) for c in current)


def _run(rule, variables, components, fibre, budget, deadline, ambient=(), dimension=None, count_strata=True,
         max_blowups=None, intrinsic=False, state=None):
    """The blowups of one rule. `state` keeps the charts and centres at the
    start of the stage being computed: a run that was stopped for its time
    continues from that stage when called again with the same rule and state."""
    if intrinsic and (count_strata or len(components) != 1 or components[0][0] != "strict"):
        raise ValueError("Intrinsic stopping requires one reduced strict equation and no embedded strata.")
    dimension = len(variables) if dimension is None else dimension
    state = {} if state is None else state
    current = state.get("current") or [
        Chart(variables, tuple(ambient), tuple((label, n, nu, tuple(ideal)) for label, n, nu, ideal in components),
              tuple(fibre), (), tuple((v.name, v) for v in variables), ())]
    centres, known, seen = state.setdefault("centres", []), state.setdefault("known", {}), state.setdefault("seen", {})
    first = state.get("stage", 1)
    stages = (count(first) if budget is None and type(rule) is _Canonical else
              range(first, rule.stages + (_Canonical.stages if type(rule) is _Order else 0) + 1))
    for stage in stages:
        state.update(current=current, stage=stage, progress=monotonic())
        report(f"Checking charts for normal crossings ({rule.name} method)", len(centres))
        if deadline is not None and monotonic() > deadline:
            raise GiveUp("The time allowed for the blowups is over.")
        retains = getattr(rule, "retains", False)
        final = hasattr(rule, "final_check") and rule.final_check()
        needs = not final and (not hasattr(rule, "needs_failures") or rule.needs_failures())
        # The charts that still own points over the fibre, and where normal crossings fail there.
        current, failures = _checked_charts(current, dimension, known if retains else None, crossings=needs,
                                            seen=seen)
        # An older intrinsic archive may continue past the first smooth stage.
        # Replay all its recorded blowups and verify smoothness at its endpoint.
        if intrinsic and (max_blowups is None or len(centres) == max_blowups):
            if _smooth_strict(current, dimension, failures, retains):
                return {}, {}, current, centres
        if final:
            from .repeated_tangent import certify_fibre
            for chart in current:
                certify_fibre(chart, dimension)
            return (*(_strata(current) if count_strata else ({}, {})), current, centres)
        if needs:
            if not any(inside for _, inside in failures):
                return (*(_strata(current) if count_strata else ({}, {})), current, centres)
        if max_blowups is not None and len(centres) >= max_blowups:
            raise GiveUp("The recorded blowups end before the canonical charts reach normal crossings.")
        report(f"Searching for the next centre ({rule.name} method)", len(centres))
        chosen, detail = rule.choose(current, failures, dimension, deadline)
        if all(centre is None for centre in chosen):
            raise GiveUp("No centre was found.")
        label = f"B{len(centres) + 1}"
        # Of a finite centre, only the points over the fibre: the others play no part
        # (on a cover that makes conjugate points rational, they lie on other sheets).
        chosen = list(chosen)
        for position, (chart, centre) in enumerate(zip(current, chosen)):
            if centre is not None and _dimension(chart.variables, (*chart.ambient, *centre)) == 0:
                over = (*chart.ambient, *centre, *chart.fibre)
                chosen[position] = None if empty_locus(chart.variables, over) else _radical_points(chart.variables, over)
        if all(centre is None for centre in chosen):
            raise GiveUp("No centre was found over the fibre.")
        sizes, through = [], []
        for chart, centre in zip(current, chosen):
            if centre is None:
                continue
            # Each centre is checked: it must be smooth.
            size = _dimension(chart.variables, (*chart.ambient, *centre))
            proof = smooth_locus(chart.variables, (*chart.ambient, *centre), size) if size >= 0 else None
            # Where the chart is no longer its own (a removed centre), the closure
            # of the centre is not asked to be smooth.
            if proof is None or not (proof["smooth"] or chart.removed and not _present(
                    chart.variables, (*chart.ambient, *proof["bad_ideal"]), chart.removed)):
                raise GiveUp("A centre is not smooth.")
            sizes.append(size)
            through = through or [name for name, _, _, ideal in chart.components
                                  if radical_subset(chart.variables, ideal, (*chart.ambient, *centre))]
        size = level_weights("common_value", (), values=sizes)
        if size is None:
            raise GiveUp("A centre has parts of different dimensions.")
        def blown(centres):
            # The parts of a centre may lie in different charts: `_blown_up`
            # checks that the new divisor gets the same numerical data from all of them.
            keep_complement = retains or getattr(rule, "keep_complement", False)
            return _blown_up(current, centres, label, stage, dimension, size, deadline, keep_complement)

        try:
            report(f"Blowing up centre {label} ({rule.name} method)", len(centres))
            following = blown(chosen)
        except Uneven:
            # Unlike parts: for curves, those inside a divisor that does not contain the
            # others; for points, those on the largest set of divisors. The rest
            # comes at later stages.
            whole, following = chosen, None
            if size <= 1:
                try:
                    chosen = _apart(current, whole) if size else _alike(current, whole)
                    try:
                        following = blown(chosen)
                    except Uneven:
                        if size:
                            raise  # The remaining centre is a curve, outside the finite-point splitter.
                        chosen = _alike(current, chosen, single=True)
                        following = blown(chosen)
                except Uneven:
                    following = None
            if following is None:
                # Any centre: its prime parts with the largest numerical data first.
                keep_complement = retains or getattr(rule, "keep_complement", False)
                chosen, kept = _by_numerical_data(current, whole, label, stage, dimension, size, deadline,
                                                  keep_complement)
                try:
                    following = blown(chosen)
                except Uneven:
                    if not keep_complement:
                        raise
                    # Parts that meet only inside the centres removed from a
                    # chart: their union is singular there, and each is blown up on its own.
                    following = _blown_up(current, chosen, label, stage, dimension, size, deadline, True,
                                          parts=kept)
        rule.created(label, detail)
        # Keep the ideals of the accepted blowup, in its actual local coordinates.
        # Stage and pivot identify the parent open even when it is no longer a final chart.
        center_charts = [{"history": list(chart.history), "coordinates": [v.name for v in chart.variables],
                          "ideal": [str(g).replace("**", "^") for g in centre],
                          **({"ambient_ideal": [str(g).replace("**", "^") for g in chart.ambient]}
                             if chart.ambient else {})}
                         for chart, centre in zip(current, chosen) if centre is not None]
        centres.append({"id": label, "stage": stage, "dimension": size, "rule": detail.get("rule", rule.name),
                        "kind": {0: "points", 1: "curve", 2: "surface"}.get(size, "subvariety"),
                        "components": detail.get("components", through), "center_charts": center_charts,
                        **({"invariant": detail["invariant"]} if "invariant" in detail else {})})
        current = following
        if budget is not None and len(current) > budget:
            raise GiveUp("Too many charts.")
    raise GiveUp("Normal crossings were not reached.")


# A deadline (of time.monotonic) for every call of acampo_over in a context,
# as if each had been given it. A verification can replay blowups deep inside
# other modules, which do not pass a deadline along.
_TIME_LIMIT = ContextVar("atlas_blowup_time_limit", default=None)


@contextmanager
def time_limit(deadline):
    """Bound the blowups of this context by `deadline` (of time.monotonic)."""
    outer = _TIME_LIMIT.get()
    token = _TIME_LIMIT.set(deadline if outer is None else min(outer, deadline))
    try:
        yield
    finally:
        _TIME_LIMIT.reset(token)


def time_limit_passed():
    """Whether the deadline of the enclosing `time_limit` has passed."""
    limit = _TIME_LIMIT.get()
    return limit is not None and monotonic() > limit


def acampo_over(variables, components, fibre, *, ambient=(), dimension=None, budget=400, deadline=None,
                rules=("complete", "direct"), count_strata=True, intrinsic=False):
    """Blow up over V(fibre) until the divisor has normal crossings there.

    `components` lists (label, N, nu, ideal) on affine space with the given
    coordinates (or on its smooth subvariety V(ambient) of the given
    dimension); those labelled "strict" or S1, S2, ... make up the reduced
    hypersurface, the others are exceptional divisors with normal crossings.
    Returns the numerical data {label: (N, nu)} of the components
    with a part over the fibre, the Euler characteristic of every stratum
    over it ({set of labels: chi} for the points lying on exactly those
    components), the final charts and the centres blown up. Raises GiveUp
    when it cannot, or when the time `deadline` (of time.monotonic) has passed.

    The complete rule is used; should it give up, the short one is tried
    from the start. Whatever the rule, the answer is the same kind of
    object, with normal crossings verified in every final chart. For a single
    reduced source labelled "strict", intrinsic=True with count_strata=False
    stops once the strict transform is smooth on the entire owned fibre.
    An enclosing `time_limit` bounds the deadline.
    """
    limit = _TIME_LIMIT.get()
    if limit is not None:
        deadline = limit if deadline is None else min(deadline, limit)
    variables = tuple(variables)
    reason = GiveUp("No rule was allowed.")
    rules = tuple(rules)
    stopped = []
    for index, name in enumerate(rules):
        if name == "tangent":
            from .repeated_tangent import RepeatedTangent
            rule = RepeatedTangent()
        elif name == "order":
            rule = _Order(components)
        else:
            rule = _Direct() if name == "direct" else _Canonical(components)
        # Each rule is fast when it succeeds and can be slow when it fails. A
        # rule followed by another keeps the share of the remaining time in
        # RULE_SHARES, so that the next one also runs.
        limit = deadline
        if deadline is not None and index + 1 < len(rules) and name in RULE_SHARES:
            limit = min(deadline, monotonic() + RULE_SHARES[name] * max(0.0, deadline - monotonic()))
        state = {}
        try:
            return _run(rule, variables, components, fibre, budget, limit, ambient, dimension, count_strata,
                        intrinsic=intrinsic, state=state)
        except GiveUp as error:
            reason = error
            now = monotonic()
            if deadline is not None and now > deadline:
                break
            if limit is not None and now > limit:
                # Stopped at its share, not for a reason of its own.
                stopped.append((now - state.get("progress", now), index, rule, state))
    else:
        # The rules that follow can give up early. The time they leave goes
        # back to the rules stopped at their share, which continue where they
        # were: first the one that had completed a blowup most recently.
        for _, _, rule, state in sorted(stopped, key=lambda item: item[:2]):
            if monotonic() > deadline:
                break
            try:
                return _run(rule, variables, components, fibre, budget, deadline, ambient, dimension, count_strata,
                            intrinsic=intrinsic, state=state)
            except GiveUp as error:
                reason = error
    raise reason


# The share of the remaining time that a rule keeps when another rule follows.
RULE_SHARES = {"complete": 0.6, "order": 0.5}


def _strata(charts, domain=None):
    """Numerical data of the components, and the Euler characteristic of the
    set of points over the fibre lying on exactly a given set of components.

    Each chart is counted on the part where its fibre and region equations
    hold. Those among them that are coordinates are set to zero beforehand.
    """
    from .chart_kernel import InconsistentSheets, owned_strata, strata_totals
    data, contributions = {}, []
    for chart in charts:
        for label, n, nu, _ in chart.components:
            if data.setdefault(label, (n, nu)) != (n, nu):
                raise GiveUp("A component has two sets of numerical data.")
        # The kernel computes the strata of a chart (native/ideal.cpp,
        # op_strata, and native/euler.cpp); the code below answers when it
        # cannot.
        native = owned_strata(chart) if domain is None or getattr(domain, "is_QQ", False) else None
        if native is not None:
            here = {frozenset(chart.components[j][0] for j in subset): chi for subset, chi in native.items()}
        else:
            here = _strata_here(chart, domain)
        degree = chart.coefficient_field[2] if chart.coefficient_field else 1
        contributions.extend((key, degree, chi) for key, chi in here.items())
    try:
        return data, strata_totals(contributions)
    except InconsistentSheets as exc:
        raise GiveUp(str(exc)) from exc


def _strata_here(chart, domain=None):
    """The strata of one chart, computed here: {labels: Euler characteristic}."""
    from .constructible_euler import euler_characteristic
    if domain is None:
        from .reference import QQ
        domain = QQ
    api = chart_expressions(chart.variables)
    from .chart_kernel import strata_totals
    contributions = []
    zero = [g for g in dict.fromkeys((*chart.fibre, *chart.region)) if g in chart.variables]
    zero = dict.fromkeys(zero[:len(chart.variables) - 1], api.S.Zero)
    variables = tuple(v for v in chart.variables if v not in zero)

    def cut(ideal):
        values = (api.sympify(g).xreplace(zero) for g in ideal)
        return tuple(g for g in values if g != 0)

    base = (*cut(chart.ambient), *cut(chart.fibre), *cut(chart.region))
    ideals = [cut(component[3]) for component in chart.components]
    count = len(chart.components)
    def equations(subset):
        return (*base, *(g for j in subset for g in ideals[j]))
    def present(task):
        return not empty_locus(variables, equations(task["indices"]), domain=domain)
    for subset in centre_search("strata", present, count=count):
        others = [ideal for j,ideal in enumerate(ideals) if j not in subset] + [cut(z) for z in chart.removed]
        chi = euler_characteristic(variables, equations(subset), others, domain=domain)
        if chi:
            key = frozenset(chart.components[j][0] for j in subset)
            contributions.append((key, 1, chi))
    return strata_totals(contributions)


def _chart_cell_counts(chart, dimension, labels):
    from . import kernel
    from .polynomial import integer_text

    own = (*chart.fibre, *chart.region)

    def intersection(subset):
        return (*chart.ambient, *(g for j in subset for g in labels[j][1]))

    def owned(prime):
        return ideal_subset(chart.variables, own, prime) and not any(
            ideal_subset(chart.variables, hole, prime) for hole in chart.removed)

    def geometric(whole, dimension):
        from .geometric_components import geometric_components
        primes = geometric_components(chart.variables, whole, dimension)
        if primes is not None:
            return [owned((*chart.ambient, *prime)) for prime in primes]
        # A chart shows its own part as it is. The rest it may show as it was
        # before later blowups, and there the branches of an intersection can
        # still meet: in a centre removed from the chart, or at points that
        # other charts own. The branches are then found one by one. Those
        # inside the chart's own part are counted when none of them meets
        # another branch outside the removed centres.
        primes = geometric_components(chart.variables, whole, dimension, reduced_support=True)
        if primes is None:
            return None
        flags = [owned((*chart.ambient, *prime)) for prime in primes]
        for i, mine in enumerate(flags):
            if mine and any(_present(chart.variables, (*chart.ambient, *primes[i], *other), chart.removed)
                            for j, other in enumerate(primes) if j != i):
                return None
        return flags

    from .native_library import _kernel, current_replay_scope
    replay = current_replay_scope()
    if (replay is None or replay.compute in (None, "cpu")) and hasattr(_kernel(), "atlas_owned_chart_incidence"):
        from .chart_gluing_kernel import rows
        from .kernel_json import kernel_json
        axes = chart.variables
        field = chart.coefficient_field
        items = kernel_json("atlas_owned_chart_incidence", "cells", width=len(axes), dimension=dimension,
                            ambient=rows(axes, chart.ambient), fibre=rows(axes, chart.fibre),
                            region=rows(axes, chart.region), holes=[rows(axes, h) for h in chart.removed],
                            labels=[rows(axes, ideal) for _,ideal in labels],
                            degree=integer_text(field[2]) if field else None,
                            relation=rows(axes, (field[1],))[0] if field else None)
        if items is not None:
            for item in items:
                subset = item["indices"]
                if "components_dimension" in item:
                    # The unresolved projection uses the same geometric decomposition
                    # and generic-point ownership checks as the full chart verifier.
                    flags = geometric(intersection(subset), item["components_dimension"])
                    item["here"] = item["rational"] = (
                        kernel_json("atlas_owned_chart_incidence", "count_owned", flags=flags)
                        if flags is not None else None)
                yield subset, item
            return

    def live(task):
        return _present(chart.variables, (*intersection(task["indices"]), *own), chart.removed)
    for subset in centre_search("strata", live, count=len(labels)):
        size = len(subset)
        whole = intersection(subset)
        def query(task):
            op = task["query"]
            if op == "over_fibre":
                return radical_subset(chart.variables,chart.fibre,whole)
            if op == "points":
                native = kernel.strata(chart.variables, (*whole, *own), [()], chart.removed)
                if native is not None:
                    return integer_text(native.get((0,), 0))
                from .constructible_euler import euler_characteristic
                return integer_text(euler_characteristic(chart.variables, (*whole, *own), chart.removed))
            if op == "plane":
                pair = kernel.plane_components(chart.variables, whole, own, chart.removed)
                return list(map(integer_text,pair)) if pair is not None else None
            if op == "owned":
                return owned(whole)
            return geometric(whole, task["dimension"])
        flat = bool(chart.coefficient_field and chart.ambient == (chart.coefficient_field[1],)
                    and all(len(labels[j][1]) == 1 and labels[j][1][0] in chart.variables[:-1] for j in subset))
        counts_here = centre_search("incidence_count",query,dimension=dimension,count=size,flat=flat,
                                    degree=integer_text(chart.coefficient_field[2]) if flat else None)
        yield subset, counts_here


def cells(charts, dimension, *, rational_counts=None):
    """The sets of exceptional divisors with a common point over the fibre,
    with the number of geometric components of their intersection:
    {set of labels: number, or None when it is not determined here}.

    The charts partition the space over the fibre: each owns the points of
    its region off the centres removed from it. An irreducible component Z
    of an intersection E_J has a first chart (in the order given) that it
    meets; there it is dense in Z, and it is contained in the part that the
    chart owns, the earlier charts being its complement. So the components
    of E_J are counted once each by counting, in every chart, the components
    of E_J on the chart that lie in the part it owns. Normal crossings make
    E_J smooth, so its components are disjoint.

    The count is meant for intersections that lie over the fibre (a divisor
    created over the fibre is one of the set); for a set whose intersection
    reaches outside the fibre the number is None, and its presence in the
    answer still says that the intersection has a point over the fibre.
    """
    from .chart_kernel import strata_totals
    from .polynomial import integer_from_decimal
    counts, contributions = {}, []
    rational_contributions = []
    for chart in charts:
        labels = [(name, ideal) for name, _, _, ideal in chart.components if not _is_strict(name)]
        for subset, counts_here in _chart_cell_counts(chart, dimension, labels):
            size = len(subset)
            key = frozenset(labels[j][0] for j in subset)
            here = integer_from_decimal(counts_here["here"]) if counts_here["here"] is not None else None
            if rational_counts is not None and size == 1:
                count = integer_from_decimal(counts_here["rational"]) if counts_here["rational"] is not None else None
                if count is None:
                    rational_counts[key] = None
                elif rational_counts.setdefault(key, 0) is not None:
                    rational_contributions.append((key, 1, chart.one_sheet(count)))
            if here is None:
                counts[key] = None
            elif counts.setdefault(key, 0) is not None:
                contributions.append((key, 1, chart.one_sheet(here)))
    totals = strata_totals(contributions)
    if rational_counts is not None:
        rational_contributions.extend((key, 1, value) for key, value in rational_counts.items() if value)
        rational_totals = strata_totals(rational_contributions)
        rational_counts.update((key, rational_totals.get(key, 0)) for key in rational_counts
                               if rational_counts[key] is not None)
    return {key: totals.get(key, 0) if value is not None else None for key, value in counts.items()}
