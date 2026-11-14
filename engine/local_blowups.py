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
from dataclasses import dataclass, replace
from itertools import combinations, islice
from time import monotonic

from .chart_expressions import chart_expressions
from .native_polynomial import native_ring

from .blowup_ideal import (blowup_charts, empty_locus, ideal_basis, ideal_subset, radical_subset, saturate,
                           smooth_locus)
from .affine_kernel import frame_function, frame_linear, pullbacks
from .centres import GiveUp, maximum
from .groebner import _MODE, colon, groebner
from .incidence import dual_cells


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

    def one_sheet(self, count):
        """A field parameter describes one fixed embedding. Over Q the kernel
        sees its conjugates, each with the same geometric count."""
        if not self.coefficient_field or count is None:
            return count
        degree = self.coefficient_field[2]
        value, remainder = divmod(count, degree)
        if remainder:
            raise GiveUp("A geometric count is inconsistent across coefficient field embeddings.")
        return value


def _covers(variables, removed):
    """Functions whose nonvanishing sets cover the complement of the removed
    centres: the products of one generator of each ideal."""
    out = [chart_expressions(variables).sympify(1)]
    for ideal in removed:
        out = [_expanded(variables, u*g) for u in out for g in ideal]
    return list(dict.fromkeys(out))


def _present(variables, ideal, removed=()):
    """Whether V(ideal) has a point outside the removed centres."""
    if empty_locus(variables, ideal):
        return False
    relevant = [z for z in removed if not empty_locus(variables, (*ideal, *z))]
    return any(not radical_subset(variables, (u,), ideal) for u in _covers(variables, relevant))


def _dimension(variables, ideal):
    """The dimension of V(ideal), from the leading monomials of a Groebner basis."""
    from . import kernel
    from .groebner import CHECK
    value = kernel.dimension(variables, ideal)
    if value is None:
        if _MODE == "native" or native_ring(variables):
            raise GiveUp("No native dimension certificate is available for this chart.")
        return _dimension_reference(variables, ideal)
    if CHECK and value != _dimension_reference(variables, ideal):
        raise AssertionError("The dimension differs from the leading-monomial reference.")
    return value


def _dimension_reference(variables, ideal):
    # Variables absent from the generators give an affine factor. This
    # also keeps the independent reference small on wide sparse charts.
    import sympy as sp
    used = set().union(*(sp.sympify(f).free_symbols for f in ideal))
    active = tuple(v for v in variables if v in used)
    if not active:
        return -1 if any(ideal) else len(variables)
    basis = sp.groebner(tuple(ideal), *active, order="grevlex", domain=sp.QQ)
    if tuple(basis) == (1,):
        return -1
    heads = [p.monoms(order="grevlex")[0] for p in basis.polys]
    best = 0
    fixed = {i for head in heads if sum(e > 0 for e in head) == 1 for i, e in enumerate(head) if e}
    heads = [head for head in heads if not any(head[i] for i in fixed)]
    relevant = sorted({i for head in heads for i, e in enumerate(head) if e})
    heads = [tuple(head[i] for i in relevant) for head in heads]
    # The largest set of variables no power product of which is a leading monomial.
    for mask in range(1, 1 << len(relevant)):
        size = bin(mask).count("1")
        if size > best and not any(all(e == 0 or (mask >> i) & 1 for i, e in enumerate(head)) for head in heads):
            best = size
    return len(variables) - len(relevant) - len(fixed) + best


def _generators(variables, ambient, ideal):
    """Few generators of the ideal modulo the ambient one."""
    basis = groebner((*ambient, *ideal), variables)
    from .affine_kernel import sparse_terms
    degree = {g: max(sum(m) for m, _ in sparse_terms(variables, g)) for g in basis.exprs}
    gens = [g for g in degree if not ideal_subset(variables, (g,), ambient)]
    for g in sorted(gens, key=lambda h: (-degree[h], chart_expressions(variables).default_sort_key(h))):
        rest = [h for h in gens if h != g]
        if rest and ideal_subset(variables, (g,), (*ambient, *rest)):
            gens = rest
    return tuple(gens)


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
    those among them seen inside the chart's own region.

    The chart W is smooth, of codimension c in its affine space. At a point
    of k of the divisors, their intersection is smooth of codimension k in W
    exactly when the Jacobian matrix of the equations of W and of the k
    divisors has rank c + k there (its dimension is at least the expected
    one). The matrix is only needed at the points over the fibre: its
    entries are reduced modulo their ideal.

    The kernel computes this (native/ideal.cpp, op_crossings); the code below
    answers when it cannot, and with ATLAS_GROEBNER=check both must agree.
    """
    from . import kernel
    from .groebner import CHECK, _MODE
    labels = [component[0] for component in chart.components]
    found = kernel.crossings(chart.variables, dimension, chart.ambient,
                             [component[3] for component in chart.components], chart.fibre, chart.region)
    if found is not None:
        over = {tuple(labels[i] for i in positions): gens for positions, _, gens in found}
        inside = {tuple(labels[i] for i in positions) for positions, flag, _ in found if flag}
    else:
        over, inside = _failures_here(chart, dimension, reference=_MODE == "sympy")
    if CHECK:
        other, within = _failures_here(chart, dimension, reference=True)
        if inside != within or {key: tuple(ideal_basis(chart.variables, bad)) for key, bad in over.items()} != \
                {key: tuple(ideal_basis(chart.variables, bad)) for key, bad in other.items()}:
            raise AssertionError("The native check of normal crossings differs from this file's.")
    return over, inside


def _failures_here(chart, dimension, *, reference=False):
    """The computation of `_failures`, done here."""
    variables = chart.variables
    labels = [component[0] for component in chart.components]
    ideals = [component[3] for component in chart.components]
    codimension = len(variables) - dimension
    over, inside = {}, set()
    alive = {()}
    for size in range(1, len(ideals) + 1):
        found = set()
        for subset in combinations(range(len(ideals)), size):
            if any(subset[:k] + subset[k + 1:] not in alive for k in range(size)):
                continue
            gens = (*chart.ambient, *(g for j in subset for g in ideals[j]))
            bad = (*gens, *chart.fibre)
            if empty_locus(variables, bad):
                continue
            found.add(subset)
            rank = codimension + size
            if rank <= len(variables):
                basis = groebner(bad, variables)
                if reference:
                    import sympy as sp
                    matrix = sp.Matrix([[basis.reduce(sp.diff(g, v))[1] for v in variables] for g in gens])
                    minors = {sp.expand(matrix.extract(list(rows), list(columns)).det(method="berkowitz"))
                              for rows in combinations(range(len(gens)), rank)
                              for columns in combinations(range(len(variables)), rank)}
                else:
                    from .affine_kernel import frame_minor
                    minors = set()
                    for rows in combinations(gens, rank):
                        active = set().union(*(getattr(g, "free_symbols", ()) for g in rows))
                        for columns in combinations((v for v in variables if v in active), rank):
                            # Determinant commutes with passage to the fibre
                            # quotient, so reduction after the minor gives the
                            # same rank-defect ideal there.
                            minor = frame_minor(variables, rows, columns)
                            if minor != 0:
                                minors.add(basis.reduce(minor)[1])
                bad = (*bad, *(m for m in minors if m != 0))
                if empty_locus(variables, bad):
                    continue
            key = tuple(labels[j] for j in subset)
            over[key] = bad
            if not empty_locus(variables, (*bad, *chart.region)):
                inside.add(key)
        if not found:
            break
        alive = found
    return over, inside


def _order(variables, ambient, pivot, ideal, limit=64):
    """The order of an ideal along the divisor pivot = 0 of V(ambient): the
    least k such that some generator is not a multiple of pivot^(k+1) there.
    (The divisor is Cartier on a smooth variety, so its powers are its
    symbolic powers.)"""
    for order in range(limit + 1):
        if not ideal_subset(variables, ideal, (*ambient, pivot**(order + 1))):
            return order
    raise GiveUp("A divisor vanishes on a whole chart, or to an order beyond the limit.")


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
    from .groebner import CHECK
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
    bad = (len(variables) - 1)*count*(count - 1)//2
    for a in range(1, bad + 2):
        form = sum(a**i*v for i, v in enumerate(variables))
        try:
            factors = _point_factors((*variables, t), (*ideal, t - form), t)
            if sum(degree for _, degree in factors) != count:
                continue
            equations = pullbacks(variables, (form,), (t,), [f for f, _ in factors])
        except ArithmeticError as exc:
            if _MODE == "native" or native_ring(variables):
                raise GiveUp("The native kernel could not separate this finite centre.") from exc
            return _point_components_reference(variables, ideal)
        parts = [(degree, ideal_basis(variables, (*ideal, eq))) for (_, degree), eq in zip(factors, equations)]
        answer = [p for _, p in sorted(parts, key=lambda part: (part[0], chart_expressions(variables).default_sort_key(part[1])))]
        if CHECK and answer != _point_components_reference(variables, ideal, form):
            raise AssertionError("The finite point components differ from the reference decomposition.")
        return answer
    raise GiveUp("The points of a centre were not separated.")


def _point_components_reference(variables, ideal, separated=None):
    """SymPy reference for finite point decomposition and fallback on kernel limits."""
    import sympy as sp
    from sympy import QQ
    from .constructible_euler import euler_characteristic
    count = euler_characteristic(variables, ideal)
    t = sp.Dummy("form")
    bad = (len(variables) - 1)*count*(count - 1)//2
    forms = (separated,) if separated is not None else (sum(a**i*v for i, v in enumerate(variables))
                                                         for a in range(1, bad + 2))
    for form in forms:
        basis = sp.groebner((*ideal, t - form), *variables, t, order="lex", domain=QQ)
        last = [g for g in basis if g.free_symbols <= {t}]
        if not last:
            raise GiveUp("A centre expected to be finite is not.")
        minimal = sp.Poly(last[-1], t).sqf_part()
        if minimal.degree() != count:
            continue
        factors = [factor.as_expr().subs(t, form) for factor, _ in minimal.factor_list()[1]]
        return sorted((tuple(sp.groebner((*ideal, f), *variables, order="grevlex", domain=QQ)) for f in factors),
                      key=lambda gens: (euler_characteristic(variables, gens), sp.default_sort_key(gens)))
    raise GiveUp("The points of a centre were not separated.")


def _linear(chart, centre, codimension):
    """Generators of a centre that are linear in the coordinates, when the
    chart is an affine space over parameters (the variables of its ambient
    equations) and the centre is cut out by coordinate = polynomial in the
    parameters, for as many coordinates as its codimension. Returns pairs
    (coordinate, polynomial), or None."""
    parameters = [v for v in chart.variables if any(v in g.free_symbols for g in chart.ambient)]
    others = [v for v in chart.variables if v not in parameters]
    if any(g.free_symbols - set(parameters) for g in chart.ambient):
        return None
    basis = groebner((*chart.ambient, *centre), (*others, *parameters), order="lex").exprs
    found = []
    for v in others:
        for g in basis:
            if v in g.free_symbols and g.free_symbols <= {v, *parameters}:
                value = _solved(chart.variables, g, v)
                if value is not None:
                    found.append((v, value))
                    break
    if len(found) != codimension:
        return None
    if not ideal_subset(chart.variables, centre, (*chart.ambient, *(v - value for v, value in found))):
        return None
    return found


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

    The kernel computes this (native/ideal.cpp, op_blowup); `_blow_up_linear_here`
    answers when it cannot, and with ATLAS_GROEBNER=check both must agree."""
    from . import kernel
    from .groebner import CHECK, _MODE
    names = _child_variables(chart, linear, stage)
    native = kernel.blowup_linear(
        chart.variables, chart.ambient, [chart.variables.index(v) for v, _ in linear], [shift for _, shift in linear],
        [component[3] for component in chart.components], chart.fibre, chart.region,
        [*(expr for _, expr in chart.coordinates), *(g for z in chart.removed for g in z)])
    if native is None:
        children = _blow_up_linear_here(chart, linear, label, stage, codimension, names,
                                        reference=_MODE == "sympy")
        if CHECK and children != _blow_up_linear_here(chart, linear, label, stage, codimension, names,
                                                      reference=True):
            raise AssertionError("The sparse blowup of a linear centre differs from its reference.")
        return children
    children = []
    for j, ((variables, w, ratios), raw) in enumerate(zip(names, native)):
        components, multiplicity, discrepancy = [], 0, codimension
        for (name, n, nu, _), (order, here, rows) in zip(chart.components, raw["divisors"]):
            if here:
                components.append((name, n, nu, kernel.exprs(variables, rows)))
            multiplicity += n*order
            discrepancy += (nu - 1)*order
        components.append((label, multiplicity, discrepancy, (w,)))
        region = (*kernel.exprs(variables, raw["region"]), *ratios[:j])
        moved = kernel.exprs(variables, raw["functions"])
        removed, at = [], len(chart.coordinates)
        for z in chart.removed:
            removed.append(tuple(g for g in moved[at:at + len(z)] if g != 0))
            at += len(z)
        children.append(Chart(variables, chart.ambient, tuple(components), kernel.exprs(variables, raw["fibre"]), region,
                              tuple((name, expr) for (name, _), expr in zip(chart.coordinates, moved)),
                              (*chart.history, (stage, j)), tuple(removed), chart.coefficient_field))
    if CHECK:
        for mine, other in zip(children, _blow_up_linear_here(chart, linear, label, stage, codimension, names,
                                                               reference=True)):
            def same(a, b):
                return ideal_basis(mine.variables, (*mine.ambient, *a)) == ideal_basis(mine.variables, (*mine.ambient, *b))
            if mine.variables != other.variables or [c[:3] for c in mine.components] != [c[:3] for c in other.components] \
                    or not all(same(a[3], b[3]) for a, b in zip(mine.components, other.components)) \
                    or not same(mine.fibre, other.fibre) or not same(mine.region, other.region) \
                    or not all(ideal_subset(mine.variables, (a[1] - b[1],), mine.ambient)
                               for a, b in zip(mine.coordinates, other.coordinates)):
                raise AssertionError("The native blowup of a linear centre differs from this file's.")
    return children


def _blow_up_linear_here(chart, linear, label, stage, codimension, names, *, reference=False):
    """The computation of `_blow_up_linear`, done here."""
    parameters = [v for v in chart.variables if any(v in g.free_symbols for g in chart.ambient)]
    children = []
    for j, (pivot, shift) in enumerate(linear):
        variables, w, ratios = names[j]
        swap = {pivot: w + shift}
        swap.update({v: value + ratios[i]*w for i, (v, value) in enumerate(linear) if i != j})
        reducer = groebner(chart.ambient, variables) if chart.ambient else None
        mapping = tuple(swap.get(v,v) for v in chart.variables)

        def move(f):
            if reference:
                import sympy as sp
                from sympy import QQ
                value = sp.Poly(sp.sympify(f).xreplace(swap), *variables, domain=QQ).as_expr()
            else:
                value, = pullbacks(variables, mapping, chart.variables, (f,))
            return reducer.reduce(value)[1] if reducer is not None and value.free_symbols & set(parameters) else value

        def ideal(gens):
            return tuple(g for g in map(move, gens) if g != 0)

        components, multiplicity, discrepancy = [], 0, codimension
        for name, n, nu, before in chart.components:
            moved = ideal(before)
            if not moved:
                raise GiveUp("A divisor contains a chart of a blowup.")
            if reference:
                import sympy as sp
                order = min(min(monomial[0] for monomial in sp.Poly(g, w).monoms()) for g in moved)
            else:
                from .affine_kernel import sparse_terms
                column = variables.index(w)
                order = min(e[column] for g in moved for e, _ in sparse_terms(variables, g))
            if order and len(moved) > 1:
                after = colon((*chart.ambient, *moved), (w**order,), variables)
                if after is None:
                    raise GiveUp("An ideal quotient is not available.")
                after = _generators(variables, chart.ambient, after)
            elif not order:
                after = moved
            elif reference:
                import sympy as sp
                after = tuple(sp.expand(sp.cancel(g/w**order)) for g in moved)
            else:
                from .affine_kernel import _request, _ring, _sparse_expression
                ring = _ring(variables)
                q, r = _request(["wide_division", ring.header, "lex", ring.text(moved[0]), ring.text(w**order)])
                if r:
                    raise ArithmeticError("The chart function was not divisible by its exceptional order.")
                after = (_sparse_expression(variables, q),)
            if not empty_locus(variables, (*chart.ambient, *after)):
                components.append((name, n, nu, after))
            multiplicity += n*order
            discrepancy += (nu - 1)*order
        components.append((label, multiplicity, discrepancy, (w,)))
        region = (*ideal(chart.region), *(r for r in ratios[:j]))
        children.append(Chart(variables, chart.ambient, tuple(components), ideal(chart.fibre), region,
                              tuple((name, move(expr)) for name, expr in chart.coordinates),
                              (*chart.history, (stage, j)), tuple(ideal(z) for z in chart.removed),
                              chart.coefficient_field))
    return children


def _strict_ideal(variables, ambient, pivot, generators):
    """Order and residual ideal of a pulled-back divisor along the new pivot.

    In the regular chart ring let g_i = p^a_i h_i, with h_i the principal
    residual ideals returned by the kernel. Their common order is m=min a_i.
    Since p is a nonzerodivisor, J:(p^m) = sum_i p^(a_i-m) (h_i). Vanishing
    on an entire component of the pivot after this division still signals
    nonuniform order; the caller checks that before accepting the centre.
    """
    from . import kernel
    from .groebner import CHECK
    nonzero = tuple(g for g in generators if not ideal_subset(variables, (g,), ambient))
    native = kernel.strict_parts(variables, ambient, ((pivot,),), nonzero) if nonzero else None
    if native:
        order = min(orders[0] for orders, _ in native)
        if len(native) == 1:
            after = native[0][1]
        else:
            residual = [_expanded(variables, g*pivot**(orders[0]-order)) for orders, ideal in native for g in ideal]
            after = ideal_basis(variables, (*ambient, *residual))
        if not CHECK:
            return order, after
    # Independent comparison and fallback for rings outside the kernel.
    expected_order = _order(variables, ambient, pivot, generators, limit=max(64, order) if native else 64)
    expected = colon((*ambient, *generators), (pivot**expected_order,), variables)
    if expected is None:
        if _MODE == "native" or native_ring(variables):
            raise GiveUp("The native kernel could not compute the residual divisor ideal.")
        from sympy import QQ
        ring = QQ.old_poly_ring(*variables, order="grevlex")
        quotient = ring.ideal(*ambient, *generators).quotient(ring.ideal(pivot**expected_order))
        expected = ideal_basis(variables, tuple(ring.to_sympy(g) for g in quotient.gens))
    if native:
        if order != expected_order or after != expected:
            raise AssertionError("The graph blowup divisor order or residual ideal differs from its reference.")
        return order, after
    return expected_order, expected


def _blow_up(chart, centre, label, stage, dimension, codimension, deadline=None):
    linear = _linear(chart, centre, codimension)
    if linear:
        return _blow_up_linear(chart, linear, label, stage, codimension)
    gens = _generators(chart.variables, chart.ambient, centre)
    children = []
    for child in blowup_charts(chart.variables, chart.ambient, None, gens):
        if deadline is not None and monotonic() > deadline:
            raise GiveUp("The time allowed for the blowups is over.")
        if empty_locus(child.variables, child.ambient):
            continue
        pivot = (child.pivot,)
        components = []
        multiplicity, discrepancy = 0, codimension
        for old, n, nu, before in chart.components:
            # The transform of a divisor is its pullback with the new divisor
            # removed as often as it occurs: an ideal quotient by that power of
            # the pivot. The order must be the same on every component of the
            # new divisor: what is left may vanish on none of them.
            order, after = _strict_ideal(child.variables, child.ambient, child.pivot, before)
            if _dimension(child.variables, (*child.ambient, child.pivot, *after)) >= dimension - 1:
                raise Uneven("A divisor has different orders along parts of one exceptional divisor.")
            if not empty_locus(child.variables, (*child.ambient, *after)):
                components.append((old, n, nu, after))
            multiplicity += n * order
            discrepancy += (nu - 1) * order
        components.append((label, multiplicity, discrepancy, pivot))
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
    gens = (chart_expressions(chart.variables).sympify(1),)
    for label, _, _, ideal in chart.components:
        if _is_strict(label):
            gens = tuple(_expanded(chart.variables, a*b) for a in gens for b in ideal)
    return _generators(chart.variables, chart.ambient, gens)


def _radical_points(variables, ideal):
    """Seidenberg's radical of a finite ideal, using the squarefree coordinate eliminants."""
    from .affine_kernel import _sparse_expression
    from .groebner import CHECK, _MODE
    from .kernel import finite_radical
    if _MODE == "sympy":
        return _radical_points_reference(variables, ideal)
    answer = finite_radical(variables, ideal)
    if answer is not None:
        if CHECK and answer != _radical_points_reference(variables, ideal):
            raise AssertionError("The native finite radical differs from its reference.")
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
        if _MODE == "native" or native_ring(variables):
            raise GiveUp("The native kernel could not reduce this finite centre.") from exc
        return _radical_points_reference(variables, ideal)
    if CHECK and answer != _radical_points_reference(variables, ideal):
        raise AssertionError("The radical of a finite locus differs from its reference.")
    return answer


def _radical_points_reference(variables, ideal):
    import sympy as sp
    from sympy import QQ
    extra = []
    for v in variables:
        order = (*(w for w in variables if w != v), v)
        last = [g for g in sp.groebner(ideal, *order, order="lex", domain=QQ) if g.free_symbols <= {v}]
        if not last:
            raise GiveUp("A centre expected to be finite is not.")
        extra.append(sp.Poly(last[-1], v).sqf_part().as_expr())
    return tuple(sp.groebner((*ideal, *extra), *variables, order="grevlex", domain=QQ))


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
        seen = set().union(*(inside for _, inside in failures))
        order = [component[0] for chart in current for component in chart.components]
        chosen = min(seen, key=lambda key: (-len(key), [order.index(label) for label in key]))
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
        points = any(s is not None for s in special)
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

    def __init__(self, components):
        self._loci, self._steps = {}, {}
        # For each exceptional divisor, the invariant of the centre that created it
        # (None for those given at the start), and the divisors in order of creation.
        self.records = {label: None for label, *_ in components if not _is_strict(label)}
        self.births = list(self.records)
        self.mode = "order"

    @staticmethod
    def _surface_of(chart, over):
        """The component of the strict transform through the points where normal
        crossings fail, when it is the only one and it is smooth at them."""
        if any(all(_is_strict(label) for label in key) for key in over):
            return None
        through = [ideal for label, _, _, ideal in chart.components if _is_strict(label)
                   and any(not empty_locus(chart.variables, (*chart.ambient, *ideal, *w)) for w in over.values())]
        return through[0] if len(through) == 1 else None

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
            if check is None or check[:2] <= item[:2]:
                break
            working.append(on_fibre)
        else:
            raise GiveUp("The locus of the invariant keeps meeting points of larger invariant.")
        self._loci[id(chart)] = chart, item
        return item

    def _on_surface(self, current, failures, deadline=None):
        from .surface import step
        found = []
        for chart, (over, _) in zip(current, failures):
            if not over:
                found.append(None)
                continue
            surface = self._surface_of(chart, over)
            if surface is None:
                raise GiveUp("The strict transform is no longer a smooth surface at the points considered.")
            exceptional = [(label, ideal) for label, _, _, ideal in chart.components if not _is_strict(label)]
            if id(chart) not in self._steps:
                self._steps[id(chart)] = chart, step(chart.variables, chart.ambient, surface, exceptional,
                                                     list(over.values()), deadline, away=chart.removed)
            found.append(self._steps[id(chart)][1])
        if all(item is None for item in found):
            raise GiveUp("Normal crossings fail where the procedure for a surface has nothing to do.")
        best = max(item[0] for item in found if item is not None)
        centres = []
        for chart, item in zip(current, found):
            if item is None or item[0] != best:
                centres.append(None)
            elif item[1] == "points":
                centres.append(_radical_points(chart.variables, item[2]))
            else:
                centres.append(item[2])
        return centres, {"rule": "surface"}

    def choose(self, current, failures, dimension, deadline):
        if self.mode == "order" and dimension == 3 and all(
                self._surface_of(chart, over) is not None for chart, (over, _) in zip(current, failures) if over):
            self.mode = "surface"
        if self.mode == "surface":
            from .surface import NoModel
            try:
                return self._on_surface(current, failures, deadline)
            except NoModel:
                # The invariant from here on, with the divisors present as the given ones.
                self.mode = "invariant"
                self._loci = {}
                labels = dict.fromkeys(label for chart in current for label, *_ in chart.components
                                       if not _is_strict(label))
                self.records, self.births = dict.fromkeys(labels), list(labels)
        # Where the strict transform is smooth at every point of a chart, the
        # invariant there begins with 1: such charts wait while it is singular elsewhere.
        singular = [bool(over) and self._surface_of(chart, over) is None for chart, (over, _) in zip(current, failures)]
        found = [self._locus(chart, list(over.values()), dimension, deadline)
                 if over and (wanted or self.mode != "order" or not any(singular)) else None
                 for chart, (over, _), wanted in zip(current, failures, singular)]
        if all(item is None for item in found):
            raise GiveUp("Normal crossings fail away from the strict transform.")
        # A key lists membership for the divisors born so far; one kept from an
        # earlier stage is shorter, and the divisors born since are not in it.
        width = len(self.births)

        def rank(item):
            return item[0], tuple(item[1]) + (0,)*(width - len(item[1]))

        best = max(rank(item) for item in found if item is not None)
        centres = [item[2] if item is not None and rank(item) == best else None for item in found]
        return centres, {"invariant": ["oo" if entry == float("inf") else str(entry) for entry in best[0]],
                         "value": best[0], "rule": "invariant"}

    def created(self, label, detail):
        if "value" in detail:
            self.records[label] = detail["value"]
        else:
            self.records[label] = None
        self.births.append(label)


def _blown_up(current, chosen, label, stage, dimension, size, deadline, retains=False):
    """The charts after the blowup of the centre given in each chart (None where it does not meet the chart).

    With `retains` the chart stays for its points off the centre, and each
    chart of the blowup owns its part of the new divisor only."""
    following = []
    for chart, centre in zip(current, chosen):
        if centre is None:
            following.append(chart)
            continue
        children = _blow_up(chart, centre, label, stage, dimension, dimension - size, deadline)
        if not retains:
            following.extend(children)
            continue
        for child in children:
            new = next((ideal for name, _, _, ideal in child.components if name == label), None)
            if new is not None:
                following.append(replace(child, region=tuple(dict.fromkeys((*child.region, *new)))))
        centre = tuple(centre)
        older = tuple(z for z in chart.removed if not radical_subset(chart.variables, centre, (*chart.ambient, *z)))
        following.append(replace(chart, removed=(*older, centre)))
    return following


def _alike(current, chosen, single=False):
    """Of a finite centre, the points lying on the largest set of divisors
    (the sets are compared by size, then by their labels): this does not
    depend on the chart. The divisors, defined over Q, have the same orders
    at conjugate points. When the same divisors pass through all the points,
    or with `single`, one component over Q is taken in each chart."""
    def signature(chart, part):
        labels = sorted(name for name, _, _, ideal in chart.components
                        if not empty_locus(chart.variables, (*chart.ambient, *part, *ideal)))
        return len(labels), tuple(labels)

    groups = [None if centre is None else
              [(signature(chart, part), part) for part in
               _point_components(chart.variables, ideal_basis(chart.variables, (*chart.ambient, *centre)))]
              for chart, centre in zip(current, chosen)]
    best = max(sign for group in groups if group for sign, _ in group)
    kept = [None if group is None else [part for sign, part in group if sign == best] for group in groups]
    if single or all(group is None or len(group) == len(parts) for group, parts in zip(groups, kept)):
        if all(not parts or len(parts) < 2 for parts in kept):
            raise Uneven("A divisor has different orders along one component of a centre.")
        kept = [parts[:1] if parts else parts for parts in kept]
    out = []
    for chart, parts in zip(current, kept):
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
    labels = sorted({name for chart in current for name, *_ in chart.components if not _is_strict(name)})
    for label in labels:
        inside, outside = [], False
        for chart, centre in zip(current, chosen):
            ideal = next((i for name, _, _, i in chart.components if name == label), None)
            if centre is None:
                inside.append(None)
                continue
            whole = tuple(ideal_basis(chart.variables, (*chart.ambient, *centre)))
            if ideal is None:
                inside.append(None)
                outside = True
                continue
            rest = tuple(ideal_basis(chart.variables,
                                     saturate(chart.variables, whole, _principal(chart.variables, chart.ambient, ideal))))
            if rest == (1,):
                inside.append(whole)
                continue
            outside = True
            part = colon(whole, rest, chart.variables)
            if part is None:
                raise GiveUp("An ideal quotient is not available.")
            part = tuple(ideal_basis(chart.variables, part))
            inside.append(None if part == (1,) else part)
        if outside and any(part is not None for part in inside):
            return inside
    raise Uneven("A divisor has different orders along curves that no divisor separates.")


def _union(variables, first, second):
    from .centres import _intersection
    return tuple(ideal_basis(variables, _intersection(variables, first, second)))


def _own_failures(chart, dimension, known):
    """The failures of normal crossings at the points that the chart owns:
    those of its region, off the centres removed from it. In the form of
    `_failures`; `known` keeps the answer for a chart that does not change."""
    if id(chart) not in known:
        over, inside = _failures(chart, dimension)
        own = {}
        for labels in inside:
            ideal = tuple(ideal_basis(chart.variables, (*over[labels], *chart.region)))
            if _present(chart.variables, ideal, chart.removed):
                own[labels] = ideal
        known[id(chart)] = chart, (own, set(own))
    return known[id(chart)][1]


def _run(rule, variables, components, fibre, budget, deadline, ambient=(), dimension=None, count_strata=True):
    dimension = len(variables) if dimension is None else dimension
    current = [Chart(variables, tuple(ambient), tuple((label, n, nu, tuple(ideal)) for label, n, nu, ideal in components),
                     tuple(fibre), (), tuple((v.name, v) for v in variables), ())]
    centres, known = [], {}
    for stage in range(1, rule.stages + 1):
        if deadline is not None and monotonic() > deadline:
            raise GiveUp("The time allowed for the blowups is over.")
        retains = getattr(rule, "retains", False)
        current = [c for c in current if _present(c.variables, (*c.ambient, *c.fibre, *c.region), c.removed)]
        failures = [_own_failures(chart, dimension, known) if retains else _failures(chart, dimension)
                    for chart in current]
        if not any(inside for _, inside in failures):
            return (*(_strata(current) if count_strata else ({}, {})), current, centres)
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
        sizes, through = set(), []
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
            sizes.add(size)
            through = through or [name for name, _, _, ideal in chart.components
                                  if radical_subset(chart.variables, ideal, (*chart.ambient, *centre))]
        if len(sizes) != 1:
            raise GiveUp("A centre has parts of different dimensions.")
        size = sizes.pop()
        def blown(centres):
            charts = _blown_up(current, centres, label, stage, dimension, size, deadline, retains)
            # The parts of a centre may lie in different charts: the new divisor
            # must get the same numerical data from all of them.
            if len({(n, nu) for chart in charts for name, n, nu, _ in chart.components if name == label}) > 1:
                raise Uneven("The parts of a centre give the new divisor different numerical data.")
            return charts

        try:
            following = blown(chosen)
        except Uneven:
            # Unlike parts: for curves, those inside a divisor that does not contain the
            # others; for points, those on the largest set of divisors. The rest
            # comes at later stages.
            if size > 1:
                raise
            fewer = _apart(current, chosen) if size else _alike(current, chosen)
            try:
                following = blown(fewer)
            except Uneven:
                following = blown(_alike(current, fewer, single=True))
        rule.created(label, detail)
        centres.append({"id": label, "stage": stage, "dimension": size, "rule": detail.get("rule", rule.name),
                        "kind": {0: "points", 1: "curve", 2: "surface"}.get(size, "subvariety"),
                        "components": detail.get("components", through),
                        **({"invariant": detail["invariant"]} if "invariant" in detail else {})})
        current = following
        if len(current) > budget:
            raise GiveUp("Too many charts.")
    raise GiveUp("Normal crossings were not reached.")


def acampo_over(variables, components, fibre, *, ambient=(), dimension=None, budget=400, deadline=None,
                rules=("complete", "direct"), count_strata=True):
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
    object, with normal crossings verified in every final chart.
    """
    variables = tuple(variables)
    reason = GiveUp("No rule was allowed.")
    for name in rules:
        rule = _Direct() if name == "direct" else _Canonical(components)   # "complete"
        try:
            return _run(rule, variables, components, fibre, budget, deadline, ambient, dimension, count_strata)
        except GiveUp as error:
            reason = error
            if deadline is not None and monotonic() > deadline:
                break
    raise reason


def _strata(charts, domain=None):
    """Numerical data of the components, and the Euler characteristic of the
    set of points over the fibre lying on exactly a given set of components.

    Each chart is counted on the part where its fibre and region equations
    hold. Those among them that are coordinates are set to zero beforehand.
    """
    from . import kernel
    from .groebner import CHECK
    data, strata = {}, {}
    for chart in charts:
        for label, n, nu, _ in chart.components:
            if data.setdefault(label, (n, nu)) != (n, nu):
                raise GiveUp("A component has two sets of numerical data.")
        # The kernel computes the strata of a chart (native/ideal.cpp,
        # op_strata, and native/euler.cpp); the code below answers when it
        # cannot, and with ATLAS_GROEBNER=check both must agree.
        native = kernel.strata(chart.variables, (*chart.ambient, *chart.fibre, *chart.region),
                               [component[3] for component in chart.components], chart.removed) \
            if domain is None or getattr(domain, "is_QQ", False) else None
        if native is not None:
            here = {frozenset(chart.components[j][0] for j in subset): chi for subset, chi in native.items()}
            if CHECK and here != _strata_here(chart, domain):
                raise AssertionError("The native Euler characteristics differ from this file's.")
        else:
            here = _strata_here(chart, domain)
        for key, chi in here.items():
            strata[key] = strata.get(key, 0) + chart.one_sheet(chi)
    return data, {key: chi for key, chi in strata.items() if chi}


def _strata_here(chart, domain=None):
    """The strata of one chart, computed here: {labels: Euler characteristic}."""
    from .constructible_euler import euler_characteristic
    if domain is None:
        from .reference import QQ
        domain = QQ
    api = chart_expressions(chart.variables)
    strata = {}
    zero = [g for g in dict.fromkeys((*chart.fibre, *chart.region)) if g in chart.variables]
    zero = dict.fromkeys(zero[:len(chart.variables) - 1], api.S.Zero)
    variables = tuple(v for v in chart.variables if v not in zero)

    def cut(ideal):
        values = (api.sympify(g).xreplace(zero) for g in ideal)
        return tuple(g for g in values if g != 0)

    base = (*cut(chart.ambient), *cut(chart.fibre), *cut(chart.region))
    ideals = [cut(component[3]) for component in chart.components]
    count = len(chart.components)
    alive = {()}
    for size in range(1, count + 1):
        found = set()
        for subset in combinations(range(count), size):
            if any(subset[:k] + subset[k + 1:] not in alive for k in range(size)):
                continue
            equations = (*base, *(g for j in subset for g in ideals[j]))
            if empty_locus(variables, equations, domain=domain):
                continue
            found.add(subset)
            others = [ideals[j] for j in range(count) if j not in subset] + [cut(z) for z in chart.removed]
            chi = euler_characteristic(variables, equations, others, domain=domain)
            if chi:
                key = frozenset(chart.components[j][0] for j in subset)
                strata[key] = strata.get(key, 0) + chi
        if not found:
            break
        alive = found
    return {key: chi for key, chi in strata.items() if chi}


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
    from . import kernel
    counts = {}
    for chart in charts:
        labels = [(name, ideal) for name, _, _, ideal in chart.components if not _is_strict(name)]
        own = (*chart.fibre, *chart.region)
        alive = {()}
        for size in range(1, len(labels) + 1):
            found = set()
            for subset in combinations(range(len(labels)), size):
                if any(subset[:k] + subset[k + 1:] not in alive for k in range(size)):
                    continue
                whole = (*chart.ambient, *(g for j in subset for g in labels[j][1]))
                if not _present(chart.variables, (*whole, *own), chart.removed):
                    continue
                found.add(subset)
                key = frozenset(labels[j][0] for j in subset)
                here = rational = None
                if radical_subset(chart.variables, chart.fibre, whole):  # the intersection lies over the fibre
                    if dimension == size:
                        native = kernel.strata(chart.variables, (*whole, *own), [()], chart.removed)
                        if native is None:
                            from .constructible_euler import euler_characteristic
                            here = euler_characteristic(chart.variables, (*whole, *own), chart.removed)
                        else:
                            here = native.get((0,), 0)
                    elif (chart.coefficient_field and chart.ambient == (chart.coefficient_field[1],)
                          and all(len(labels[j][1]) == 1 and labels[j][1][0] in chart.variables[:-1] for j in subset)):
                        # A coordinate flat on each embedding. Its generic point
                        # is owned on all sheets or on none by conjugacy.
                        owned = ideal_subset(chart.variables, own, whole) and not any(
                            ideal_subset(chart.variables, z, whole) for z in chart.removed)
                        here = chart.coefficient_field[2] if owned else 0
                    elif dimension - size == 1 and (plane := kernel.plane_components(
                            chart.variables, whole, own, chart.removed)) is not None:
                        here, rational = plane
                    elif dimension > size:
                        from .geometric_components import geometric_components
                        primes = geometric_components(chart.variables, whole, dimension - size)
                        if primes is not None:
                            here = sum(1 for prime in primes if ideal_subset(chart.variables, own, (*chart.ambient, *prime))
                                       and not any(ideal_subset(chart.variables, z, (*chart.ambient, *prime))
                                                   for z in chart.removed))
                if rational_counts is not None and size == 1:
                    # The plane operation counts rational factors as well as
                    # their geometric constituents. Explicit geometric primes
                    # from the general routine are already defined over Q.
                    count = rational if rational is not None else here if dimension > size else None
                    current = rational_counts.get(key, 0)
                    rational_counts[key] = None if current is None or count is None else current + chart.one_sheet(count)
                if key not in counts:
                    counts[key] = 0
                counts[key] = None if here is None or counts[key] is None else counts[key] + chart.one_sheet(here)
            if not found:
                break
            alive = found
    return counts
