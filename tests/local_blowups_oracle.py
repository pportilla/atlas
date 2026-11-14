"""Independent symbolic checks for local blowups, finite centres, and crossing loci."""
from itertools import combinations

from engine.blowup_ideal import empty_locus, ideal_basis
from engine.centres import GiveUp
from engine.groebner import groebner, colon
from engine.local_blowups import Chart, _generators


def present_outside(variables, ideal, removed):
    from itertools import product
    if empty_locus(variables, ideal):
        return False
    basis = groebner(ideal, variables) if removed else None
    relevant = []
    for hole in removed:
        # On the closed locus, use normal forms before forming products of
        # generators. The complement of V(J) is the union of their D(g).
        residues = tuple(dict.fromkeys(basis.reduce(f)[1] for f in hole))
        if any(f != 0 and not getattr(f, "free_symbols", ()) for f in residues):
            continue
        choices = tuple(f for f in residues if f != 0)
        if not choices:
            return False
        if not empty_locus(variables, (*ideal, *choices)) and choices not in relevant:
            relevant.append(choices)
    if not relevant:
        return True
    from .affine_kernel import frame_locus
    return any(not frame_locus(variables, tuple(ideal), units) for units in product(*relevant))


def _dimension_reference(variables, ideal):
    # Variables absent from the generators give an affine factor. This
    # also keeps the independent reference small on wide sparse charts.
    import sympy as sp
    from engine import native_polynomial as np
    if np.native_ring(variables):
        symbols = tuple(sp.Dummy(v.name) for v in variables)
        ideal = [sp.Poly.from_dict(np.terms(variables, f), symbols, domain=sp.QQ).as_expr() for f in ideal]
        variables = symbols
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


def _point_components_reference(variables, ideal, separated=None):
    """SymPy reference for finite point decomposition and fallback on kernel limits."""
    import sympy as sp
    from sympy import QQ
    from engine.constructible_euler import euler_characteristic
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


def _failures_here(chart, dimension):
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
                import sympy as sp
                # In the fibre ring, duplicate or zero Jacobian rows
                # contribute only zero minors and repeated generators.
                gradients = {tuple(basis.reduce(sp.diff(g,v))[1] for v in variables) for g in set(gens)}
                gradients.discard((0,)*len(variables))
                minors = set()
                for rows in combinations(gradients, rank):
                    active = [i for i in range(len(variables)) if any(row[i] for row in rows)]
                    for columns in combinations(active,rank):
                        block = [[row[i] for i in columns] for row in rows]
                        minor = sp.expand(sp.Matrix(block).det(method="berkowitz"))
                        if minor:
                            minors.add(minor)
                            if not minor.free_symbols:
                                break
                    else:
                        continue
                    # A nonzero constant generates the unit ideal.
                    break
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


def _blow_up_linear_here(chart, linear, label, stage, codimension, names):
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
            import sympy as sp
            from sympy import QQ
            value = sp.Poly(sp.sympify(f).xreplace(swap), *variables, domain=QQ).as_expr()
            return reducer.reduce(value)[1] if reducer is not None and value.free_symbols & set(parameters) else value

        def ideal(gens):
            return tuple(g for g in map(move, gens) if g != 0)

        components, multiplicity, discrepancy = [], 0, codimension
        for name, n, nu, before in chart.components:
            moved = ideal(before)
            if not moved:
                raise GiveUp("A divisor contains a chart of a blowup.")
            import sympy as sp
            order = min(min(monomial[0] for monomial in sp.Poly(g, w).monoms()) for g in moved)
            if order and len(moved) > 1:
                after = colon((*chart.ambient, *moved), (w**order,), variables)
                if after is None:
                    raise GiveUp("An ideal quotient is not available.")
                after = _generators(variables, chart.ambient, after)
            elif not order:
                after = moved
            else:
                import sympy as sp
                after = tuple(sp.expand(sp.cancel(g/w**order)) for g in moved)
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


def residual_quotient(variables, ambient, generators, pivot, expected_order):
    from sympy import QQ
    ring = QQ.old_poly_ring(*variables, order="grevlex")
    quotient = ring.ideal(*ambient, *generators).quotient(ring.ideal(pivot**expected_order))
    return ideal_basis(variables, tuple(ring.to_sympy(g) for g in quotient.gens))
