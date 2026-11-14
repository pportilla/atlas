"""Normal crossings of a smooth surface with exceptional divisors, in dimension three.

W is a smooth threefold, X a smooth surface in it and E a collection of
smooth divisors with normal crossings, none containing X. Each E restricts
to a divisor D_E = sum_C m(E, C) C on X, over the irreducible curves C of X;
R is the union of these curves, with its reduced structure.

Lemma. X and E have normal crossings at a point q of X exactly when R is
smooth or has a node at q, every curve C through q has sum_E m(E, C) = 1,
and the two branches of a node do not belong to the same divisor.
(Then the divisors through q restrict to the branches of R at q, one each,
and their equations restricted to X are part of a coordinate system of X.)

Two facts used below. Since the differentials of the divisors through q are
independent in W, and those of the divisors tangent to X at q (the E with
D_E of order at least 2 at q) lie on the line of forms vanishing on the
tangent plane of X: at most one divisor is tangent to X at a point; and two
divisors that share a curve through q exclude a tangent one there.

The blowup of W at a point q of X blows up X at q; the new curve A has
m(E, A) = (order of D_E at q) - 1 for the old divisors through q and
m(new, A) = 1. The blowup of W along a curve C of X leaves X as it is;
it lowers m(E, C) by one for the divisors containing C and gives
m(new, C) = 1. Both keep X smooth.

Procedure. Its steps, in this order of priority:

 (a) R has a singular point that is not a node: blow up these points.
     Finitely often, by the embedded resolution of a curve on a surface;
     the new curves belong to the new divisors, so R stays the whole trace.
 (b) R has only nodes, and some curve has sum_E m(E, C) >= 2. Let M be the
     largest multiplicity m(E, C). If M >= 2, take the last label E that
     attains it and the union G of the curves with m(E, C) = M; if M = 1,
     the last pair of labels with a common curve and the union G of their
     common curves. If G is singular (two of its curves cross, or one
     crosses itself), blow up its singular points: they are nodes of R,
     their number goes down, and the new curves do not belong to G.
     Otherwise blow up G, the curves contained in a given other divisor
     first when the others are not. A curve C of G has normal crossings
     with E: a divisor through one of its points q that does not contain C
     contains the other branch C' of R at q, and is transverse to C unless
     m(E', C') >= 2, that is, unless E' is tangent to X at q. If C has a
     divisor of multiplicity >= 2, that divisor is tangent at q and E' is
     not; if it has none, no curve has one (the largest is taken first).
     Two divisors through q that both contain C' and not C are excluded by
     the second fact, or by the count of divisors through a point.
     A curve with multiplicities (m) becomes (m-1, 1), then (m-2, 1), ...,
     then (1, 1), then a single divisor with multiplicity 1: finitely many
     blowups, and the other curves are not affected.
 (c) Every curve has sum_E m(E, C) = 1, and some divisor contains both
     branches of a node: blow up these nodes. The new curve has
     multiplicities (1, 1); after its blowup by (b) the two branches and
     the new curve belong to three different divisors.

When none applies, the Lemma gives normal crossings. Each step either
lowers the number of points of R that are not nodes after finitely many
repetitions, or a sum of multiplicities, or the number of nodes of (c),
without undoing the earlier ones: the procedure is finite.

Computation. The curves and multiplicities are those of polynomials in two
variables when X is the graph of a rational function over a coordinate
plane: an equation of X (or of W) that is linear in a variable, with a
coefficient that does not vanish at the points considered, determines that
variable. Functions on X are then polynomials in the two remaining
variables, up to powers of the coefficients, which are units.

Otherwise X is used as it is, with two of the variables as coordinates and
derivations by implicit differentiation (engine/centres.py). A principal
ideal J of X has radical J : (J + its derivatives): if J = (prod p^e), the
second ideal is prod p^(e-1) times an ideal of finite colength. So the
union of the curves with m(E, C) >= k is obtained from the trace of E by
k - 1 quotients, and the curves of one divisor contained in another by a
saturation and a quotient. A singular point of R is a node when the
Hessian of some generator of its ideal is not degenerate there.
"""
from time import monotonic

from .chart_expressions import chart_expressions
from .centre_weights import centre_search, cover_products, level_weights, ordered_parts, surface_maxima

from .affine_kernel import frame_derivative, frame_function, frame_linear, frame_locus, frame_minor, frame_restrict
from .blowup_ideal import empty_locus, ideal_basis, radical_subset, saturate
from .centres import GiveUp, _intersection, _principal
from .groebner import groebner


class NoModel(GiveUp):
    """The surface is not a rational graph over a coordinate plane near some point."""


def _empty(variables, units, gens):
    """Whether the ideal has no zero where the units are invertible."""
    return frame_locus(tuple(variables), tuple(gens), tuple(units))


def _models(variables, equations, functions, working, threefold=0):
    """Describe the surface, near the points of the working set, with as few
    variables as possible: an equation linear in a variable with invertible
    coefficient determines that variable. The first `threefold` equations
    are those of the chart, the others cut the surface in it. Yields
    (variables, units, functions, working, equations left, how many of these
    are equations of the chart: they come first), with the functions and the
    working ideals in those variables; with no equation left the variables
    are two and the surface is an open set of their plane. On the open
    defined by a nonvanishing coefficient a, substitution and clearing
    powers of a induce an isomorphism after localization. Removing any
    further common powers of a preserves every zero locus there. The locus
    a=0 stays on the pending list in the original coordinates."""
    pending = [(tuple(variables), [(e, i < threefold) for i, e in enumerate(equations) if e != 0], dict(functions), (),
                list(working))]
    while pending:
        names, tagged, fns, units, work = pending.pop()
        eqs = [e for e, _ in tagged]
        if not eqs:
            if len(names) != 2:
                raise NoModel("A chart does not reduce to a plane.")
            yield names, units, fns, work, (), 0
            continue
        selected = {}
        def linear_open(task):
            e, y = tagged[task["i"]][0], names[task["axis"]]
            linear = frame_linear(names, e, y)
            if linear is None:
                return None
            a, b = linear
            hit = [j for j,w in enumerate(work) if not _empty(names, (*units, a), (*eqs, *w))]
            if not hit:
                return None
            selected["linear"] = linear
            return hit

        choice = centre_search("surface_model", linear_open, width=len(names),
                               costs=[chart_expressions(variables).count_ops(e) for e,_ in tagged])
        if choice is not None:
            e, y = tagged[choice["i"]][0], names[choice["axis"]]
            a, b = selected["linear"]
            hit = [work[j] for j in choice["hit"]]
            rest = tuple(v for v in names if v != y)

            def move(f, y=y, a=a, b=b, names=names):
                return frame_restrict(names, f, y, a, b)

            def ideal(gens, move=move):
                return tuple(g for g in map(move, gens) if g != 0)

            moved = [(move(other), flag) for other, flag in tagged if other is not e]
            pending.append((rest, [(g, flag) for g, flag in moved if g != 0],
                            {label: move(f) for label, f in fns.items()}, (*ideal(units), a),
                            [ideal(w) for w in hit]))
            left = [w for w in work if w not in hit] + ([] if a.is_number else [(*w, a) for w in hit])
            left = [w for w in left if not _empty(names, units, (*eqs, *w))]
            if left:
                pending.append((names, tagged, fns, units, left))
        else:
            work = [w for w in work if not _empty(names, units, (*eqs, *w))]
            if work:
                # The equations left are used as they are, those of the chart first.
                first = [e for e, flag in tagged if flag]
                yield names, units, fns, work, (*first, *(e for e, flag in tagged if not flag)), len(first)


def _surface_search(variables, labels, query, *, implicit=False):
    from .kernel import exprs
    from .binomial_kernel import _integer
    result = centre_search("surface_plan", query, labels=labels, implicit=implicit)
    if result is None:
        return None
    key = (result["kind"], _integer(result["top"]), int(result["point"]), tuple(labels[i] for i in result["labels"]))
    return key, result["type"], exprs(variables, result["ideal"])


def _plane_step(names, units, functions, working):
    """The step of the procedure on one plane model: (key, kind, generators), or None."""
    from .affine_kernel import rational_factor_rows, _sparse_expression
    s, t = names

    def expand(f):
        return frame_function(names, f)

    def derivative(f, v):
        return frame_derivative(names, f, {v: 1})

    def factors(f):
        return [(_sparse_expression(names, row), m) for row, m, _ in rational_factor_rows(names, f)[1]]

    def here(gens, everywhere=False):
        if everywhere:
            return not _empty(names, units, gens)
        return any(not _empty(names, units, (*gens, *w)) for w in working)

    # The curves of the traces through the points considered, with their multiplicities.
    curves = {}
    for label, f in functions.items():
        if f == 0:
            raise GiveUp("A divisor contains the surface.")
        for factor, power in factors(f):
            p = factor.as_expr() if hasattr(factor, "as_expr") else factor
            if p.free_symbols and here((p,)):
                curves.setdefault(p, {})[label] = power
    if not curves:
        return None
    from .binomial_kernel import run
    from .centre_weights import centre_rows
    columns = level_weights("label_order", (), labels=list(functions))

    def query(task):
        operation = task["query"]
        if operation == "nonordinary":
            whole = expand(chart_expressions(names).prod(curves))
            first = [derivative(whole, v) for v in names]
            bad = (whole, *first, frame_minor(names, first, names))
            return centre_rows(names,bad) if here(bad) else None
        if operation == "curve_choice":
            choice, *matches = run(f"surface_curve_matches {len(curves)} {len(columns)}",
                                   *(tuple(o.get(label,0) for label in columns) for o in curves.values()))
            if choice == "none":
                return None
            top, *indices = choice.split()
            equations = tuple(curves)
            together = expand(chart_expressions(names).prod(equations[j] for j in map(int,matches[0].split())))
            return dict(top=top, labels=list(map(int,indices)), ideal=centre_rows(names,(together,)))
        if operation == "curve_singular":
            from .kernel import exprs
            together, = exprs(names, task["ideal"])
            singular = (together, derivative(together, s), derivative(together, t))
            return centre_rows(names,singular) if here(singular,everywhere=True) else None
        label = columns[task["i"]]
        trace = expand(chart_expressions(names).prod(p for p,m in curves.items() if label in m))
        nodes = (trace, derivative(trace, s), derivative(trace, t))
        return centre_rows(names,nodes) if trace.free_symbols and here(nodes) else None

    return _surface_search(names, columns, query)


def _implicit_step(frame, functions, working, deadline=None):
    """The same step on a surface given by equations, with two of the variables
    as coordinates: (key, kind, generators in the ring of the frame), or None."""
    from .groebner import _MODE, colon
    if _MODE == "sympy":
        from tests.surface_oracle import _implicit_step as reference
        return reference(frame, functions, working, deadline)
    # Ideals are taken in the variables of the chart, over the equations of the
    # surface; the open set of the frame only enters the tests for emptiness.
    ring, base = frame.variables, frame.rows

    def here(gens, everywhere=False):
        if deadline is not None and monotonic() > deadline:
            raise GiveUp("The time allowed for the blowups is over.")
        if everywhere:
            return not frame.empty(*gens)
        return any(not frame.empty(*gens, *w) for w in working)

    def quotient(ideal, by):
        answer = colon(tuple(ideal), tuple(by), ring)
        if answer is None:
            raise NoModel("An ideal quotient is not available.")
        return tuple(answer)

    def derived(ideal):
        # Each derivation is tangent to the surface. A generator which is
        # zero on the open frame contributes only vanishing derivatives
        # there. It is safe to retain them, and we need no saturation to
        # recognize zero functions outside the working locus.
        return [d for g in ideal for k in frame.coords if (d := frame.derivative(g, k)) != 0]

    def radical(ideal):
        return quotient(ideal, (*ideal, *derived(ideal)))

    def expanded(f):
        return frame_function(ring, f)

    labels = [label for label, f in functions.items() if here((f,))]
    if not labels:
        return None
    from .centre_weights import centre_rows
    from .kernel import exprs
    def nonordinary():
        product = expanded(chart_expressions(ring).prod((functions[label] for label in labels)))
        whole = radical((*base, product))
        a, b = frame.coords
        gradients = []
        for g in whole:
            if g not in base:
                gradients.append((frame.derivative(g, a), frame.derivative(g, b)))
        # The Hessian is only tested at singular points in the working set.
        # Reducing its entries modulo each such slice before multiplication
        # preserves the determinant there and keeps the checked equations small.
        singular = (*whole, *(d for gradient in gradients for d in gradient if d != 0))
        slices = [w for w in working if not frame.empty(*singular, *w)]
        if slices:
            # Differentiating a representative of the first derivative is valid:
            # the frame fields are tangent to its equations.
            da, db = frame.field(a), frame.field(b)
            entries = [(frame_derivative(ring, ga, da), frame_derivative(ring, ga, db),
                        frame_derivative(ring, gb, db)) for ga, gb in gradients if ga != 0 or gb != 0]
            from .affine_kernel import _request, _ring, _sparse_expression
            wire = _ring(ring)

            def determinants(rows):
                lines = (wire.text(value) for triple in rows for value in triple)
                result = _request(["symmetric_determinants", wire.header, *lines])
                return tuple(_sparse_expression(ring, row) for row in result)

            for w in slices:
                slice_basis = groebner((*base, *singular, *w), ring)
                hessians = determinants(tuple(slice_basis.reduce(g)[1] for g in row) for row in entries)
                if not frame.empty(*singular, *w, *hessians):
                    # The centre itself must still use the whole singular locus;
                    # the working ideals only select which step has priority.
                    whole_hessians = determinants(entries)
                    return centre_rows(ring, (*singular, *whole_hessians))
        return None

    labels = level_weights("label_order", (), labels=labels)
    levels = {label: [] for label in labels}
    current = {label: (*base, functions[label]) for label in labels}

    def query(task):
        action = task["query"]
        if action == "nonordinary":
            return nonordinary()
        if action == "next_level":
            label = labels[task["i"]]
            reduced = radical(current[label])
            if not here(reduced):
                return False
            levels[label].append(reduced)
            current[label] = quotient(current[label], reduced)
            return True
        if action == "highest_curve":
            return centre_rows(ring, levels[labels[task["i"]]][-1])
        if action == "shared_curve":
            first, second = labels[task["a"]], labels[task["b"]]
            outside = saturate(ring,levels[first][0],functions[second])
            inside = levels[first][0] if tuple(outside)==(1,) else quotient(levels[first][0],outside)
            return centre_rows(ring,inside) if here(inside) else None
        if action == "curve_singular":
            chosen = exprs(ring,task["ideal"])
            singular = (*chosen,*derived(chosen))
            return centre_rows(ring,singular) if here(singular,everywhere=True) else None
        trace = levels[labels[task["i"]]][0]
        nodes = (*trace,*derived(trace))
        return centre_rows(ring,nodes) if here(nodes) else None

    return _surface_search(ring, labels, query, implicit=True)


def step(variables, ambient, strict, divisors, working, deadline=None, away=()):
    """One step of the procedure on a chart.

    `strict` generates the ideal of the surface X on the chart W = V(ambient),
    `divisors` lists (label, ideal) for the exceptional divisors and `working`
    the ideals of the points where normal crossings fail. Returns
    (key, kind, generators of the centre on the chart), with kind "points" or
    "curve" and keys compared between charts (the largest is done first), or
    None when the procedure has nothing to do. Raises NoModel when the
    surface has no two variables as coordinates near those points.

    Variables that an equation determines are eliminated first (the models
    below); the step on each model is the kernel's (native/ideal.cpp,
    op_surface). This file's own two forms answer when the kernel cannot
    (ATLAS_GROEBNER=sympy), and with ATLAS_GROEBNER=check both are computed
    and must agree.
    """
    variables, ambient = tuple(variables), tuple(ambient)
    working = [tuple(w) for w in working if not empty_locus(variables, (*ambient, *strict, *w))]
    if not working:
        return None
    away = [z for z in away if any(not empty_locus(variables, (*ambient, *strict, *z, *w)) for w in working)]
    if away:
        # The points of the closed sets V(Z), Z in `away`, are left out: the
        # rest is covered by the open sets u != 0, u a product of one generator
        # of each Z. Such an open set is the chart t*u = 1 with one variable
        # more, which the models below eliminate again.
        covers = cover_products(variables, away)
        t = chart_expressions(variables).Dummy("off")
        found, left = [], working
        for unit in dict.fromkeys(covers):
            inner = step((t, *variables), (*ambient, t*unit - 1), strict, divisors, left, deadline)
            if inner is not None:
                key, kind, centre = inner
                closed = [g for g in groebner(centre, (t, *variables), order="first").exprs if t not in g.free_symbols]
                found.append((key, kind, tuple(ideal_basis(variables, closed))))
            left = [w for w in ((*w, unit) for w in left) if not empty_locus(variables, (*ambient, *strict, *w))]
            if not left:
                break
        return _best(variables, found)
    functions = {}
    for label, ideal in divisors:
        if any(not empty_locus(variables, (*ambient, *strict, *ideal, *w)) for w in working):
            functions[label] = _principal(variables, ambient, ideal)
    return _step(variables, ambient, strict, functions, working, deadline)


def _best(variables, found):
    """Of the steps found on the open sets used, the one with the largest key; its locus is the union of theirs."""
    if not found:
        return None
    indices = surface_maxima(found)
    key, kind, _ = found[indices[0]]
    centres = list(dict.fromkeys(found[i][2] for i in indices))
    order = ordered_parts(variables, centres)
    centre = centres[order[0]]
    for i in order[1:]:
        other = centres[i]
        centre = tuple(ideal_basis(variables, _intersection(variables, centre, other)))
    return key, kind, centre


def _on_model(names, units, fns, work, left, threefold, deadline):
    """The steps found on one model, as (key, kind, generators in its variables)."""
    from . import kernel
    from .centres import _frames
    from .groebner import CHECK, _MODE
    labels = level_weights("label_order", (), labels=list(fns))
    native = kernel.surface_step(names, left, [fns[label] for label in labels], work, units, threefold)
    if native == "nomodel":
        raise NoModel("No two variables are coordinates of the surface near some point.")
    if native is not None:
        native = [((kind, top, flag, tuple(labels[i] for i in positions)),
                   "curve" if kind == 2 and not flag else "points", tuple(gens))
                  for kind, top, flag, positions, gens in native]
        if not CHECK:
            return native
    if CHECK or _MODE == "sympy":
        from tests.surface_oracle import _implicit_step as implicit, _plane_step as plane
    else:
        implicit, plane = _implicit_step, _plane_step
    if not left:
        mine = [plane(names, units, fns, work)]
    else:
        # Two of the remaining variables as coordinates, the others determined by the equations left.
        try:
            # Only the points where the units are invertible are to be covered.
            frames = list(_frames(names, left, 2, (), work,
                                  units=tuple(u for u in units if not getattr(u, "is_number", True))))
        except GiveUp as error:
            raise NoModel(str(error)) from error
        mine = [implicit(frame, fns, part, deadline) for frame, part in frames]
    mine = [item for item in mine if item is not None]
    if native is not None:
        from tests.surface_oracle import compare_chart_steps
        compare_chart_steps(names, left, work, native, mine)
        return native
    return mine


def _step(variables, ambient, strict, functions, working, deadline=None):
    """The step on the chart: the largest of those found on its models."""
    found = []
    for names, units, fns, work, left, threefold in _models(variables, (*ambient, *strict), functions, working,
                                                              len(ambient)):
        for key, kind, gens in _on_model(names, units, fns, work, left, threefold, deadline):
            # The locus on the surface, then its closure in the chart.
            centre = ideal_basis(variables, (*ambient, *strict, *gens))
            for unit in units:
                if not getattr(unit, "is_number", True):
                    centre = saturate(variables, centre, unit)
            centre = tuple(ideal_basis(variables, centre))
            if centre != (1,):
                found.append((key, kind, centre))
    return _best(variables, found)
