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

from .affine_kernel import frame_derivative, frame_function, frame_linear, frame_locus, frame_restrict
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
        for e, _ in sorted(tagged, key=lambda item: chart_expressions(variables).count_ops(item[0])):
            done = False
            for y in names:
                linear = frame_linear(names, e, y)
                if linear is None:
                    continue
                a, b = linear
                # The substitution y=-b/a is defined precisely on this open.
                hit = [w for w in work if not _empty(names, (*units, a), (*eqs, *w))]
                if not hit:
                    continue
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
                done = True
                break
            if done:
                break
        else:
            work = [w for w in work if not _empty(names, units, (*eqs, *w))]
            if work:
                # The equations left are used as they are, those of the chart first.
                first = [e for e, flag in tagged if flag]
                yield names, units, fns, work, (*first, *(e for e, flag in tagged if not flag)), len(first)


def _plane_step(names, units, functions, working, *, reference=False):
    """The step of the procedure on one plane model: (key, kind, generators), or None."""
    if reference:
        import sympy as sp
    s, t = names
    def expand(f):
        return sp.expand(f) if reference else frame_function(names, f)

    def derivative(f, v):
        return sp.diff(f, v) if reference else frame_derivative(names, f, {v: 1})

    def factors(f):
        if reference:
            return sp.factor_list(f, s, t)[1]
        from .affine_kernel import rational_factor_rows, _sparse_expression
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
    whole = expand(chart_expressions(names).prod(curves))
    first = [derivative(whole, v) for v in names]
    hessian = expand(derivative(first[0], s)*derivative(first[1], t) - derivative(first[0], t)**2)
    if here((whole, *first, hessian)):
        return (3, 0, 0, ()), "points", (whole, *first, hessian)
    if any(sum(m.values()) >= 2 for m in curves.values()):
        top = max(max(m.values()) for m in curves.values())
        if top >= 2:
            labels = (max(label for m in curves.values() for label, power in m.items() if power == top),)
            chosen = [p for p, m in curves.items() if m.get(labels[0]) == top]
        else:
            labels = max(tuple(sorted(m)) for m in curves.values() if len(m) >= 2)
            chosen = [p for p, m in curves.items() if all(label in m for label in labels)]
        together = expand(chart_expressions(names).prod(chosen))
        singular = (together, derivative(together, s), derivative(together, t))
        if here(singular, everywhere=True):
            return (2, top, 1, labels), "points", singular
        return (2, top, 0, labels), "curve", (together,)
    for label in sorted(functions, reverse=True):
        trace = expand(chart_expressions(names).prod((p for p, m in curves.items() if label in m)))
        nodes = (trace, derivative(trace, s), derivative(trace, t))
        if trace.free_symbols and here(nodes):
            return (1, 0, 0, (label,)), "points", nodes
    return None


def _implicit_step(frame, functions, working, deadline=None, *, reference=False):
    """The same step on a surface given by equations, with two of the variables
    as coordinates: (key, kind, generators in the ring of the frame), or None."""
    from .groebner import colon
    # Ideals are taken in the variables of the chart, over the equations of the
    # surface; the open set of the frame only enters the tests for emptiness.
    if reference:
        import sympy as sp
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
            from .native_polynomial import native_ring
            if native_ring(ring):
                raise NoModel("An ideal quotient is not available.")
            from sympy import QQ
            polynomials = QQ.old_poly_ring(*ring, order="grevlex")
            quotient = polynomials.ideal(*ideal).quotient(polynomials.ideal(*by))
            answer = tuple(polynomials.to_sympy(g) for g in quotient.gens)
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
        return sp.Poly(f, *ring, domain=sp.QQ).as_expr() if reference else frame_function(ring, f)

    labels = [label for label, f in functions.items() if here((f,))]
    if not labels:
        return None
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
        for w in slices:
            slice_basis = groebner((*base, *singular, *w), ring)
            hessians = []
            for row in entries:
                aa, ab, bb = (slice_basis.reduce(g)[1] for g in row)
                hessians.append(expanded(aa*bb - ab**2))
            if not frame.empty(*singular, *w, *hessians):
                # The centre itself must still use the whole singular locus;
                # the working ideals only select which step has priority.
                whole_hessians = [expanded(aa*bb - ab**2) for aa, ab, bb in entries]
                return (3, 0, 0, ()), "points", (*singular, *whole_hessians)
    # For each divisor, the unions of its curves of multiplicity at least 1, 2, ...
    levels = {}
    for label in labels:
        current, found = (*base, functions[label]), []
        while len(found) < 64:
            reduced = radical(current)
            if not here(reduced):
                break
            found.append(reduced)
            current = quotient(current, reduced)
        levels[label] = found
    top = max(len(found) for found in levels.values())
    chosen = None
    if top >= 2:
        key = (max(label for label, found in levels.items() if len(found) == top),)
        chosen = levels[key[0]][top - 1]
    else:
        for first, second in sorted(((a, b) for a in labels for b in labels if a < b), reverse=True):
            outside = saturate(ring, levels[first][0], functions[second])
            inside = levels[first][0] if tuple(outside) == (1,) else quotient(levels[first][0], outside)
            if here(inside):
                key, chosen = (first, second), tuple(inside)
                break
    if chosen is not None:
        singular = (*chosen, *derived(chosen))
        if here(singular, everywhere=True):
            return (2, top, 1, key), "points", singular
        return (2, top, 0, key), "curve", chosen
    for label in sorted(labels, reverse=True):
        nodes = (*levels[label][0], *derived(levels[label][0]))
        if here(nodes):
            return (1, 0, 0, (label,)), "points", nodes
    return None


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
        covers = [chart_expressions(variables).sympify(1)]
        for ideal in away:
            covers = [frame_function(variables, u*g) for u in covers for g in ideal]
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
    key = max(item[0] for item in found)
    kind = next(item[1] for item in found if item[0] == key)
    centres = sorted({item[2] for item in found if item[0] == key}, key=chart_expressions(variables).default_sort_key)
    centre = centres[0]
    for other in centres[1:]:
        centre = tuple(ideal_basis(variables, _intersection(variables, centre, other)))
    return key, kind, centre


def _on_model(names, units, fns, work, left, threefold, deadline):
    """The steps found on one model, as (key, kind, generators in its variables)."""
    from . import kernel
    from .centres import _frames
    from .groebner import CHECK, _MODE
    labels = sorted(fns)
    native = kernel.surface_step(names, left, [fns[label] for label in labels], work, units, threefold)
    if native == "nomodel":
        raise NoModel("No two variables are coordinates of the surface near some point.")
    if native is not None:
        native = [((kind, top, flag, tuple(labels[i] for i in positions)),
                   "curve" if kind == 2 and not flag else "points", tuple(gens))
                  for kind, top, flag, positions, gens in native]
        if not CHECK:
            return native
    if not left:
        mine = [_plane_step(names, units, fns, work, reference=CHECK or _MODE == "sympy")]
    else:
        # Two of the remaining variables as coordinates, the others determined by the equations left.
        try:
            # Only the points where the units are invertible are to be covered.
            frames = list(_frames(names, left, 2, (), work,
                                  units=tuple(u for u in units if not getattr(u, "is_number", True))))
        except GiveUp as error:
            raise NoModel(str(error)) from error
        mine = [_implicit_step(frame, fns, part, deadline, reference=CHECK or _MODE == "sympy") for frame, part in frames]
    mine = [item for item in mine if item is not None]
    if native is not None:
        # Compared after closure, where generators are canonical.
        best = max((item[0] for item in native), default=None)
        same = best == max((item[0] for item in mine), default=None)
        if same and best is not None:
            # The loci, over the points considered (elsewhere they may differ by
            # parts that play no role): the same zero sets.
            ours = [gens for key, _, gens in native if key == best]
            theirs = [gens for key, _, gens in mine if key == best]
            for w in work:
                same = same and all(
                    radical_subset(names, (*left, *first, *w), (*left, *second, *w)) and
                    radical_subset(names, (*left, *second, *w), (*left, *first, *w))
                    for first in ours[:1] for second in theirs[:1])
        if not same:
            raise AssertionError("The native step of the surface procedure differs from this file's.")
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
