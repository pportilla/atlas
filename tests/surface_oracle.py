"""Independent symbolic plane and implicit models for surface blowup selection."""
from time import monotonic

from engine.affine_kernel import frame_derivative
from engine.blowup_ideal import saturate
from engine.centres import GiveUp
from engine.chart_expressions import chart_expressions
from engine.groebner import groebner
from engine.surface import NoModel, _empty


def _plane_step(names, units, functions, working):
    """The step of the procedure on one plane model: (key, kind, generators), or None."""
    import sympy as sp
    s, t = names

    def expand(f):
        return sp.expand(f)

    def derivative(f, v):
        return sp.diff(f, v)

    def factors(f):
        return sp.factor_list(f, s, t)[1]

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


def _implicit_step(frame, functions, working, deadline=None):
    """The same step on a surface given by equations, with two of the variables
    as coordinates: (key, kind, generators in the ring of the frame), or None."""
    from engine.groebner import colon
    # Ideals are taken in the variables of the chart, over the equations of the
    # surface; the open set of the frame only enters the tests for emptiness.
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
            from engine.native_polynomial import native_ring
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
        return sp.Poly(f, *ring, domain=sp.QQ).as_expr()

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


def compare_chart_steps(names, left, work, native, mine):
    from engine.blowup_ideal import radical_subset
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
