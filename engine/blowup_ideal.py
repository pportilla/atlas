from __future__ import annotations
"""Exact affine blowup charts and fixed-dimension smoothness/SNC tests.

All inputs are SymPy expressions representing polynomials in the listed
variables, over QQ or a characteristic-zero algebraic number field. Ideals
are tuples of generators: () is zero and (1,) is the unit ideal. Closed
loci and smoothness statements are geometric (over an algebraic closure).

A graph chart is the closure of the graph of f_j / f_i in D(f_i).
Standard smooth presentations give a Jacobian criterion independent of a
global complete-intersection or equidimensionality assumption.
See https://stacks.math.columbia.edu/tag/01OF and
https://stacks.math.columbia.edu/tag/01V4.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from . import native_polynomial as np
from .chart_expressions import chart_expressions
from .reference import sp
from .reference import QQ

from . import kernel
from .groebner import CHECK, _MODE, _canonical_qq, colon, groebner


def _ring(variables, domain):
    variables = tuple(variables)
    if not variables or not all(getattr(v, "is_Symbol", False) for v in variables):
        raise ValueError("The polynomial ring needs one or more coordinate symbols.")
    if len(set(variables)) != len(variables):
        raise ValueError("Coordinates must be distinct symbols.")
    if not (domain == QQ or getattr(domain, "is_AlgebraicField", False)):
        raise ValueError("Use QQ or a finite algebraic extension of QQ.")
    return variables


@lru_cache(maxsize=50_000)
def _canonical(variables, expr, domain):
    """Canonical polynomial; preserve zero and the positions of chart generators."""
    if isinstance(expr, str):
        raise TypeError("Supply parsed polynomial expressions, not strings.")
    if np.native_ring(variables) and domain == QQ:
        return _canonical_qq(variables, expr)
    expr = chart_expressions(variables).sympify(expr)
    if domain == QQ:
        value = _canonical_qq(variables, expr)
        if value is not None:
            return value
    elif _MODE != "sympy":
        from .field_factors import compose
        value, = compose(variables, variables, variables, (expr,), domain)
        return value
    return sp.Poly(expr, *variables, domain=domain).as_expr()


def _expressions(variables, ideal, domain):
    """Coerce coefficients explicitly; textual expressions are not accepted."""
    rows = (_canonical(variables, expr, domain) for expr in ideal)
    return tuple(dict.fromkeys(g for g in rows if g != 0))


def _native(function, reference, domain, *arguments):
    """The kernel's answer when it gives one, otherwise SymPy's.

    The functions of engine/kernel.py return None for what they cannot take.
    With ATLAS_GROEBNER=check both are computed and must agree.
    """
    if domain == QQ:
        answer = function(*arguments)
    elif domain.is_AlgebraicField and function in (kernel.smooth_locus, kernel.verify_snc):
        answer = _field_smoothness(function, arguments, domain)
    else:
        answer = None
    if answer is None:
        return reference(*arguments, domain)
    if CHECK and answer != reference(*arguments, domain):
        raise AssertionError(f"The native {function.__name__} differs from SymPy's.")
    return answer


def _field_smoothness(function, arguments, domain):
    """Base change to Qbar splits Q[theta]/m as disjoint copies, one per
    embedding of K. Since m is separable, adding m to the ambient equations
    adds one variable and one independent Jacobian row on every sheet.
    All dimensions and smoothness loci are therefore those of the K-chart.
    """
    from .field_polynomial import lift, lower, modulus
    if _MODE == "sympy":
        return None
    variables, equations, *rest = arguments
    theta = chart_expressions(variables).Dummy("smooth_field")
    extended = (*variables, theta)
    def move(ideal):
        return tuple(lift(g, variables, domain, theta) for g in ideal)
    ambient = (*move(equations), modulus(domain, theta))
    if function is kernel.smooth_locus:
        result = function(extended, ambient, *rest)
        witnesses = [result]
    else:
        strict, exceptional, dimension = rest
        result = function(extended, ambient, None if strict is None else move(strict),
                          tuple(map(move, exceptional)), dimension)
        witnesses = result["strata"] if result is not None else ()
    if result is None:
        return None
    for item in witnesses:
        if item["bad_ideal"] != (1,):
            rows = tuple(lower(g, variables, domain, theta) for g in item["bad_ideal"])
            item["bad_ideal"] = ideal_basis(variables, rows, domain=domain)
    return result


@lru_cache(maxsize=128)
def _gb(variables, ideal, domain, order="grevlex", known=0):
    return groebner(ideal, variables, order=order, domain=domain, known=known)


def _basis(variables, ideal, domain, known=0):
    """known counts leading generators that already form a grevlex Groebner basis."""
    return tuple(_gb(variables, tuple(ideal), domain, "grevlex", known).exprs)


def ideal_basis(variables, ideal, *, domain=QQ):
    variables = _ring(variables, domain)
    return _basis(variables, _expressions(variables, ideal, domain), domain)


@lru_cache(maxsize=128)
def _packed_present(variables, ideal):
    return kernel.present_outside(variables, ideal, ())


def empty_locus(variables, ideal, *, domain=QQ):
    """Decide geometric emptiness by the unit ideal test."""
    variables = _ring(variables, domain)
    equations = _expressions(variables, ideal, domain)
    if _MODE == "native" and domain == QQ and np.native_ring(variables):
        from .algebra_execution import current_execution
        from .native_library import current_replay_scope
        scope = current_replay_scope()
        if current_execution() is None and (scope is None or scope.compute == "cpu"):
            present = _packed_present(variables, equations)
            if present is not None:
                return not present
    return _basis(variables, equations, domain) == (1,)


def ideal_subset(variables, smaller, larger, *, domain=QQ):
    variables = _ring(variables, domain)
    basis = _gb(variables, _expressions(variables, larger, domain), domain)
    return all(basis.contains(f) for f in _expressions(variables, smaller, domain))


def radical_subset(variables, smaller, larger, *, domain=QQ):
    """Decide sqrt(smaller) subset sqrt(larger); equivalently V(larger) subset V(smaller)."""
    variables = _ring(variables, domain)
    return _native(kernel.radical_subset, _radical_subset, domain, variables, tuple(smaller), tuple(larger))


def _radical_subset(variables, smaller, larger, domain):
    small = _expressions(variables, smaller, domain)
    large = _expressions(variables, larger, domain)
    if ideal_subset(variables, small, large, domain=domain):
        return True
    t = chart_expressions(variables).Dummy("inverse")
    return all(_basis((t, *variables), (*large, 1 - t*f), domain) == (1,) for f in small)


def saturate(variables, ideal, by, *, domain=QQ):
    """Return generators of I : by^infinity via exact lexicographic elimination."""
    variables = _ring(variables, domain)
    return _native(kernel.saturate, _saturate, domain, variables, tuple(ideal), by)


def _saturate(variables, ideal, by, domain):
    generators = _expressions(variables, ideal, domain)
    if isinstance(by, str):
        raise TypeError("The saturating element must be a parsed expression.")
    divisor = _canonical(variables, by, domain)
    if divisor == 0:
        return (chart_expressions(variables).S.One,)
    if not divisor.free_symbols or not generators:
        return _basis(variables, generators, domain)
    t = chart_expressions(variables).Dummy("inverse")
    basis = _gb((t, *variables), (*generators, 1 - t*divisor), domain, "lex")
    return tuple(g for g in basis.exprs if t not in g.free_symbols)


@dataclass(frozen=True)
class BlowupChart:
    """Projection keeps the old coordinates; ratios dehomogenize the center generators."""
    variables: tuple
    ambient: tuple
    strict: tuple | None
    exceptional: tuple
    pivot: sp.Expr
    pivot_index: int
    ratios: tuple
    projection: tuple


def blowup_charts(variables, ambient, strict, center, exceptional=(), *, domain=QQ):
    """Cover the blowup along the supplied ideal by its standard affine opens.

    Original generators (including zeros) determine the projective coordinate
    ordering. Zero generators give empty charts. Each old exceptional ideal
    is replaced by its strict transform and the new exceptional ideal is
    appended. This function constructs the blowup for *any* ideal and does
    not by itself assert admissibility of that ideal as a resolution center.
    """
    variables = _ring(variables, domain)
    center = tuple(center)
    return _native(_kernel_charts, _blowup_charts, domain, variables, tuple(ambient),
                   None if strict is None else tuple(strict), center, tuple(tuple(ideal) for ideal in exceptional))


def chart_coordinates(variables, count, i):
    """The ratios f_j/f_i of the i-th chart of a centre with `count` generators
    (one at j = i), and the chart's coordinates: the old ones, then the ratios."""
    names = {v.name for v in variables}
    ratios = []
    for j in range(count):
        if j == i:
            ratios.append(chart_expressions(variables).S.One)
            continue
        name = f"r{i+1}_{j+1}"
        while name in names:
            name += "_"
        names.add(name)
        ratios.append(chart_expressions(variables).Symbol(name))
    return tuple(ratios), variables + tuple(x for j, x in enumerate(ratios) if j != i)


def _kernel_charts(variables, ambient, strict, center, exceptional):
    if any(isinstance(f, str) for f in center):
        return None
    layout = [chart_coordinates(variables, len(center), i) for i in range(len(center))]
    answer = kernel.blowup_charts(variables, ambient, strict, center, exceptional,
                                  [coordinates for _, coordinates in layout])
    if answer is None:
        return None
    generators = tuple(_canonical(variables, f, QQ) for f in center)
    return tuple(BlowupChart(variables=coordinates, ambient=chart_ambient, strict=chart_strict,
                             exceptional=transforms + ((pivot,),), pivot=pivot, pivot_index=i,
                             ratios=ratios, projection=variables)
                 for i, (pivot, (ratios, coordinates), (chart_ambient, chart_strict, transforms))
                 in enumerate(zip(generators, layout, answer)))


def _blowup_charts(variables, ambient, strict, center, exceptional, domain):
    ambient = _expressions(variables, ambient, domain)
    strict = None if strict is None else _expressions(variables, strict, domain)
    exceptional = tuple(_expressions(variables, ideal, domain) for ideal in exceptional)
    if any(isinstance(f, str) for f in center):
        raise TypeError("Center generators must be parsed expressions.")
    generators = tuple(_canonical(variables, f, domain) for f in center)
    charts = []
    for i, pivot in enumerate(generators):
        ratios, coordinates = chart_coordinates(variables, len(generators), i)
        graph = list(ambient)
        graph.extend(f - pivot*ratios[j] for j, f in enumerate(generators) if j != i)
        chart_ambient = saturate(coordinates, graph, pivot, domain=domain)

        def transform(ideal):
            return saturate(coordinates, (*chart_ambient, *ideal), pivot, domain=domain)

        charts.append(BlowupChart(
            variables=coordinates, ambient=chart_ambient,
            strict=None if strict is None else transform(strict),
            exceptional=tuple(transform(ideal) for ideal in exceptional) + ((pivot,),),
            pivot=pivot, pivot_index=i, ratios=ratios, projection=variables,
        ))
    return tuple(charts)


def resolution_charts(variables, f, center, projective, boundary):
    """The kernel's verdict on a candidate resolution of the hypersurface f over Q.

    None when the kernel cannot take the candidate and False when it rejects
    it. Otherwise the charts of the blowup of the centre and the number of
    strata proved smooth: the centre lies in the singular locus of f and, on
    every chart, the boundary ideals (homogeneous in the projective
    coordinates) cut out the exceptional divisor and make normal crossings
    with the strict transform.
    """
    variables, center = tuple(variables), tuple(center)
    return assemble_charts(variables, center, kernel.verify(variables, f, center, projective, boundary))


def assemble_charts(variables, center, answer):
    """The charts and the number of strata from the kernel's answer to a verification."""
    if not answer:
        return answer
    strata, rows = answer
    charts = []
    generators = tuple(_canonical(variables, g, QQ) for g in center)
    for i, (pivot, (ambient, strict)) in enumerate(zip(generators, rows)):
        ratios, coordinates = chart_coordinates(variables, len(center), i)
        charts.append(BlowupChart(variables=coordinates, ambient=kernel.exprs(coordinates, ambient),
                                  strict=kernel.exprs(coordinates, strict), exceptional=((pivot,),), pivot=pivot,
                                  pivot_index=i, ratios=ratios, projection=variables))
    return tuple(charts), strata


def _colon(variables, selected, ideal, domain):
    """(selected):ideal. Multipliers in this ideal make the two ideals locally equal."""
    sub = _gb(variables, selected, domain)
    others = [g for g in ideal if not sub.contains(g)]
    if not others:
        return (chart_expressions(variables).S.One,)
    quotient = colon(selected, others, variables, domain=domain)
    if quotient is not None and not CHECK:
        return quotient
    if np.native_ring(variables):
        raise ArithmeticError("The native ideal quotient is unavailable for this ring.")
    # SymPy's ideal quotient uses exact module syzygies in a polynomial ring.
    ring = domain.old_poly_ring(*variables, order="grevlex")
    reference = tuple(ring.to_sympy(f) for f in ring.ideal(*selected).quotient(ring.ideal(*others)).gens)
    if quotient is not None and _basis(variables, quotient, domain) != _basis(variables, reference, domain):
        raise AssertionError("The native ideal quotient differs from SymPy's.")
    return reference if quotient is None else quotient


def smooth_locus(variables, ideal, expected_dimension, *, domain=QQ):
    """Compute the closed complement of the smooth locus of a fixed dimension.

    For a c-element subset F of the generators of I, local equations F
    generate I on D(q) for q in (F):I. A nonzero c-column Jacobian minor m
    therefore certifies an open D(q*m) of smooth dimension n-c. Conversely,
    at any smooth point of dimension n-c, some c generators form a basis of
    the conormal space and generate the local ideal by Nakayama. Some minor
    and some colon multiplier are units there. These opens are thus exact.
    'bad_ideal' cuts out the complement within V(I). In particular this test
    does not silently assume purity, reducedness, or global complete intersection.
    """
    variables = _ring(variables, domain)
    if not isinstance(expected_dimension, int):
        raise TypeError("Expected dimension must be an integer.")
    return _native(kernel.smooth_locus, _smooth_locus, domain, variables, tuple(ideal), expected_dimension)


def _smooth_locus(variables, ideal, expected_dimension, domain):
    raw = _expressions(variables, ideal, domain)
    generators = _basis(variables, raw, domain)
    n = len(variables)
    codim = n - expected_dimension
    bad = generators

    def result():
        return {"expected_dimension": expected_dimension, "empty": generators == (1,),
                "smooth": bad == (1,), "bad_ideal": bad}

    if bad == (1,) or codim < 0 or codim > n:
        return result()
    if codim == 0:
        if not generators:  # affine space is irreducible; its only open n-dimensional closed subscheme is itself
            bad = (chart_expressions(variables).S.One,)
        return result()
    # A shorter presentation can turn a smoothness check into one determinant.
    equations = raw if len(raw) < len(generators) else generators
    if len(equations) < codim:
        return result()
    from .affine_kernel import frame_minor
    reference = sp.Matrix(equations).jacobian(variables) if _MODE == "sympy" else None
    if domain == QQ or reference is not None:
        lifted = equations
        coordinates = variables
    else:
        from .field_polynomial import lift, lower
        theta = chart_expressions(variables).Dummy("smooth_coefficient")
        coordinates = (*variables, theta)
        lifted = tuple(lift(f, variables, domain, theta) for f in equations)
    for rows in combinations(range(len(equations)), codim):
        selected = tuple(equations[r] for r in rows)
        current = _gb(variables, bad, domain)
        minors = []
        active = set().union(*(g.free_symbols for g in selected))
        for columns in combinations((j for j,v in enumerate(variables) if v in active), codim):
            if reference is not None:
                minor = reference.extract(rows, columns).det()
            else:
                minor = frame_minor(coordinates, tuple(lifted[r] for r in rows), tuple(variables[j] for j in columns))
                if domain != QQ:
                    minor = lower(minor, variables, domain, theta)
            if minor != 0:
                remainder = current.reduce(minor)[1]
                if remainder != 0 and remainder not in minors:
                    minors.append(remainder)
        if not minors:
            continue
        # On V(I) a minor and its remainder modulo the current bad ideal
        # coincide on all points not already covered by earlier open sets.
        multipliers = _colon(variables, selected, equations, domain)
        bad = _basis(variables, (*bad, *(q*m for q in multipliers for m in minors)), domain, known=len(bad))
        if bad == (1,):
            break
    return result()


def verify_snc(variables, ambient, strict, exceptional, ambient_dimension, *, domain=QQ):
    """Check smoothness of the ambient chart and transversality of all given strata.

    Supply separately the ideals for each proposed divisor piece. Disconnected
    smooth pieces are allowed here. Their decomposition into geometric
    components, labels and gluing belong to the dual-complex calculation.
    Pass strict=None to skip it. Empty intersections satisfy transversality.
    """
    variables = _ring(variables, domain)
    return _native(kernel.verify_snc, _verify_snc, domain, variables, tuple(ambient),
                   None if strict is None else tuple(strict), tuple(tuple(ideal) for ideal in exceptional),
                   ambient_dimension)


def _verify_snc(variables, ambient, strict, exceptional, ambient_dimension, domain):
    ambient = _expressions(variables, ambient, domain)
    labels, divisors = [], []
    if strict is not None:
        labels.append("strict")
        divisors.append(_expressions(variables, strict, domain))
    for index, ideal in enumerate(exceptional, 1):
        labels.append(f"E{index}")
        divisors.append(_expressions(variables, ideal, domain))
    strata = []

    def descend(indices, equations, start):
        witness = smooth_locus(variables, equations, ambient_dimension - len(indices), domain=domain)
        item = {"components": [labels[i] for i in indices], **witness}
        strata.append(item)
        # Descendants of an empty intersection are empty by inclusion and
        # need no separate computation. This also handles unit divisors.
        if witness["empty"]:
            return
        for j in range(start, len(divisors)):
            descend((*indices, j), (*equations, *divisors[j]), j+1)

    descend((), ambient, 0)
    failures = [s for s in strata if not s["smooth"]]
    return {"ok": not failures, "strata": strata, "bad_strata": failures}
