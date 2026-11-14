"""Strata of a certified embedded projective ideal blowup, for A'Campo's
formula and for the topological zeta function."""
from collections import Counter
from itertools import combinations
import re

import sympy as sp

from .blowup_ideal import ideal_subset
from .general_resolution import expression
from .nonreduced import _valuation
from .polynomial import InputError, parse_expression


def _field(record):
    field = record.get("coefficient_field", {})
    name = field.get("generator")
    if not name:
        return sp.QQ, {}
    # The Euler calculation uses only field operations and geometric root
    # counts. Any one consistent embedding of this coefficient field suffices.
    x = sp.Symbol("x")
    coefficients = field.get("minimal_polynomial_coefficients")
    if coefficients:
        if not isinstance(coefficients, list) or not all(
                re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", str(c)) for c in coefficients):
            raise InputError("Expected rational coefficients for the graph's number field.")
        polynomial = sp.Poly.from_list([sp.Rational(c) for c in coefficients], x)
        root = sp.CRootOf(polynomial.as_expr(), 0)
    else:
        # The older graph exporter stored the exact SymPy root as text.
        exact = field.get("exact_root", "")
        indexed = re.fullmatch(r"CRootOf\((.+),\s*(\d+)\)", exact)
        radical = re.fullmatch(r"(-?)sqrt\(([-\d/]+)\)", exact)
        if indexed:
            p = parse_expression(indexed[1], ["x"])
            root = sp.CRootOf(expression(p, (x,)), int(indexed[2]))
        elif radical:
            root = (-1 if radical[1] else 1) * sp.sqrt(sp.Rational(radical[2]))
        elif exact in ("I", "-I"):
            root = sp.I if exact == "I" else -sp.I
        else:
            raise InputError("The archived graph needs its coefficient field's minimal polynomial.")
    return sp.QQ.algebraic_field(root), {name: root}


def graph_contributions(record, polynomial):
    if record.get("engine") != "verified-ideal-blowup" or record.get("mode") != "embedded":
        return None
    from .constructible_euler import euler_characteristic

    domain, constants = _field(record)
    orders = {}
    euler = Counter()
    for chart in record["charts"]:
        # A CRootOf coefficient can itself use x as a bound name. Keep
        # geometric coordinates distinct even when the user's input uses x.
        coords = tuple(map(sp.Dummy, chart["coordinates"]))
        bindings = dict(zip(chart["coordinates"], coords))
        if not set(record["variables"]).issubset(bindings):
            raise InputError("A graph chart must retain the original coordinates.")
        base = tuple(bindings[name] for name in record["variables"])
        bindings.update(constants)
        def read(source):
            return expression(parse_expression(source, list(bindings)), tuple(bindings.values()))

        ambient = tuple(map(read, chart["ambient_ideal"]))
        strict = tuple(map(read, chart.get("reduced_strict_transform", chart["strict_transform"])))
        primes = {e["id"]: tuple(map(read, e["ideal"])) for e in chart["exceptional"]}
        # Take the first nonzero homogeneous coordinate as pivot. Setting
        # earlier ratios to zero partitions the fiber into disjoint pieces.
        projective = list(map(read, chart["projective_coordinates"]))
        pivot = projective.index(1)
        zeros = dict.fromkeys((*base, *projective[:pivot]), sp.S.Zero)
        if any(v not in coords for v in zeros):
            raise InputError("Expected coordinate ratios in a standard graph chart.")
        free = tuple(v for v in coords if v not in zeros)
        def fiber(ideal):
            return tuple(g.xreplace(zeros) for g in ideal)

        if not primes and read(chart["pivot_generator"]).subs(dict.fromkeys(base, 0)) != 0:
            # The center misses the origin; locally this is an isomorphism
            # and the certificate proves smoothness of the strict support.
            return [{"component":"strict", "multiplicity":min(map(sum, polynomial)),
                     "euler_characteristic":1}]
        for comp in chart["exceptional"]:
            label = comp["id"]
            if label not in orders:
                weight = comp.get("total_multiplicity")
                if weight is None:
                    f = expression(polynomial, base)
                    if ideal_subset(coords, (f,), ambient, domain=domain):
                        raise InputError("The input pullback vanishes identically on the ambient chart.")
                    if ideal_subset(coords, (1,), (*ambient, *primes[label]), domain=domain):
                        raise InputError("The exceptional ideal is empty on its chart.")
                    weight = _valuation(coords, ambient, primes[label], f, domain)
                orders[label] = weight
            excluded = [fiber(strict)] + [fiber(ideal) for other, ideal in primes.items() if other != label]
            euler[label] += euler_characteristic(free, fiber((*ambient, *primes[label])), excluded,
                                                   domain=domain)
    return [{"component":label, "multiplicity":weight, "euler_characteristic":euler[label]}
            for label, weight in orders.items()]


def _jacobian_order(coords, base, ambient, prime, domain):
    """The order along V(prime) of the Jacobian of the projection of the chart
    W = V(ambient) to the base coordinates.

    With t the other coordinates, the module of relative differentials of W
    over the base is presented by the matrix of the derivatives of the
    ambient equations in t, and also by the Jacobian matrix of the
    projection, a map between smooth varieties of the same dimension. Its
    zeroth Fitting ideal does not depend on the presentation: the maximal
    minors of the first matrix generate the ideal of the Jacobian
    determinant. The order of an ideal along a divisor is the least order of
    its generators.
    """
    extra = [v for v in coords if v not in base]
    if not extra:
        return 0
    rows = [[sp.diff(h, t) for t in extra] for h in ambient]
    best = None
    for chosen in combinations(range(len(rows)), len(extra)):
        minor = sp.expand(sp.Matrix([rows[i] for i in chosen]).det())
        if minor == 0 or ideal_subset(coords, (minor,), ambient, domain=domain):
            continue
        order = _valuation(coords, ambient, prime, minor, domain)
        best = order if best is None else min(best, order)
        if best == 0:
            break
    if best is None:
        raise InputError("The projection of a graph chart is not generically a local isomorphism.")
    return best


def graph_strata(record, polynomial):
    """Numerical data {label: (N, nu)} and strata {labels: chi} over the origin
    of a certified embedded blowup of an ideal, with the method used; None
    when the record is of another kind."""
    if record.get("engine") != "verified-ideal-blowup" or record.get("mode") != "embedded":
        return None
    from .blowup_ideal import empty_locus, saturate
    from .local_blowups import Chart, _strata

    domain, constants = _field(record)
    factors = record.get("input_divisor", {}).get("factors")
    numbers = {}
    charts = []
    for chart in record["charts"]:
        coords = tuple(map(sp.Dummy, chart["coordinates"]))
        bindings = dict(zip(chart["coordinates"], coords))
        if not set(record["variables"]).issubset(bindings):
            raise InputError("A graph chart must retain the original coordinates.")
        base = tuple(bindings[name] for name in record["variables"])
        bindings.update(constants)
        def read(source):
            return expression(parse_expression(source, list(bindings)), tuple(bindings.values()))

        ambient = tuple(map(read, chart["ambient_ideal"]))
        primes = {e["id"]: tuple(map(read, e["ideal"])) for e in chart["exceptional"]}
        projective = list(map(read, chart["projective_coordinates"]))
        earlier = tuple(projective[:projective.index(1)])
        if any(v not in coords for v in earlier):
            raise InputError("Expected coordinate ratios in a standard graph chart.")
        pivot = read(chart["pivot_generator"])
        if not primes and pivot.subs(dict.fromkeys(base, 0)) != 0:
            # The center misses the origin: locally an isomorphism, and the
            # certificate proves smoothness of the support.
            return {"strict": (min(map(sum, polynomial)), 1)}, {frozenset({"strict"}): 1}, \
                "The centre misses the origin, where the support is smooth"
        f = expression(polynomial, base)
        components = []
        for comp in chart["exceptional"]:
            label = comp["id"]
            if label not in numbers:
                weight = comp.get("total_multiplicity")
                if weight is None:
                    if ideal_subset(coords, (f,), ambient, domain=domain):
                        raise InputError("The input pullback vanishes identically on the ambient chart.")
                    if ideal_subset(coords, (1,), (*ambient, *primes[label]), domain=domain):
                        raise InputError("The exceptional ideal is empty on its chart.")
                    weight = _valuation(coords, ambient, primes[label], f, domain)
                numbers[label] = (weight, 1 + _jacobian_order(coords, base, ambient, primes[label], domain))
            components.append((label, *numbers[label], primes[label]))
        if factors and len(factors) > 1:
            # The strict transform of each factor: its pullback with the
            # exceptional divisor of the chart removed.
            names = dict(zip(record["variables"], base))
            for k, factor in enumerate(factors, 1):
                g = expression(parse_expression(factor["equation"], list(names)), tuple(names.values()))
                ideal = saturate(coords, (*ambient, g), pivot, domain=domain)
                if not empty_locus(coords, ideal, domain=domain):
                    components.append((f"S{k}", factor["multiplicity"], 1, tuple(ideal)))
        else:
            strict = tuple(map(read, chart.get("reduced_strict_transform", chart["strict_transform"])))
            components.append(("strict", factors[0]["multiplicity"] if factors else 1, 1, strict))
        charts.append(Chart(coords, ambient, tuple(components), base, earlier, (), ()))
    data, strata = _strata(charts, domain)
    return data, strata, "Euler integration on disjoint pieces of the projective origin fiber"
