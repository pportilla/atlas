"""Monodromy on coordinate strata of a binomial hypersurface over C."""
from itertools import combinations
from math import gcd, lcm

from .polynomial import parse


def nearby_binomial_witnesses(polynomial, variables, missing_orders):
    """Return one transverse Milnor fiber certificate per detected root order.

    Set x_i=x_j=0 and all other coordinates nonzero. If only the first
    monomial uses x_i and only the second uses x_j, each coefficient in
    the transverse binomial is an analytic unit in the remaining variables.
    Taking local roots of these units gives X^a + Y^b times a smooth
    parameter space, at points of the stratum arbitrarily close to zero.

    In the Atlas convention the plane curve zeta is
    (1-T^lcm(a,b))^gcd(a,b) / ((1-T^a)(1-T^b)).
    Three zero coordinates can also give X*Y + Z^m. This double suspension
    of Z^m has the same reduced monodromy shifted by two cohomological
    degrees, hence its zeta is 1/(1-T^m).
    """
    if not missing_orders or len(variables) < 3:
        return {}
    terms = parse(polynomial, list(variables), bounded=False)
    if len(terms) != 2:
        return {}
    first, second = sorted(terms, reverse=True)
    left = [(v, a) for v, a, b in zip(variables, first, second) if a and not b]
    right = [(v, b) for v, a, b in zip(variables, first, second) if b and not a]
    witnesses = {}
    for x, a in left:
        for y, b in right:
            period, copies = lcm(a, b), gcd(a, b)
            for q in missing_orders:
                if q <= 1 or q in witnesses or period % q:
                    continue
                exponent = copies - int(a % q == 0) - int(b % q == 0)
                if exponent:
                    witnesses[q] = {"model": "X^a + Y^b", "coordinates": (x, y), "powers": (a, b),
                                    "cyclotomic_exponent": exponent}
    if len(variables) >= 4:
        for product_side, power_side in ((left, right), (right, left)):
            linear = [v for v, a in product_side if a == 1]
            for x, y in combinations(linear, 2):
                for z, m in power_side:
                    for q in missing_orders:
                        if q > 1 and q not in witnesses and m % q == 0:
                            witnesses[q] = {"model": "X*Y + Z^m", "coordinates": (x, y, z),
                                            "powers": (1, 1, m), "cyclotomic_exponent": -1}
    return witnesses
