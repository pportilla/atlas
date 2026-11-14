"""Monodromy on coordinate strata of a binomial hypersurface over C."""
from .monodromy import _request
from .point_geometry import encode_terms

from .polynomial import integer_from_decimal, integer_text, parse


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
    rows = _request("binomial", terms=encode_terms(terms), width=len(variables),
                    orders=list(map(integer_text, missing_orders)))
    return {integer_from_decimal(row["order"]): {
                "model": row["model"], "coordinates": tuple(variables[i] for i in row["axes"]),
                "powers": tuple(map(integer_from_decimal, row["powers"])),
                "cyclotomic_exponent": integer_from_decimal(row["exponent"])} for row in rows}
