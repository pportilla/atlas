"""The local topological zeta function of Denef and Loeser at the origin.

For an embedded resolution of f whose total transform has the components
E_i, with f vanishing to order N_i along E_i and the pulled-back volume form
to order nu_i - 1,

    Z_top,0(s) = sum over sets I of chi(E_I^o meet the fibre over 0) * prod_{i in I} 1/(N_i*s + nu_i),

where E_I^o is the set of points lying on the components of I and on no
other. The strict transform is one of the components, with N = nu = 1.
(Denef and Loeser, J. Amer. Math. Soc. 5 (1992).)
"""
import sympy as sp

from .polynomial import InputError, parse


def _acampo(data, strata):
    """The exponents {N: -chi} of A'Campo's formula, from the strata on one component."""
    orders = {row["component"]: row["multiplicity"] for row in data}
    powers = {}
    for row in strata:
        if len(row["components"]) == 1:
            n = orders[row["components"][0]]
            powers[n] = powers.get(n, 0) - row["euler_characteristic"]
    return {n: e for n, e in powers.items() if e}


def topological_zeta(record, *, polynomial=None, monodromy=None):
    """Compute from an Atlas record of an embedded resolution.

    `monodromy` is the monodromy zeta function of the same germ, when known
    (by default the one stored in the record). It is used to mark the poles
    and to check the strata: they must give the same zeta function.
    """
    if not isinstance(record, dict) or "status" not in record:
        raise InputError("Expected an Atlas resolution record.")
    if record["status"] != "resolved":
        return {"status": "unavailable", "reason": "The resolution has not been certified."}
    from .strata import resolution_strata
    found = resolution_strata(record, polynomial)
    if found is None:
        return {"status": "unavailable", "reason":
                "The strata of the total transform over the origin are not determined by this record. "
                "Run atlas resolve with --mode embedded --monodromy."}
    data, strata, method = found
    numbers = {row["component"]: (row["multiplicity"], row["nu"]) for row in data}
    s = sp.Symbol("s")
    # Exact arithmetic of polynomials in s over a common denominator: the
    # product of N*s + nu over all the components.
    linear = {name: sp.Poly(order*s + nu, s, domain=sp.ZZ) for name, (order, nu) in numbers.items()}
    numerator = sp.Poly(0, s, domain=sp.ZZ)
    for row in strata:
        term = sp.Poly(row["euler_characteristic"], s, domain=sp.ZZ)
        for name, factor in linear.items():
            if name not in row["components"]:
                term *= factor
        numerator += term
    denominator = sp.Poly(1, s, domain=sp.ZZ)
    for factor in linear.values():
        denominator *= factor
    numerator, denominator = numerator.cancel(denominator, include=True)
    value = sp.factor(numerator.as_expr()/denominator.as_expr())
    numerator, denominator = sp.fraction(value)
    poles = []
    for root, order in sorted(sp.roots(sp.Poly(denominator, s)).items()):
        candidates = [name for name, (n, nu) in numbers.items() if sp.Rational(-nu, n) == root]
        poles.append({"value": str(root), "order": order, "components": candidates})
    # Two checks of the strata. At s = 0 the sum is the Euler characteristic
    # of the point (the formula for the function 1), and the strata on a
    # single component must give the monodromy zeta function found before.
    if value.subs(s, 0) != 1:
        return {"status": "unavailable", "reason": "The strata of this record fail the identity Z(0) = 1."}
    monodromy = monodromy or record.get("monodromy_zeta") or {}
    if monodromy.get("status") == "computed":
        known = {factor["multiplicity"]: factor["exponent"] for factor in monodromy["factors"]}
        if known != _acampo(data, strata):
            return {"status": "unavailable", "reason":
                    "The strata of this record do not give its monodromy zeta function."}
        # exp(2*pi*i*s0) is a zero or pole of the monodromy zeta function at
        # the origin exactly when the cyclotomic polynomial of its order
        # divides an unbalanced product of the factors 1 - T^N.
        for pole in poles:
            q = sp.Rational(pole["value"]).q
            balance = sum(factor["exponent"] for factor in monodromy["factors"] if factor["multiplicity"] % q == 0)
            pole["in_monodromy_zeta_at_origin"] = balance != 0
    return {"status": "computed", "variable": "s",
            "formula": "sum_I chi(E_I^open intersect pi^(-1)(0)) * prod_{i in I} 1/(N_i*s + nu_i)",
            "expression": str(value).replace("**", "^"),
            "numerator": str(sp.expand(numerator)).replace("**", "^"),
            "denominator": str(sp.factor(denominator)).replace("**", "^"),
            "method": method, "poles": poles, "numerical_data": data, "strata": strata}
