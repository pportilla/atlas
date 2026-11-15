"""Independent symbolic reduction of the Denef–Loeser sum."""

def reduce(numbers, strata):
    """Independent polynomial reduction and factorization for comparison."""
    import sympy as sp
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
    if value.subs(s, 0) != 1:
        return None
    return (str(value).replace("**", "^"), str(sp.expand(numerator)).replace("**", "^"),
            str(sp.factor(denominator)).replace("**", "^"), poles)


