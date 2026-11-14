"""Independent component counts by plane factorization and closed differential forms."""
from engine.reference import sp


def plane_components(variables, equations, region, removed):
    """Factor over Q and solve Gao's closed-form equations symbolically."""
    zero = {v: 0 for v in variables if any(sp.sympify(g).as_coeff_Mul()[1] == v for g in equations)}
    free = [v for v in variables if v not in zero]
    if len(free) != 2:
        return None
    x, y = free

    def restrict(g):
        return sp.Poly(sp.sympify(g).xreplace(zero), x, y, domain=sp.QQ)

    remaining = [g for g in map(restrict, equations) if not g.is_zero]
    if len(remaining) != 1:
        return None
    equation, = remaining
    geometric = rational = 0
    for p, _ in equation.factor_list()[1]:
        def vanishes(ideal):
            return all(restrict(g).rem(p).is_zero for g in ideal)

        if not vanishes(region) or any(vanishes(z) for z in removed):
            continue
        rational += 1
        m, n = p.degree(x), p.degree(y)
        if not m or not n:
            geometric += m + n
            continue
        f = p.as_expr()
        # Column polynomials of f*(g_y-h_x) - g*f_y + h*f_x.
        columns = [f*q.diff(y) - q*f.diff(y) for i in range(m) for j in range(n+1)
                   for q in (x**i*y**j,)]
        columns.extend(q*f.diff(x) - f*q.diff(x) for i in range(m+1) for j in range(n)
                       for q in (x**i*y**j,))
        sparse = [sp.Poly(q, x, y).as_dict() for q in columns]
        powers = sorted(set().union(*sparse))
        matrix = sp.Matrix([[column.get(power, 0) for column in sparse] for power in powers])
        geometric += len(columns) - matrix.to_DM().rank()
    return geometric, rational
