"""Independent rational calculations for comparison of BM coefficient presentations."""
from fractions import Fraction
from math import ceil


def residual(table):
    alpha = tuple(min(row[j]/w for row, w in zip(table.exponents, table.weights)) for j in range(len(table.labels)))
    return (alpha, *(tuple(x-a*w for x, a in zip(row, alpha)) for row, w in zip(table.exponents, table.weights)))


def values(table, strata, bounds):
    values = {Fraction(0)}
    for chosen, orders in zip(strata, bounds):
        for row, w, order in zip(table.exponents, table.weights, orders):
            shift = sum(row[j] for j in chosen)
            values.update((k+shift)/w for k in range(order+1))
    return sorted(values)


def requirements(table, strata, value):
    return [tuple(ceil(value*w-sum(row[j] for j in ids)) for w, row in zip(table.weights, table.exponents))
            for ids in strata]


def jet(frame, f, order):
    found, layer = [f], [f]
    for _ in range(order - 1):
        layer = list(dict.fromkeys(d for g in layer for k in frame.coords
                                   if not frame.vanishes(d := frame.derivative(g, k))))
        found.extend(layer)
        if not layer:
            break
    return tuple(dict.fromkeys(found))


def _divide(frame, f, x):
    """A function q on N with f = x*q on N, up to a function without zeros,
    or None when x does not divide f there.

    N is the part of V(rows) where the product u of the units is invertible,
    and x is the coordinate of a divisor of N. The functions on N are the
    polynomials modulo the rows with powers of u as denominators.

    Whether x divides f: f lies in x*O(N) exactly when f*u^k lies in
    I0 = (rows) + (x) for some k, that is, f lies in I0 : u^infinity. In
    that test u may be replaced by its normal form modulo I0.

    The quotient: let J = (rows) + (t*x - 1), with t compared first. If
    f = x*q on N, write q = Q/u^k; then (f*u^k - x*Q)*u^m lies in (rows) for
    some m, so t*f*u^(k+m) is congruent to Q*u^m modulo J and its normal form
    has no t. Conversely, if the normal form p of t*f*u^K has no t, then
    f*u^K - x*p lies in J and has no t, so it vanishes on every component of
    V(rows) not inside x = 0, hence on N: f = x*(p/u^K) there. So p is found
    by trying K = 0, 1, ...; it differs from the quotient by the unit u^K,
    which changes neither orders nor loci. No basis contains the units,
    which are large polynomials.
    """
    from engine.affine_kernel import frame_function, frame_quotient
    from engine.centres import GiveUp
    if not frame.rows:
        return frame_quotient(frame.variables, f, x)
    first, second = frame.cut(x)
    if second is False:
        return None  # x has no zero on N
    if not first.contains(f) and (second is None or not second.contains(f)):
        return None
    t, basis = frame.over(x)
    g = frame.reduced(f)
    for _ in range(12):
        q = basis.reduce(frame_function((t, *frame.variables), t*g))[1]
        if t not in q.free_symbols:
            return q
        g = frame.reduced(g*frame.unit())
    raise GiveUp("A quotient by the equation of a divisor was not found.")


def remove_coordinate_powers(frame, poly, divisors):
    orders = []
    for v in divisors:
        order = 0
        while (q := _divide(frame, poly, v)) is not None:
            poly, order = q, order + 1
        orders.append(order)
    return orders, poly
