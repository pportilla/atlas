"""SymPy term orders matching the native kernel's monomial orders, for reference calculations."""


def reference_order(order):
    if order != "first":
        return order
    from sympy.polys.orderings import ProductOrder, grevlex, lex
    return ProductOrder((lex, lambda monomial: monomial[:1]), (grevlex, lambda monomial: monomial[1:]))


