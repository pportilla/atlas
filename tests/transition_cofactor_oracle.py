"""Independent rational cofactor comparison for marked-sheet transitions."""

def check(variables, a, b, reduced):
    from sympy import Dummy, QQ
    from sympy.polys.rings import ring as symbolic_ring
    from engine.affine_kernel import sparse_terms
    reference, *_ = symbolic_ring(tuple(Dummy(v.name) for v in variables), QQ)
    left, right = [reference.from_dict(dict(sparse_terms(variables, f))) for f in (a,b)]
    _, primitive = left.gcd(right).monic().clear_denoms()
    for f, answer in zip((left, right), reduced):
        if f.exquo(primitive) != reference.from_dict(dict(sparse_terms(variables, answer))):
            raise AssertionError("The transition cofactors differ from the rational reference.")
