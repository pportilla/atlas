"""Independent symbolic checks of component and projection certificates."""
from itertools import product
import sympy as sp

from engine.blowup_ideal import empty_locus
from engine.geometric_components import _factor_equations


def _absolute_reference(poly, effort):
    variables = poly.gens
    degree = poly.total_degree()
    homogeneous = poly.is_homogeneous
    hvars = variables if homogeneous else (*variables, sp.Dummy("homogenizer"))
    h = poly if homogeneous else poly.homogenize(hvars[-1])
    if degree == 2:
        # A homogeneous quadratic splits into two linear forms over Kbar
        # precisely when its symmetric matrix has rank at most two.
        return sp.hessian(h.as_expr(), hvars).to_DM(domain=poly.domain).rank() > 2
    # A smooth projective hypersurface of dimension >=1 is geometrically
    # integral: distinct positive-degree factors must meet in projective space.
    if len(hvars)>=3 and degree <= 3+effort and len(hvars)<=3+effort:
        gradient = [h.diff(x).as_expr() for x in hvars]
        for x in hvars:
            rest = tuple(t for t in hvars if t!=x)
            if not empty_locus(rest,tuple(g.subs(x,1) for g in gradient),domain=poly.domain):
                break
        else:
            return True
    return _factor_equations(poly, effort)


def _monomials(bounds, degree, leading):
    """Monomials that can occur below a prescribed graded lex leading term."""
    def visit(j, remaining, powers):
        if j == len(bounds):
            if (sum(powers), powers) < (degree, leading):
                yield powers
            return
        for e in range(min(bounds[j],remaining)+1):
            yield from visit(j+1,remaining-e,(*powers,e))
    return visit(0, degree, ())


def _reference_factor_layouts(coeffs, d, effort):
    lead = max(coeffs, key=lambda e: (sum(e), e))
    bounds = tuple(map(max, zip(*coeffs)))
    max_unknowns = 6 + 2*effort
    max_pivots = 32*effort
    checked = 0
    layouts = []
    for alpha in product(*(range(i+1) for i in lead)):
        a_degree = sum(alpha)
        if not 1 <= a_degree <= d//2:
            continue
        checked += 1
        if checked > max_pivots:
            return lead, layouts, False
        beta = tuple(e-i for e,i in zip(lead,alpha))
        items = []
        for mon, degree in [(alpha,a_degree),(beta,d-a_degree)]:
            row = [mon]
            for k in _monomials(bounds,degree,mon):
                if len(row) > max_unknowns:
                    return lead, layouts, False
                row.append(k)
            items.append(row)
        if sum(map(len, items)) - 2 > max_unknowns:
            return lead, layouts, False
        layouts.append(items)
    return lead, layouts, True


def _projection_matrices_reference(n, effort):
    """Independent dense-matrix reference for the ordered projection schedule."""
    identity = sp.eye(n)
    yield identity, identity
    if n<=1:
        return
    for k in range(1,n):
        permutation = identity[:,tuple((i+k)%n for i in range(n))]
        yield permutation, permutation.T
    # Separate finite fibers immediately in common examples such as grids of
    # rational points. The exhaustive shear sequence below is still needed.
    primitive = sp.eye(n)
    for i in range(n-1):
        primitive[n-1,i] = 2**(i+1)
    yield primitive, 2*identity - primitive
    shears = [(i,j,s) for i in range(n) for j in range(n) if i!=j for s in (1,-1)]
    for code in range(1,effort+1):
        matrix, inverse = sp.eye(n), sp.eye(n)
        while code:
            code, index = divmod(code-1,len(shears))
            i,j,s = shears[index]
            matrix[i,:] += s*matrix[j,:]
            # Left multiplication by an elementary shear changes the
            # inverse by the inverse elementary column operation on the right.
            inverse[:,j] -= s*inverse[:,i]
        yield matrix, inverse
