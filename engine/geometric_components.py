"""Exact geometric components of a smooth affine stratum over a number field.

The shape calculation localizes in independent projection parameters. Every
candidate returned to the caller is contracted by saturation and its union
is checked against the *original* ideal. This catches components lying over
special parameter values. None means a field extension or more work is needed.
"""
from itertools import combinations, product

import sympy as sp
from sympy import QQ

from .blowup_ideal import empty_locus, ideal_basis, ideal_subset, saturate
from .groebner import groebner


def _absolute_factor(poly, effort):
    """True only for proved absolute irreducibility over the coefficient field."""
    all_variables = poly.gens
    variables = tuple(x for x in all_variables if poly.degree(x) > 0)
    poly = sp.Poly(poly.as_expr(), *variables, domain=poly.domain)
    degree = poly.total_degree()
    if degree == 1:
        return True
    if len(variables) == 1:
        return False
    # Called only on K-irreducible factors, so a polynomial of degree one in x
    # is primitive in x. The content stays a unit over any constant extension.
    if any(poly.degree(x) == 1 for x in variables):
        return True
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


def _factor_equations(poly, effort):
    """Test existence of A*B=f over Kbar by exact coefficient comparison.

    Monic A and fixed leading coefficient of B remove the scalar action.
    Every possible leading monomial split is included. A positive-dimensional
    solution set still gives a factorization, by the weak Nullstellensatz.
    """
    d = poly.total_degree()
    lead, lc = poly.terms(order="grlex")[0]
    coeffs = poly.as_dict()
    bounds = tuple(poly.degree(v) for v in poly.gens)
    max_unknowns = 6 + 2*effort
    max_pivots = 32*effort
    checked = 0
    for alpha in product(*(range(i+1) for i in lead)):
        a_degree = sum(alpha)
        if not 1 <= a_degree <= d//2:
            continue
        checked += 1
        if checked > max_pivots:
            return None
        beta = tuple(e-i for e,i in zip(lead,alpha))
        items = []
        for mon, top, degree in [(alpha,1,a_degree),(beta,lc,d-a_degree)]:
            row = {mon:top}
            for k in _monomials(bounds,degree,mon):
                if len(row) > max_unknowns:
                    return None
                row[k] = sp.Dummy("coefficient")
            items.append(row)
        left, right = items
        unknowns = tuple(left.values())[1:] + tuple(right.values())[1:]
        if len(unknowns)>max_unknowns:
            return None
        equations = {m:-c for m,c in coeffs.items()}
        for ma, ca in left.items():
            for mb, cb in right.items():
                key = tuple(i+j for i,j in zip(ma,mb))
                equations[key] = equations.get(key,0) + ca*cb
        constraints = tuple(expr for expr in equations.values() if expr != 0)
        if not unknowns:
            if not constraints:
                return False
            continue
        # Dummies are distinct, including their printed names, for ring helpers.
        names = tuple(sp.Dummy(f"factor_{i}") for i in range(len(unknowns)))
        replacements = dict(zip(unknowns,names))
        constraints = tuple(sp.sympify(f).xreplace(replacements) for f in constraints)
        if not empty_locus(names,constraints, domain=poly.domain):
            return False
    return True


def _disjoint(variables, primes, domain):
    return all(empty_locus(variables,a+b,domain=domain) for a,b in combinations(primes,2))


def _intersection(variables, a, b, domain):
    selector = sp.Dummy("union_gate")
    basis = groebner(tuple(selector*f for f in a) + tuple((1-selector)*g for g in b),
                     (selector,*variables), order="lex",domain=domain)
    return tuple(p.as_expr() for p in basis.polys if p.degree(selector)==0)


def _coverage(variables, components, ideal, domain):
    if not components or not _disjoint(variables,components,domain):
        return False
    if not all(ideal_subset(variables,ideal,p,domain=domain) for p in components):
        return False
    union = components[0]
    for prime in components[1:]:
        union = _intersection(variables, union, prime, domain)
    return ideal_subset(variables,union,ideal,domain=domain)


def _projections(n, effort):
    """Coordinate permutations then rationally dense elementary linear changes.

    Successive efforts enumerate all finite words in elementary integer shears.
    SL_n(Z) is Zariski dense in SL_n; generic projections and primitive linear
    forms therefore appear in this enumeration. Each effort is finite.
    """
    identity = sp.eye(n)
    yield identity
    if n<=1:
        return
    for k in range(1,n):
        yield identity[:,tuple((i+k)%n for i in range(n))]
    # Separate finite fibers immediately in common examples such as grids of
    # rational points. The exhaustive shear sequence below is still needed.
    primitive = sp.eye(n)
    for i in range(n-1):
        primitive[n-1,i] = 2**(i+1)
    yield primitive
    shears = [(i,j,s) for i in range(n) for j in range(n) if i!=j for s in (1,-1)]
    for code in range(1,effort+1):
        matrix = sp.eye(n)
        while code:
            code, index = divmod(code-1,len(shears))
            i,j,s = shears[index]
            matrix[i,:] += s*matrix[j,:]
        yield matrix


def _shape(variables, generators, dimension, domain, effort, matrix):
    u, z, t = variables[:dimension], variables[dimension:-1], variables[-1]
    coord = sp.Matrix(variables)
    forward = dict(zip(variables,matrix*coord))
    ideal = tuple(f.xreplace(forward) for f in generators)
    fraction_field = domain.frac_field(*u) if u else domain
    gb = sp.groebner(ideal,*z,t, order="lex",domain=fraction_field)
    relations = [p.as_expr() for p in gb.polys if all(p.degree(x)==0 for x in z)]
    if len(relations)!=1 or not relations[0].has(t):
        return None
    h = []
    graph = []
    for x in z:
        lift = gb.reduce(x)[1]
        if any(lift.has(v) for v in z):
            return None
        numerator,denominator = sp.fraction(sp.together(lift))
        if denominator.has(t):
            return None
        graph.append(x*denominator-numerator)
        h.append(denominator)
    numerator = sp.together(relations[0]).as_numer_denom()[0]
    factors = sp.Poly(numerator,*u,t,domain=domain).factor_list()[1]
    primes = []
    for factor,multiplicity in factors:
        if factor.degree(t)==0:
            continue  # content is invertible in K(u)
        if multiplicity!=1 or _absolute_factor(factor,effort) is not True:
            return None
        # On D(product h), the z's are expressed in u,t and the coordinate
        # ring is a localization of a geometrically integral hypersurface.
        # Saturation gives its geometrically prime closure in affine space.
        prime = saturate(variables,(*graph,factor.as_expr()),sp.prod(set(h)),domain=domain)
        primes.append(prime)
    if not _coverage(variables, primes, ideal, domain):
        return None
    reverse = dict(zip(variables, matrix.inv()*coord))
    return [ideal_basis(variables,[f.xreplace(reverse) for f in prime],domain=domain) for prime in primes]


def geometric_components(variables, ideal, dimension, *, domain=QQ, effort=1):
    """List pairwise disjoint geometric prime ideals, [] for empty, or None.

    Designed for smooth equidimensional strata. Returned coordinates and ideals
    are over the supplied field K. A geometric component that is not individually
    definable over K requires a new call over a larger number field. Higher
    integer effort considers more projections and larger exact factor tests.
    """
    variables = tuple(variables)
    if not isinstance(effort,int) or effort<1:
        raise ValueError("effort must be a positive integer")
    generators = ideal_basis(variables,ideal,domain=domain)
    if generators==(1,):
        return []
    if not 0<=dimension<=len(variables):
        return None
    if not generators:
        return [()] if dimension==len(variables) else None
    if dimension==len(variables):
        return None
    # Eliminate globally defined graph coordinates before a projection.
    for f in generators:
        for x in variables:
            coefficient = f.diff(x)
            if coefficient == 0 or any(coefficient.has(v) for v in variables):
                continue
            others = tuple(v for v in variables if v!=x)
            if not others:
                return [generators] if dimension==0 else None
            value = sp.expand(-(f-x*coefficient)/coefficient)
            smaller = geometric_components(others,[g.subs(x,value) for g in generators if g!=f],
                                          dimension,domain=domain,effort=effort)
            if smaller is None:
                return None
            return [ideal_basis(variables,(*prime, f),domain=domain) for prime in smaller]
    if len(generators)==1:
        if dimension!=len(variables)-1:
            return None
        factors = sp.Poly(generators[0],*variables,domain=domain).factor_list()[1]
        if any(m!=1 or _absolute_factor(p,effort) is not True for p,m in factors):
            return None
        candidates = [(p.as_expr(),) for p,_ in factors]
        return candidates if _disjoint(variables,candidates,domain) else None
    for projection in _projections(len(variables),effort):
        found = _shape(variables,generators,dimension,domain,effort,projection)
        if found is not None:
            return found
    return None
