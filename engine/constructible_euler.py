"""Exact compactly supported Euler characteristic of a complex affine constructible set.

Project out one coordinate. On strata of the base, polynomial Euclidean
division in that coordinate gives the greatest common divisor of the fiber
equations and its number of distinct roots. Integrating these fiber counts
gives the Euler characteristic from those of the lower-dimensional strata.
"""
from collections import Counter
from functools import cache
from math import prod

import sympy as sp
from sympy import QQ


class _Euler:
    def __init__(self, domain):
        self.domain = domain
        # Keep caches local to the operation and its coefficient field.
        self.condition = cache(self._condition)
        self.closed = cache(self._closed)

    def _condition(self, variables, expression):
        """Canonicalize a zero/nonzero condition, including its multiplicities."""
        if not variables:
            return sp.Integer(bool(self.domain.convert(expression)))
        p = sp.Poly(expression,*variables,domain=self.domain)
        if p.is_zero:
            return sp.S.Zero
        if p.total_degree()==0:
            return sp.S.One
        return p.sqf_part().monic().as_expr()

    def conditions(self, variables, equations):
        return frozenset(self.condition(variables,e) for e in equations)-{sp.S.Zero}

    def open_part(self, variables, zeros, nonzeros):
        if sp.S.One in zeros or not zeros.isdisjoint(nonzeros):
            return 0
        value = self.closed(variables,zeros)
        if not nonzeros:
            return value
        # Simultaneous inequations are the complement of one hypersurface.
        forbidden = self.condition(variables,prod(nonzeros))
        return value-self.closed(variables,zeros|{forbidden})

    def _degrees(self, p, base, zeros, nonzeros):
        """Disjoint cases with certified degree after specialization in the base."""
        while not p.is_zero:
            degree = p.degree()
            c = self.condition(base,p.LC())
            if c not in zeros:
                if c==1 or c in nonzeros:
                    yield p,zeros,nonzeros
                    return
                yield p,zeros,nonzeros|{c}
                zeros = zeros|{c}
            # On this stratum the leading coefficient vanishes identically.
            p = p.slice(0,degree) if degree else p.zero
        yield None,zeros,nonzeros

    def _gcd_cases(self, polynomials, base, zeros, nonzeros):
        """GCD up to invertible scalar, with None denoting the zero polynomial."""
        todo = [(None,tuple(polynomials),zeros,nonzeros)]
        while todo:
            g, pending, z, u = todo.pop()
            if not pending:
                yield g,z,u
                continue
            p, rest = pending[0],pending[1:]
            for h,z1,u1 in self._degrees(p,base,z,u):
                if h is None:
                    todo.append((g,rest,z1,u1))
                elif h.degree()==0:
                    yield h,z1,u1
                elif g is None:
                    todo.append((h,rest,z1,u1))
                else:
                    a,b = (g,h) if g.degree()>=h.degree() else (h,g)
                    # Pseudo-remainder specializes to the ordinary remainder
                    # times a power of lc(b), certified nonzero on this case.
                    remainder = a.prem(b)
                    todo.append((b,(remainder,*rest),z1,u1))

    def _closed(self, variables, equations):
        if sp.S.One in equations:
            return 0
        if not equations:
            return 1  # Every complex affine space, including a point, has χ_c=1.
        used = set().union(*(e.free_symbols for e in equations))
        active = tuple(v for v in variables if v in used)
        if active != variables:
            return self.closed(active,equations)
        polys = [sp.Poly(e,*variables,domain=self.domain)
                 for e in sorted(equations,key=sp.default_sort_key)]
        if len(variables)==1:
            g = polys[0]
            for p in polys[1:]:
                g = g.gcd(p)
            return int(g.sqf_part().degree())

        # Special fibers of graph charts prescribe several original
        # coordinates simultaneously. Remove these in a single substitution.
        fixed = {}
        for p in polys:
            if p.total_degree()!=1:
                continue
            support = p.as_expr().free_symbols.intersection(variables)
            if len(support)==1:
                v, = support
                fixed[v] = -p.as_expr().subs(v,0)/p.diff(v).as_expr()
        if fixed:
            base = tuple(v for v in variables if v not in fixed)
            remaining = (p.as_expr().xreplace(fixed) for p in polys)
            return self.closed(base,self.conditions(base,remaining))

        # A variable with an invertible constant linear coefficient can be
        # eliminated globally. This is frequent on projective graph fibers.
        for p in polys:
            for v in variables:
                if p.degree(v)!=1:
                    continue
                coefficient = p.diff(v)
                if coefficient.total_degree()==0:
                    base = tuple(s for s in variables if s!=v)
                    replacement = -p.as_expr().subs(v,0)/coefficient.as_expr()
                    remaining = (q.as_expr().subs(v,replacement) for q in polys if q!=p)
                    return self.closed(base,self.conditions(base,remaining))

        # A free fiber contributes one; a nonzero polynomial contributes
        # its number of distinct complex roots. Degree drops and repeated
        # roots form separate constructible base strata.
        def projection_cost(v):
            degrees = [p.degree(v) for p in polys]
            return max(degrees),sum(degrees)
        variable = min(variables,key=projection_cost)
        base = tuple(v for v in variables if v!=variable)
        coeffs = self.domain.poly_ring(*base)
        fiber = tuple(sp.Poly(p.as_expr(),variable,domain=coeffs) for p in polys
                      if p.degree(variable)>0)
        base_eqs = self.conditions(base,(p.as_expr() for p in polys if p.degree(variable)==0))

        value = 0
        for g,z,u in self._gcd_cases(fiber,base,base_eqs,frozenset()):
            if g is None or g.degree()==1:
                value += self.open_part(base,z,u)
            elif g.degree()>1:
                for repeated,z1,u1 in self._gcd_cases((g,g.diff()),base,z,u):
                    value += int(g.degree()-repeated.degree())*self.open_part(base,z1,u1)
        return value


def euler_characteristic(variables, equations, excluded=(), *, domain=QQ):
    """Compute χ_c(V(equations) minus the union of V(I) for I in excluded).

    Variables are SymPy symbols. Each excluded ideal is an iterable of
    generators. Coefficients are rational or in the supplied algebraic number
    field. The result concerns complex geometric points and ignores nilpotents.
    """
    variables = tuple(variables)
    # A CRootOf coefficient may bind the same printed symbol as a geometric
    # variable. Fresh generators keep SymPy's polynomial parser from treating
    # the bound symbol in that exact algebraic number as a chart coordinate.
    aliases = {v:sp.Dummy("euler_coordinate") for v in variables} if domain.is_AlgebraicField else {}
    if aliases:
        variables = tuple(aliases.values())
    def symbols(ideal):
        return (sp.sympify(e).as_expr().xreplace(aliases) for e in ideal) if aliases else ideal
    calculator = _Euler(domain)
    closed = calculator.conditions(variables,symbols(equations))
    pieces = Counter({closed:1})
    inequations = set()
    for ideal in excluded:
        forbidden = calculator.conditions(variables,symbols(ideal))
        if not forbidden:
            return 0
        if sp.S.One in forbidden:
            continue
        if len(forbidden)==1:
            inequations.update(forbidden)
            continue
        # Indicator of X minus V(I) equals 1_X − 1_(X ∩ V(I)).
        for locus,weight in tuple(pieces.items()):
            pieces[locus|forbidden] -= weight
        pieces = Counter({locus:w for locus,w in pieces.items() if w})
    return sum(w*calculator.open_part(variables,z,frozenset(inequations))
               for z,w in pieces.items())
