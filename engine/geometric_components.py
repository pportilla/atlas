"""Exact geometric components of a smooth affine stratum over a number field.

The shape calculation localizes in independent projection parameters. Every
candidate returned to the caller is contracted by saturation and its union
is checked against the *original* ideal. This catches components lying over
special parameter values. None means a field extension or more work is needed.
"""
from itertools import combinations, product

from .reference import sp
from .reference import QQ
from .chart_expressions import chart_expressions
from .native_polynomial import native_ring

from . import field_factors, kernel
from .affine_kernel import (_primitive_support, _sparse_expression, _sparse_terms, frame_derivative, rational_factor_rows,
                            sparse_terms)
from .blowup_ideal import empty_locus, ideal_basis, ideal_subset, radical_subset, saturate
from .groebner import CHECK, _MODE, groebner


def _factor_records(variables, f, domain):
    """Keep geometric Q-factors in the native sparse representation, including their certificates."""
    if domain == QQ and _MODE != "sympy":
        return [(_sparse_expression(variables, row), m, (row, primitive))
                for row, m, primitive in rational_factor_rows(variables, f)[1]]
    from .field_polynomial import polynomial
    source = sp.Poly(f, *variables, domain=domain) if domain == QQ else polynomial(f, variables, domain)
    return [(p.as_expr(), m, None) for p, m in field_factors.factors(source)]


def _certify(variables, expression, data, domain, effort):
    if data is None:
        from .field_polynomial import polynomial
        source = (sp.Poly(expression, *variables, domain=domain) if domain == QQ
                  else polynomial(expression, variables, domain))
        return _absolute_factor(source, effort)
    row, primitive = data
    if primitive:
        return True
    terms = _sparse_terms(row)
    used = tuple(i for i in range(len(variables)) if any(e[i] for e, _ in terms))
    coords = tuple(variables[i] for i in used)
    terms = {tuple(e[i] for i in used): c for e, c in terms}
    degree = max(map(sum, terms))
    if degree == 1:
        return True
    if len(used) == 1:
        return False
    if any(max(e[j] for e in terms) == 1 for j in range(len(used))):
        return True
    try:
        answer = _projective_or_plane(coords, expression, terms, degree, QQ, effort)
    except ArithmeticError:
        answer = None
    if answer is None:
        answer = _coefficient_system(coords, terms, degree, QQ, effort)
    if CHECK:
        expected = _absolute_reference(sp.Poly(expression, *coords, domain=QQ), effort)
        if expected is not None and answer != expected:
            raise AssertionError("The sparse geometric irreducibility test differs from its reference.")
    return answer


def _absolute_factor(poly, effort):
    """True only for proved absolute irreducibility over the coefficient field."""
    all_variables = poly.gens
    variables = tuple(x for x in all_variables if poly.degree(x) > 0)
    if variables != all_variables:
        if native_ring(variables):
            from .field_polynomial import polynomial
            poly = polynomial(poly.as_expr(), variables, poly.domain)
        else:
            used = [i for i, x in enumerate(all_variables) if x in variables]
            terms = {tuple(m[i] for i in used): c for m, c in poly.rep.to_dict().items()}
            poly = sp.Poly.from_dict(terms, *variables, domain=poly.domain)
    degree = poly.total_degree()
    if degree == 1:
        return True
    if len(variables) == 1:
        return False
    # Called only on K-irreducible factors, so a polynomial of degree one in x
    # is primitive in x. The content stays a unit over any constant extension.
    if any(poly.degree(x) == 1 for x in variables):
        return True
    if _primitive_vertex_coordinates(poly):
        return True
    if _MODE != "sympy":
        try:
            answer = _native_absolute(poly, effort)
        except ArithmeticError:
            answer = None
        if answer is not None:
            if CHECK:
                expected = _absolute_reference(poly, effort)
                if expected is not None and answer != expected:
                    raise AssertionError("The native irreducibility certificate differs from its reference.")
            return answer
        # Coefficient comparison is valid in every dimension over any finite field extension.
        return _factor_equations(poly, effort)
    return _absolute_reference(poly, effort)


def _primitive_vertex_coordinates(poly):
    """Certify absolute irreducibility of a K-irreducible input from its support.

    The irreducible factors after extension to Qbar are Galois conjugate
    up to scalars, so they have a common Newton polytope P. The Newton
    polytope of their product is the Minkowski sum rP, where r is the
    number of factors. Every coordinate of every vertex is divisible by r.
    Any collection of exposed vertices whose coordinate gcd is 1 suffices.
    """
    return _primitive_support(poly.monoms())


def _native_absolute(poly, effort):
    return _projective_or_plane(poly.gens, poly.as_expr(), poly.as_dict(), poly.total_degree(), poly.domain, effort)


def _projective_or_plane(variables, expression, terms, degree, domain, effort):
    """Gao's factor count on a plane, then projective tests in higher dimension.

    In characteristic zero the bounded logarithmic-differential system has
    one independent solution per absolute factor of a squarefree bivariate
    polynomial (KERNEL.md, “Geometric component counts on coordinate plane
    curves”). Thus a rational irreducible plane factor is geometrically
    irreducible exactly when this count is one.

    A homogeneous quadratic splits into linear forms over the algebraic
    closure exactly when its rank is at most two. Its partial derivatives
    are the linear equations of the Hessian kernel; their ideal therefore
    has as many independent linear generators in reduced basis as its rank.
    For higher degrees any distinct projective factors meet in dimension
    at least zero when the ambient projective space has dimension at least
    two. At their common point the entire gradient vanishes. Smoothness
    therefore certifies absolute irreducibility, and singular inputs defer.
    """
    if len(variables) == 2 and domain == QQ:
        result = kernel.plane_components(variables, (expression,), (), ())
        return None if result is None else result[0] == 1
    hvars = variables if all(sum(e) == degree for e in terms) else (*variables, chart_expressions(variables).Dummy("homogenizer"))
    if degree > 2 and (degree > 3+effort or len(hvars) > 3+effort):
        return None
    # Homogenization extends the exponent vector of each input term.
    if hvars == variables:
        h = expression
    else:
        if native_ring(hvars):
            api = chart_expressions(hvars)
            if domain == QQ:
                h = api.from_terms(hvars, {(*e, degree-sum(e)): c for e,c in terms.items()})
            else:
                # Homogenize only geometric coordinates, keeping the stored
                # rational coefficient rows in the same marked power basis.
                source = api.terms((*variables, domain.parameter), expression)
                h = api.from_terms((*hvars, domain.parameter),
                                   {(*e[:-1], degree-sum(e[:-1]), e[-1]): c for e,c in source.items()})
        else:
            api = chart_expressions(hvars)
            h = api.Add(*(c*api.prod(v**k for v,k in zip(hvars, (*e, degree-sum(e)))) for e,c in terms.items()))
    if domain == QQ:
        gradient = tuple(frame_derivative(hvars, h, {v: 1}) for v in hvars)
    else:
        from .field_polynomial import lift, lower
        theta = chart_expressions(variables).Dummy("projective_field")
        lifted = lift(h, hvars, domain, theta)
        gradient = tuple(lower(frame_derivative((*hvars, theta), lifted, {v:1}), hvars, domain, theta)
                         for v in hvars)
    if degree == 2:
        return len(groebner(gradient, hvars, domain=domain).exprs) > 2
    if len(hvars) < 3:
        return None
    for x in hvars:
        rest = tuple(v for v in hvars if v != x)
        images = tuple(1 if v == x else v for v in hvars)
        affine = field_factors.compose(rest, images, hvars, gradient, domain)
        if not empty_locus(rest, affine, domain=domain):
            return None
    return True


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


def _factor_equations(poly, effort):
    return _coefficient_system(poly.gens, poly.as_dict(), poly.total_degree(), poly.domain, effort)


def _coefficient_system(variables, coeffs, d, domain, effort):
    """Test existence of A*B=f over Kbar by exact coefficient comparison.

    Monic A and fixed leading coefficient of B remove the scalar action.
    Every possible leading monomial split is included. A positive-dimensional
    solution set still gives a factorization, by the weak Nullstellensatz.
    """
    expressions = chart_expressions(variables)
    lead = max(coeffs, key=lambda e: (sum(e), e))
    lc = coeffs[lead]
    bounds = tuple(map(max, zip(*coeffs)))
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
                row[k] = expressions.Dummy("coefficient")
            items.append(row)
        left, right = items
        unknowns = tuple(left.values())[1:] + tuple(right.values())[1:]
        if len(unknowns)>max_unknowns:
            return None
        names = tuple(expressions.Dummy(f"factor_{i}") for i in range(len(unknowns)))
        replacements = dict(zip(unknowns, names))
        if _MODE != "sympy":
            from .affine_kernel import frame_function
            ring = (*variables, *names)
            if domain != QQ:
                from .field_polynomial import lift, modulus
                theta = expressions.Dummy("factor_field")
                ring = (*ring, theta)

            def coefficient(c):
                if c in replacements:
                    return replacements[c]
                return c if domain == QQ else lift(c, (), domain, theta)

            parameters = ring[len(variables):]

            def polynomial(terms):
                if native_ring(ring):
                    return expressions.from_terms(ring, {(*m, *e): q for m,c in terms.items()
                                                  for e,q in expressions.terms(parameters, coefficient(c)).items()})
                return expressions.Add(*(coefficient(c)*expressions.prod(v**e for v,e in zip(variables,k)) for k,c in terms.items()))

            # FLINT multiplies the two unknown factors, then the exponents of
            # x distinguish coefficient equations in the unknowns and theta.
            residual = frame_function(ring, polynomial(left)*polynomial(right) - polynomial(coeffs))
            groups = {}
            for exponents, c in sparse_terms(ring, residual):
                groups.setdefault(exponents[:len(variables)], {})[exponents[len(variables):]] = c
            if native_ring(ring):
                constraints = [expressions.from_terms(parameters, terms) for terms in groups.values()]
            else:
                constraints = [expressions.Add(*(c*expressions.prod(v**k for v,k in zip(parameters,e))
                                                for e,c in terms.items())) for terms in groups.values()]
            if domain != QQ:
                constraints.append(modulus(domain, theta))
            exists = not empty_locus(parameters, constraints) if parameters else not any(constraints)
            if not CHECK:
                if exists:
                    return False
                continue
        equations = {m:-c for m,c in coeffs.items()}
        for ma, ca in left.items():
            for mb, cb in right.items():
                key = tuple(i+j for i,j in zip(ma,mb))
                equations[key] = equations.get(key,0) + ca*cb
        constraints = tuple(sp.sympify(f).xreplace(replacements) for f in equations.values() if f != 0)
        expected = (tuple(sp.groebner(constraints, *names, domain=domain).exprs) != (1,)
                    if names else not constraints)
        if CHECK and exists != expected:
            raise AssertionError("The coefficient equations differ from their independent factor test.")
        if expected:
            return False
    return True


def _disjoint(variables, primes, domain):
    return all(empty_locus(variables,a+b,domain=domain) for a,b in combinations(primes,2))


def _intersection(variables, a, b, domain):
    selector = chart_expressions(variables).Dummy("union_gate")
    basis = groebner(tuple(selector*f for f in a) + tuple((1-selector)*g for g in b),
                     (selector,*variables), order="lex",domain=domain)
    return tuple(g for g in basis.exprs if selector not in g.free_symbols)


def _coverage(variables, components, ideal, domain, reduced_support):
    if not components or not reduced_support and not _disjoint(variables,components,domain):
        return False
    if not all(ideal_subset(variables,ideal,p,domain=domain) for p in components):
        return False
    union = components[0]
    for prime in components[1:]:
        union = _intersection(variables, union, prime, domain)
    contains = radical_subset if reduced_support else ideal_subset
    return contains(variables,union,ideal,domain=domain)


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


def _projection_maps(variables, effort):
    """Permutations, a primitive linear form, then every finite word in integer shears."""
    n = len(variables)
    yield variables, variables
    if n <= 1:
        return
    for k in range(1,n):
        yield variables[-k:] + variables[:-k], variables[k:] + variables[:k]
    form = sum(2**(i+1)*v for i,v in enumerate(variables[:-1]))
    yield (*variables[:-1], variables[-1]+form), (*variables[:-1], variables[-1]-form)

    shears = [(i,j,s) for i in range(n) for j in range(n) if i!=j for s in (1,-1)]
    for code in range(1,effort+1):
        images, back = list(variables), list(variables)
        word = []
        while code:
            code, index = divmod(code-1, len(shears))
            i,j,s = shears[index]
            word.append((i,j,s))
            images[i] += s*images[j]
        # The inverse of a product applies the opposite shears in reverse order.
        for i,j,s in reversed(word):
            back[i] -= s*back[j]
        yield tuple(images), tuple(back)


def _projections(variables, effort):
    """SL_n(Z) is Zariski dense. Increasing effort enumerates generic projections."""
    maps = _projection_maps(variables, effort)
    if _MODE == "native":
        yield from maps
        return
    coords = sp.Matrix(variables)
    reference = ((tuple(matrix*coords), tuple(inverse*coords))
                 for matrix,inverse in _projection_matrices_reference(len(variables), effort))
    if _MODE == "sympy":
        yield from reference
    else:
        for actual, expected in zip(maps, reference, strict=True):
            if actual != expected:
                raise AssertionError("The projection or its inverse differs from the reference linear map.")
            yield actual


def _coordinate_images(variables, projection):
    """Accept explicit matrices from direct callers and coordinate maps from the search."""
    if not hasattr(projection, "todok"):
        return tuple(projection)
    if projection.shape != (len(variables), len(variables)):
        raise ValueError("The projection matrix must be square in the given coordinates.")
    if _MODE == "sympy":
        return tuple(projection * sp.Matrix(variables))
    rows = [[] for _ in variables]
    for (i,j), c in projection.todok().items():
        rows[i].append(c * variables[j])
    images = tuple(chart_expressions(variables).Add(*row) for row in rows)
    if CHECK and images != tuple(projection * sp.Matrix(variables)):
        raise AssertionError("The given projection differs from its reference matrix product.")
    return images


def _shape(variables, generators, dimension, domain, effort, matrix, inverse=None, *, reduced_support=False):
    u, z, t = variables[:dimension], variables[dimension:-1], variables[-1]
    images = _coordinate_images(variables, matrix)
    if inverse is None:
        # A caller can also supply a matrix without its construction.
        from .affine_kernel import inverse_map
        images_back = tuple(a/b for a,b in inverse_map(variables, images, variables))
    else:
        images_back = _coordinate_images(variables, inverse)
    ideal = field_factors.compose(variables, images, variables, generators, domain)
    answer = None
    if _MODE != "sympy":
        data = _rational_shape(ideal, u, z, t, domain)
        if data is not None:
            answer = _shape_components(variables, dimension, domain, effort, images_back, ideal, data, reduced_support)
        if not CHECK:
            return answer
    reference = _fraction_shape(ideal, u, z, t, domain)
    expected = None if reference is None else _shape_components(
        variables, dimension, domain, effort, images_back, ideal, reference, reduced_support)
    if CHECK:
        if expected is not None and (answer is None or set(answer) != set(expected)):
            raise AssertionError("The two elimination methods give different geometric components.")
        return answer
    return expected


def _shape_components(variables, dimension, domain, effort, images, ideal, data, reduced_support):
    u, t = variables[:dimension], variables[-1]
    graph, h, numerator = data
    factors = _factor_records((*u, t), numerator, domain)
    primes = []
    for factor,multiplicity,certificate in factors:
        if t not in factor.free_symbols:
            continue  # content is invertible in K(u)
        if (multiplicity != 1 and not reduced_support) or _certify((*u, t), factor, certificate, domain, effort) is not True:
            return None
        # The graph over D(product h) is a geometrically integral
        # hypersurface there. Saturation contracts its prime to the chart.
        prime = saturate(variables,(*graph,factor),chart_expressions(variables).prod(set(h)),domain=domain)
        primes.append(prime)
    if not _coverage(variables, primes, ideal, domain, reduced_support):
        return None
    return [ideal_basis(variables, field_factors.compose(variables, images, variables, prime, domain),
                        domain=domain) for prime in primes]


def _rational_shape(ideal, u, z, t, domain):
    """Lex elimination over K gives candidate graph equations before localization.
    On D(product h) each h(u)*z_j = g(u,t) has a unique solution. Together
    with a geometric prime of the eliminant this defines an integral graph.
    Coverage of the original ideal is verified independently in _shape."""
    order = (*z, t, *u)
    basis = groebner(ideal, order, order="lex", domain=domain).exprs
    parameters = set(u)
    # A native field coefficient is a reduced polynomial in the marked
    # primitive element. It belongs to K, independently of the projection.
    if native_ring(order) and domain != QQ:
        parameters.add(domain.parameter)
    eliminated = parameters | {t}
    relations = [g for g in basis if g.free_symbols <= eliminated]
    if not relations:
        return None
    eliminant = relations[0]
    if len(relations) > 1:
        # Gauss's lemma: the gcd of the eliminants in Q[u,t] gives their
        # gcd in Q(u)[t] up to a unit in Q(u). Ignore pure-u factors below.
        def factors(g):
            return {p: m for p, m, _ in _factor_records((*u, t), g, domain)}
        shared = factors(eliminant)
        for g in relations[1:]:
            other = factors(g)
            shared = {p: min(m, other[p]) for p, m in shared.items() if p in other}
        eliminant, = field_factors.compose((*u,t), (*u,t), (*u,t), (chart_expressions(order).prod(p**m for p,m in shared.items()),), domain)
    if t not in eliminant.free_symbols:
        return None
    graph, h = [], []
    for x in z:
        def equation(rows):
            for g in rows:
                if x in g.free_symbols and g.free_symbols <= eliminated | {x}:
                    linear = field_factors.linear(order, g, x, domain)
                    if linear and linear[0].free_symbols <= parameters:
                        return g, linear[0]
            return None
        result = equation(basis)
        if result is None and x != z[-1]:
            # Eliminate the other graph coordinates first. If x belongs
            # to Q(u)[t]/(p), clearing denominators gives a relation
            # d(u)*x - h(u,t); a reduced lex basis of this projection
            # must have a leading monomial x*u^a dividing its leading term.
            first = tuple(v for v in z if v != x)
            result = equation(groebner(ideal, (*first, x, t, *u), order="lex", domain=domain).exprs)
        if result is None:
            return None
        graph.append(result[0])
        h.append(result[1])
    return graph, h, eliminant


def _fraction_shape(ideal, u, z, t, domain):
    fraction_field = domain.frac_field(*u) if u else domain
    # Interreduce the input as univariate coefficients in K(u), before
    # Buchberger's algorithm: the same monic rows may be supplied many
    # times by a full basis of the original affine ideal. Raw fractions
    # in K(u) otherwise accumulate uncancelled numerators and denominators.
    # Sequential normal forms replace a generator by a residue modulo its
    # predecessors, which preserves the generated ideal.
    generic = []
    for g in ideal:
        p = sp.Poly(g,*z,t,domain=fraction_field)
        if generic:
            p = sp.reduced(p, generic, *z,t, order="lex", domain=fraction_field)[1]
        if not p.is_zero:
            generic.append(p.monic())
    gb = sp.groebner(generic,*z,t, order="lex",domain=fraction_field)
    relations = [p.as_expr() for p in gb.polys if all(p.degree(x)==0 for x in z)]
    if len(relations)!=1 or not relations[0].has(t):
        return None
    h = []
    graph = []
    for x in z:
        lift = gb.reduce(x)[1].as_expr()
        if any(lift.has(v) for v in z):
            return None
        numerator,denominator = sp.fraction(sp.together(lift))
        if denominator.has(t):
            return None
        graph.append(x*denominator-numerator)
        h.append(denominator)
    numerator = sp.together(relations[0]).as_numer_denom()[0]
    return graph, h, numerator


def geometric_components(variables, ideal, dimension, *, domain=QQ, effort=1, reduced_support=False):
    """List certified geometric prime ideals, [] for empty, or None.

    By default the primes must be disjoint and give the original ideal.
    A reduced-support decomposition allows meeting branches and nilpotents.
    Absolute primality and radical containment of the intersection ideal
    prove that every geometric component is accounted for on the entire chart.

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
            pair = field_factors.linear(variables, f, x, domain)
            if pair is None or pair[0].free_symbols.intersection(variables):
                continue
            others = tuple(v for v in variables if v!=x)
            if not others:
                return [generators] if dimension==0 else None
            if domain == QQ or pair[0].is_Rational:
                value = -pair[1]/pair[0]
            else:
                # On a graph over K the leading coefficient is a field
                # unit. Lex reduction sends x to its unique coordinate
                # expression, inverting the coefficient modulo its relation.
                value = groebner((f,), (x, *others), order="lex", domain=domain).reduce(x)[1]
            remaining = [g for g in generators if g!=f]
            equations = field_factors.compose(others, tuple(value if v==x else v for v in variables),
                                              variables, remaining, domain)
            smaller = geometric_components(others,equations,dimension,domain=domain,effort=effort,
                                           reduced_support=reduced_support)
            if smaller is None:
                return None
            return [ideal_basis(variables,(*prime, f),domain=domain) for prime in smaller]
    if len(generators)==1:
        if dimension!=len(variables)-1:
            return None
        factors = _factor_records(variables, generators[0], domain)
        if any((m != 1 and not reduced_support) or _certify(variables, p, data, domain, effort) is not True
               for p, m, data in factors):
            return None
        candidates = [(p,) for p, _, _ in factors]
        return candidates if reduced_support or _disjoint(variables,candidates,domain) else None
    for projection,inverse in _projections(variables,effort):
        found = _shape(variables,generators,dimension,domain,effort,projection,inverse,reduced_support=reduced_support)
        if found is not None:
            return found
    return None
