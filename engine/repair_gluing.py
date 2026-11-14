"""Recover affine opens over conjugate repair centres from their maps to the parent.

After base change to a splitting field, a repair chart supplies one birational
chart for each root. Each such open must exclude the preimages of every
other centre in its parent. The images to exclude are found by eliminating
the child coordinates on its owned fibre, including its open conditions.
Only finite, disjoint closed images with union equal to the deleted parent
centres are used. Thus closing the graph of a transition cannot identify a
child over one centre with the unmodified copy of a different centre.
"""
from dataclasses import replace
from functools import lru_cache

from .affine_kernel import decode, pullbacks
from .atlas_finite import _extend_chart, _field_mark, _integer_polynomial
from .blowup_ideal import empty_locus, radical_subset
from .chart_expressions import chart_expressions
from .groebner import groebner
from .local_blowups import _covers, _dimension, _present
from .marked_field import common_fields, flint_literal


@lru_cache(maxsize=256)
def _birational_parameters(chart, mapping):
    """Certify generic injectivity on an affine space over its chosen field.

    The parameter is copied as a base coordinate before inversion. Specializing
    an inverse from Q(theta) to the marked field can kill its denominators.
    Pull every fraction back modulo the irreducible relation: a nonzero
    denominator times each coordinate must equal the numerator identically.
    Their common nonvanishing locus is then dense on this integral field sheet.
    """
    from .affine_kernel import inverse_map
    variables = chart.variables
    field = chart.coefficient_field
    modulus = field[1] if field else None
    if tuple(chart.ambient) != ((modulus,) if field else ()):
        return None
    mapping = tuple(mapping) + ((field[0],) if field else ())
    if len(mapping) != len(variables) or field and field[0] != variables[-1]:
        return None
    api = chart_expressions(variables)
    base = api.symbols(f"atlasGeneric0:{len(variables)}")
    try:
        inverse = inverse_map(variables, mapping, base)
        fractions = pullbacks(variables, mapping, base, (g for pair in inverse for g in pair), modulus=modulus)
    except ArithmeticError:
        return None
    for coordinate, numerator, denominator in zip(variables, fractions[::2], fractions[1::2]):
        if not denominator:
            return None
        residual, = pullbacks(variables, variables, variables, (numerator-coordinate*denominator,), modulus=modulus)
        if residual != 0:
            return None
    return base, inverse


@lru_cache(maxsize=4096)
def _reduced_fraction(chart, a, b):
    """Cancel in the field polynomial ring. Its UFD then gives the regular locus."""
    from .affine_kernel import _ring, _request, _sparse_expression
    from .atlas_finite import _power_domain
    from .field_factors import factors
    from .field_polynomial import lift, polynomial
    from .native_polynomial import native_ring
    from .groebner import _MODE
    from .reference import sp
    if not b:
        raise ArithmeticError("A transition denominator vanished on the marked sheet.")
    if not a:
        return a, chart_expressions(chart.variables).S.One
    field = chart.coefficient_field
    t = field[0] if field else None
    coords = tuple(v for v in chart.variables if v != t)
    if not b.free_symbols.intersection(coords):
        return a,b
    if field:
        domain, generator = _power_domain(*field[:3])
        def into(f):
            return f if native_ring(chart.variables) else f.xreplace({t: generator})
        def out(f):
            return f if native_ring(chart.variables) else lift(f,coords,domain,t)
        a,b = into(a),into(b)
        for factor, multiplicity in factors(polynomial(b,coords,domain)):
            basis = groebner((factor.as_expr(),),coords,order="lex",domain=domain)
            for _ in range(multiplicity):
                qa, ra = basis.reduce(a)
                if ra != 0:
                    break
                qb, rb = basis.reduce(b)
                if rb != 0:
                    raise ArithmeticError("A denominator factor did not divide in its marked field.")
                a,b = qa[0],qb[0]
        return out(a),out(b)
    if _MODE == "sympy":
        return sp.cancel(a/b).as_numer_denom()
    ring = _ring(coords)
    header = [ring.header, "lex"]
    rows = _request(["primitive_gcd_cofactors", *header, ring.text(a), ring.text(b)])
    reduced = tuple(_sparse_expression(coords, row) for row in rows)
    if _MODE == "check":
        from tests.transition_cofactor_oracle import check
        check(coords, a, b, reduced)
    return reduced


def _transition_substitution(source, coordinates, fractions, coefficient=None):
    """Evaluate on a regular open, clearing powers of its nonzero denominators."""
    from .affine_kernel import frame_restrict
    variables = source.variables
    modulus = source.coefficient_field[1] if source.coefficient_field else None
    api = chart_expressions(variables)
    aliases = {v:api.Dummy("transition_axis") for v in coordinates}
    if coefficient is not None:
        aliases[coefficient] = source.coefficient_field[0]
    substitutions = [(aliases[v], a, b) for v,(a,b) in zip(coordinates,fractions)]
    @lru_cache(maxsize=1024)
    def transport(g, basis=None):
        f = g.xreplace(aliases)
        ring = (*variables, *(v for v,_,_ in substitutions))
        for v,a,b in substitutions:
            if v in f.free_symbols:
                if basis is not None:
                    a,b = basis.reduce(a)[1],basis.reduce(b)[1]
                f = frame_restrict(ring, f, v, b, -a)
            ring = tuple(x for x in ring if x != v)
        if basis is not None:
            return basis.reduce(f)[1]
        if modulus is not None:
            f, = pullbacks(variables,variables,variables,(f,),modulus=modulus)
        return f
    return transport


def transition_on_field_sheet(source, target, source_map, target_map, *, target_parent=None):
    """Give the regular overlap and transport target equations to the source.

    In affine space over the common marked field, a coprime coordinate
    fraction is regular precisely off its denominator. The transition and
    its rational inverse agree with the parent maps at the generic point.
    The overlap also requires the inverse transition to be regular. On that
    mutual locus their compositions give the identity by separatedness.
    """
    target_inverse = _birational_parameters(target,target_map)
    if not target_inverse or _birational_parameters(source,source_map) is None:
        return None
    base,inverse = target_inverse
    field = source.coefficient_field
    if field != target.coefficient_field:
        return None
    modulus = field[1] if field else None
    theta = field[0] if field else None
    if target_parent is None:
        mapping = tuple(source_map) + ((theta,) if field else ())
        specialized = pullbacks(source.variables,mapping,base,(g for pair in inverse for g in pair),modulus=modulus)
    else:
        # Source and target enter the same original base through different
        # affine parents. Compose the *two low-degree inverses* before pulling
        # back, without eliminating the high-degree original map of the child.
        parent_images = tuple(g for _,g in target_parent.coordinates)
        parent_inverse = _birational_parameters(target_parent,parent_images)
        if parent_inverse is None:
            return None
        outer, back = parent_inverse
        source_images = dict(source.coordinates)
        if source_images.keys() != dict(target_parent.coordinates).keys():
            return None
        maps = tuple(source_images[name] for name,_ in target_parent.coordinates) + ((theta,) if field else ())
        mid = pullbacks(source.variables,maps,outer,(g for pair in back for g in pair),modulus=modulus)
        if any(not d for d in mid[1::2]):
            return None
        mid = tuple(_reduced_fraction(source,n,d) for n,d in zip(mid[::2],mid[1::2]))
        # After substitution an inverse fraction is a fraction of fractions.
        # Clear exactly the SAME powers on its numerator and denominator.
        from .affine_kernel import _polynomial, sparse_terms
        from .kernel_json import kernel_json
        from .point_geometry import decode_terms, encode_terms
        api = chart_expressions(source.variables)
        homogenizers = tuple(api.Dummy("transition_den") for _ in base)
        all_coords = (*base, *homogenizers)
        images = tuple(n for n, _ in mid) + tuple(d for _, d in mid)
        def lift_pair(a,b):
            rows = kernel_json("atlas_component_geometry", "homogenize_fraction", width=len(base),
                               pair=[encode_terms(dict(sparse_terms(base, g))) for g in (a, b)])
            homogenized = [_polynomial(decode_terms(row), all_coords) for row in rows]
            return pullbacks(source.variables, images, all_coords, homogenized, modulus=modulus)
        specialized = tuple(g for a,b in inverse for g in lift_pair(a,b))
    if any(not b for b in specialized[1::2]):
        return None
    fractions = tuple(_reduced_fraction(source,a,b) for a,b in zip(specialized[::2],specialized[1::2]))
    coordinates = tuple(v for v in target.variables if v != theta)
    transfer = _transition_substitution(source,coordinates,fractions,theta)
    source_axes = set(source.variables) - {theta}
    holes = tuple(dict.fromkeys((d,) for _,d in fractions if d.free_symbols & source_axes))
    return transfer,(*source.removed,*holes)


def _point_images(chart, parent, mapping):
    from .constructible_euler import euler_characteristic
    theta = chart.coefficient_field[0]
    api = chart_expressions(chart.variables)
    variables = tuple(v for v in chart.variables if v != theta)
    targets = tuple(api.Dummy("parent_axis") for _ in parent.variables[:-1])
    coords = (*variables, *targets, theta)
    graph = (*chart.ambient, *chart.region, *chart.fibre, *(v-g for v,g in zip(targets,mapping)))
    parent_axes = dict(zip(targets, parent.variables))
    images = []
    for unit in _covers(chart.variables, chart.removed):
        inverse = api.Dummy("open_unit")
        basis = groebner((*graph, 1-inverse*unit), (inverse,*coords), order="lex")
        if basis.exprs == [1]:
            continue
        ideal = tuple(g.xreplace(parent_axes) for g in basis.exprs if g.free_symbols <= {*targets,theta})
        # A constructible set whose closure is zero-dimensional is closed.
        if (_dimension(parent.variables, ideal) != 0
                or parent.one_sheet(euler_characteristic(parent.variables, ideal)) != 1):
            return None
        if ideal not in images:
            images.append(ideal)
    return images


def repair_charts(charts):
    links = [(i,c.repair_projection) for i,c in enumerate(charts) if c.repair_projection]
    if not links:
        return charts
    covers = {}
    marks = []
    for i,(j,mapping) in links:
        child = charts[i]
        if (not 0 <= j < len(charts) or charts[j].repair_projection or charts[j].coefficient_field
                or charts[j].ambient or not charts[j].removed or child.coefficient_field
                or len(mapping) != len(charts[j].variables)):
            return None
        choices = [(v, f, coeff) for f in child.ambient if len(f.free_symbols) == 1
                   for v in f.free_symbols for coeff in _integer_polynomial(v,f)
                   if len(coeff) > 2]
        if not choices:
            continue
        if len(choices) != 1:
            return None
        v,f,coeff = choices[0]
        from .atlas import _field_relation
        if not _field_relation(v,f,len(coeff)-1):
            return None
        roots = tuple(flint_literal(coeff,k) for k in range(1,len(coeff)))
        marks.extend(roots)
        covers[i] = v,f,roots
    if not covers:
        return [replace(c,repair_projection=()) for c in charts]
    existing = {c.coefficient_field:_field_mark(c.coefficient_field) for c in charts if c.coefficient_field}
    if any(mark is None for mark in existing.values()):
        return None
    marks = tuple(dict.fromkeys([*marks,*existing.values()]))
    api = chart_expressions(charts[0].variables)
    theta = api.Dummy("repair_field")
    mark, text, degree, images = common_fields(marks,"T")
    T = api.Symbol("T")
    def read(word):
        return decode((T,),word).xreplace({T:theta})
    field = theta,read(text),degree,mark
    values = {s:read(word) for s,word in zip(marks,images)}
    ordinary = {i:_extend_chart(c,field,values.get(existing.get(c.coefficient_field)))
                for i,c in enumerate(charts) if not c.repair_projection}
    sheets = []
    for i,(j,mapping) in links:
        v,relation,roots = covers.get(i,(None,None,(None,)))
        child = charts[i]
        parent = ordinary[j]
        conjugates = []
        for root in roots:
            cover = replace(child, coefficient_field=(v,relation,len(roots),root) if root else (),
                            repair_projection=())
            sheet = _extend_chart(cover,field,values.get(root))
            coords = sheet.variables
            substitution = tuple(values[root] if x == v else x for x in child.variables)
            parent_map = pullbacks(coords,substitution,child.variables,mapping,modulus=field[1])
            parent_names, maps = zip(*parent.coordinates)
            sheet_images = dict(sheet.coordinates)
            if sheet_images.keys() != set(parent_names) or len(sheet_images) != len(sheet.coordinates):
                return None
            def restrict(polys):
                return pullbacks(coords,(*parent_map,theta),parent.variables,polys,modulus=field[1])
            if not radical_subset(coords, [p-sheet_images[name] for name,p in zip(parent_names,restrict(maps))],
                                  sheet.ambient):
                return None
            centres = _point_images(sheet,parent,parent_map)
            if centres is None or len(centres) != 1:
                return None
            centre, = centres
            if any(not empty_locus(parent.variables,(*centre,*other)) for other in conjugates):
                return None  # two copies of the same centre would double-count its owned fibre
            conjugates.append(centre)
            sheets.append((replace(sheet,repair_projection=(j,parent_map)),j,parent_map,centre))

    # All images of a parent are finite. Two owned pieces over the same
    # centre may report its ideal more than once, in different generators.
    by_parent = {j:[] for _,(j,_) in links}
    for _,j,_,ideal in sheets:
        parent = ordinary[j]
        found = by_parent[j]
        if _present(parent.variables,ideal,parent.removed):
            return None
        for other in found:
            if empty_locus(parent.variables,(*ideal,*other)):
                continue
            if not (radical_subset(parent.variables,ideal,other) and radical_subset(parent.variables,other,ideal)):
                return None
            break
        else:
            found.append(ideal)
    for j,centres in by_parent.items():
        parent = ordinary[j]
        if any(_present(parent.variables,(*parent.ambient,*hole),centres) for hole in parent.removed):
            return None
    # A fan records a centre in its first owning chart. Blowup modifies
    # every open containing that point, including all the other parent
    # opens in the star. Pull the verified finite sets to those opens.
    ordinary = {i:replace(c,repair_projection=(i,c.variables[:-1])) for i,c in ordinary.items()}
    transitions = parent_transitions(ordinary.values())
    from .blowup_ideal import ideal_basis, saturate
    additions = {}
    for i, target in ordinary.items():
        additions[i] = []
        for j,centres in by_parent.items():
            if i == j:
                continue
            source = ordinary[j]
            pairs = _parent_pullback(target, source, transitions)
            inverse = _parent_pullback(source, target, transitions)
            if pairs is None or inverse is None:
                return None
            move, poles = pairs
            _, inverse_poles = inverse
            units = api.Mul(*poles)
            for ideal in centres:
                if any(not empty_locus(source.variables, (*ideal, d)) for d in inverse_poles):
                    continue
                equations = (*target.ambient, *(move(g) for g in ideal))
                closed = (saturate(target.variables,equations,units) if units != 1
                          else ideal_basis(target.variables,equations))
                if closed != (1,):
                    # Distinct source repairs must own disjoint parent centres.
                    if _dimension(target.variables,closed) != 0 or any(
                            not empty_locus(target.variables,(*closed,*old))
                            for old in (*target.removed,*additions[i])):
                        return None
                    additions[i].append(closed)
    result = []
    for sheet,j,mapping,centre in sheets:
        parent = ordinary[j]
        other = [f for f in by_parent[j] if empty_locus(parent.variables,(*f,*centre))] + additions[j]
        holes = tuple(pullbacks(sheet.variables,(*mapping,theta),parent.variables,ideal,modulus=field[1])
                      for ideal in other)
        result.append(replace(sheet,removed=(*sheet.removed,*holes)))
    return [*(replace(c,removed=(*c.removed,*additions[i])) for i,c in ordinary.items()),*result]


def _parent_pullback(source, target, toric_transition):
    """Pull target equations back on the regular open of the parent transition.

    Use the parents before deleting their repaired points. A centre is visible
    in both opens precisely where the map and its inverse are regular.
    """
    if source.coefficient_field != target.coefficient_field:
        return None
    pairs = toric_transition(source, target)
    if pairs is not None:
        field_parameter = source.coefficient_field[0]
        axes = tuple(v for _,v,_ in pairs)
        fractions = tuple((n,d) for n,_,d in pairs)
        move = _transition_substitution(source, axes, fractions, field_parameter)
        return move, tuple(d for _,d in fractions if d != 1)

    # A saved affine chart may have only its map to the original base.
    # Invert its peer on the common marked field and specialize to this map.
    # The coprime coordinate fractions determine its full regular locus.
    source_map = dict(source.coordinates)
    if source_map.keys() != dict(target.coordinates).keys():
        return None
    left = tuple(source_map[name] for name,_ in target.coordinates)
    right = tuple(g for _,g in target.coordinates)
    transition = transition_on_field_sheet(replace(source, removed=()), replace(target, removed=()), left, right)
    if transition is None:
        return None
    move, holes = transition
    return move, tuple(denominator for (denominator,) in holes)


def parent_transitions(charts):
    """Pull the regular toric parent diagonals back to pairs of repair charts.

    If u and v are unimodular monomial coordinates on a common torus, each
    v_j is a Laurent monomial in u. Its negative powers are units on the
    common chart open. Clearing their denominators gives the coordinate
    equalities there. The child overlap still needs saturation away from
    its exceptional divisors and restriction to both retained opens.
    """
    from functools import cache
    from .affine_kernel import _polynomial, sparse_terms
    from .binomial_kernel import signed_powers
    from .fan_geometry import _determinant, lattice_characters
    parents = {c.repair_projection[0]:c for c in charts if c.repair_projection and c.torus
               and c.repair_projection[1] == c.variables[:-1]}
    @cache
    def certified(index):
        c = parents.get(index)
        if c is None:
            return False
        rays, linear = c.torus
        variables = c.variables[:-1]
        if (len(rays) != len(variables) or len(linear) != len(c.coordinates)
                or len(linear) != len(rays) or any(len(row) != len(rays) for row in (*rays,*linear))
                or any(k < 0 for row in rays for k in row) or not _determinant(linear)):
            return False
        try:
            lattice_characters(rays,tuple(tuple(int(i==j) for i in range(len(rays))) for j in range(len(rays))))
        except (ArithmeticError, ValueError):
            return False
        # On this field sheet, reduce the supplied maps modulo its modulus.
        # In a unimodular chart each matrix column names a distinct monomial;
        # the appended zero is its exponent in the field parameter.
        columns = tuple((*column, 0) for column in zip(*rays))
        actual = pullbacks(c.variables, c.variables, c.variables, (g for _, g in c.coordinates),
                           modulus=c.coefficient_field[1])
        for row, image in zip(linear, actual):
            expected = {power: coefficient for power, coefficient in zip(columns, row) if coefficient}
            if expected != dict(sparse_terms(c.variables, image)):
                return False
        return True

    @cache
    def powers(i,j):
        if (not certified(i) or not certified(j) or parents[i].torus[1] != parents[j].torus[1]
                or tuple(dict(parents[i].coordinates)) != tuple(dict(parents[j].coordinates))):
            return None
        # A common character a has local exponents R*a. For target v_j
        # solve R_target*a=e_j, then apply the source matrix.
        left, right = parents[i].torus[0], parents[j].torus[0]
        unit = tuple(tuple(int(a==b) for a in range(len(right))) for b in range(len(right)))
        characters = lattice_characters(right,unit)
        from .newton_backend import _valuations
        return tuple(zip(*_valuations(left, characters)))

    def equations(a,b):
        i, source = a.repair_projection
        j, target = b.repair_projection
        exponents = powers(i,j)
        if exponents is None:
            return None
        base = parents[i].variables[:-1]
        monomials = (_polynomial({powers: 1}, base) for e in exponents for powers in signed_powers(e))
        factors = iter(pullbacks(a.variables, source, base, monomials))
        return [(next(factors), t, next(factors)) for t in target]
    return equations
