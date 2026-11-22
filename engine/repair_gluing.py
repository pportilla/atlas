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
from .atlas_finite import _extend_chart, _field_mark
from .blowup_ideal import empty_locus, radical_subset
from .chart_expressions import chart_expressions
from .groebner import groebner
from .local_blowups import _covers, _dimension, _present
from .marked_field import common_fields, flint_literal


def _marked_operation(op, chart, **arguments):
    from .chart_gluing_kernel import rows
    from .kernel_json import kernel_json
    field = chart.coefficient_field
    axes = chart.variables
    return kernel_json("atlas_marked_chart",op,width=len(axes),theta=axes.index(field[0]) if field else -1,
                       relation=rows(axes,(field[1],))[0] if field else None,**arguments)


@lru_cache(maxsize=256)
def _birational_parameters(chart, mapping):
    """Recover and verify the rational coordinate inverse on the marked sheet."""
    from .chart_gluing_kernel import rows,values
    axes = chart.variables
    result = _marked_operation("faithful_inverse",chart,mapping=rows(axes,tuple(mapping)),ambient=rows(axes,chart.ambient))
    if result is None:
        return None
    base = chart_expressions(axes).symbols(f"atlasGeneric0:{len(axes)}")
    return base, tuple(values(base,pair) for pair in result)


@lru_cache(maxsize=4096)
def _reduced_fraction(chart, a, b):
    """Cancel the geometric gcd on the marked sheet."""
    from .chart_gluing_kernel import rows,values
    from .polynomial import InputError
    try:
        result = _marked_operation("cofactors",chart,pair=rows(chart.variables,(a,b)))
    except InputError as exc:
        raise ArithmeticError(str(exc)) from exc
    return values(chart.variables,result)


def _transition_substitution(source, coordinates, fractions, coefficient=None):
    """Expose exact polynomial transport and the chart fractions used by native gluing."""
    from .chart_gluing_kernel import rows,values
    axes=source.variables
    mapping=dict(zip(coordinates,fractions))
    if coefficient is not None:
        mapping[coefficient]=(source.coefficient_field[0],chart_expressions(axes).S.One)
    source_map=[rows(axes,pair) for pair in mapping.values()]
    target_axes=tuple(mapping)

    @lru_cache(maxsize=1024)
    def transport(g, basis=None):
        equation, = rows(target_axes,(g,))
        prime = rows(axes,tuple(basis.exprs)) if basis is not None else None
        result = _marked_operation("transfer",source,map=source_map,equation=equation,prime=prime)
        return values(axes,result)[0]

    transport.coordinate_fractions=mapping
    return transport


def transition_on_field_sheet(source, target, source_map, target_map, *, target_parent=None):
    """Specialize two certified ambient inverses and transport their regular open."""
    from .chart_gluing_kernel import rows,values
    target_inverse=_birational_parameters(target,target_map)
    if target_inverse is None or _birational_parameters(source,source_map) is None:
        return None
    if source.coefficient_field != target.coefficient_field:
        return None
    base,inverse=target_inverse
    mapping=tuple(source_map)
    parent=None
    if target_parent is not None:
        parent_map=dict(target_parent.coordinates)
        source_coordinates=dict(source.coordinates)
        if source_coordinates.keys()!=parent_map.keys():
            return None
        parent_inverse=_birational_parameters(target_parent,tuple(parent_map.values()))
        if parent_inverse is None:
            return None
        outer,back=parent_inverse
        parent=[rows(outer,pair) for pair in back]
        mapping=tuple(source_coordinates[name] for name in parent_map)
    axes=source.variables
    result=_marked_operation("transition",source,inverse=[rows(base,pair) for pair in inverse],parent=parent,
                             mapping=rows(axes,mapping))
    if result is None:
        return None
    theta = source.coefficient_field[0] if source.coefficient_field else None
    coordinates = tuple(v for v in target.variables if v != theta)
    transfer=_transition_substitution(source,coordinates,[values(axes,pair) for pair in result["fractions"]],theta)
    return transfer,(*source.removed,*((f,) for f in values(axes,result["holes"])))


def _repair_geometry(name, **arguments):
    from .kernel_json import kernel_json
    return kernel_json("atlas_repair_geometry", name, **arguments)


def _point_images(chart, parent, mapping):
    from .constructible_euler import euler_characteristic
    from .newton_repair import supported_rows
    from .polynomial import integer_text
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
        ring = (inverse, *coords)
        columns = [ring.index(x) for x in (*targets, theta)]
        ideal = tuple(g.xreplace(parent_axes) for g in supported_rows(ring, basis.exprs, columns))
        # A constructible set whose closure is zero-dimensional is closed.
        dimension = _dimension(parent.variables, ideal)
        count = parent.one_sheet(euler_characteristic(parent.variables, ideal)) if dimension == 0 else 0
        if not _repair_geometry("point", dimension=dimension, count=integer_text(count)):
            return None
        if ideal not in images:
            images.append(ideal)
    return images


def repair_charts(charts, *, order_components=False):
    links = [(i,c.repair_projection) for i,c in enumerate(charts) if c.repair_projection]
    if not links:
        return charts
    covers = {}
    from .centre_weights import centre_rows
    from .polynomial import integer_from_decimal
    relation_rows = _repair_geometry("relations", charts=[
        dict(parent=c.repair_projection[0] if c.repair_projection else None, marked=bool(c.coefficient_field),
             width=len(c.variables), mapping=len(c.repair_projection[1]) if c.repair_projection else None,
             ambient=centre_rows(c.variables,c.ambient), removed=bool(c.removed)) for c in charts])
    if relation_rows is None:
        return None
    for item in relation_rows:
        i = item["sheet"]
        c = charts[i]
        coeff = tuple(map(integer_from_decimal,item["coefficients"]))
        roots = tuple(flint_literal(coeff,k) for k in range(1,len(coeff)))
        covers[i] = c.variables[item["axis"]],c.ambient[item["relation"]],roots
    if not covers:
        return [replace(c,repair_projection=()) for c in charts]
    order = range(len(charts))
    if order_components:
        from .chart_gluing_kernel import chart_input
        order = _repair_geometry("chart_order", charts=[chart_input(c) for c in charts])
        links = [(i,charts[i].repair_projection) for i in order if charts[i].repair_projection]
    existing = {field:_field_mark(field) for i in order if (field := charts[i].coefficient_field)}
    if any(mark is None for mark in existing.values()):
        return None
    roots = [root for i in order if i in covers for root in covers[i][2]]
    marks = tuple(dict.fromkeys([*roots,*existing.values()]))
    api = chart_expressions(charts[0].variables)
    theta = api.Dummy("repair_field")
    mark, text, degree, images = common_fields(marks,"T")
    T = api.Symbol("T")
    def read(word):
        return decode((T,),word).xreplace({T:theta})
    field = theta,read(text),degree,mark
    values = {s:read(word) for s,word in zip(marks,images)}
    ordinary = {i:_extend_chart(charts[i],field,values.get(existing.get(charts[i].coefficient_field)))
                for i in order if not charts[i].repair_projection}
    sheets = []
    for i,(j,mapping) in links:
        v,relation,roots = covers.get(i,(None,None,(None,)))
        child = charts[i]
        parent = ordinary[j]
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
            sheets.append((replace(sheet,repair_projection=(j,parent_map)), j, parent_map, centre, i))

    ordinary = {i:replace(c,repair_projection=(i,c.variables[:-1])) for i,c in ordinary.items()}
    transitions = parent_transitions(ordinary.values())
    from .blowup_ideal import ideal_basis, saturate
    from .chart_gluing_kernel import operation
    transport = {}
    closed = []

    def query(task):
        action = task["query"]
        if action in ("disjoint", "same", "outside", "covers"):
            parent = ordinary[task["parent"]]
            axes = parent.variables
            if action == "covers":
                centres = [sheets[k][3] for k in task["centres"]]
                return all(not _present(axes, (*parent.ambient,*hole), centres) for hole in parent.removed)
            ideal = sheets[task["sheet"]][3]
            if action == "outside":
                return _present(axes, ideal, parent.removed)
            other = sheets[task["other"]][3]
            if action == "disjoint":
                return empty_locus(axes, (*ideal,*other))
            return radical_subset(axes,ideal,other) and radical_subset(axes,other,ideal)

        i, j = task["target"], task["source"]
        target, source = ordinary[i], ordinary[j]
        if action == "transition":
            forward = _parent_pullback(target, source, transitions)
            inverse = _parent_pullback(source, target, transitions)
            if forward is None or inverse is None:
                return False
            move, poles = forward
            transport[i,j] = move, api.Mul(*poles), inverse[1]
            return True
        move, unit, poles = transport[i,j]
        if action == "regular":
            ideal = sheets[task["sheet"]][3]
            return all(empty_locus(source.variables, (*ideal,d)) for d in poles)
        if action == "close":
            ideal = sheets[task["sheet"]][3]
            equations = (*target.ambient, *(move(g) for g in ideal))
            image = (saturate(target.variables,equations,unit) if unit != 1
                     else ideal_basis(target.variables,equations))
            if image == (1,):
                return None
            closed.append(image)
            return len(closed)-1
        image = closed[task["closed"]]
        if action == "point":
            return _dimension(target.variables,image) == 0
        return all(empty_locus(target.variables,(*image,*old))
                   for old in (*target.removed, *(closed[k] for k in task["previous"])))

    plan = operation(dict(parents=list(ordinary), sheets=[dict(parent=j,child=i) for _,j,_,_,i in sheets]),
                     query, symbol="atlas_repair_ownership")
    if plan is None:
        return None
    additions = {i: [closed[k] for k in indices] for i,indices in zip(ordinary, plan["additions"])}
    result = []
    for (sheet,j,mapping,_,_), indices in zip(sheets,plan["holes"]):
        parent = ordinary[j]
        other = [sheets[k][3] for k in indices] + additions[j]
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
    from .centre_weights import centre_rows
    from .kernel import exprs
    from .polynomial import rational_text
    parents = {c.repair_projection[0]:c for c in charts if c.repair_projection and c.torus
               and c.repair_projection[1] == c.variables[:-1]}

    @cache
    def certified(index):
        c = parents.get(index)
        if c is None or not c.coefficient_field:
            return None
        rays, linear = c.torus
        return _repair_geometry("parent", width=len(c.variables), rays=[list(map(rational_text,row)) for row in rays],
                                linear=[list(map(rational_text,row)) for row in linear],
                                modulus=centre_rows(c.variables,[c.coefficient_field[1]])[0],
                                names=[name for name,_ in c.coordinates],
                                images=centre_rows(c.variables,[p for _,p in c.coordinates]))

    @cache
    def powers(i,j):
        return _repair_geometry("transition", source=certified(i), target=certified(j))

    def equations(a,b):
        i, source = a.repair_projection
        j, target = b.repair_projection
        rows = powers(i,j)
        if rows is None:
            return None
        base = parents[i].variables[:-1]
        factors = iter(pullbacks(a.variables, source, base, exprs(base,rows)))
        return [(next(factors), t, next(factors)) for t in target]
    return equations
