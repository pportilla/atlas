"""Glue geometric intersection components on overlaps of rational affine charts.

Equal original coordinates give the transition graph off the exceptional
divisors. Saturating by one nonzero equation from each exceptional ideal
closes that graph in the chart product. Its closure in the supplied chart
models can have contracted fibres. A common open requires every target
coordinate to be regular at the source generic point and conversely.
Elimination gives all linear relations d(X)t-n(X) for each coordinate t.
Their denominators generate its regularity ideal on the source chart.
Matching requires the two prime ideals to pull back into each other on
these regular loci and the generic point to survive every removed centre.

For a fixed number field, adjoining one common parameter to every chart and
equating its two copies in the diagonal matches the chosen sheets. The
kernel can work on all embeddings together because the tests of nonempty
intersection and ideal containment commute with conjugacy. Local components
are used in this argument only when their ideals define a single geometric
prime on each sheet. For a finite cover of several repair centres,
first separate the marked sheets and verify the projection to each centre.
For affine charts, a section M²=d of a coordinate flat with M a primitive
monomial and d a nonzero rational nonsquare supplies its own quadratic
splitting field. Sections in that square class then have two explicit
geometric primes whose identifications are tested on the same diagonal.

For an intrinsic boundary, first split each restricted divisor into
geometric primes and test the reduced branches for simple normal crossings
on the owned fibre. Primes on two charts agree precisely when they have the
same generic point on a common chart open. Gluing before
intersecting with the origin fibre retains the parts of a component that
join two visible charts. A face is the unique containing component of the
smaller intersection, tested by containment of ideals on a common chart.
"""
from dataclasses import replace
from functools import cache

from . import native_polynomial as np
from .chart_expressions import chart_expressions

from .affine_kernel import decode
from .atlas_finite import _extend_chart, _field_mark, marked_primes, splitting_charts
from .geometric_components import geometric_components
from .chart_gluing_kernel import chart_complex, chart_input, rows, values
from .kernel_json import kernel_json
from .local_blowups import _is_strict, _own_failures


def _common_field(charts):
    """Present the chosen embeddings using one common parameter. Equality of
    that parameter on the overlap identifies the sheets, even when two
    charts have independently named conjugate roots of the same polynomial."""
    fields = list(dict.fromkeys(c.coefficient_field for c in charts if c.coefficient_field))
    if not fields:
        return charts
    if len(fields) == 1 and all(c.coefficient_field == fields[0] for c in charts):
        return charts
    api = chart_expressions(charts[0].variables)
    theta = api.Symbol("atlas_theta")
    used = {v for c in charts for v in c.variables}
    while theta in used:
        theta = api.Symbol(theta.name + "_")
    identities = [(minimal.xreplace({v: theta}), degree, embedding) for v, minimal, degree, embedding in fields]
    if len(set(identities)) == 1:
        relation, degree, embedding = identities[0]
        values = dict.fromkeys(fields, theta)
    else:
        from .plane_backend import coefficient_field
        exact = tuple(_field_mark(field) for field in fields)
        if any(value is None for value in exact):
            return None
        common = coefficient_field(exact)
        def read(source):
            return decode((api.Symbol("T"),), source).xreplace({api.Symbol("T"): theta})
        values = {field: read(word) for field, word in zip(fields, common["values"])}
        relation, degree, embedding = read(common["minimal_polynomial"]), common["degree"], common["exact"]
    field = theta, relation, degree, embedding
    return [_extend_chart(c, field, values.get(c.coefficient_field)) for c in charts]


def _character_field(charts):
    """Use one marked quadratic embedding for all affine charts."""
    result = kernel_json("atlas_chart_geometry","character_field",charts=[chart_input(c) for c in charts])
    if result is None:
        return charts,None
    api = chart_expressions(charts[0].variables)
    parameter = api.Symbol("atlas_theta")
    used = {v for chart in charts for v in chart.variables}
    while parameter in used:
        parameter = api.Symbol(parameter.name + "_")
    relation, = values((parameter,),[result["relation"]])
    field = (parameter,relation,2,result["embedding"])
    return [replace(c,variables=(*c.variables,parameter),ambient=(relation,),coefficient_field=field)
            for c in charts],api.Rational(result["d"])


def _geometric_primes(chart, equations, dimension, character=None, *, reduced_support=False):
    """Transport free affine factors, rational characters and fixed field sheets."""
    field = chart.coefficient_field
    axes = chart.variables
    plan = kernel_json("atlas_chart_geometry","prime_plan",width=len(axes),ideal=rows(axes,tuple(equations)),
                       dimension=dimension, theta=axes.index(field[0]) if field else -1,
                       relation=rows(axes,(field[1],))[0] if field else None,
                       character=str(character) if character is not None else None)
    if plan and "primes" in plan:
        return [values(axes,ideal) for ideal in plan["primes"]]
    if plan and "axes" in plan:
        coords = tuple(axes[j] for j in plan["axes"])
        result = geometric_components(coords,values(axes,plan["ideal"]),plan["dimension"],
                                      reduced_support=reduced_support)
        if result is not None:
            return result
    return marked_primes(chart,equations,dimension,branches=reduced_support) if field else None


def _prime_groups(charts, pieces, overlap, transition):
    """Glue irreducible primes by equality on a nonempty open of the diagonal."""
    return chart_complex(charts, overlap, transition, pieces=pieces)


class _NeedsSplittingField(Exception):
    """Carry the exact failed section into a proposed base change."""


def _ordinary_overlaps(charts, boundaries):
    """Prepare ordinary affine chart maps once and close each requested diagonal."""
    from .kernel import exprs
    from .native_library import Unavailable, _kernel, ideal_request
    if not hasattr(_kernel(), "atlas_affine_overlap_abi"):
        return None

    @cache
    def prepare(i):
        chart = charts[i]
        if (chart.ambient or chart.coefficient_field or chart.repair_projection
                or not np.native_ring(chart.variables)):
            return None
        images = {name: np.row(chart.variables, value) for name, value in chart.coordinates}
        ideals = tuple(boundaries[i].values())
        sizes = " ".join(map(str, (len(ideals), *(len(ideal) for ideal in ideals))))
        rows = tuple(np.row(chart.variables, value) for ideal in ideals for value in ideal)
        return images, sizes, rows

    @cache
    def block(i, prefix):
        variables = charts[i].variables
        coordinates = np.symbols(f"{prefix}0:{len(variables)}")
        rename = dict(zip(variables, coordinates))
        # Deleted centres restrict the open *after* closure of the diagonal.
        removed = tuple(tuple(g.xreplace(rename) for g in ideal) for ideal in charts[i].removed)
        return coordinates, rename, removed

    def overlap(i, j):
        a, b = prepare(i), prepare(j)
        if a is None or b is None:
            return None
        images_a, sizes_a, divisors_a = a
        images_b, sizes_b, divisors_b = b
        if images_a.keys() != images_b.keys():
            raise ValueError("The chart maps use different base coordinates.")
        va, rename_a, holes_a = block(i, "atlasLeft")
        vb, rename_b, holes_b = block(j, "atlasRight")
        header = f"affine_overlap {len(va)} {len(vb)} {len(images_a)} {sizes_a} {sizes_b}"
        rows = (*images_a.values(), *(images_b[name] for name in images_a), *divisors_a, *divisors_b)
        try:
            diagonal = ideal_request(header, rows)
        except Unavailable:
            return None
        coords = (*va, *vb)
        return coords, rename_a, rename_b, exprs(coords, diagonal), holes_a + holes_b

    return overlap


def glued_complex(charts, dimension, intrinsic=False, *, counts=None, _descended=False, global_boundary_labels=False,
                  canonical_cover=False):
    """Return the full complex when local geometric primes can be identified.

    On charts over marked fields, use the diagonal of the common coefficient
    field and local primes whose restriction to a single embedding is proved
    geometrically integral. Off deleted centres both the existence of an
    overlap and the final fibre test use the actual open sets. A marked field
    is distinct from a finite cover representing several geometric points:
    the latter fails the generic-injectivity test on implicit charts.
    """
    plan = kernel_json("atlas_chart_geometry", "preflight", charts=[chart_input(c) for c in charts],
                       dimension=dimension, intrinsic=intrinsic)
    if not plan["valid"]:
        return None
    if not _descended and not canonical_cover and (intrinsic or any(c.repair_projection for c in charts)):
        from .repair_replay import restore_repair_maps
        charts = restore_repair_maps(charts)
        if charts is None:
            return None
    if not _descended and any(c.repair_projection for c in charts):
        from .repair_gluing import repair_charts
        charts = repair_charts(charts, order_components=intrinsic)
        if charts is None:
            return None
        _descended = True
    if counts is not None and not intrinsic and not _descended:
        from .atlas_finite import finite_complex
        finite = finite_complex(charts, dimension, counts)
        if finite is not None:
            return finite
    charts = _common_field(charts)
    if charts is None:
        return None
    character = None
    if plan["restricted"]:
        charts, character = _character_field(charts)
    carrier_dimension = plan["carrier"]
    # Repair and base change may introduce new charts or new presentations.
    if (intrinsic or _descended) and not plan["restricted"]:
        plan = kernel_json("atlas_chart_geometry","preflight",charts=[chart_input(c) for c in charts],
                           dimension=dimension,intrinsic=intrinsic)
    for i, ideal in plan["check"]:
        chart = charts[i]
        boundary = tuple(row for row in chart.components if not _is_strict(row[0]))
        owned = replace(chart,ambient=(*chart.ambient,*values(chart.variables,ideal)),components=boundary,
                        fibre=tuple(dict.fromkeys((*chart.fibre,*chart.region))),region=())
        if _own_failures(owned,carrier_dimension,{})[0]:
            return None
    if _descended:
        from .repair_gluing import parent_transitions
        parent_equations = parent_transitions(charts)
        ordinary = None
    else:
        parent_equations = None
        boundaries = [{k: ideal for k, _, _, ideal in c.components if not _is_strict(k)} for c in charts]
        ordinary = _ordinary_overlaps(charts, boundaries)

    @cache
    def overlap(i, j):
        if ordinary is not None:
            result = ordinary(i, j)
            if result is not None:
                return result
        a, b = charts[i], charts[j]
        # All functions on each side are renamed simultaneously. The two
        # coordinate blocks span a fresh product ring, so positional names
        # can also reuse cached expressions for successive chart products.
        api = chart_expressions(a.variables)
        va = api.symbols(f"atlasLeft0:{len(a.variables)}")
        vb = api.symbols(f"atlasRight0:{len(b.variables)}")
        rename_a, rename_b = dict(zip(a.variables, va)), dict(zip(b.variables, vb))
        images_a, images_b = dict(a.coordinates), dict(b.coordinates)
        if images_a.keys() != images_b.keys():
            raise ValueError("The chart maps use different base coordinates.")
        source, target = chart_input(a), dict(chart_input(b))
        target["maps"] = rows(b.variables,tuple(images_b[name] for name in images_a))
        relations = (parent_equations(a,b) if parent_equations and a.repair_projection and b.repair_projection
                     and a.repair_projection[0] != b.repair_projection[0] else None)
        triplets = None if relations is None else [
            (rows(a.variables,(n,))[0],rows(b.variables,(h,))[0],rows(a.variables,(d,))[0])
            for n,h,d in relations]
        result = kernel_json("atlas_chart_geometry","diagonal",a=source,b=target,transition=triplets,
                             descended=_descended)
        coords = (*va,*vb)
        return coords,rename_a,rename_b,values(coords,result["graph"]),tuple(values(coords,h) for h in result["removed"])

    parents_by_link = {}
    for i, chart in enumerate(charts):
        self_check = kernel_json("atlas_chart_geometry","self",chart=chart_input(chart),descended=_descended)
        if self_check["ordinary_parent"]:
            parents_by_link[chart.repair_projection[0]] = chart
        if self_check["identity"]:
            continue
        if "inverse" in self_check:
            from .repair_gluing import _birational_parameters
            if _birational_parameters(chart,values(chart.variables,self_check["inverse"])) is not None:
                continue
            if self_check["ordinary_parent"]:
                return None
        coords, _, _, diagonal, _ = overlap(i,i)
        if not kernel_json("atlas_chart_geometry","faithful",width=len(chart.variables),graph=rows(coords,tuple(diagonal)),
                           ambient=rows(chart.variables,chart.ambient)):
            return None

    @cache
    def transition(i, j):
        if not _descended:
            return None
        a,b = charts[i],charts[j]
        if not a.repair_projection or not b.repair_projection:
            return None
        pa,pb = a.repair_projection[0],b.repair_projection[0]
        target_parent = None
        if pa != pb:
            if pa not in parents_by_link or pb not in parents_by_link:
                return None
            target_parent = parents_by_link[pb]
        from .repair_gluing import transition_on_field_sheet
        return transition_on_field_sheet(a,b,a.repair_projection[1],b.repair_projection[1],target_parent=target_parent)

    from .absolute_sections import ExtendSectionField, extend_section_field

    def retry(enlarged):
        if enlarged is None:
            return None
        return glued_complex(enlarged, dimension, intrinsic=intrinsic, _descended=_descended,
                             global_boundary_labels=global_boundary_labels, canonical_cover=canonical_cover)

    if plan["restricted"]:
        try:
            return chart_complex(charts, overlap, transition, carrier_dimension, True, character=character,
                                 global_boundary_labels=global_boundary_labels)
        except _NeedsSplittingField as failed:
            enlarged = splitting_charts(charts, failed.args)
            if enlarged is not None:
                return retry(enlarged)
            try:
                return chart_complex(charts, overlap, transition, carrier_dimension, True, character=character,
                                     absolute=True, global_boundary_labels=global_boundary_labels)
            except ExtendSectionField as extension:
                return retry(extend_section_field(charts, extension.args[0]))
            except _NeedsSplittingField:
                return None

    try:
        return chart_complex(charts, overlap, transition, carrier_dimension, intrinsic, character=character)
    except ExtendSectionField as extension:
        return retry(extend_section_field(charts, extension.args[0]))
