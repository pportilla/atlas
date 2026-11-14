"""Identify strata with finite geometric parameters on the resolution atlas.

A native computation identifies finite tuples of regular coordinate ratios
on the supporting charts, proves the local affine-space presentations and
checks generic ownership on every geometric sheet.

For intrinsic strata, a finite coordinate eliminant or a univariate
specialization can supply a candidate splitting field. Lines with collinear
exponents supply a polynomial in their primitive Laurent character.
FLINT finds a common primitive element and embeds the original marked
coefficient fields in it. A coordinate section over this field splits by
imposing each of its exact roots on its original equations. Linear
triangular graph leaves are geometrically integral. More general leaves
use absolute prime certificates and ideal-coverage tests. A specialization
never certifies a component or an incidence. Those require proofs on the
unspecialized ideal and on the actual chart overlaps.
"""
from dataclasses import replace
from functools import lru_cache
import re

from .chart_expressions import chart_expressions

from .chart_gluing_kernel import operation, rows, values
from .kernel_json import kernel_json
from .polynomial import integer_from_decimal, integer_text, kernel_fraction
from .local_blowups import _is_strict, _own_failures


def finite_complex(charts, dimension, counts):
    """Identify geometric sheets of exceptional intersections over finite parameter bases."""
    from .atlas import _record_text
    from .incidence import _intersections
    from .kernel_json import kernel_json
    payload = []
    for chart in charts:
        def text(f):
            return _record_text(f, chart.variables)
        payload.append({"coordinates": [v.name for v in chart.variables],
                        "ambient_ideal": [text(f) for f in chart.ambient],
                        "components": {label: [text(f) for f in ideal] for label, _, _, ideal in chart.components
                                       if not _is_strict(label)},
                        "map": {name: text(f) for name, f in chart.coordinates},
                        "region": [text(f) for f in chart.region], "fibre": [text(f) for f in chart.fibre],
                        "removed": [[text(f) for f in hole] for hole in chart.removed],
                        "coefficient_field": chart.coefficient_field[3] if chart.coefficient_field else None})
    result = kernel_json("atlas_finite_sheets", "complex", charts=payload, counts=_intersections(counts),
                         dimension=dimension)
    if result is not None:
        for chart in charts:
            boundary = tuple(row for row in chart.components if not _is_strict(row[0]))
            if _own_failures(replace(chart, components=boundary), dimension, {})[0]:
                return None
    return result


def geometric_chart_records(record):
    """Expose the certified local sheet ideals while retaining rational chart equations for replay."""
    from .polynomial import InputError
    charts = record["charts"]
    dual = record.get("dual_complex", {})
    sections = dual.get("finite_sheet_certificate", {}).get("chart_sections")
    if sections is None:
        return charts
    if len(charts) != len(sections):
        raise InputError("The finite sheet certificate has a different number of charts.")
    result = []
    for chart, entry in zip(charts, sections):
        if chart["coordinates"] != entry["coordinates"]:
            raise InputError("The finite sheet chart coordinates differ from the saved resolution.")
        sheets = entry["divisors"]
        if not sheets:
            result.append(chart)
            continue
        rational = chart.get("rational_exceptional", chart["exceptional"])
        rows = []
        for divisor in rational:
            label = divisor["id"]
            if label not in sheets:
                rows.append(divisor)
                continue
            ideal = divisor.get("ideal")
            if ideal is None:
                ideal = [divisor["equation"]]
            for section in sheets[label]:
                generators = [*ideal, *section["restrictions"]]
                rows.append({**divisor, "id": section["id"], "ambient_divisor": label,
                             "ideal": generators, "equation": ", ".join(generators)})
        result.append({**chart, "exceptional": rows, "rational_exceptional": rational})
    return result


def _integer_polynomial(variable, polynomial):
    f, = rows((variable,),(polynomial,))
    factors = kernel_json("atlas_marked_chart","integers",f=f)
    return [tuple(map(integer_from_decimal,row)) for row in factors]


@lru_cache(maxsize=256)
def _field_mark(field):
    """Transport archived FLINT indices and validate the defining marked polynomial."""
    from .marked_field import description
    from .polynomial import InputError
    t,relation,degree,exact = field
    parsed = re.fullmatch(r"root ([1-9][0-9]*) \(FLINT order\)", exact)
    coefficients = None
    if parsed:
        index = int(parsed[1])
    else:
        try:
            coefficients, index = description(exact)
        except InputError:
            return None
        coefficients = list(map(integer_text,coefficients))
    return kernel_json("atlas_marked_chart","mark",relation=rows((t,),(relation,))[0],
                       degree=degree,index=integer_text(index),description=coefficients)


@lru_cache(maxsize=2048)
def _root_coordinates(mark,root):
    from .marked_field import program
    from .native_library import field_request
    result = field_request("atlas_field_coefficient",[program(mark),program(root)])
    return None if result is None or result=="outside" else result.split()


@lru_cache(maxsize=4096)
def _root_test(row,mark,root):
    from .marked_field import program
    return kernel_json("atlas_marked_chart","on_sheet",equation=row,mark_program=program(mark),
                       root_program=program(root))


def _root_request(task):
    if task["op"]=="coordinates":
        return _root_coordinates(task["mark"],task["root"])
    if task["op"]=="satisfies":
        return _root_test(task["equation"],task["mark"],task["root"])
    raise ValueError("Unknown marked root request.")


@lru_cache(maxsize=256)
def _power_domain(t, relation, degree):
    from .candidate_field import NativeField
    from .native_polynomial import native_ring
    row = tuple(map(kernel_fraction,kernel_json("atlas_marked_chart","minimal",f=rows((t,),(relation,))[0])))
    if native_ring((t,)):
        return NativeField(row,0,t),t
    from .field_coefficient import indexed_field
    return indexed_field(row,t,0)


def marked_primes(chart, equations, dimension, branches=False):
    """Read certified absolute components on the chart's chosen field sheet."""
    from .field_polynomial import lift
    from .geometric_components import geometric_components
    from .native_polynomial import native_ring
    t,relation,degree,_ = chart.coefficient_field
    axes = chart.variables
    plan = kernel_json("atlas_marked_chart","normalize",width=len(axes),theta=axes.index(t),dimension=dimension,
                       ideal=rows(axes,tuple(equations)),relation=rows(axes,(relation,))[0])
    if plan.get("final"):
        return None if plan["result"] is None else [values(axes,ideal) for ideal in plan["result"]]
    coords = tuple(axes[j] for j in plan["axes"])
    ring = (*coords,t)
    equations = values(ring,plan["ideal"])
    dim = plan["dimension"]
    mark = _field_mark(chart.coefficient_field)
    if mark:
        sections = operation(dict(operation="sections",width=len(ring),dimension=dim,mark=mark,
                                  relation=rows(ring,(relation,))[0],ideal=plan["ideal"]),_root_request,
                             symbol="atlas_marked_sections")
        if sections is not None:
            return [values(ring,ideal) for ideal in sections]
    if plan["rational"]:
        result = geometric_components(coords,equations,dim,reduced_support=branches)
        if result is not None:
            return result
    field,generator = _power_domain(t,relation,degree)
    if not native_ring(axes):
        equations = tuple(g.xreplace({t:generator}) for g in equations)
    result = geometric_components(coords,equations,dim,domain=field,reduced_support=branches)
    if result is None or native_ring(axes):
        return result
    return [tuple(lift(g,coords,field,t) for g in piece) for piece in result]


def _extend_chart(chart, field, old_value):
    from .affine_kernel import pullbacks
    theta, relation, _, _ = field
    if not chart.coefficient_field:
        return replace(chart, variables=(*chart.variables,theta), ambient=(*chart.ambient,relation),
                       coefficient_field=field)
    t, old_relation, _, _ = chart.coefficient_field
    coords = tuple(v for v in chart.variables if v != t) + (theta,)
    mapping = tuple(old_value if v == t else v for v in chart.variables)

    def move(polys):
        return pullbacks(coords, mapping, chart.variables, polys, modulus=relation)

    ambient = tuple(f for f in chart.ambient if f != old_relation)
    return replace(chart, variables=coords, ambient=(*move(ambient),relation),
                   repair_projection=(chart.repair_projection[0],move(chart.repair_projection[1]))
                                     if chart.repair_projection else (),
                   coordinates=tuple(zip((name for name,_ in chart.coordinates),
                                         move(tuple(g for _,g in chart.coordinates)))),
                   components=tuple((label,n,nu,move(ideal)) for label,n,nu,ideal in chart.components),
                   fibre=move(chart.fibre), region=move(chart.region), removed=tuple(move(hole) for hole in chart.removed),
                   coefficient_field=field)


def splitting_charts(charts, witness):
    """Try a common splitting field on a failed section, and certify before retrying gluing."""
    from .affine_kernel import decode, pullbacks
    from .plane_backend import coefficient_field
    index, equations, dimension, *branch_mode = witness
    chart = charts[index]
    branches = branch_mode[0] if branch_mode else True
    api = chart_expressions(chart.variables)
    theta = api.Dummy("boundary_field")
    fields = tuple(dict.fromkeys(c.coefficient_field for c in charts if c.coefficient_field))
    marks = tuple(_field_mark(field) for field in fields)
    if any(mark is None for mark in marks):
        return None
    T = api.Symbol("T")

    def read(text):
        return decode((T,), text).xreplace({T:theta})

    def attempt(roots):
        generators = tuple(dict.fromkeys((*marks,*roots)))
        # FLINT proves every inclusion in the primitive field it returns.
        common = coefficient_field(generators)
        if not kernel_json("atlas_marked_chart","larger",degree=common["degree"],degrees=[f[2] for f in fields]):
            return None
        field = (theta, read(common["minimal_polynomial"]), common["degree"], common["exact"])
        images = {m:read(v) for m,v in zip(generators, common["values"])}
        substitutions = {f:images[m] for f,m in zip(fields,marks)}
        grown = _extend_chart(chart,field,substitutions.get(chart.coefficient_field))
        checks = equations
        if chart.coefficient_field:
            t = chart.coefficient_field[0]
            mapping = tuple(substitutions[chart.coefficient_field] if v == t else v for v in chart.variables)
            checks = pullbacks(grown.variables, mapping, chart.variables, equations, modulus=field[1])
        if marked_primes(grown, checks, dimension, branches=branches) is None:
            return None
        return [_extend_chart(c,field,substitutions.get(c.coefficient_field)) for c in charts]

    accepted = None
    def handle(task):
        nonlocal accepted
        if task["op"]=="attempt":
            accepted = attempt(task["roots"])
            return accepted is not None
        return _root_request(task)
    axes = chart.variables
    field = chart.coefficient_field
    operation(dict(operation="propose",width=len(axes),ideal=rows(axes,tuple(equations)),
                   theta=axes.index(field[0]) if field else -1,relation=rows(axes,(field[1],))[0] if field else None,
                   mark=_field_mark(field) if field else ""),handle,symbol="atlas_marked_sections")
    return accepted
