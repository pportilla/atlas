"""Factor the original effective divisor and reconstruct its chart pullbacks over Qbar."""
import re
from functools import lru_cache

from . import native_library
from .kernel_json import kernel_json
from .point_geometry import point_operation
from .polynomial import format_poly, integer_from_decimal, integer_text, parse_expression, rational_text, sparse_row, InputError


def _product(parts):
    return "*".join(f"({g})^{integer_text(m)}" if m != 1 else f"({g})" for g,m in parts) or "1"


def crossing_input_divisor(poly, names, crossing):
    """Describe the original weights and local unit of a certified transverse divisor."""
    weights = kernel_json("atlas_identity_divisor", "weight_profile", weights=[integer_text(m) for _, m in crossing])
    if not weights["repeated"]:
        return None
    # The crossing solver already knows all branches passing through the
    # origin. Exact division recovers the unit without factoring its support.
    quotient = f"({format_poly(poly, names)})/({_product(crossing)})"
    row, = native_library.affine_request(["function", " ".join(names), quotient])
    support = parse_expression(_product([(f, 1) for f, _ in crossing]), names)
    return dict(reduced_equation=format_poly(support, names), unit_at_origin=format_poly(sparse_row(row), names),
                factors=[dict(equation=f, multiplicity=m) for f, m in crossing])


@lru_cache(maxsize=256)
def _factor(source, names):
    from .affine_kernel import archived_factor_order
    from .identity_atlas import origin_indices
    rows = native_library.affine_request(["factor", " ".join(names), source])
    zero = (0,)*len(names)
    scalar = sparse_row(rows[0])[zero]
    parts = [(sparse_row(row),integer_from_decimal(m)) for m,row in zip(rows[1::2],rows[2::2])]
    if len(parts) > 1:
        parts = [parts[i] for i in archived_factor_order(parts, len(names))]
    active = set(origin_indices((poly for poly, _ in parts), len(names)))
    factors, units = [], []
    for i, (poly, m) in enumerate(parts):
        (factors if i in active else units).append((format_poly(poly, names), m))
    if scalar != 1:
        units.insert(0,(rational_text(scalar),1))
    unit = _product(units)
    return tuple(factors), unit


def verify_factors(polynomial, names, records):
    from .strata import InvalidChart
    names = tuple(names)
    expected,_ = _factor(format_poly(polynomial, names), names)
    parts = []
    for entry in records:
        w = entry["multiplicity"]
        if type(w) != int or w <= 0:
            raise InvalidChart("A weighted factor needs a positive integral multiplicity.")
        canonical = format_poly(parse_expression(entry["equation"], names), names)
        parts.append((canonical,w))
    if sorted(parts) != sorted(expected):
        raise InvalidChart("Recorded factors and orders differ from the input divisor.")
    return parts


def _weighted_orders(components, rows):
    values = kernel_json("atlas_identity_divisor", "weighted_orders", width=len(components), rows=rows)
    return {c["id"]: integer_from_decimal(v) for c, v in zip(components, values)}


def _write_free(record, chart, weighted, unit, base, uniform):
    from .marked_field import record_field, display
    names = tuple(chart["coordinates"])
    texts = [chart["map"][v] for v in base]
    theta = "weight_generator"
    allnames = set(names) | set(base) | set(record.get("algebraic_numbers", {}))
    while theta in allnames:
        theta += "_"
    data = record.get("coefficient_field", {})
    if data.get("generator") and not data.get("exact_root"):
        name = data["generator"]
        match = re.fullmatch(re.escape(name)+r"\^2-\((-?\d+(?:/\d+)?)\)",data.get("minimal_polynomial",""))
        if match:
            data = dict(data, exact_root=f"sqrt({match[1]})")
    field, values = record_field(data, record.get("algebraic_numbers", {}), texts, theta)
    replace = lambda g: re.sub(r"[A-Za-z][A-Za-z0-9_]*",lambda m:"("+values[m[0]]+")" if m[0] in values else m[0],g)
    mark = (theta,) if field else ()
    coords = (*names,*mark)
    mod = field["minimal_polynomial"] if field else "0"
    args = [" ".join(coords)," ".join((*base,*mark)), *map(replace,texts), *mark, mod]
    boundary = chart.get("ambient_components",chart.get("exceptional", ()))
    axis = [names.index(c["equation"]) for c in boundary]
    valuations = []
    strict, total = [],[]
    pulled = native_library.affine_request(["pullbacks", *args, unit, *(s for s,_ in weighted)])
    scalar = format_poly(sparse_row(pulled[0]), coords)
    if scalar != "1":
        strict.append((scalar,1))
        total.append((scalar,1))
    for (f,m), row in zip(weighted,pulled[1:]):
        result = native_library.affine_request(["chart", *args, f, " ".join(map(str,axis))])
        valuations.append([integer_text(m), [n.split()[0] for n in result[:-1]]])
        strict.append((format_poly(sparse_row(result[-1]),coords),m))
        total.append((format_poly(sparse_row(row),coords),m))
    local = _weighted_orders(boundary, valuations)
    original = chart["strict_transform"]
    chart["reduced_strict_transform"] = original
    if "strict_transform_on_exceptional" in chart:
        chart["reduced_strict_transform_on_exceptional"] = chart.pop("strict_transform_on_exceptional")
    if uniform:
        # The support's strict equation carries the producer's displayed mark.
        # Compose its unit in the same unreduced names for this display alone.
        charts_names = (*names,*(k for k in values if k not in names))
        raw, = native_library.affine_request(["pullbacks"," ".join(charts_names)," ".join(base),*texts,"0",unit])
        raw_unit = format_poly(sparse_row(raw),charts_names)
        chart["strict_transform"] = _product(([(raw_unit,1)] if raw_unit != "1" else [])+[(original,weighted[0][1])])
    else:
        chart["strict_transform"] = _product(strict)
        if field:
            chart["weight_field"] = {"generator":theta,"exact":display(field["embedding"])}
    chart["pullback_factor"] = _product([(v["equation"],local[v["id"]]) for v in boundary if local[v["id"]]])
    chart["ambient_total_transform"] = f"({chart['pullback_factor']})*({chart['strict_transform']})" if uniform else _product(total)
    return local


def _write_local(saved, chart, factors, unit, base):
    """Remove each exceptional Cartier ideal from the original factor pullbacks."""
    from .monomial_atlas import _row
    coords = tuple(saved["coordinates"])
    ambient = [_row(g,coords) for g in saved["ambient_ideal"]]
    exceptional = [c for c in saved["components"] if c["id"] != "strict" and not re.fullmatch(r"S\d+",c["id"])]
    divisors = [[_row(g,coords) for g in c["ideal"]] for c in exceptional]
    pulled = native_library.affine_request(["pullbacks"," ".join(coords)," ".join(base),
                *(saved["map"][v] for v in base),"0",unit,*(f for f,_ in factors)])
    header = f"strict_parts {len(coords)} {len(ambient)} {len(divisors)}"+"".join(f" {len(d)}" for d in divisors)
    parts = iter(native_library.ideal_request(header,[*ambient,*(g for d in divisors for g in d),*pulled[1:]]))
    valuations = []
    components = []
    for i,(_,m) in enumerate(factors,1):
        *orders,size = map(integer_from_decimal,next(parts).split())
        ideal = [format_poly(sparse_row(next(parts)),coords) for _ in range(size)]
        valuations.append([integer_text(m), list(map(integer_text, orders))])
        if ideal != ["1"]:
            components.append(dict(id="strict" if len(factors)==1 else f"S{i}", multiplicity=m, nu=1, ideal=ideal))
    weights = _weighted_orders(exceptional, valuations)
    saved["components"] = [{**c,"multiplicity":weights[c["id"]]} for c in exceptional]+components
    scalar = format_poly(sparse_row(pulled[0]),coords)
    terms = [(format_poly(sparse_row(row),coords),m) for row,(_,m) in zip(pulled[1:],factors)]
    prefix = [(scalar,1)] if scalar != "1" else []
    chart["ambient_total_transform"] = _product(prefix+terms)
    chart["reduced_strict_transform"] = chart["strict_transform"]
    chart["strict_transform"] = kernel_json("atlas_identity_divisor", "strict_product_generators", unit=scalar,
                                          components=[{"ideal": c["ideal"], "multiplicity": integer_text(c["multiplicity"])}
                                                      for c in components])
    return weights


def nonreduced_resolution(poly, names, mode, resolve_support):
    names = tuple(names)
    factors,unit = _factor(format_poly(poly,names),names)
    weights = kernel_json("atlas_identity_divisor", "weight_profile", weights=[integer_text(m) for _, m in factors])
    if not weights["repeated"]:
        return None
    if mode == "intrinsic":
        raise InputError("The germ has a repeated component through the origin. An intrinsic resolution requires the underlying reduced hypersurface.")
    support = parse_expression(_product([(f,1) for f,_ in factors]),names)
    record = resolve_support(support)
    metadata = dict(reduced_equation=format_poly(support,names),unit_at_origin=unit,
                    factors=[dict(equation=f,multiplicity=m) for f,m in factors])
    record["input_divisor"] = metadata
    if record["status"] != "resolved":
        return record
    if record["engine"] == "hyperplane-arrangement-fan":
        from .newton_trials import hyperplane_fan
        lifted = hyperplane_fan(poly, list(names), record["fan"], None, False, factor_original=True)
        if lifted is None:
            raise ArithmeticError("Could not verify the original weighted divisor on its hyperplane fan.")
        record.update(lifted)
        return record
    if record["engine"] == "conjugate-plane-crossing":
        # Both marked branches lie over the same irreducible rational factor.
        # Keep their equations in the splitting field for original-divisor replay.
        (_, power), = factors
        chart, = record["charts"]
        owned, = record["atlas"]
        for component in (*chart["strict_transform_components"], *owned["components"]):
            component["multiplicity"] = power
        chart["reduced_strict_transform"] = chart["strict_transform"]
        chart["strict_transform"] = format_poly(poly, names)
        record["multiplicity"] = integer_from_decimal(point_operation("order", [0] * len(names), poly))
        return record
    raw = record["charts"]
    saved = record.get("atlas",[]) if record.get("engine") == "canonical-blowups" else []
    if not saved:
        record.pop("atlas",None)
    uniform = weights["uniform"]
    weights = {}
    if record.get("engine") == "verified-ideal-blowup":
        from .graph_record import lift_graph
    for i,c in enumerate(raw):
        owned = c if "fibre" in c and "components" in c else saved[i] if saved else None
        if owned is not None:
            local = _write_local(owned,c,factors,unit,names)
        elif record.get("engine") == "verified-ideal-blowup":
            local = lift_graph(record,c,factors,unit,names)
        else:
            local = _write_free(record,c,factors,unit,names,uniform)
        for label,n in local.items():
            if weights.setdefault(label,n) != n:
                raise ArithmeticError("Weighted orders disagree between charts.")
        for comp in (*c.get("exceptional",()), *c.get("ambient_components",())):
            for key in ("multiplicity","ambient_order","total_multiplicity"):
                if key in comp: comp[key] = local[comp["id"]]
            comp["total_multiplicity"] = local[comp["id"]]
    for vertex in record["dual_complex"]["vertices"]:
        for key in ("multiplicity","ambient_order","total_multiplicity"):
            if key in vertex: vertex[key] = weights[vertex["id"]]
        vertex["total_multiplicity"] = weights[vertex["id"]]
    for step in record.get("steps",[]):
        if step.get("exceptional_id") in weights: step["total_multiplicity"] = weights[step["exceptional_id"]]
        for key in ("strict_multiplicity","multiplicity"):
            if key in step: step["reduced_support_"+key] = step.pop(key)
    record["multiplicity"] = integer_from_decimal(point_operation("order", [0] * len(names), poly))
    record["summary"] = "Embedded resolution of the divisor with multiplicities. "+record["summary"]
    record.setdefault("notes",[]).append("The geometric dual complex and smoothness certificate refer to the reduced support. The strict-transform equations and exceptional orders include the original multiplicities.")
    return record
