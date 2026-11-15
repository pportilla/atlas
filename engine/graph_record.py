"""Reconstruct projective graph charts and their Cartier orders from sparse equations."""
import re

from . import native_library
from .marked_field import record_field
from .polynomial import (InputError as PolynomialInputError, format_poly, integer_from_decimal,
                         native_expression_source, parse_expression)
from .strata import InvalidChart as InputError
from .monomial_atlas import _row


def _graph_request(operation, coords, **inputs):
    from .kernel_json import kernel_json
    try:
        return kernel_json("atlas_graph_replay", operation, coords=coords, **inputs)
    except PolynomialInputError as exc:
        raise InputError(str(exc)) from exc


def _graph_ring(variables, domain):
    """Present one marked coefficient as a rational coordinate for native graph operations."""
    from .affine_kernel import _ring
    from .reference import QQ
    if domain == QQ:
        ring = _ring(tuple(variables))
        return ring.header.split(), ring.text, []
    from .field_polynomial import _parameter, lift, modulus
    theta = _parameter(domain)
    ring = _ring((*variables, theta))
    def text(g):
        return ring.text(lift(g, variables, domain, theta))
    return ring.header.split(), text, [ring.text(modulus(domain, theta))]


def graph_record_charts(record, polynomial):
    if record.get("engine") != "verified-ideal-blowup":
        return None
    base = tuple(record["variables"])
    source = record["charts"]
    theta = record.get("coefficient_field", {}).get("generator", "atlas_field")
    if any(theta in c["coordinates"] for c in source):
        if record.get("coefficient_field", {}).get("generator"):
            raise InputError("The marked field and graph coordinates share a variable.")
        while any(theta in c["coordinates"] for c in source):
            theta += "_"
    steps = record.get("steps", ())
    if len(steps) != 1 or not steps[0].get("generators"):
        raise InputError("The projective graph needs its centre generators.")
    generators = steps[0]["generators"]
    if not source:
        raise InputError("The projective graph has no affine opens.")
    seen = set()
    texts = list(generators)
    for chart in source:
        for values in (chart["ambient_ideal"], chart["map"].values(), chart["projective_coordinates"],
                       (chart["pivot_generator"],), chart.get("reduced_strict_transform", chart["strict_transform"])):
            texts.extend([values] if isinstance(values, str) else values)
        for divisor in chart.get("ambient_components", chart["exceptional"]):
            texts.extend(divisor["ideal"])
    try:
        field, constants = record_field(record.get("coefficient_field", {}), record.get("algebraic_numbers", {}), texts, theta)
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    factors = record.get("input_divisor", {}).get("factors", [])
    if factors:
        from .weighted_record import verify_factors
        verify_factors(polynomial, base, factors)
    f = format_poly(polynomial, base)
    charts = []

    def expression(g, aliases):
        if aliases:
            g = re.sub(r"[A-Za-z][A-Za-z0-9_]*",
                       lambda m: "(" + aliases[m[0]] + ")" if m[0] in aliases else m[0], g)
        return native_expression_source(g, base)

    centre = [expression(g, constants) for g in generators]
    relation = field["minimal_polynomial"] if field else None
    for src in source:
        names = tuple(src["coordinates"])
        if (len(set(names)) != len(names) or not set(base).issubset(names)
                or set(src["map"]) != set(base)):
            raise InputError("Invalid graph coordinates or projection.")
        coords = (*base, *(v for v in names if v not in base), *((theta,) if field else ()))
        exceptional = src.get("ambient_components", src["exceptional"])
        aliases = dict(constants)
        weight = src.get("weight_field")
        if weight:
            from .marked_field import description
            if not field or description(weight["exact"]) != description(field["embedding"]):
                raise InputError("The weighted strict transform changed its marked coefficient field.")
            name = weight["generator"]
            if name in names:
                raise InputError("The weighted coefficient and chart coordinates share a variable.")
            aliases[name] = theta

        def equations(values, mapping=constants):
            return [expression(g, mapping) for g in values]

        strict = src["strict_transform"]
        args = dict(base=len(base), centre=centre, relation=relation, equation=f,
                    projection=equations(src["map"][v] for v in base),
                    projective=equations(src["projective_coordinates"]),
                    pivot=expression(src["pivot_generator"], constants), ambient=equations(src["ambient_ideal"]),
                    exceptional=[equations(d["ideal"]) for d in exceptional],
                    factors=[e["equation"] for e in factors],
                    strict=equations([strict] if isinstance(strict, str) else strict, aliases))
        if "reduced_strict_transform" in src:
            reduced = src["reduced_strict_transform"]
            args["reduced_strict"] = equations([reduced] if isinstance(reduced, str) else reduced)
        verified = _graph_request("open", coords, **args)
        position = verified["position"]
        if position in seen:
            raise InputError("The affine chart repeats a projective pivot.")
        seen.add(position)
        components = []
        for divisor, proof in zip(exceptional, verified["exceptional"]):
            n, nu = map(integer_from_decimal, proof["orders"])
            components.append(dict(id=divisor["id"], multiplicity=n, nu=nu, ideal=proof["ideal"]))
        for i, ideal in enumerate(verified["strict"]):
            if ideal is not None:
                label = "strict" if len(factors) <= 1 else f"S{i+1}"
                components.append(dict(id=label, multiplicity=factors[i]["multiplicity"] if factors else 1,
                                       nu=1, ideal=ideal))
        chart = dict(coordinates=list(coords), ambient_ideal=verified["ambient"], components=components,
                     fibre=list(base), region=verified["region"], removed=[], map={v:v for v in base})
        if field:
            chart["coefficient_field"] = field
        charts.append(chart)
    for pos in set(range(len(generators))) - seen:
        ratios = []
        for j in range(len(generators)):
            name = f"graphRatio{j}"
            while name in (*base, theta, *constants):
                name += "_"
            ratios.append("1" if pos == j else name)
        coords = (*base, *(v for v in ratios if v != "1"), *((theta,) if field else ()))
        _graph_request("omitted", coords, base=len(base), centre=centre, relation=relation, projective=ratios)
    return charts


def _graph_bundle_outputs(record, charts, *, complex, intrinsic, integrals):
    from .chart_kernel import crossings_rows, strata_rows, strata_totals
    from .incidence import incidence, stratum_dimension
    from .strata import records
    if (intrinsic or record.get("coefficient_field")
            or record.get("input_divisor")
            or not hasattr(native_library._library(), "atlas_coordinate_exceptional_abi")):
        return None
    # graph_record_charts has verified every saturated graph open and its
    # pivot. The native certificate checks the actual exceptional ideals.
    base = tuple(record["variables"])
    centre = record["steps"][0]["generators"]
    pieces = [{"exceptional": [c for c in chart["components"] if c["id"] != "strict"]} for chart in charts]
    label = incidence("coordinate_exceptional", variables=base, centre=centre, charts=pieces)
    if label is None and hasattr(native_library._library(), "atlas_polynomial_graph_exceptional_abi"):
        # Compare the exceptional ideal on each open with the very same base
        # polynomials used to prove the centre is a graph. A ratio variable in
        # a boundary equation requires the general component calculation.
        from .polynomial import InputError as PolynomialInputError
        try:
            centre_rows = [_row(g, base) for g in centre]
            for chart in pieces:
                chart["exceptional"] = [
                    {"id": c["id"], "ideal": [_row(g, base) for g in c["ideal"]]}
                    for c in chart["exceptional"]
                ]
        except PolynomialInputError:
            return None
        label = incidence("polynomial_graph_exceptional", variables=base, centre=centre_rows, charts=pieces)
    if label is None:
        return None
    data, contributions = {}, []
    for chart in charts:
        coords = tuple(chart["coordinates"])
        ambient = [_row(eq, coords) for eq in chart["ambient_ideal"]]
        fibre = [_row(eq, coords) for eq in (*chart["region"], *chart["fibre"])]
        components = chart["components"]
        ideals = [[_row(eq, coords) for eq in c["ideal"]] for c in components]
        if crossings_rows(len(coords), len(record["variables"]), ambient, ideals, fibre, ()) != []:
            return None
        if integrals:
            for c in components:
                numbers = (c["multiplicity"], c["nu"])
                if data.setdefault(c["id"], numbers) != numbers:
                    return None
            strata = strata_rows(len(coords), [*ambient, *fibre], ideals, ())
            if strata is None:
                return None
            for indices, chi in strata.items():
                labels = frozenset(components[i]["id"] for i in indices)
                contributions.append((labels, 1, chi))
    answer = {"reason": None, "cells": None}
    if complex:
        vertex = dict(id=label, dimension=stratum_dimension(len(record["variables"]), 1))
        boundary = dict(status="complete", vertices=[vertex], cells=[],
                        convention="Ambient exceptional divisor in the germ over the origin over Qbar.")
        answer.update(dual_complex=boundary, cells=boundary["cells"])
    if integrals:
        values, strata = records({"data": data, "strata": strata_totals(contributions)})
        answer.update(numerical_data=values, strata=strata)
    return answer


def graph_outputs(record, polynomial, *, complex=True, intrinsic=False, integrals=True):
    if record.get("engine") != "verified-ideal-blowup":
        return None
    expected = graph_record_charts(record, polynomial)
    saved = record.get("atlas")
    if saved:
        if len(saved) != len(expected):
            raise InputError("The saved graph has an incorrect number of affine charts.")
        for old, new in zip(saved, expected):
            coords = tuple(new["coordinates"])
            old_field = old.get("coefficient_field")
            new_field = new.get("coefficient_field")
            if bool(old_field) != bool(new_field):
                raise InputError("The saved graph changed its coefficient field.")
            if old_field:
                from .marked_field import description
                if (old_field["generator"] != new_field["generator"] or old_field["degree"] != new_field["degree"]
                        or _row(old_field["minimal_polynomial"], coords) != _row(new_field["minimal_polynomial"], coords)
                        or description(old_field["embedding"]) != description(new_field["embedding"])):
                    raise InputError("The saved graph changed its marked coefficient field.")
            if tuple(old["coordinates"]) != coords or old["removed"] or set(old["map"]) != set(new["map"]):
                raise InputError("The saved graph changed its affine presentation.")
            if any(_row(g,coords) != _row(new["map"][v],coords) for v,g in old["map"].items()):
                raise InputError("The saved graph changed its projection.")
            ambient = new["ambient_ideal"]
            pairs = [[old["ambient_ideal"], ambient]]
            for slot in ("fibre", "region"):
                pairs.append([[*ambient, *old[slot]], [*ambient, *new[slot]]])
            actual = {v["id"]:v for v in old["components"]}
            if len(actual) != len(old["components"]) or set(actual) != {v["id"] for v in new["components"]}:
                raise InputError("The saved graph changed its boundary labels.")
            for v in new["components"]:
                given = actual[v["id"]]
                if any(type(given[k]) != int or given[k] != v[k] for k in ("multiplicity", "nu")):
                    raise InputError("The saved graph changed a divisor order.")
                pairs.append([[*ambient, *given["ideal"]], [*ambient, *v["ideal"]]])
            validated = [[[native_expression_source(g, coords) for g in ideal] for ideal in pair] for pair in pairs]
            if not _graph_request("equal_ideals", coords, pairs=validated):
                raise InputError("The saved graph changed its ambient, fibre, ownership or divisor ideals.")
    try:
        bundle = _graph_bundle_outputs(record, expected, complex=complex, intrinsic=intrinsic, integrals=integrals)
    except native_library.Unavailable:
        bundle = None
    if bundle is not None:
        return (saved or expected), bundle
    from .general_atlas import native_general_outputs
    output = native_general_outputs(record, expected, complex=complex, intrinsic=intrinsic, integrals=integrals)
    if output is None:
        return None
    if complex and record.get("input_divisor") and not intrinsic:
        # Fresh weighted resolutions attach total orders to exceptional
        # vertices. Recover them from the independently reconstructed charts.
        orders = {}
        for chart in expected:
            for component in chart["components"]:
                label, order = component["id"], component["multiplicity"]
                if orders.setdefault(label, order) != order:
                    raise InputError("The weighted graph has inconsistent divisor orders across charts.")
        for vertex in output["dual_complex"]["vertices"]:
            if vertex["id"] in orders:
                vertex["total_multiplicity"] = orders[vertex["id"]]
    return (saved or expected), output


def lift_graph(record, src, factors, unit, base):
    """Compute a weighted graph's displayed equations and all local Cartier orders."""
    from .marked_field import display
    from .weighted_record import _product
    theta = "weight_generator"
    while theta in {*src["coordinates"],*base}:
        theta += "_"
    texts = [*src["map"].values(), *src["ambient_ideal"], src["pivot_generator"],
             *(g for c in src["exceptional"] for g in c["ideal"])]
    field, constants = record_field(record.get("coefficient_field",{}), record.get("algebraic_numbers",{}), texts, theta)
    coords = (*base, *(n for n in src["coordinates"] if n not in base), *((theta,) if field else ()))
    def read(g):
        return format_poly(parse_expression(re.sub(r"[A-Za-z][A-Za-z0-9_]*", lambda m:
            "("+constants[m[0]]+")" if m[0] in constants else m[0], g),coords),coords)
    ambient = [read(g) for g in src["ambient_ideal"]]
    if field:
        ambient.append(field["minimal_polynomial"])
        exact = record.get("coefficient_field",{}).get("exact_root")
        src["weight_field"] = dict(generator=theta, exact=exact if exact and not record.get("algebraic_numbers") else display(field["embedding"]))
    if tuple(read(src["map"][v]) for v in base) != tuple(base):
        raise InputError("A graph chart must retain its original coordinates.")
    total = _product(([(unit,1)] if unit != "1" else [])+list(factors))
    pivot = read(src["pivot_generator"])
    strict = _graph_request("basis", coords, ideal=[*ambient, total], pivot=pivot)
    src["reduced_strict_transform"] = src["strict_transform"]
    src["strict_transform"] = [g for g in strict if not _graph_request(
        "equal_ideals", coords, pairs=[[ambient, [*ambient, g]]])]
    src["ambient_total_transform"] = total
    weights = {}
    for component in src["exceptional"]:
        n, _ = _graph_request("orders", coords, base=len(base), ambient=ambient,
                              prime=[read(g) for g in component["ideal"]], equation=total)
        weights[component["id"]] = integer_from_decimal(n)
    return weights
