"""Reconstruct projective graph charts and their Cartier orders from sparse equations."""
import re

from . import native_library
from .marked_field import record_field
from .polynomial import format_poly, parse_expression, sparse_row
from .strata import InvalidChart as InputError
from .monomial_atlas import _row


_PACKED_LIMITS = {"Unsupported number of variables.", "Degree out of range.", "Exponent out of range."}


def _coordinates(names):
    from .chart_expressions import runtime_expressions
    return tuple(map(runtime_expressions().Symbol, names))


def _saturate(coords, equations, pivot):
    try:
        rows = native_library.ideal_request(f"saturate {len(coords)} {len(equations)}", [*equations, pivot])
    except native_library.Unavailable as exc:
        if str(exc) not in _PACKED_LIMITS:
            raise
        from .affine_kernel import _sparse_expression
        from .blowup_ideal import saturate
        from .groebner import _line
        ring = _coordinates(coords)
        result = saturate(ring, tuple(_sparse_expression(ring,q) for q in equations), _sparse_expression(ring,pivot))
        rows = [_line(ring, g) for g in result]
    expressions = [format_poly(sparse_row(row), coords) for row in rows if row and row != "none"]
    reduced = native_library.affine_request(["wide_basis", " ".join(coords), "lex", *expressions])
    # A reduced lex basis is a canonical set. Packed and FLINT basis
    # extraction can enumerate its rows in a different order.
    return sorted(format_poly(sparse_row(row),coords) for row in reduced if row)


def _orders(coords, base, ambient, prime, total):
    try:
        data, = native_library.ideal_request(f"graph_orders {len(coords)} {len(base)} {len(ambient)} {len(prime)}",
                                             [*ambient,*prime,total])
        return tuple(map(int, data.split()))
    except native_library.Unavailable as exc:
        if str(exc) not in _PACKED_LIMITS:
            raise
        from .affine_kernel import _sparse_expression
        from .graph_atlas import _orders as chart_orders
        ring = _coordinates(coords)
        def convert(q):
            return _sparse_expression(ring, q)
        return chart_orders(ring, ring[:len(base)], tuple(map(convert,ambient)), tuple(map(convert,prime)), convert(total))


def _same_locus(coords, left, right):
    from .monomial_atlas import _same_locus as packed
    try:
        return packed(len(coords), left, right)
    except native_library.Unavailable as exc:
        if str(exc) not in _PACKED_LIMITS:
            raise
        from .affine_kernel import _sparse_expression
        from .blowup_ideal import radical_subset
        ring = _coordinates(coords)
        left, right = [tuple(_sparse_expression(ring,q) for q in ideal) for ideal in (left,right)]
        return radical_subset(ring,left,right) and radical_subset(ring,right,left)


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
            texts.extend(values)
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
    for src in source:
        names = tuple(src["coordinates"])
        if len(set(names)) != len(names) or not set(base).issubset(names):
            raise InputError("Invalid graph coordinates.")
        coords = (*base, *(v for v in names if v not in base), *((theta,) if field else ()))
        def equation(g):
            return format_poly(parse_expression(re.sub(r"[A-Za-z][A-Za-z0-9_]*",
                lambda m: "("+constants[m[0]]+")" if m[0] in constants else m[0], g), coords), coords)
        if set(src["map"]) != set(base) or any(equation(src["map"][v]) != v for v in base):
            raise InputError("The projective graph does not project to its base coordinates.")
        ambient = list(map(equation, src["ambient_ideal"]))
        if field: ambient.append(field["minimal_polynomial"])
        amb_rows = [_row(g, coords) for g in ambient]
        projective = list(map(equation, src["projective_coordinates"]))
        if projective.count("1") != 1 or any(q not in coords for q in projective if q != "1"):
            raise InputError("Expected one unit and coordinate ratios in the projective graph.")
        position = projective.index("1")
        ratios = [v for v in projective if v != "1"]
        if (position in seen or len(projective) != len(generators) or len(set(ratios)) != len(ratios)
                or set(ratios) != set(names)-set(base)):
            raise InputError("The affine chart is not a standard open of the declared projective graph.")
        seen.add(position)
        centre = [equation(g) for g in generators]
        pivot = centre[position]
        if equation(src["pivot_generator"]) != pivot:
            raise InputError("The pivot differs from its centre generator.")
        graph = [_row(f"({g})-({pivot})*({r})",coords) for g,r in zip(centre,projective)]
        if field:
            graph.append(_row(field["minimal_polynomial"],coords))
        if _saturate(coords,graph,_row(pivot,coords)) != _saturate(coords,amb_rows,_row("1",coords)):
            raise InputError("The ambient ideal differs from the saturated projective graph.")
        region = projective[:position]
        components = []
        covered = []
        for d in src.get("ambient_components", src["exceptional"]):
            prime = list(map(equation, d["ideal"]))
            n,nu = _orders(coords, base, amb_rows, [_row(g, coords) for g in prime], _row(f, coords))
            covered.append(prime)
            components.append(dict(id=d["id"], multiplicity=n, nu=nu, ideal=prime))
        # On the origin fibre the exceptional locus is V(pivot). Test both
        # inclusions against its listed prime ideals in the chart ring.
        from itertools import product
        by = _row(pivot, coords)
        closed = [*amb_rows,*(_row(v,coords) for v in base)]
        union = [_row("*".join(f"({g})" for g in selection) or "1",coords)
                 for selection in product(*covered)]
        if not _same_locus(coords,[*closed,by],[*closed,*union]):
            raise InputError("The exceptional prime ideals do not cover the pivot over the origin.")
        for index, factor in enumerate(factors or [dict(equation=f, multiplicity=1)], 1):
            ideal = _saturate(coords, [*amb_rows, _row(factor["equation"], coords)], by)
            if ideal != ["1"]:
                label = "strict" if len(factors) <= 1 else f"S{index}"
                components.append(dict(id=label, multiplicity=factor["multiplicity"], nu=1, ideal=ideal))
        chart = dict(coordinates=list(coords), ambient_ideal=ambient, components=components, fibre=list(base),
                     region=region, removed=[], map={v:v for v in base})
        if field: chart["coefficient_field"] = field
        charts.append(chart)
    # Charts whose affine graph misses the origin fibre may have been omitted
    # by the exporter. Check emptiness on each absent standard projective open.
    for pos in set(range(len(generators)))-seen:
        ratios = []
        for j in range(len(generators)):
            r = f"graphRatio{j}"
            while r in (*base,theta,*constants):
                r += "_"
            ratios.append("1" if pos == j else r)
        coords = (*base, *(r for r in ratios if r != "1"), *((theta,) if field else ()))
        centre = [equation(g) for g in generators]
        pivot = centre[pos]
        rows = [_row(f"({g})-({pivot})*({r})",coords) for g,r in zip(centre,ratios)]
        if field:
            rows.append(_row(field["minimal_polynomial"],coords))
        graph = _saturate(coords,rows,_row(pivot,coords))
        if _saturate(coords,[_row(g,coords) for g in (*graph,*base)],_row("1",coords)) != ["1"]:
            raise InputError("The graph omits an affine open meeting the origin fibre.")
    return charts


def _coordinate_outputs(record, charts, *, complex, intrinsic, integrals):
    from .chart_kernel import crossings_rows, strata_rows, strata_totals
    from .incidence import incidence, stratum_dimension
    from .strata import records
    if (intrinsic or native_library._MODE != "native" or record.get("coefficient_field")
            or record.get("input_divisor")
            or not hasattr(native_library._library(), "atlas_coordinate_exceptional_abi")):
        return None
    # graph_record_charts has verified every saturated graph open and its
    # pivot. The native certificate checks the actual exceptional ideals.
    pieces = [{"exceptional": [c for c in chart["components"] if c["id"] != "strict"]} for chart in charts]
    label = incidence("coordinate_exceptional", variables=record["variables"],
                      centre=record["steps"][0]["generators"], charts=pieces)
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
    if native_library._MODE == "sympy" or record.get("engine") != "verified-ideal-blowup":
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
            def basis(gens):
                return _saturate(coords, [_row(g,coords) for g in gens], _row("1",coords))
            if basis(old["ambient_ideal"]) != basis(new["ambient_ideal"]):
                raise InputError("The saved graph changed its ambient ideal.")
            ambient = new["ambient_ideal"]
            for slot in ("fibre","region"):
                if basis([*ambient,*old[slot]]) != basis([*ambient,*new[slot]]):
                    raise InputError("The saved graph changed its fibre or ownership.")
            actual = {v["id"]:v for v in old["components"]}
            if len(actual) != len(old["components"]) or set(actual) != {v["id"] for v in new["components"]}:
                raise InputError("The saved graph changed its boundary labels.")
            for v in new["components"]:
                given = actual[v["id"]]
                if (any(type(given[k]) != int or given[k] != v[k] for k in ("multiplicity","nu"))
                        or basis([*ambient,*given["ideal"]]) != basis([*ambient,*v["ideal"]])):
                    raise InputError("The saved graph changed a divisor ideal or its orders.")
    try:
        coordinate = _coordinate_outputs(record, expected, complex=complex, intrinsic=intrinsic, integrals=integrals)
    except native_library.Unavailable:
        coordinate = None
    if coordinate is not None:
        return (saved or expected), coordinate
    from .general_atlas import native_general_outputs
    output = native_general_outputs(record, expected, complex=complex, intrinsic=intrinsic, integrals=integrals)
    if output is None:
        return None
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
    amb = [_row(g,coords) for g in ambient]
    proj = [read(src["map"][v]) for v in base]
    if tuple(proj) != tuple(base):
        raise InputError("A graph chart must retain its original coordinates.")
    total = _product(([(unit,1)] if unit != "1" else [])+list(factors))
    row = _row(total,coords)
    strict = _saturate(coords,[*amb,row],_row(read(src["pivot_generator"]),coords))
    one = _row("1",coords)
    original = _saturate(coords,amb,one)
    src["reduced_strict_transform"] = src["strict_transform"]
    src["strict_transform"] = [g for g in strict if _saturate(coords,[*amb,_row(g,coords)],one) != original]
    src["ambient_total_transform"] = total
    weights = {}
    for c in src["exceptional"]:
        prime = [_row(read(g),coords) for g in c["ideal"]]
        weights[c["id"]] = _orders(coords, base, amb, prime, row)[0]
    return weights
