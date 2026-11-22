"""Verify affine fan charts with monomial projection in a common linear coordinate system.

A maximal regular cone gives an affine-space chart whose exceptional divisors
are the noncoordinate rays. The toric orbit closure for a face is integral,
including for a ray on the boundary of the orthant; the closure meets the
origin fibre at the fixed point of any containing maximal cone. Thus exceptional
faces give the embedded incidence complex. For intrinsic complexes with internal
exceptional rays, common torus characters glue the sections. On surfaces,
coordinate curves and closures of plane-curve factors can also meet in the
same ambient divisor. The reduced branches are checked on their smooth
strict surface and their intersections counted on the owned regions.
A common invertible linear
map preserves the origin fibre and changes the volume form by a nonzero scalar.
The original input is pulled back through the composite map on every chart.
The same native chart, crossings and Euler operations
supply the numerical data for fresh charts and for saved equations.
"""
from functools import lru_cache
from itertools import groupby, islice
import re

from . import native_library
from .chart_kernel import InconsistentSheets, crossings_rows, strata_rows, strata_totals
from .fan_geometry import _determinant, complete_fan, fan_cover, split_cone
from .polynomial import (InputError, encode_sparse_row, format_poly, integer_from_decimal, integer_text,
                         parse_expression, sparse_row)
from .strata import InvalidChart, records


def _unit(i, n):
    return tuple(int(i == j) for j in range(n))


@lru_cache(maxsize=16384)
def _row(text, coords):
    return encode_sparse_row(parse_expression(text, coords))


def _strict_equation(row, coords):
    """Decode a kernel pullback for chart serialization and orbit restriction."""
    terms = sparse_row(row)
    return terms, format_poly(terms, coords)


# Retain at most 8 MiB of combined sparse rows, names and formatted
# equations, and at most 131072 decoded terms across all cached charts.
@lru_cache(maxsize=256)
def _cached_strict_equation(row, coords):
    return _strict_equation(row, coords)


def _chart_equation(row, coords):
    names = sum(map(len, coords))
    terms = row.count(";") + 1
    # A rendered term uses at most all coordinate names plus a product
    # and exponent marker per axis. Its coefficients and exponent digits
    # already occur in the sparse row. Bound both texts before caching.
    text_bound = 2 * len(row) + names + terms * (names + 2*len(coords) + 1)
    decode = _cached_strict_equation if terms <= 512 and text_bound <= 32768 else _strict_equation
    return decode(row, coords)


def _charts(record, n):
    identity = tuple(_unit(i, n) for i in range(n))
    saved = record.get("atlas")
    items = saved if saved is not None else record.get("charts", ())
    if items and any("split_cone" in c for c in items):
        if any("split_cone" not in c for c in items):
            raise InvalidChart("Every chart of the split fan needs its cone.")
        if any(c.get(k) for c in items for k in ("blown_up", "removed", "exclude_points", "point_blowup")):
            if saved is not None:
                raise InvalidChart("A saved complete split fan must include every point of its maximal charts.")
            return None
        if any(c.get("torus") for c in items):
            raise InvalidChart("A chart must use a consistent common torus description.")
        if not saved and any(c.get("ambient_ideal") for c in items):
            return None
        return [(c, split_cone(c, n)) for c in items], None
    if saved is not None:
        if not isinstance(saved, list) or not saved:
            return None
        if any(c.get(k) for c in saved for k in ("coefficient_field", "removed", "ambient_ideal")):
            return None  # includes the toric parents joined to local blowup charts
        marked = next((c["torus"] for c in saved if c.get("torus")), None)
        linear = tuple(map(tuple, marked["linear_map"])) if marked else identity
        result = []
        for chart in saved:
            if chart.get("torus"):
                if tuple(map(tuple, chart["torus"]["linear_map"])) != linear:
                    raise InvalidChart("The charts disagree on their common torus coordinates.")
                cone = tuple(map(tuple, chart["torus"]["rays"]))
            else:
                # Older rational fan charts sometimes carry only the map.
                # Its exponent columns *are* the rays in the identity frame.
                images = [parse_expression(chart["map"][name], chart["coordinates"]) for name in record["variables"]]
                if linear != identity or any(len(f) != 1 or next(iter(f.values())) != 1 for f in images):
                    if marked:
                        raise InvalidChart("The claimed fan has a chart with no verified common coordinates.")
                    return None
                cone = tuple(zip(*(next(iter(f)) for f in images)))
            result.append((chart, cone))
        if not marked and (len(saved) == 1 or not complete_fan(tuple(cone for _, cone in result), n)):
            return None
        return result, linear
    fan = record.get("fan")
    items = record.get("charts", ())
    if not fan or not items or record.get("input_divisor") or record.get("coefficient_field"):
        return None
    if any("ray_indices" not in c or any(c.get(k) for k in ("blown_up", "ambient_ideal", "coefficient_field"))
           for c in items):
        return None
    linear = tuple(map(tuple, record.get("certificate", {}).get("linear_map_to_input") or identity))
    rays = fan["rays"]
    if tuple(map(tuple, rays[:n])) != identity:
        return None
    if any(type(i) is not int or not 0 <= i < len(rays) for c in items for i in c["ray_indices"]):
        raise InvalidChart("A fan chart refers to an unknown ray.")
    return [(c, tuple(tuple(rays[i]) for i in c["ray_indices"])) for c in items], linear


def _same_locus(n, left, right):
    if set(left) == set(right):
        return True
    return (native_library.ideal_request(f"radical {n} {len(left)}", [*left, *right]) == ["1"] and
            native_library.ideal_request(f"radical {n} {len(right)}", [*right, *left]) == ["1"])


def _same_principal(coords, equations, expected):
    # Equal monic reduced bases characterize equality of ideals, including
    # multiplicities. Accept duplicate and nonminimal archived generators.
    if set(equations) == {expected}:
        return True
    def basis(polys):
        return native_library.affine_request(["wide_basis", " ".join(coords), "lex", *polys])
    return basis(equations) == basis([expected])


def _named_strict_factor(equations, coords, n, field, cone):
    """Principal ideal and monic Laurent equation in the common coefficient field."""
    from .fan_geometry import canonical_laurent_terms
    if not equations:
        raise InvalidChart("A strict component needs a nonzero equation.")
    if len(equations) == 1:
        generator = format_poly(parse_expression(equations[0], coords), coords)
    else:
        # With theta last, lex first compares geometric monomials. For a
        # principal ideal over K its monic generator and m(theta) form the
        # reduced triangular basis over Q. This also accepts redundant lists
        # in which none of the individual generators generates the ideal.
        ambient = [field["minimal_polynomial"]] if field else []
        basis = native_library.affine_request(["wide_basis", " ".join(coords), "lex", *ambient, *equations])
        rows = [sparse_row(row) for row in basis]
        factors = [row for row in rows if any(any(e[:n]) for e in row)]
        if len(factors) != 1 or len(rows) != len(ambient) + 1:
            raise InvalidChart("The saved strict component does not generate a principal divisor.")
        generator = format_poly(factors[0], coords)
        if field:
            from .point_atlas import _same_ideal
            if not _same_ideal(coords, ambient, equations, generator):
                raise InvalidChart("The marked strict equations differ from their principal divisor.")
    values = parse_expression(generator, coords)
    if field:
        row, = native_library.affine_request(["pullbacks", " ".join(coords), " ".join(coords), *coords,
                                              field["minimal_polynomial"], generator])
        values = sparse_row(row)
    groups = {}
    for exponent, c in values.items():
        groups.setdefault(exponent[:n], {})[exponent[n:]] = c
    if not groups:
        raise InvalidChart("The strict equation vanishes on the whole chart.")
    if len(groups) == 1:
        e, = groups
        if not any(e):
            raise InvalidChart("A strict branch cannot be a unit.")
        # The closure of a coordinate hypersurface has empty dense-torus
        # trace. Its global identity is the ray of its generic orbit.
        return generator, frozenset((cone[i], k) for i, k in enumerate(e) if k)
    from .binomial_kernel import run
    if run(f"common_coordinate {n} {len(groups)}", *groups) == ["1"]:
        raise InvalidChart("A named torus branch contains a separate coordinate divisor.")
    coefficients = canonical_laurent_terms(cone, {e:format_poly(v, coords[n:]) for e,v in groups.items()})
    return generator, coefficients


def _same_branch(a, b, field):
    if a == b:
        return True
    if not isinstance(a, dict) or not isinstance(b, dict) or a.keys() != b.keys():
        return False
    t = field["generator"] if field else "t"
    modulus = field["minimal_polynomial"] if field else "0"
    e = min(a)
    differences = [f"({a[k]})*({b[e]})-({b[k]})*({a[e]})" for k in a]
    return not any(native_library.affine_request(["pullbacks", t, t, t, modulus, *differences]))


def _prime_branch(generator, coords, field):
    if not field:
        rows = native_library.affine_request(["factor", " ".join(coords), generator])
        return len(rows) == 3 and rows[1] == "1"
    from . import native_polynomial as np
    from .candidate_field import FieldPolynomial, indexed_field
    from .field_factors import factors
    t = field["generator"]
    relation = parse_expression(field["minimal_polynomial"], (t,))
    # Factorization in the abstract quotient by m is independent of the
    # embedding; the actual common mark has already been validated by _field.
    domain, theta = indexed_field(tuple(relation.get((i,),0) for i in range(field["degree"],-1,-1)), 0)
    axes = tuple(map(np.Symbol, coords[:-1]))
    value = np.from_terms((*axes,theta), parse_expression(generator, coords))
    primes = factors(FieldPolynomial(value, axes, domain))
    return len(primes) == 1 and primes[0][1] == 1


def _chart_parts(charts, regions, marked, dimension):
    for (chart, cone), excluded in zip(charts, regions):
        coords = tuple(chart["coordinates"]) + marked
        if (len(coords) != dimension or len(set(coords)) != dimension or any(not isinstance(v, str) or
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", v) for v in coords)):
            raise InputError("Each regular chart needs distinct coordinates.")
        yield chart, cone, excluded, coords


def _owned_supports(monomials, excluded):
    from .quotient_monomials import encode_row
    rows = list(monomials)
    width = len(rows[0])
    rows.extend(tuple(int(i in face) for i in range(width)) for face in excluded)
    masks, = native_library.affine_request(["monomial_support_masks", str(width), *map(encode_row, rows)])
    return masks.strip()


def _adapted_equation(equation, term_count, base, linear):
    """The equation in the fan's common linear coordinates, and its number of terms."""
    n = len(base)
    units = tuple(_unit(j, n) for j in range(n))
    adapted = equation
    if linear != units:
        names = " ".join(base)
        images = [format_poly(dict(zip(units, row)), base) for row in linear]
        row, = native_library.affine_request(["pullbacks", names, names, *images, "0", adapted])
        terms = sparse_row(row)
        term_count = len(terms)
        adapted = format_poly(terms, base)
    if not term_count:
        raise InvalidChart("Expected a nonzero fan equation.")
    return adapted, term_count


_PER_CHART = object()


def _verify_fan_natively(charts, regions, equation, term_count, base, linear, axes, complex, bounded):
    """Verify a fresh embedded fan atlas in one native call.

    The kernel runs the per-chart steps of _verify_charts for every cone and
    reports each chart's first failing check. This loop raises the same
    exceptions in the same chart order. _PER_CHART selects the per-chart
    verifier, for example when the plan needs orbit proofs.
    """
    n = len(base)
    try:
        parts = list(_chart_parts(charts, regions, (), n))
    except InputError:
        return _PER_CHART  # reported at its chart by the per-chart verifier
    adapted, term_count = _adapted_equation(equation, term_count, base, linear)
    ids, rays, requests, exceptional = {}, [], [], []
    for chart, cone, excluded, coords in parts:
        cone_ids = []
        for ray in cone:
            if ray not in ids:
                ids[ray] = len(ids)
                rays.append([integer_text(x) for x in ray])
            cone_ids.append(ids[ray])
        positions = [j for j, ray in enumerate(cone) if ray not in axes]
        exceptional.append(positions)
        labels = [""] * n
        for j in positions:
            labels[j] = f"E{chart['ray_indices'][j]-n+1}"
        requests.append({"coordinates": list(coords), "cone": cone_ids, "labels": labels,
                         "excluded": [sorted(face) for face in excluded]})
    from .kernel_json import kernel_json
    try:
        answer = kernel_json("atlas_fan_verification", "verify", names=list(base), equation=adapted,
                             frame=[[integer_text(c) for c in row] for row in linear], rays=rays, charts=requests,
                             block=max(1, min(64, 16384 // term_count)), bounded=bounded, complex=complex)
    except InputError:
        return _PER_CHART
    failure, results = answer.get("failure"), answer["charts"]
    output = []
    for index, (chart, cone, excluded, coords) in enumerate(parts):
        if failure is not None and index == failure["chart"]:
            raise (native_library.Unavailable if failure["kind"] == "unavailable" else ArithmeticError)(
                failure["message"])
        if index >= len(results):
            raise ArithmeticError("The native fan verification returned incomplete chart data.")
        verified = results[index]
        if set(chart["map"]) != set(base):
            raise InvalidChart("The projection map differs from its monomial chart.")
        projection = tuple(verified["projection"])
        if any(actual != expected and _row(actual, coords) != _row(expected, coords)
               for actual, expected in zip((chart["map"][v] for v in base), projection)):
            raise InvalidChart("The projection map differs from its monomial chart.")
        status = verified["status"]
        if status == "label":
            raise InvalidChart("One exceptional ray must have a single global component label.")
        if status == "data":
            raise InvalidChart("The charts disagree about exceptional numerical data.")
        if status == "unavailable":
            raise native_library.Unavailable(verified["message"])
        if status == "unknown":
            return None
        if status == "defect":
            raise InvalidChart("Normal crossings fail on the owned origin fibre.")
        if status != "ok":
            raise ArithmeticError("Unknown native fan verification status.")
        components = [dict(id=f"E{chart['ray_indices'][j]-n+1}", multiplicity=integer_from_decimal(order),
                           nu=integer_from_decimal(nu), ideal=[coords[j]])
                      for j, (order, nu) in zip(exceptional[index], verified["orders"], strict=True)]
        if verified["strict"]:
            components.append(dict(id="strict", multiplicity=1, nu=1, ideal=[verified["strict_text"]]))
        output.append({"coordinates": list(coords), "ambient_ideal": [], "components": components,
                       "fibre": list(projection), "region": verified["region"], "removed": [],
                       "map": dict(zip(base, projection)), "torus": {"rays": cone, "linear_map": linear}})
    found = {"reason": None, "cells": None}
    if complex:
        plan = answer.get("plan")
        if plan is None:
            return None
        if "complex" not in plan:
            return _PER_CHART
        found.update(cells=plan["complex"]["cells"], dual_complex=plan["complex"])
    return output, found


def _fan_chart_rows(parts, equation, term_count, base, linear, *, boundary_terms=False):
    """Pull back the original equation through the common linear map, then through each cone."""
    names = " ".join(base)
    adapted, term_count = _adapted_equation(equation, term_count, base, linear)
    # Keep only a block of chart responses in memory, even for large equations.
    block_size = max(1, min(64, 16384 // term_count))
    with_projection = hasattr(native_library._library(), "atlas_fan_projection_abi")
    operation = "fan_charts_with_projection" if with_projection else "fan_charts"
    if boundary_terms:
        operation = "fan_charts_with_boundary"
    frame = [" ".join(map(integer_text, row)) for row in linear] if with_projection else []
    width = 4 + len(frame) + int(boundary_terms)
    for coords, group in groupby(parts, key=lambda part: part[3]):
        while block := list(islice(group, block_size)):
            request = [operation, names, " ".join(coords), adapted, *frame]
            for _, cone, excluded, _ in block:
                request.extend(" ".join(map(integer_text, ray)) for ray in cone)
                request.append(_owned_supports(zip(*cone), excluded))
            rows = native_library.affine_request(request)
            if len(rows) != width * len(block):
                raise ArithmeticError("The native fan operation returned incomplete chart data.")
            for i, part in enumerate(block):
                yield part, rows[width*i:width*(i+1)]


def _fan_orbit_strata(polynomial, base, linear, charts, cones, axes, saved):
    from .newton_repair import strata as orbit_strata
    # Rebuild the fan from the actual chart maps. In particular, saved
    # fan/strata summaries are not a source of geometry or Euler numbers.
    n = len(base)
    indices = {ray: i for i, ray in enumerate(linear)}
    if saved:
        labels = {}
        for chart, cone in charts:
            for c in chart["components"]:
                if c["id"] != "strict" and len(c["ideal"]) == 1 and c["ideal"][0] in chart["coordinates"]:
                    labels[cone[chart["coordinates"].index(c["ideal"][0])]] = c["id"]
    else:
        labels = {ray: f"E{i-n+1}" for chart, cone in charts for i,ray in zip(chart["ray_indices"], cone) if i >= n}
    for cone in cones:
        for ray in cone:
            indices.setdefault(ray, len(indices))
    if all(ray in labels for ray in indices if ray not in axes):
        fan = {"rays": list(indices), "cones": [tuple(indices[r] for r in cone) for cone in cones]}
        orbit_data = orbit_strata(polynomial, base, fan, names=[labels.get(ray) for ray in indices],
                                  certify_subfaces=True)
        if orbit_data and not orbit_data["repairs"]:
            return orbit_data["strata"]
    return None


def _verify_charts(record, polynomial, charts, linear, complex, intrinsic, integrals, bounded=False,
                   *, pullbacks_only=False, local_sections=False, verified_parent_partition=False):
    from .point_atlas import _field, _invertible, _same_ideal, monomial_matrix
    base = record["variables"]
    n = len(base)
    split = linear is None
    saved = "atlas" in record
    declared = _field([c for c,_ in charts], base, record.get("coefficient_field"), saved) if split else None
    field, source = declared if declared else (None, None)
    marked = (field["generator"],) if field else ()
    dimension = n + len(marked)
    ambient = [field["minimal_polynomial"]] if field else []
    modulus = ambient[0] if field else "0"
    if not split and (len(linear) != n or any(len(row) != n or any(type(c) is not int for c in row) for row in linear)
                      or not _determinant(linear)):
        raise InvalidChart("The common change of torus coordinates must be invertible.")
    axes = {_unit(i, n) for i in range(n)}
    cones = tuple(cone for _, cone in charts)
    regions = fan_cover(cones, n)
    if regions is None:
        raise InvalidChart("The monomial chart cones do not form a complete regular subdivision of the orthant.")
    internal_surface = n == 3 and all(ray in axes or min(ray) > 0 for cone in cones for ray in cone)
    defer_complex = False
    sections, surface_charts = [], []
    embedded = True
    branch_equations = {}
    named = saved and any(re.fullmatch(r"S[0-9]+", c["id"]) for chart,_ in charts for c in chart["components"])
    output = []
    data, contributions = {}, []
    by_ray = {}
    f = format_poly(polynomial, base)
    from .fan_centres import singular_ray_images
    exceptional_rays = tuple(dict.fromkeys(ray for cone in cones for ray in cone if ray not in axes))
    original_equation = f

    def check_centres(matrix):
        if singular_ray_images(original_equation, base, exceptional_rays, matrix, field) is None:
            raise InvalidChart("The fan modifies the smooth locus of the reduced hypersurface.")

    if not split:
        check_centres(linear)
    weight, reduced = 1, None
    weighted = saved and (record.get("input_divisor") or any(
        c["id"] == "strict" and c["multiplicity"] != 1 for chart,_ in charts for c in chart["components"]))
    if named or weighted:
        from .weighted_record import _factor, _product, verify_factors
        parts, _ = _factor(f, tuple(base))
        if record.get("input_divisor"):
            verify_factors(polynomial, base, record["input_divisor"]["factors"])
        if intrinsic and any(m > 1 for _,m in parts):
            raise InvalidChart("An intrinsic resolution requires a reduced hypersurface.")
        if not named:
            if len(parts) != 1:
                raise InvalidChart("The saved weighted strict divisor needs its distinct input factors.")
            reduced, weight = parts[0]
        else:
            f = _product(parts)
    base_header = " ".join((*base, *marked))
    direct_jet = not named
    # The surface fallback can require the full strict equation off its boundary.
    boundary_terms = complex and intrinsic and n != 3 and hasattr(
        native_library._library(), "atlas_fan_boundary_terms_abi")
    if (not split and not saved and not intrinsic and not integrals and direct_jet and reduced is None
            and all(hasattr(native_library._library(), abi) for abi in
                    ("atlas_fan_charts_abi", "atlas_fan_projection_abi", "atlas_fan_verification_abi"))):
        native = _verify_fan_natively(charts, regions, f, len(polynomial), base, linear, axes, complex, bounded)
        if native is not _PER_CHART:
            return native
    parts = _chart_parts(charts, regions, () if saved else marked, dimension)
    if not split and direct_jet and reduced is None and hasattr(native_library._library(), "atlas_fan_charts_abi"):
        verified = _fan_chart_rows(parts, f, len(polynomial), base, linear, boundary_terms=boundary_terms)
    else:
        verified = ((part, None) for part in parts)
    orbit_strata = None
    try_chart_orbits = (integrals and not intrinsic and not split and not named and not weighted and
                        record.get("engine") == "verified-monomial-modification" and
                        linear == tuple(_unit(j, n) for j in range(n)))
    for (chart, cone, excluded, coords), fan_rows in verified:
        monomials = tuple(zip(*cone))
        if set(chart["map"]) != set(base):
            raise InvalidChart("The projection map differs from its monomial chart.")
        if split:
            source_coords = (*coords[:n], source) if field and not saved else coords
            projection = tuple(format_poly(parse_expression(chart["map"][v], source_coords), coords) for v in base)
            matrix = monomial_matrix(projection, coords, monomials, field)
            if matrix is None or (linear is not None and matrix != linear):
                raise InvalidChart("The split fan maps disagree on their common linear coordinates.")
            if linear is None:
                if not _invertible(matrix, coords, field):
                    raise InvalidChart("The common linear coordinates of the fan are singular.")
                check_centres(matrix)
                linear = matrix
        else:
            projection = (fan_rows[4:4+n] if fan_rows is not None and len(fan_rows) > 4 else
                          tuple(format_poly(dict(zip(monomials, row)), coords) for row in linear))
            # Most producers already use the canonical text of the verified
            # projection. Compare other spellings in the polynomial ring.
            if any(actual != expected and _row(actual, coords) != _row(expected, coords)
                   for actual, expected in zip((chart["map"][v] for v in base), projection)):
                raise InvalidChart("The projection map differs from its monomial chart.")
        region = [format_poly({tuple(int(j in face) for j in range(n)): 1}, coords)
                  for face in excluded]
        exceptional = [j for j, ray in enumerate(cone) if ray not in axes]
        if fan_rows is not None:
            header, strict_text, strict_row, jet = fan_rows[:4]
            numbers = iter(map(integer_from_decimal, header.split()))
            has_strict = bool(next(numbers))
            orders = zip(numbers, numbers, strict=True)
            strict = sparse_row(fan_rows[-1] if boundary_terms else strict_row) if complex and intrinsic else None
        else:
            chart_header = ["chart", " ".join(coords), base_header, *projection, *marked, modulus]
            indices = " ".join(map(str, exceptional))
            jet_request = [_owned_supports(monomials, excluded)] if direct_jet else []
            rows = native_library.affine_request([*chart_header, f, indices, *(jet_request if reduced is None else ())])
            strict_rows = rows if reduced is None else native_library.affine_request(
                [*chart_header, reduced, indices, *jet_request])
            strict_row = strict_rows[-2] if direct_jet else strict_rows[-1]
            jet = strict_rows[-1] if direct_jet else None
            strict, strict_text = _chart_equation(strict_row, coords)
            has_strict = any(any(e[:n]) for e in strict)
            orders = (map(integer_from_decimal, row.split()) for row in rows[:len(exceptional)])
        ambient_rows = [_row(g, coords) for g in ambient]
        def same_ideal(gens, expected):
            return (_same_ideal(coords, ambient, gens, expected) if field
                    else _same_principal(coords, gens, expected))
        if saved:
            if set(ambient_rows) != {_row(g, coords) for g in chart["ambient_ideal"]}:
                raise InvalidChart("The fan chart has an inconsistent ambient equation.")
            saved_fibre = [*ambient_rows, *(_row(g, coords) for g in chart["fibre"])]
            true_fibre = [*ambient_rows, *(_row(g, coords) for g in projection)]
            saved_region = [*true_fibre, *(_row(g, coords) for g in chart["region"])]
            true_region = [*true_fibre, *(_row(g, coords) for g in region)]
            if not _same_locus(dimension, saved_fibre, true_fibre) or (
                    not verified_parent_partition and not _same_locus(dimension, saved_region, true_region)):
                raise InvalidChart("The fibre or partition differs from the chart projection and overlaps.")
            labels = {}
            candidate, branches = [], {}
            for comp in chart["components"]:
                label = comp["id"]
                if label == "strict":
                    candidate.append(comp)
                elif re.fullmatch(r"S[0-9]+", label):
                    weight = comp["multiplicity"]
                    if (label in branches or type(weight) is not int or weight < 1 or
                            type(comp["nu"]) is not int or comp["nu"] != 1):
                        raise InvalidChart("A named strict branch has inconsistent label or numerical data.")
                    g, signature = _named_strict_factor(comp["ideal"], coords, n, field, cone)
                    if isinstance(signature, frozenset) and (len(signature) != 1 or
                            any(ray not in axes or power != 1 for ray, power in signature)):
                        raise InvalidChart("A coordinate strict branch must be a reduced nonexceptional divisor.")
                    branches[label] = (g, weight)
                    if label not in branch_equations:
                        if not isinstance(signature, frozenset) and not _prime_branch(g, coords, field):
                            raise InvalidChart("A named strict equation must define a reduced prime divisor.")
                        if any(_same_branch(signature, sig, field) for sig,_ in branch_equations.values()):
                            raise InvalidChart("The same strict component carries two different labels.")
                        branch_equations[label] = (signature, weight)
                    previous, previous_weight = branch_equations[label]
                    if previous_weight != weight or not _same_branch(signature, previous, field):
                        raise InvalidChart("A named strict branch differs on the common torus.")
                else:
                    ideal = list(dict.fromkeys(comp["ideal"]))
                    at = coords.index(ideal[0]) if len(ideal) == 1 and ideal[0] in coords else next(
                        (j for j in exceptional if same_ideal(ideal, coords[j])), None)
                    if at is None:
                        raise InvalidChart("The exceptional ideal differs from the coordinate divisor.")
                    if at in labels:
                        raise InvalidChart("Duplicate boundary label in one chart.")
                    labels[at] = comp
            if named:
                if candidate:
                    raise InvalidChart("The charts disagree about the named strict branches.")
                product = "*".join(f"({g})^{w}" for g,w in branches.values()) or "1"
                if not same_ideal([product], strict_text):
                    raise InvalidChart("The named strict branches do not give the original pullback.")
            if set(labels) != set(exceptional) or (not named and
                    len(candidate) != int(has_strict)):
                raise InvalidChart("The monomial chart omits a boundary or strict component.")
            if candidate:
                given, = candidate
                if (type(given["multiplicity"]) is not int or type(given["nu"]) is not int or
                        (given["multiplicity"], given["nu"]) != (weight, 1)):
                    raise InvalidChart("The strict transform has incorrect numerical data.")
                if not same_ideal(given["ideal"], strict_text):
                    raise InvalidChart("The strict transform differs from the pullback on the chart.")
        elif split:
            labels = {}
            for comp in chart.get("ambient_components", chart.get("exceptional", ())):
                gens = comp.get("ideal") or [comp.get("equation")]
                if len(gens) != 1 or gens[0] not in coords[:n]:
                    raise InvalidChart("The exceptional ideal differs from the coordinate divisor.")
                j = coords.index(gens[0])
                if (j in labels or not isinstance(comp.get("id"), str) or not comp["id"] or
                        comp["id"] == "strict" or re.fullmatch(r"S[0-9]+", comp["id"])):
                    raise InvalidChart("Duplicate or invalid exceptional ray label.")
                labels[j] = comp
            if set(labels) != set(exceptional):
                raise InvalidChart("The split cone and its exceptional divisors disagree.")
        else:
            labels = {j: {"id": f"E{chart['ray_indices'][j]-n+1}"} for j in exceptional}
        components = []
        here = []
        for j, (order, nu) in zip(exceptional, orders, strict=True):
            c = labels[j]
            label = c["id"]
            if split and not saved and any(k in c and (type(c[k]) != int or c[k] != value)
                                           for k,value in (("multiplicity",order), ("nu",nu))):
                raise InvalidChart("The exceptional orders differ from the split chart pullback.")
            if (by_ray.setdefault(cone[j], label) != label or label in data and
                    data[label][2] != cone[j]):
                raise InvalidChart("One exceptional ray must have a single global component label.")
            if (saved and (type(c["multiplicity"]) is not int or type(c["nu"]) is not int or
                           (c["multiplicity"], c["nu"]) != (order, nu))):
                raise InvalidChart("The exceptional orders differ from the pullback or its Jacobian.")
            if data.setdefault(label, (order, nu, cone[j])) != (order, nu, cone[j]):
                raise InvalidChart("The charts disagree about exceptional numerical data.")
            components.append(dict(id=label, multiplicity=order, nu=nu, ideal=[coords[j]]))
            here.append(label)
        if named:
            components.extend(dict(id=label, multiplicity=w, nu=1, ideal=[g])
                              for label,(g,w) in branches.items())
        if pullbacks_only:
            # The parent can be singular at a centre that certified children
            # replace. Its projection, divisor orders and strict equation must
            # still agree with the original polynomial before the repair.
            continue
        if named and intrinsic:
            # The general intrinsic gluer handles the several surfaces.
            # Finish validating maps and branch identities on every chart.
            defer_complex = True
            continue
        if complex and intrinsic:
            if field and any(e[-1] for e in strict):
                defer_complex = True  # section_complex requires rational Laurent coefficients
            if defer_complex:
                continue  # still validate the remaining charts before using general gluing
            terms = tuple(sorted((e[:n], c) for e,c in strict.items()))
            details = cone, dict(zip(exceptional, here)), terms
            sections.append(details)
            if n == 3:
                # After removing the marked parameter these equations are
                # defined over Q, on each of the identical geometric sheets.
                fibre = [encode_sparse_row({e: 1}) for e in monomials]
                partition = [_row(g, coords[:n]) for g in region]
                surface_charts.append((*details, tuple(dict.fromkeys([*fibre, *partition]))))
        elif complex:
            sections.append((cone, dict(zip(exceptional, here)), ()))
        if has_strict and not named:
            components.append(dict(id="strict", multiplicity=weight, nu=1, ideal=[strict_text]))
        ideals = [(strict_row if v["id"] == "strict" else _row(v["ideal"][0], coords),) for v in components]
        # Invertibility of the verified constant matrix gives (L*m)=(m).
        # Monomial fibre generators expose the coordinate restrictions to
        # the native crossings and Euler algorithms.
        owned = list(dict.fromkeys([*(encode_sparse_row({e + (0,)*len(marked): 1}) for e in monomials),
                                    *(_row(g, coords) for g in region)]))
        crossing_ideals = ideals
        if has_strict and not named:
            crossing_ideals = [*ideals[:-1], (jet,)]
        defects = crossings_rows(dimension, n, ambient_rows, crossing_ideals, owned, (), bounded=bounded)
        if defects is None:
            return None
        if defects:
            if not intrinsic:
                raise InvalidChart("Normal crossings fail on the owned origin fibre.")
            if has_strict and crossings_rows(dimension, n, ambient_rows, [crossing_ideals[-1]], owned, ()):
                raise InvalidChart("The strict transform is singular on the owned origin fibre.")
            embedded = False
        if integrals and embedded and orbit_strata is None:
            fibre = [*ambient_rows, *owned]
            strata = None
            if try_chart_orbits:
                if hasattr(native_library._library(), "atlas_monomial_strata_abi"):
                    try:
                        strata = strata_rows(dimension, fibre, ideals, (), monomial_only=True)
                    except native_library.Unavailable:
                        # The Laurent fan calculation also supports exponents outside the ideal ring.
                        pass
                if strata is None:
                    # One unresolved chart justifies trying the global Newton-orbit formula.
                    orbit_strata = _fan_orbit_strata(polynomial, base, linear, charts, cones, axes, saved)
                    try_chart_orbits = False
            if orbit_strata is None:
                if strata is None:
                    strata = strata_rows(dimension, fibre, ideals, ())
                degree = field["degree"] if field else 1
                for indices, chi in strata.items():
                    labels = frozenset(components[i]["id"] for i in indices)
                    contributions.append((labels, degree, chi))
        if not saved or local_sections:
            header = {"split_cone": cone} if split else {"torus": {"rays": cone, "linear_map": linear}}
            output.append({"coordinates": list(coords), "ambient_ideal": ambient, "components": components,
                       "fibre": list(projection), "region": region, "removed": [], "map": dict(zip(base, projection)),
                       **({"coefficient_field": field} if field else {}), **header})
    if named:
        for chart, cone in charts:
            expected = {k for k,(sig, _) in branch_equations.items() if isinstance(sig, dict) or
                        any(ray in cone for ray,_ in sig)}
            if {c["id"] for c in chart["components"] if c["id"] in branch_equations} != expected:
                raise InvalidChart("A fan chart has omitted a named strict branch.")
    if pullbacks_only:
        return True
    if defer_complex:
        return None
    found = {"reason": None, "cells": None}
    if integrals and embedded:
        numbers = {name: value[:2] for name, value in data.items()}
        if named:
            numbers.update({k:(m,1) for k,(_,m) in branch_equations.items()})
        else:
            numbers["strict"] = (weight, 1)
        try:
            strata = orbit_strata if orbit_strata is not None else strata_totals(contributions)
        except InconsistentSheets as exc:
            raise ArithmeticError("The conjugate fan strata disagree on their degree.") from exc
        rows, listed = records({"data": numbers, "strata": strata})
        found.update(numerical_data=rows, strata=listed)
    if complex:
        from .atlas_torus import section_complex
        local_coordinates = [chart["coordinates"] for chart, _ in charts] if local_sections else None
        options = dict(common_source=True, embedded_snc=embedded, local_coordinates=local_coordinates)
        boundary = section_complex(sections, n, intrinsic, defer_relative_coordinates=bool(surface_charts), **options)
        if boundary is None and surface_charts:
            from .fan_surface import surface_complex
            # Preserve the affine factor ordering of previously certified curve
            # ideals. Relative orbit closures can also certify nonsplit charts.
            boundary = surface_complex(surface_charts, local_coordinates=local_coordinates)
            if boundary is None and embedded:
                boundary = section_complex(sections, n, intrinsic, **options)
        if boundary is None or local_sections and "chart_sections" not in boundary:
            return None
        if (intrinsic and internal_surface and record.get("engine") == "native-toric"
                and linear == tuple(_unit(i, n) for i in range(n))):
            from .toric_backend import diagonal_annotations
            boundary = diagonal_annotations(polynomial, cones, by_ray, boundary)
        if local_sections:
            found["chart_sections"] = boundary.pop("chart_sections")
            if "root_families" in boundary:
                found["root_families"] = boundary.pop("root_families")
        found.update(cells=boundary["cells"], dual_complex=boundary)
    return (record["atlas"] if saved and not local_sections else output), found


def fan_outputs(record, polynomial, *, complex=True, intrinsic=False, integrals=True, bounded=False,
                local_sections=False):
    if record.get("engine") == "hyperplane-arrangement-fan" or any(
            "ray_indices" in c and any("ideal" in v for v in c.get("strict_transform_components", ()))
            for c in record.get("charts", ())):
        from .newton_trials import restore_hyperplane_model
        record = restore_hyperplane_model(record, polynomial)
    n = len(record["variables"])
    if not 2 <= n <= 8:
        return None
    translation = record.get("certificate", {}).get("coordinate_translation")
    if translation is None:
        translation = next((c["torus"]["coordinate_translation"] for c in record.get("atlas", ())
                            if "coordinate_translation" in c.get("torus", {})), None)
    if translation is not None:
        from .quadratic_fan import translated_outputs
        return translated_outputs(record, polynomial, translation, fan_outputs,
                                  dict(complex=complex, intrinsic=intrinsic, integrals=integrals, bounded=bounded,
                                       local_sections=local_sections))
    if record.get("input_divisor") and not record.get("atlas"):
        return None
    charts = _charts(record, n)
    if charts is None:
        return None
    try:
        # New intrinsic trial records expose geometric chart ideals separately
        # from the owned ambient atlas. Both presentations refer to the same
        # verified cones, even when a replay only requests Euler integrals.
        displayed = record.get("charts", ())
        if (record.get("mode") == "intrinsic" and displayed and ("atlas" in record or "polynomial" in record)
                and (record.get("engine") == "verified-monomial-modification" or
                     any("ray_indices" in c and "ambient_components" in c for c in displayed))):
            from .newton_trials import verify_intrinsic_presentation
            verified = _verify_charts(record, polynomial, *charts, True, True, integrals and intrinsic, bounded,
                                      local_sections=True)
            if verified is None:
                raise InvalidChart("The displayed intrinsic fan sections could not be certified.")
            atlas, output = verified
            verify_intrinsic_presentation(record, atlas, output)
            if intrinsic:
                if not local_sections:
                    output.pop("chart_sections")
                    output.pop("root_families", None)
                if not complex:
                    output.pop("dual_complex")
                    output["cells"] = None
                return record.get("atlas", atlas), output
        result = _verify_charts(record, polynomial, *charts, complex, intrinsic, integrals, bounded,
                                local_sections=local_sections)
    except native_library.Unavailable:
        return None
    return result


def verified_parent_pullbacks(record, polynomial, charts):
    """Check the complete parent fan and its divisors before local chart verification."""
    from .atlas import to_json
    from .repair_parent_cover import _cone, certified_parent_cover

    n = len(record["variables"])
    if not 2 <= n <= 8:
        return False
    parents = [c for c in charts if not c.repair_projection]
    if not parents or any(c.removed for c in parents) and len(parents) == len(charts):
        return False  # No replacement for the deleted points has been certified.
    saved = to_json(parents)
    for c in saved:
        c["removed"] = []  # The repair verifier checked the children over these centres.
    if not any(c.torus or c.split_cone for c in parents):
        # Recover a common monomial frame from the actual coordinate maps.
        # The split-fan verifier will also check that its linear change is
        # invertible and identical on every open.
        for chart, item in zip(parents, saved):
            frame = _cone(chart, record["variables"])
            if frame is None:
                return False
            item["split_cone"] = frame[1]
    parent_record = dict(record, atlas=saved)
    family = _charts(parent_record, n)
    if family is None:
        return False
    # A repaired record can reorder the parent charts while retaining its
    # original partition. The cover proof checks each orbit has exactly one
    # owner using the actual region equations in that record.
    checked_partition = len(parents) < len(charts) and certified_parent_cover(parents)
    return _verify_charts(parent_record, polynomial, *family, complex=False, intrinsic=False, integrals=False,
                          pullbacks_only=True, verified_parent_partition=checked_partition)
