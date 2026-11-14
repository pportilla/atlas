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
from collections import Counter
from functools import lru_cache
from itertools import combinations, combinations_with_replacement
import re

from . import native_library
from .chart_kernel import crossings_rows, strata_rows
from .fan_geometry import _determinant, complete_fan, fan_cover, split_cone
from .polynomial import InputError, format_poly, parse_expression, sparse_row, rational_text
from .strata import InvalidChart, records


def _unit(i, n):
    return tuple(int(i == j) for j in range(n))


def _polynomial_row(terms):
    return ";".join(rational_text(c) + " " + " ".join(map(str, e)) for e, c in terms.items())


@lru_cache(maxsize=16384)
def _row(text, coords):
    return _polynomial_row(parse_expression(text, coords))


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


def _crossing_strict(coords, equation, monomials, excluded):
    """Compatibility with kernels predating the direct sparse crossing jet."""
    # The radical of the owned monomial ideal J is generated by the
    # squarefree supports of its generators. On V(J), every element of
    # (rad J)^2 and its differential vanish. The generators of this
    # square are monomials, so ordinary FLINT division gives the jet.
    supports = {frozenset(i for i, k in enumerate(e) if k) for e in monomials}
    supports.update(map(frozenset, excluded))
    minimal = (s for s in supports if not any(t < s for t in supports))
    generators = ["*".join(coords[i] for i in sorted(s)) or "1" for s in minimal]
    square = [f"({a})*({b})" for a, b in combinations_with_replacement(generators, 2)]
    return native_library.affine_request(["wide_division", " ".join(coords), "lex", equation, *square])[-1]


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
    if any(min(e[j] for e in groups) for j in range(n)):
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


def _verify_charts(record, polynomial, charts, linear, complex, intrinsic, integrals):
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
    data, strata = {}, Counter()
    names, by_ray, faces = [], {}, set()
    f = format_poly(polynomial, base)
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
    direct_jet = not named and hasattr(native_library._library(), "atlas_crossing_jet_abi")
    for (chart, cone), excluded in zip(charts, regions):
        coords = tuple(chart["coordinates"]) + (() if saved else marked)
        if (len(coords) != dimension or len(set(coords)) != dimension or any(not isinstance(v, str) or
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", v) for v in coords)):
            raise InputError("Each regular chart needs distinct coordinates.")
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
                linear = matrix
        else:
            projection = tuple(format_poly(dict(zip(monomials, row)), coords) for row in linear)
            if any(_row(chart["map"][v], coords) != _row(g, coords) for v, g in zip(base, projection)):
                raise InvalidChart("The projection map differs from its monomial chart.")
        region = [format_poly({tuple(int(j in face) for j in range(n)): 1}, coords)
                  for face in excluded]
        exceptional = [j for j, ray in enumerate(cone) if ray not in axes]
        chart_header = ["chart", " ".join(coords), base_header, *projection, *marked, modulus]
        indices = " ".join(map(str, exceptional))
        jet_request = []
        if direct_jet:
            supports = [sum(1 << i for i, k in enumerate(e) if k) for e in monomials]
            supports.extend(sum(1 << j for j in face) for face in excluded)
            jet_request = [" ".join(map(str, supports))]
        rows = native_library.affine_request([*chart_header, f, indices, *(jet_request if reduced is None else ())])
        strict_rows = rows if reduced is None else native_library.affine_request(
            [*chart_header, reduced, indices, *jet_request])
        strict_row = strict_rows[-2] if direct_jet else strict_rows[-1]
        strict, strict_text = _chart_equation(strict_row, coords)
        has_strict = any(any(e[:n]) for e in strict)
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
            if not _same_locus(dimension, saved_fibre, true_fibre) or not _same_locus(dimension, saved_region, true_region):
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
        for pos, j in enumerate(exceptional):
            order, nu = map(int, rows[pos].split())
            c = labels[j]
            label = c["id"]
            if split and not saved and any(k in c and (type(c[k]) != int or c[k] != value)
                                           for k,value in (("multiplicity",order), ("nu",nu))):
                raise InvalidChart("The exceptional orders differ from the split chart pullback.")
            if (by_ray.setdefault(cone[j], label) != label or label in names and
                    data[label][2] != cone[j]):
                raise InvalidChart("One exceptional ray must have a single global component label.")
            if (saved and (type(c["multiplicity"]) is not int or type(c["nu"]) is not int or
                           (c["multiplicity"], c["nu"]) != (order, nu))):
                raise InvalidChart("The exceptional orders differ from the pullback or its Jacobian.")
            if label not in names:
                names.append(label)
            if data.setdefault(label, (order, nu, cone[j])) != (order, nu, cone[j]):
                raise InvalidChart("The charts disagree about exceptional numerical data.")
            components.append(dict(id=label, multiplicity=order, nu=nu, ideal=[coords[j]]))
            here.append(label)
        if named:
            components.extend(dict(id=label, multiplicity=w, nu=1, ideal=[g])
                              for label,(g,w) in branches.items())
            if intrinsic:
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
            if internal_surface:
                # After removing the marked parameter these equations are
                # defined over Q, on each of the identical geometric sheets.
                fibre = [_polynomial_row({e: 1}) for e in monomials]
                partition = [_row(g, coords[:n]) for g in region]
                surface_charts.append((*details, tuple(dict.fromkeys([*fibre, *partition]))))
        elif complex:
            for size in range(1, len(here) + 1):
                faces.update(map(frozenset, combinations(here, size)))
        if has_strict and not named:
            components.append(dict(id="strict", multiplicity=weight, nu=1, ideal=[strict_text]))
        ideals = [(strict_row if v["id"] == "strict" else _row(v["ideal"][0], coords),) for v in components]
        # Invertibility of the verified constant matrix gives (L*m)=(m).
        # Monomial fibre generators expose the coordinate restrictions to
        # the native crossings and Euler algorithms.
        owned = list(dict.fromkeys([*(_polynomial_row({e + (0,)*len(marked): 1}) for e in monomials),
                                    *(_row(g, coords) for g in region)]))
        crossing_ideals = ideals
        if has_strict and not named:
            jet = strict_rows[-1] if direct_jet else _crossing_strict(coords, strict_text, monomials, excluded)
            crossing_ideals = [*ideals[:-1], (jet,)]
        if crossings_rows(dimension, n, ambient_rows, crossing_ideals, owned, ()):
            if not intrinsic:
                raise InvalidChart("Normal crossings fail on the owned origin fibre.")
            if has_strict and crossings_rows(dimension, n, ambient_rows, [crossing_ideals[-1]], owned, ()):
                raise InvalidChart("The strict transform is singular on the owned origin fibre.")
            embedded = False
        if integrals and embedded:
            for indices, chi in strata_rows(dimension, [*ambient_rows, *owned], ideals, ()).items():
                if field:
                    chi, remainder = divmod(chi, field["degree"])
                    if remainder:
                        raise ArithmeticError("The conjugate fan strata disagree on their degree.")
                strata[frozenset(components[i]["id"] for i in indices)] += chi
        if not saved:
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
    if defer_complex:
        return None
    found = {"reason": None, "cells": None}
    if integrals and embedded:
        numbers = {name: value[:2] for name, value in data.items()}
        if named:
            numbers.update({k:(m,1) for k,(_,m) in branch_equations.items()})
        else:
            numbers["strict"] = (weight, 1)
        rows, listed = records({"data": numbers, "strata": {key: chi for key, chi in strata.items() if chi}})
        found.update(numerical_data=rows, strata=listed)
    if complex and intrinsic:
        from .atlas_torus import section_complex
        boundary = section_complex(sections, n, True, common_source=True)
        if boundary is None and surface_charts:
            from .fan_surface import surface_complex
            boundary = surface_complex(surface_charts)
        if boundary is None:
            return None
        found.update(cells=boundary["cells"], dual_complex=boundary)
    elif complex:
        vertices = [dict(id=name, dimension=n-1) for name in names]
        def cell_id(face):
            return face[0] if len(face) == 1 else "C" + "-".join(face) + ".1"
        cells = []
        for size in range(2, n+1):
            subsets = [sorted(face, key=names.index) for face in faces if len(face) == size]
            for face in sorted(subsets, key=lambda f: list(map(names.index, f))):
                cells.append(dict(id=cell_id(face), dimension=size-1, vertices=face,
                                  faces=[cell_id(sub) for sub in combinations(face, size-1)]))
        found.update(cells=cells, dual_complex={"status": "complete", "vertices": vertices, "cells": cells,
                     "convention": "Ambient exceptional divisor in the germ over the origin over Qbar."})
    return (record["atlas"] if saved else output), found


def fan_outputs(record, polynomial, *, complex=True, intrinsic=False, integrals=True):
    n = len(record["variables"])
    if not 2 <= n <= 8 or native_library._MODE == "sympy" or (
            record.get("input_divisor") and not record.get("atlas")):
        return None
    charts = _charts(record, n)
    if charts is None:
        return None
    try:
        result = _verify_charts(record, polynomial, *charts, complex, intrinsic, integrals)
    except native_library.Unavailable:
        return None
    if result is not None and native_library.CHECK:
        from . import atlas
        all_charts = atlas.from_record(record, polynomial)
        actual = result[1]
        if integrals and "strata" not in actual:
            from .local_blowups import GiveUp
            try:
                atlas.outputs(all_charts, n, complex=False, intrinsic=intrinsic, integrals=True)
            except GiveUp:
                pass
            else:
                raise AssertionError("The general verifier accepted the embedded normal crossings.")
        expected = atlas.outputs(all_charts, n, complex=complex, intrinsic=intrinsic, integrals="strata" in actual)
        if "strata" in actual and (expected["numerical_data"] != actual["numerical_data"] or expected["strata"] != actual["strata"]):
            raise AssertionError("The monomial and general chart verifiers disagree about the strata.")
        if complex:
            def faces(g):
                return ({v["id"] for v in g["vertices"]},
                        {frozenset(c["vertices"]) for c in g["cells"]})
            if ((intrinsic and expected["dual_complex"] != actual["dual_complex"]) or
                    expected["dual_complex"]["status"] != "complete" or
                    faces(expected["dual_complex"]) != faces(actual["dual_complex"])):
                raise AssertionError("The fan incidence differs from the general chart computation.")
    return result
