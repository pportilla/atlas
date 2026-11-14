"""Verify the projective charts of a blowup of the origin from their equations.

In chart i, the split coordinates on the base are u_i and u_i*u_j. Thus
the coefficients of the projection on these n distinct monomials recover
its matrix before blowing up. One common nonsingular matrix is required;
over K=Q(theta) equality and its determinant are tested modulo the defining
irreducible polynomial. This identifies the charts with all the standard
opens of P^(n-1). Chart i owns the directions whose earlier coordinates
vanish. On a marked separable coefficient cover, conjugate sheets have
the same Euler integrals. Restricting to one divides those counts by [K:Q].

Transversality on all the owned opens proves that the section of the
exceptional projective space is a smooth hypersurface there. For n>=3 any
two positive-degree projective hypersurfaces meet over Qbar, so a nonempty
smooth hypersurface in P^(n-1) is geometrically irreducible. Both the order
and the section equations come from the verified chart pullbacks.
"""
import re
from . import native_library
from .chart_kernel import crossings_rows, strata_rows
from .fan_geometry import split_cone
from .monomial_atlas import _row, _same_locus, _unit
from .polynomial import InputError, format_poly, parse_expression, sparse_row
from .strata import InvalidChart, records


def _field(charts, base, raw, saved):
    if saved:
        data = charts[0].get("coefficient_field")
        if any(c.get("coefficient_field") != data for c in charts):
            raise InvalidChart("The chart coefficient fields disagree.")
        if not data:
            return None
        name = data["generator"]
        if name in base or any(c["coordinates"][-1] != name for c in charts):
            raise InputError("The coefficient generator must be a separate chart coordinate.")
        degree, declared = data["degree"], data["minimal_polynomial"]
        embedding = data["embedding"]
        source = name
    else:
        if not raw or not raw.get("generator"):
            return None
        source = raw["generator"]
        name = "atlas_field"
        while name in base or any(name in c["coordinates"] for c in charts):
            name += "_"
        declared, degree, embedding = raw["minimal_polynomial"], 2, "root 1 (FLINT order)"
    if not isinstance(source, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", source):
        raise InputError("Invalid coefficient parameter.")
    terms = parse_expression(declared, (source,))
    if type(degree) is not int or degree < 1 or max((e[0] for e in terms), default=-1) != degree:
        raise InputError("The coefficient relation has the wrong degree.")
    relation = format_poly(terms, (name,))
    factors = native_library.affine_request(["factor", name, relation])
    if len(factors) != 3 or factors[1] != "1":
        raise InputError("The marked coefficient relation must be irreducible.")
    if saved:
        indexed = re.fullmatch(r"root ([1-9][0-9]*) \(FLINT order\)", embedding) if isinstance(embedding, str) else None
        if indexed:
            if int(indexed[1]) > degree:
                raise InvalidChart("The saved coefficient embedding is outside the roots of its relation.")
        else:
            from .marked_field import description
            if not isinstance(embedding, str):
                raise InvalidChart("The saved fan needs an exact coefficient embedding.")
            try:
                coefficients, _ = description(embedding)
            except (InputError, ValueError) as exc:
                raise InvalidChart("The saved fan has an invalid coefficient embedding.") from exc
            minimal = format_poly({(i,):c for i,c in enumerate(reversed(coefficients)) if c}, (name,))
            if not _same_ideal((name,), [], [relation], minimal):
                raise InvalidChart("The marked root is not a root of the fan's coefficient relation.")
    return dict(generator=name, minimal_polynomial=relation, degree=degree, embedding=embedding), source


def monomial_matrix(mapping, coords, monomials, field):
    """Recover a coefficient polynomial in theta for each base entry and monomial."""
    n = len(monomials)
    indices = {monomial: j for j,monomial in enumerate(monomials)}
    rows = []
    for f in mapping:
        groups = {}
        for powers, c in parse_expression(f, coords).items():
            groups.setdefault(powers[:n], {})[powers[n:]] = c
        coefficients = [format_poly(g, coords[n:]) for g in groups.values()]
        if field:
            parameter = field["generator"]
            reduced = native_library.affine_request(["pullbacks", parameter, parameter, parameter,
                                                      field["minimal_polynomial"], *coefficients])
            coefficients = [format_poly(sparse_row(r), (parameter,)) for r in reduced]
        entries = ["0"] * n
        for exp, coefficient in zip(groups, coefficients):
            if exp in indices:
                entries[indices[exp]] = coefficient
            elif coefficient != "0":
                return None
        rows.append(tuple(entries))
    return tuple(rows)


def _invertible(matrix, coords, field):
    n = len(matrix)
    forms = ["+".join(f"({c})*{v}" for c, v in zip(row, coords[:n])) for row in matrix]
    determinant, = native_library.affine_request(["minor", " ".join(coords), " ".join(map(str, range(n))), *forms])
    if not determinant:
        return False
    if field:
        param = field["generator"]
        value = format_poly(sparse_row(determinant), coords)
        determinant, = native_library.affine_request(["pullbacks", param, param, param, field["minimal_polynomial"], value])
    return bool(determinant)


def _same_ideal(coords, ambient, given, expected):
    if len(given) == 1 and _row(given[0], coords) == _row(expected, coords):
        return True
    # Different generating sets, or different nonzero field scalars, are
    # valid if the ideals agree on the marked affine chart. Reduced lex
    # bases including its ambient equation give an exact equality test.
    def basis(gens):
        rows = [_row(g, coords) for g in (*ambient, *gens)]
        # Saturating by 1 leaves an ideal unchanged and returns its reduced
        # monic lex basis. This operation also exists in archived kernels.
        one = _row("1", coords)
        return native_library.ideal_request(f"saturate {len(coords)} {len(rows)}", [*rows, one])
    return basis(given) == basis((expected,))


def _calculate(record, polynomial, charts, *, saved, intrinsic, integrals, complex):
    base = tuple(record["variables"])
    n = len(base)
    fields = _field(charts, base, record.get("coefficient_field"), saved)
    field, source = fields if fields else (None, None)
    degree = field["degree"] if field else 1
    modulus = field["minimal_polynomial"] if field else "0"
    count_vars = n + int(bool(field))
    marked = (field["generator"],) if field else ()
    results, strata, numbers = [], {}, {}
    common = None
    for i, item in enumerate(charts):
        coords = tuple(item["coordinates"]) if saved else (*item["coordinates"], *marked)
        if (len(coords) != count_vars or len(set(coords)) != len(coords)
                or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", v) for v in coords)):
            return None
        if set(item["map"]) != set(base) or item.get("removed"):
            return None
        ambient = [modulus] if field else []
        if saved and set(_row(g, coords) for g in item["ambient_ideal"]) != set(_row(g, coords) for g in ambient):
            return None
        def map_entry(v):
            source_coords = (*coords[:n], source) if field and not saved else coords
            return format_poly(parse_expression(item["map"][v], source_coords), coords)
        mapping = [map_entry(v) for v in base]
        monomials = tuple(tuple(int(k == i) + int(j != i and k == j) for k in range(n)) for j in range(n))
        matrix = monomial_matrix(mapping, coords, monomials, field)
        if matrix is None:
            return None
        if common is None:
            if not _invertible(matrix, coords, field):
                raise InvalidChart("The point blowup has a singular base coordinate change.")
            common = matrix
        elif common != matrix:
            raise InvalidChart("The point blowup charts have different base coordinate changes.")
        # The first nonzero projective direction gives the disjoint partition.
        radial, region = coords[i], list(coords[:i])
        output = native_library.affine_request(["chart", " ".join(coords), " ".join((*base,*marked)),
                                                *mapping, *marked, modulus, format_poly(polynomial, base), str(i)])
        order, nu = map(int, output[0].split())
        strict = format_poly(sparse_row(output[1]), coords)
        if order < 1 or nu != n:
            raise InvalidChart("The point blowup has incorrect exceptional data.")
        if saved:
            comps = item["components"]
            by_label = {c["id"]: c for c in comps}
            if len(comps) != 2 or set(by_label) != {"E1","strict"}:
                raise InvalidChart("The point blowup must carry its exceptional and strict components.")
            for label, pair, equation in (("E1",(order,nu),radial),("strict",(1,1),strict)):
                c = by_label[label]
                if (type(c["multiplicity"]) != int or type(c["nu"]) != int
                        or (c["multiplicity"],c["nu"]) != pair):
                    raise InvalidChart("The saved component orders differ from the pullback.")
                if not _same_ideal(coords, ambient, c["ideal"], equation):
                    raise InvalidChart("The saved component ideal differs from its verified pullback.")
            base_rows = [_row(g, coords) for g in ambient]
            true = [*base_rows, *(_row(g, coords) for g in mapping)]
            if (not _same_locus(count_vars, true, [*base_rows, *(_row(g, coords) for g in item["fibre"])])
                    or not _same_locus(count_vars, [*true, *(_row(g, coords) for g in region)],
                                       [*true, *(_row(g, coords) for g in item["region"])])):
                raise InvalidChart("The saved fibre or region differs from the point blowup partition.")
        else:
            boundary = item.get("ambient_components", item.get("exceptional", ()))
            if (len(boundary) != 1 or boundary[0].get("id") != "E1"
                    or (boundary[0].get("ideal") or [boundary[0].get("equation")]) != [radial]):
                return None
        components = [dict(id="E1", multiplicity=order, nu=nu, ideal=[radial]),
                      dict(id="strict", multiplicity=1, nu=1, ideal=[strict])]
        if numbers.setdefault("E1", (order, nu)) != (order, nu):
            raise InvalidChart("The radial orders disagree between point blowup charts.")
        numbers["strict"] = (1,1)
        ambient_rows = [_row(g, coords) for g in ambient]
        ideals = [(_row(c["ideal"][0], coords),) for c in components]
        owned = [_row(g, coords) for g in (radial, *region)]
        if crossings_rows(count_vars, n, ambient_rows, ideals, owned, ()):
            if intrinsic:
                return None
            raise InvalidChart("Normal crossings fail on the owned point blowup chart.")
        if integrals:
            for indices, value in strata_rows(count_vars, [*ambient_rows, *owned], ideals, ()).items():
                chi, rest = divmod(value, degree)
                if rest:
                    raise ArithmeticError("The conjugate chart strata disagree on their degree.")
                key = frozenset(components[j]["id"] for j in indices)
                strata[key] = strata.get(key, 0) + chi
        if not saved:
            results.append(dict(coordinates=list(coords), ambient_ideal=ambient, components=components,
                                fibre=mapping, region=region, removed=[], map=dict(zip(base, mapping)),
                                **({"split_cone": split_cone(item, n)} if "split_cone" in item else {}),
                                **({"coefficient_field": field} if field else {})))
    answer = {"cells": None, "reason": None}
    if complex:
        # The exceptional is P^(n-1), and the hyperplane-section argument in
        # the module docstring gives one intrinsic component for n >= 3.
        boundary = {"status": "complete", "vertices": [{"id": "E1", "dimension": n-1-int(intrinsic)}], "cells": [],
                    "convention": ("Exceptional divisor on the smooth strict transform" if intrinsic else
                                   "Ambient exceptional divisor") + " in the germ over the origin over Qbar."}
        answer.update(cells=[], dual_complex=boundary)
    if integrals:
        rows, listed = records(dict(data=numbers, strata={k:chi for k,chi in strata.items() if chi}))
        answer.update(numerical_data=rows, strata=listed)
    return charts if saved else results, answer


def point_outputs(record, polynomial, *, complex=True, intrinsic=False, integrals=True):
    n = len(record["variables"])
    if native_library._MODE == "sympy" or n < 3 or n > 8 or record.get("input_divisor"):
        return None
    saved = bool(record.get("atlas"))
    charts = record["atlas"] if saved else record.get("charts")
    if not charts or len(charts) != n:
        return None
    any_split = any("split_cone" in c for c in charts)
    if any_split:
        diagonal = (1,) * n
        for i,c in enumerate(charts):
            expected = tuple(diagonal if i == j else _unit(j,n) for j in range(n))
            if split_cone(c, n) != expected or c.get("torus"):
                return None  # the full fan verifier checks the supplied cones
    if saved:
        # Recognize the presentation by its individual exceptional ideals,
        # without depending on a producer name. Other graph charts can have
        # three coordinate opens and unrelated fibre maps.
        for i, c in enumerate(charts):
            coords = c["coordinates"]
            if (c.get("torus") or len(coords) != n + int(bool(c.get("coefficient_field")))
                    or not any(v["id"] == "E1" and coords[i] in v["ideal"] for v in c["components"])):
                return None
            if any(re.fullmatch(r"S[0-9]+", v["id"]) or v["id"] == "strict" and v["multiplicity"] != 1
                   for v in c["components"]):
                return None  # the fan verifier checks branch products and their identifications
    else:
        for i,c in enumerate(charts):
            if c.get("pivot") != record["variables"][i] and not any_split:
                return None
            if c.get("ambient_ideal") or c.get("blown_up") or c.get("exclude_points"):
                return None
    try:
        result = _calculate(record, polynomial, charts, saved=saved, intrinsic=intrinsic,
                            integrals=integrals, complex=complex)
    except native_library.Unavailable:
        return None
    if result and native_library.CHECK:
        from . import atlas
        general = atlas.from_record(record, polynomial)
        expected = atlas.outputs(general, n, complex=complex, intrinsic=intrinsic, integrals=integrals)
        if result[1] != expected or (not saved and result[0] != atlas.to_json(general)):
            raise AssertionError("The point blowup and general chart calculations disagree.")
    return result
