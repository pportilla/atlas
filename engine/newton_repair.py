"""Strata of a regular Newton modification over the origin, with blowups where normal crossings fail.

On the torus orbit of a cone the strict transform is cut out by the face
polynomial of that cone, in the coordinates of the orbit. It is smooth and
transverse to the toric strata there exactly where that hypersurface of the
orbit is smooth. For a degenerate face it has singular points. When they are
finitely many, the total transform has normal crossings on the fibre over the
origin except at those points, and they are blown up until it has
(engine/local_blowups.py).

The result is the data that A'Campo's formula and the topological zeta
function of Denef and Loeser need: for every component of the total
transform, the order N of the function and the number nu (one more than the
order of the pulled-back volume form), and for every set of components the
Euler characteristic of the points over the origin lying on exactly those.

  - A toric divisor of the ray a has N = min <a, exponents> and nu = sum a_i.
  - An orbit over the origin lies on the divisors of its exceptional rays.
    Its part off the strict transform has chi(orbit) - chi(cut), and its part
    on the strict transform chi(cut), where chi(orbit) is 1 for a point and 0
    otherwise. When the face of the orbit and all the faces of that face
    are nondegenerate, chi(cut) is (-1)^(d-1) times the normalized volume
    of the face (Danilov and Khovanskii; the hypothesis on the smaller
    faces is needed: it says that the cut is transverse to the boundary of
    the orbit). Otherwise chi(cut) is computed exactly, and the singular
    points of the cut are left out.
  - Over the singular points the strata are those of the blowups made there.

That the strata over the blown-up points may be added to the others follows
from the theorem of Gusein-Zade, Luengo and Melle-Hernandez on partial
resolutions (Comment. Math. Helv. 72 (1997)) for the monodromy zeta function;
for the topological zeta function the blown-up space is an embedded
resolution over a neighbourhood of the origin and the formula is applied to it.
"""
from dataclasses import replace

from .chart_expressions import chart_expressions
from .native_polynomial import native_ring
from .binomial_kernel import _integer, _newton, minimum_faces, separating_forms
from .chart_batch import strict_valuations
from .chart_kernel import strata_totals
from .polynomial import parse_expression

from . import kernel
from .affine_kernel import _polynomial, frame_derivative, frame_linear
from .blowup_ideal import saturate
from .groebner import CHECK, groebner
from .local_blowups import GiveUp, _dimension, _point_components, _radical_points, acampo_over
from .monodromy_toric import _volume
from .newton_backend import _valuations
from .newton_faces import certify_face
from .strata import records


def _text(expr):
    return str(expr).replace("**", "^")


def _euler(coords, equations, excluded=()):
    native = kernel.strata(coords, equations, ((),), excluded)
    if native is not None:
        return native.get((0,), 0)
    from .constructible_euler import euler_characteristic
    return euler_characteristic(coords, equations, excluded)



def _point_count(free, points):
    """The compactly supported Euler characteristic counts the geometric points."""
    return _euler(free, points)


def _rational(coords, free, points, count):
    """Conjugate points made rational. For a set of conjugate points that are
    not rational, a linear form with distinct values at them is adjoined as
    a variable theta: on the points it satisfies its minimal polynomial m,
    and the coordinates are polynomials in it. Over the finite cover
    m(theta) = 0 of the chart, the set of the points (p, theta(p))
    is cut out by equations linear in the coordinates, one point on each
    sheet, and what is done over it on one sheet is conjugate to the others.

    Returns the variables and (m, coordinate - polynomial in theta, ...), or
    the coordinates and () for a rational point.
    """
    if count == 1:
        return coords, ()
    theta = chart_expressions(coords).Symbol("theta")
    ring = (*free, theta)
    # For each pair of the distinct geometric points the difference of these
    # forms is a nonzero polynomial in a of degree at most len(free)-1.
    # Beyond the sum of their root counts at least one form separates.
    for form in separating_forms(free, count, first=0):
        basis = groebner((*points, theta - form), ring, order="lex").exprs
        minimal = [g for g in basis if g.free_symbols <= {theta}]
        solved = []
        for v in free:
            for g in basis:
                if v in g.free_symbols and g.free_symbols <= {v, theta}:
                    linear = frame_linear(ring, g, v)
                    if linear is not None and linear[0] == 1:
                        solved.append(g)  # the reduced lex basis is monic
                        break
        if len(minimal) == 1 and len(solved) == len(free):
            return (*coords, theta), (minimal[0], *solved)
    raise GiveUp("No separating linear form was found for the finite reduced Newton centre.")


def strata(poly, variables, fan, parts=None, names=None, deadline=None, *, count_strata=True,
           certify_subfaces=False):
    """The numerical data, the strata over the origin and the blowups made.

    `poly` is the reduced equation, in the coordinates of the fan. `parts`
    lists (label, N, polynomial) for the factors of a divisor with
    multiplicities supported on it (by default the single part "strict" with
    N = 1), and `names` gives the labels of the exceptional rays (by default
    E1, E2, ... in the order of the rays). The blowups stop, without an
    answer, when the time `deadline` (of time.monotonic) has passed.

    Returns {"data": {label: (N, nu)}, "strata": {frozenset of labels: chi},
    "repairs": [...]}, or None when the modification is not of the expected
    kind or normal crossings fail along more than finitely many points.
    With count_strata=False only the local blowup charts are needed.
    """
    n = len(variables)
    rays, cones = fan["rays"], fan["cones"]
    if len(rays) <= n:
        return None
    from .chart_expressions import runtime_expressions
    api = runtime_expressions()
    coords = tuple(api.Symbol(f"u{i+1}") for i in range(n))
    certified = {}

    class Part:
        """A factor: its terms, their values on the rays, and the terms of least value."""
        def __init__(self, label, weight, terms):
            self.label, self.weight, self.terms = label, weight, terms
            self.exponents = sorted(terms)
            self.values = _valuations(rays, self.exponents)
            self.orders, masks = minimum_faces(self.values)
            self.least = list(map(frozenset, masks))
            self.strict_values = strict_valuations(self.exponents, rays, self.orders, self.values)

        def face(self, orbit):
            return frozenset.intersection(*(self.least[r] for r in orbit))

        def strict(self, cone, face=None):
            chosen = range(len(self.exponents)) if face is None else sorted(face)
            terms = {tuple(self.strict_values[r][j] for r in cone): self.terms[self.exponents[j]] for j in chosen}
            return _polynomial(terms, coords)

    support = Part("strict", 1, poly)
    orders, exponents = support.orders, support.exponents
    if any(orders[:n]) or min(orders[n:]) < 1:
        return None
    parts = [support] if parts is None else [Part(*part) for part in parts]
    rows = _newton("repair_orders", n, (len(rays), len(parts)), orders,
                   tuple(part.weight for part in parts), *(part.orders for part in parts), *rays)
    if rows == ["none"]:
        return None
    numbers = [tuple(map(_integer, row.split())) for row in rows]

    def nondegenerate(face):
        if face not in certified:
            status = certify_face({exponents[j]: poly[exponents[j]] for j in face})["status"]
            if status == "undetermined":
                raise GiveUp("A face could not be decided.")
            certified[face] = status == "nondegenerate"
        return certified[face]

    def label(ray):
        return names[ray] if names else f"E{ray - n + 1}"

    data = {label(i): numbers[i] for i in range(n, len(rays))}
    for part in parts:
        data[part.label] = (part.weight, 1)
    contributions = []

    def add(key, chi, sign=1):
        if chi:
            contributions.append((key, 1, chi, sign))

    repairs = []
    volumes = {}
    try:
        # The orbits over the origin, each once, with the cone that gives its chart.
        rows = _newton("repair_orbits", n, (len(rays), len(cones)), *rays, *cones)
        if rows == ["none"]:
            return None
        orbits = []
        for row in rows:
            ci, dimension, orientation, *orbit = map(int, row.split())
            orbits.append((ci, dimension, orientation, orbit, support.face(orbit)))
        # When every orbit has a term of least value on all its rays, the fan
        # refines the dual fan of the Newton polyhedron: the faces of the
        # polytope of a face polynomial are then faces of other orbits.
        refines = all(item[-1] for item in orbits)
        if certify_subfaces and not refines:
            from .newton_fan import build_fan
            from .newton_faces import exhaustive_faces
            from .polynomial import LimitError
            # A common minimum on the orbit rays is the initial form along
            # their positive sum. All subfaces of that compact face occur
            # in the regular fan of the *original* Newton polyhedron.
            # Certifying those initial forms justifies the torus volume
            # formula even if other orbits lie entirely in the strict divisor.
            try:
                original = build_fan(exponents, exhaustive=exhaustive_faces())
            except LimitError:
                return None
            values = _valuations(original["rays"], exponents)
            faces = _newton("faces", n, (len(values), len(original["cones"]), len(exponents)),
                            *values, *original["cones"])
            masks = set()
            for row in faces[len(values):]:
                size, *entries = map(int, row.split())
                masks.add(frozenset(entries[size:]))
            if not all(nondegenerate(mask) for mask in masks):
                return None
        degenerate = [face for face in {item[-1] for item in orbits} if face and not nondegenerate(face)]
        for ci, d, orientation, orbit, face in orbits:
            cone, index = cones[ci], ci+1
            if not count_strata and face not in degenerate:
                continue  # the orbit cut is smooth; only the singular points need local runs
            exceptional = [i for i in orbit if i >= n]
            off = frozenset(map(label, exceptional))
            faces = [part.face(orbit) for part in parts]
            if not face:
                # No term has least value on all the rays: the equation of
                # the strict transform vanishes on the orbit, which lies in it.
                inside = [part for part, own in zip(parts, faces) if not own]
                if len(inside) != 1 or (d and any(len(own) > 1 for own in faces)):
                    raise GiveUp("Two components of the strict transform meet over the origin.")
                add(off | {inside[0].label}, 1 if d == 0 else 0)
                continue
            if d == 0:
                add(off, 1)  # a point, off the strict transform
                continue
            if count_strata and (refines or certify_subfaces or len(face) <= 2) and not any(other <= face for other in degenerate):
                # The cuts of the parts are disjoint: their union is smooth.
                for k, (part, own) in enumerate(zip(parts, faces)):
                    if len(own) <= d:
                        continue
                    key = (k, own, d)
                    if key not in volumes:
                        remaining = [r for r in cone if r not in orbit]
                        volumes[key] = _volume([tuple(part.values[r][j] for r in remaining) for j in sorted(own)], d)
                    add(off | {part.label}, volumes[key], orientation)
                    add(off, volumes[key], -orientation)
                continue
            free = tuple(u for u, r in zip(coords, cone) if r not in orbit)
            torus = [(u,) for u in free]
            on_orbit = support.strict(cone, face)
            partials = (frame_derivative(free, on_orbit, {u: 1}) for u in free)
            invert = _polynomial({(1,)*len(free): 1}, free)
            singular = saturate(free, (on_orbit, *partials), invert)
            if singular != (1,) and _dimension(free, singular) != 0:
                raise GiveUp("The strict transform fails normal crossings along a curve of an orbit.")
            if count_strata:
                cut = _euler(free, (on_orbit,), torus)
                add(off, cut, -1)
                for part, face_terms in zip(parts, faces):
                    own = part.strict(cone, face_terms)
                    chi = cut if len(parts) == 1 else _euler(free, (own,), torus)
                    if singular != (1,):
                        add(off | {part.label}, _euler(free, (*singular, own)), -1)
                    add(off | {part.label}, chi)
            if singular == (1,):
                continue
            components = [(label(r), data[label(r)][0], data[label(r)][1], (u,))
                          for u, r in zip(coords, cone) if r >= n]
            components.extend((part.label, part.weight, 1, (part.strict(cone),)) for part in parts)
            # Each component over Q of the set of singular points is a problem of
            # its own: the blowups made over one do not touch the others.
            for points in _point_components(free, _radical_points(free, singular)):
                orbit_ideal = tuple(u for u, r in zip(coords, cone) if r in orbit)
                parent_ideal = (*orbit_ideal, *points)
                count = _point_count(free, points)
                space, relation = _rational(coords, free, points, count)
                fibre = (*orbit_ideal, *relation[1:]) if relation else parent_ideal
                if CHECK and count != _euler(space, (*relation[:1], *fibre)):
                    raise AssertionError("The finite cover must lift each Newton centre point exactly once.")
                local_data, local_strata, charts, centres = acampo_over(
                    space, components, fibre, ambient=relation[:1], dimension=n, deadline=deadline,
                    count_strata=count_strata)
                name = f"P{len(repairs) + 1}"

                def rename(text, name=name):
                    return f"{name}.{text}" if text.startswith("B") else text

                for text, numbers in local_data.items():
                    if data.setdefault(rename(text), numbers) != numbers:
                        raise GiveUp("A component has two sets of numerical data.")
                for key, chi in local_strata.items():
                    add(frozenset(map(rename, key)), chi)
                repairs.append({
                    "id": name, "chart": f"N{index}", "orbit_rays": [rays[r] for r in orbit],
                    "points": count,
                    "ideal": [_text(e) for e in parent_ideal],
                    "blowups": [{**centre, "id": rename(centre["id"]),
                                 "components": [rename(c) for c in centre["components"]]} for centre in centres],
                    "charts": [replace(c, components=tuple((rename(label), m, nu, ideal)
                                                          for label, m, nu, ideal in c.components))
                               for c in charts],
                })
    except GiveUp:
        return None
    return {"data": data, "strata": strata_totals(contributions), "repairs": repairs}


def _chart_polynomial(text, variables):
    backend = chart_expressions(variables)
    names = tuple(v.name for v in variables)
    if not native_ring(variables):
        from .atlas import _expression
        return _expression(text, dict(zip(names, variables)))
    return backend.from_terms(variables, parse_expression(text, names))


def patch_record(chart, parent, chart_id, parent_index=None):
    """Express a local blowup chart over the original coordinate space.

    Its local fibre cuts out the chosen Newton points, also on the cover:
    there it is the graph of the separating parameter, one lift per point.
    This condition now belongs in the ownership region. The fibre over the
    original origin is defined by the composite map itself. Keeping these
    ideals distinct also lets the cell count recognize a toric divisor whose
    image is the origin but whose image in the Newton chart is a whole divisor.
    """
    from .affine_kernel import pullbacks
    from .atlas import to_json
    backend = chart_expressions(chart.variables)
    parent_variables = tuple(map(backend.Symbol, parent["coordinates"]))
    substitutions = dict(chart.coordinates)
    parent_map = tuple(_chart_polynomial(text, parent_variables) for text in parent["map"].values())
    mapping = pullbacks(chart.variables, tuple(substitutions[v.name] for v in parent_variables),
                        parent_variables, parent_map)
    projection = ((parent_index, tuple(substitutions[v.name] for v in parent_variables))
                  if parent_index is not None else ())
    patch = replace(chart, fibre=tuple(mapping), region=tuple(dict.fromkeys((*chart.region, *chart.fibre))),
                    coordinates=tuple(zip(parent["map"], mapping)), repair_projection=projection)
    item, = to_json([patch])
    item.update(id=chart_id, parent=parent["id"])
    item["exceptional"] = [{"id": c["id"], "ideal": c["ideal"], "equation": ", ".join(c["ideal"]),
                            "total_multiplicity": c["multiplicity"]} for c in item["components"] if c["id"] != "strict"]
    item["strict_transform"] = next((c["ideal"] for c in item["components"] if c["id"] == "strict"), ["1"])
    return item


def join_atlas(record, toric_charts):
    """Replace the bad points in the partition by the local runs over them.

    A fan chart contains the orbit of a face precisely when the face belongs
    to its cone. Thus its first cone in the fan order owns that entire orbit.
    This is exactly the cone chosen in `strata`. Removing the points from its
    owned region removes them once, in their equations over Q before passing
    to a cover. The local runs partition the inverse image of the removed
    points, including all conjugates. The complement and these inverse
    images are disjoint and cover the resolved fibre over the origin.
    """
    from .atlas import from_json
    parents = [item for item in record["charts"] if "ray_indices" in item]
    patches = [item for item in record["charts"] if "ray_indices" not in item]
    if len(parents) != len(toric_charts):
        return None
    if any("region" not in item for item in patches):
        from .archived_newton import archived_repair_charts
        return archived_repair_charts(record, toric_charts)
    result = []
    for item, chart in zip(parents, toric_charts):
        centres = [tuple(_chart_polynomial(text, chart.variables) for text in centre["ideal"])
                   for centre in item.get("blown_up", ())]
        result.append(replace(chart, removed=(*chart.removed, *centres)))
    return result + from_json(patches, native=native_ring(toric_charts[0].variables) if toric_charts else None)
