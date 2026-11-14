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
from itertools import combinations
from math import prod
import re

import sympy as sp

from .blowup_ideal import saturate
from .groebner import groebner
from .constructible_euler import euler_characteristic
from .local_blowups import GiveUp, _dimension, _point_components, _radical_points, acampo_over, cells
from .monodromy_toric import _volume
from .newton_faces import certify_face


def _text(expr):
    return str(expr).replace("**", "^")


def _rational(coords, free, points):
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
    if euler_characteristic(free, points) == 1:
        return coords, ()
    theta = sp.Symbol("theta")
    for attempt in range(1, 12):
        form = sum((1 + (attempt*(i + 1)) % 7)*v for i, v in enumerate(free))
        basis = groebner((*points, theta - form), (*free, theta), order="lex").exprs
        minimal = [g for g in basis if g.free_symbols <= {theta}]
        solved = []
        for v in free:
            found = [g for g in basis if sp.Poly(g, v).degree() == 1 and sp.Poly(g, v).LC().is_number
                     and g.free_symbols <= {v, theta}]
            if found:
                solved.append(sp.expand(found[0]/sp.Poly(found[0], v).LC()))
        if len(minimal) == 1 and len(solved) == len(free):
            return (*coords, theta), (minimal[0], *solved)
    return coords, ()


def strata(poly, variables, fan, parts=None, names=None, deadline=None):
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
    """
    n = len(variables)
    rays, cones = fan["rays"], fan["cones"]
    if len(rays) <= n:
        return None
    exponents = sorted(poly)
    valuations = [[sum(r*t for r, t in zip(ray, e)) for e in exponents] for ray in rays]
    orders = [min(row) for row in valuations]
    if any(orders[:n]) or min(orders[n:]) < 1:
        return None
    minimizers = [frozenset(j for j, q in enumerate(row) if q == m) for row, m in zip(valuations, orders)]
    ray_support = [sum(1 << j for j, r in enumerate(ray) if r) for ray in rays]
    full_support = (1 << n) - 1
    coords = sp.symbols(f"u1:{n + 1}")
    certified = {}

    class Part:
        """A factor: its terms, their values on the rays, and the terms of least value."""
        def __init__(self, label, weight, terms):
            self.label, self.weight, self.terms = label, weight, terms
            self.exponents = sorted(terms)
            self.values = [[sum(r*t for r, t in zip(ray, e)) for e in self.exponents] for ray in rays]
            self.orders = [min(row) for row in self.values]
            self.least = [frozenset(j for j, q in enumerate(row) if q == m)
                          for row, m in zip(self.values, self.orders)]

        def face(self, orbit):
            return frozenset.intersection(*(self.least[r] for r in orbit))

        def strict(self, cone, face=None):
            """The strict transform in the chart of the cone, or its restriction to a face."""
            chosen = range(len(self.exponents)) if face is None else sorted(face)
            return sp.Add(*(sp.Rational(self.terms[self.exponents[j]].numerator, self.terms[self.exponents[j]].denominator)
                            * prod(u**(self.values[r][j] - self.orders[r]) for u, r in zip(coords, cone))
                            for j in chosen))

    support = Part("strict", 1, poly)
    parts = [support] if parts is None else [Part(*part) for part in parts]
    if any(sum(part.orders[i] for part in parts) != orders[i] for i in range(len(rays))):
        return None  # the parts are not the factors of the reduced equation

    def nondegenerate(face):
        if face not in certified:
            status = certify_face({exponents[j]: poly[exponents[j]] for j in face})["status"]
            if status == "undetermined":
                raise GiveUp("A face could not be decided.")
            certified[face] = status == "nondegenerate"
        return certified[face]

    def label(ray):
        return names[ray] if names else f"E{ray - n + 1}"

    data = {label(i): (sum(part.weight*part.orders[i] for part in parts), sum(rays[i]))
            for i in range(n, len(rays))}
    for part in parts:
        data[part.label] = (part.weight, 1)
    found = {}

    def add(key, chi):
        if chi:
            found[key] = found.get(key, 0) + chi

    repairs = []
    volumes = {}
    try:
        # The orbits over the origin, each once, with the cone that gives its chart.
        orbits, seen = [], set()
        for index, cone in enumerate(cones, 1):
            if not any(i >= n for i in cone):
                return None
            for size in range(1, len(cone) + 1):
                for orbit in combinations(sorted(cone), size):
                    if orbit in seen or not any(i >= n for i in orbit):
                        continue
                    covered = 0
                    for i in orbit:
                        covered |= ray_support[i]
                    if covered != full_support:
                        continue  # not over the origin
                    seen.add(orbit)
                    orbits.append((index, cone, orbit, frozenset.intersection(*(minimizers[r] for r in orbit))))
        # When every orbit has a term of least value on all its rays, the fan
        # refines the dual fan of the Newton polyhedron: the faces of the
        # polytope of a face polynomial are then faces of other orbits.
        refines = all(face for _, _, _, face in orbits)
        degenerate = [face for face in {face for _, _, _, face in orbits} if face and not nondegenerate(face)]
        for index, cone, orbit, face in orbits:
            exceptional = [i for i in orbit if i >= n]
            d = n - len(orbit)
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
            if (refines or len(face) <= 2) and not any(other <= face for other in degenerate):
                # The cuts of the parts are disjoint: their union is smooth.
                total = 0
                for k, (part, own) in enumerate(zip(parts, faces)):
                    if len(own) <= d:
                        continue
                    key = (k, own, d)
                    if key not in volumes:
                        remaining = [r for r in cone if r not in orbit]
                        volumes[key] = _volume([tuple(part.values[r][j] for r in remaining) for j in sorted(own)], d)
                    cut = (-1)**(d - 1) * volumes[key]
                    add(off | {part.label}, cut)
                    total += cut
                add(off, -total)
                continue
            free = tuple(u for u, r in zip(coords, cone) if r not in orbit)
            torus = [(u,) for u in free]
            on_orbit = support.strict(cone, face)
            cuts = [part.strict(cone, own) for part, own in zip(parts, faces)]
            cut = euler_characteristic(free, (on_orbit,), torus)
            singular = saturate(free, (on_orbit, *(on_orbit.diff(u) for u in free)), prod(free))
            add(off, -cut)
            if singular != (1,) and _dimension(free, singular) != 0:
                raise GiveUp("The strict transform fails normal crossings along a curve of an orbit.")
            for part, own in zip(parts, cuts):
                chi = cut if len(parts) == 1 else euler_characteristic(free, (own,), torus)
                if singular != (1,):
                    chi -= euler_characteristic(free, (*singular, own))
                add(off | {part.label}, chi)
            if singular == (1,):
                continue
            components = [(label(r), data[label(r)][0], sum(rays[r]), (u,))
                          for u, r in zip(coords, cone) if r >= n]
            components.extend((part.label, part.weight, 1, (part.strict(cone),)) for part in parts)
            # Each component over Q of the set of singular points is a problem of
            # its own: the blowups made over one do not touch the others.
            for points in _point_components(free, _radical_points(free, singular)):
                fibre = (*(u for u, r in zip(coords, cone) if r in orbit), *points)
                space, relation = _rational(coords, free, points)
                if relation:
                    fibre = (*(u for u, r in zip(coords, cone) if r in orbit), *relation[1:])
                local_data, local_strata, charts, centres = acampo_over(
                    space, components, fibre, ambient=relation[:1], dimension=n, deadline=deadline)
                name = f"P{len(repairs) + 1}"

                def rename(text, name=name):
                    return f"{name}.{text}" if text.startswith("B") else text

                for text, numbers in local_data.items():
                    if data.setdefault(rename(text), numbers) != numbers:
                        raise GiveUp("A component has two sets of numerical data.")
                for key, chi in local_strata.items():
                    add(frozenset(map(rename, key)), chi)
                # The intersections of the divisors over the point, with their numbers of
                # components; over conjugate points the cover does not give them.
                try:
                    met = None if relation else [[sorted(map(rename, key)), count]
                                                 for key, count in cells(charts, n).items()]
                except GiveUp:
                    met = None
                repairs.append({
                    "cells": met, "births": [rename(centre["id"]) for centre in centres],
                    "id": name, "chart": f"N{index}", "orbit_rays": [rays[r] for r in orbit],
                    "points": euler_characteristic(space, (*relation[:1], *fibre)),
                    "ideal": [_text(e) for e in (*relation[:1], *fibre)],
                    "blowups": [{**centre, "id": rename(centre["id"]),
                                 "components": [rename(c) for c in centre["components"]]} for centre in centres],
                    "charts": [{
                        "id": f"{name}.{k}", "parent": f"N{index}",
                        "coordinates": [v.name for v in chart.variables],
                        "ambient_ideal": [_text(e) for e in chart.ambient],
                        "parent_coordinates": {old: _text(expr) for old, expr in chart.coordinates
                                               if old in {v.name for v in coords}},
                        "divisors": [{"id": rename(text), "multiplicity": m, "ideal": [_text(e) for e in ideal]}
                                     for text, m, _, ideal in chart.components],
                        **({"blown_up": [{"ideal": [_text(e) for e in z]} for z in chart.removed]}
                           if chart.removed else {}),
                    } for k, chart in enumerate(charts, 1)],
                })
    except GiveUp:
        return None
    return {"data": data, "strata": {key: chi for key, chi in found.items() if chi}, "repairs": repairs}


def records(answer):
    """The answer of `strata` as JSON data: the numerical data of the
    components that occur and the strata, in a fixed order (exceptional
    components, those of later blowups, then the strict transform)."""
    used = set().union(*answer["strata"]) if answer["strata"] else set()

    def position(text):
        kind = 2 if text == "strict" or re.fullmatch(r"S\d+", text) else 1 if text.startswith("P") else 0
        return kind, [int(w) for w in re.findall(r"\d+", text)], text

    data = [{"component": text, "multiplicity": answer["data"][text][0], "nu": answer["data"][text][1]}
            for text in sorted(used, key=position)]
    order = {row["component"]: k for k, row in enumerate(data)}
    listed = [{"components": sorted(key, key=order.get), "euler_characteristic": chi}
              for key, chi in answer["strata"].items()]
    listed.sort(key=lambda row: (len(row["components"]), [order[c] for c in row["components"]]))
    return data, listed
