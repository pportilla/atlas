"""Incidence of divisor sections on an atlas with a common dense torus.

Each regular cone supplies the lattice basis of its chart and a common
linear map from torus coordinates to the original input. The maps and
strict equations are checked. All exceptional rays must be internal, so
their intersections lie over the origin. Transversality to every boundary
stratum excludes components supported on smaller orbits.

An orbit section of Newton dimension at least two is geometrically
connected by the toric Lefschetz theorem, hence integral by smoothness.
For a segment it is the zero set of a squarefree polynomial in a primitive
character. That character is represented in the common lattice. Its root
index propagates to a containing split section only when the character
and polynomial agree. Single-component faces need no root matching.
See native/KERNEL.md for the hypotheses and reference.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

from .fan_geometry import canonical_laurent_terms, complete_fan as _complete_fan
from .newton_faces import certify_face


def _normalized(terms):
    first = terms[min(terms)]
    if first == 1:
        return tuple(sorted(terms.items()))
    from .native_library import affine_request
    from .polynomial import checked_native, format_poly, sparse_row
    names = tuple(f"x{i}" for i in range(len(next(iter(terms)))))
    def native():
        row, = affine_request(["function", " ".join(names), f"({format_poly(terms,names)})/({first})"])
        return tuple(sorted(sparse_row(row).items()))
    return checked_native(native, lambda: tuple((e, Fraction(c)/first) for e, c in sorted(terms.items())))


@lru_cache(maxsize=4096)
def _orbit(rays, terms):
    return _section(_normalized(canonical_laurent_terms(rays, dict(terms))))


@lru_cache(maxsize=4096)
def _section(terms):
    common = dict(terms)
    proof = certify_face(common)
    if proof["status"] != "nondegenerate":
        return None
    dimension = proof["face_dimension"]
    if dimension == 0:
        return 0, None
    if dimension > 1:
        return 1, None
    edge = proof["edge"]
    coefficients = {int(k): Fraction(v) for k, v in edge["root_polynomial"].items()}
    # Torus nondegeneracy in one primitive character means every nonzero
    # root is simple. Translating the segment to start at zero makes its
    # constant coefficient nonzero, so its degree counts all its roots.
    signature = tuple(edge["direction"]), _normalized({(k,): c for k, c in coefficients.items()})
    return edge["degree"], signature


def _chart_terms(chart, dimension, common_linear):
    from .chart_expressions import chart_expressions
    from .affine_kernel import pullbacks, sparse_terms
    from .blowup_ideal import radical_subset
    from .local_blowups import _is_strict
    if (chart.ambient or chart.removed or chart.coefficient_field or len(chart.variables) != dimension or
            len(chart.coordinates) != dimension or len(chart.torus) != 2):
        return None
    rays, linear = chart.torus
    if linear != common_linear or any(len(row) != dimension for row in (*rays, *linear)):
        return None
    if len(rays) != dimension or len(linear) != dimension or any(
            type(entry) is not int for row in (*rays, *linear) for entry in row):
        return None
    axes = chart.variables
    sp = chart_expressions(axes)
    monomials = tuple(sp.Mul(*(v**r[j] for v, r in zip(axes, rays))) for j in range(dimension))
    mapping = tuple(sum(c*m for c, m in zip(row, monomials)) for row in linear)
    actual = tuple(g for _, g in chart.coordinates)
    if any(pullbacks(axes, axes, axes, tuple(a-b for a, b in zip(mapping, actual)))):
        return None
    if set(chart.fibre) != set(actual) and not (
            radical_subset(axes, chart.fibre, actual) and radical_subset(axes, actual, chart.fibre)):
        return None
    strict = [ideal for label, _, _, ideal in chart.components if _is_strict(label)]
    if len(strict) != 1 or len(strict[0]) != 1:
        return None
    boundary = {}
    for label, _, _, ideal in chart.components:
        if _is_strict(label):
            continue
        if len(ideal) != 1 or ideal[0] not in axes:
            return None
        at = axes.index(ideal[0])
        if min(rays[at]) <= 0 or at in boundary:
            return None
        boundary[at] = label
    if any((min(ray) > 0) != (i in boundary) for i, ray in enumerate(rays)):
        return None
    return rays, boundary, sparse_terms(axes, strict[0][0])


def torus_complex(charts, dimension, intrinsic):
    """Return a certified common-torus complex, or defer to general gluing."""
    from .chart_expressions import chart_expressions
    from .affine_kernel import frame_minor
    if not charts or not 2 <= dimension <= 8 or any(len(c.torus) != 2 for c in charts):
        return None
    if not _complete_fan(tuple(c.torus[0] for c in charts), dimension):
        return None
    linear = charts[0].torus[1]
    if (len(linear) != dimension or any(len(row) != dimension for row in linear) or
            any(type(c) is not int for row in linear for c in row)):
        return None
    base_names = tuple(name for name, _ in charts[0].coordinates)
    if len(set(base_names)) != dimension or any(
            tuple(name for name, _ in chart.coordinates) != base_names for chart in charts):
        return None
    axes = chart_expressions(charts[0].variables).symbols(f"torus_0:{dimension}")
    if not frame_minor(axes, [sum(c*v for c, v in zip(row, axes)) for row in linear], axes):
        return None
    return section_complex((_chart_terms(c, dimension, linear) for c in charts), dimension, intrinsic)


def section_complex(chart_terms, dimension, intrinsic):
    """Orbit sections of a verified regular cover with its maps in the common torus."""
    sections, labels, rays_by_label = {}, [], {}
    global_equation = None
    for details in chart_terms:
        if details is None:
            return None
        rays, boundary, terms = details
        try:
            original = _normalized(canonical_laurent_terms(rays, dict(terms)))
        except ArithmeticError:
            return None
        if global_equation is not None and original != global_equation:
            return None
        global_equation = original
        for at, name in boundary.items():
            if name in rays_by_label and rays_by_label[name] != rays[at]:
                return None
            if name not in rays_by_label:
                if rays[at] in rays_by_label.values():
                    return None
                labels.append(name)
                rays_by_label[name] = rays[at]
        # Restrict to each coordinate hyperplane once. The terms on an
        # intersection survive on every participating hyperplane, so the
        # smallest restriction suffices to find their common subset.
        hyperplanes = [tuple(term for term in terms if term[0][i] == 0) for i in range(dimension)]
        for size in range(1, dimension+1):
            for indices in combinations(range(dimension), size):
                if not any(i in boundary for i in indices):
                    continue
                if size == 1:
                    restricted = hyperplanes[indices[0]]
                else:
                    first = min(indices, key=lambda i: len(hyperplanes[i]))
                    other = tuple(i for i in indices if i != first)
                    restricted = tuple(term for term in hyperplanes[first] if all(term[0][i] == 0 for i in other))
                if not restricted:
                    return None
                orbit = _orbit(rays, restricted)
                if orbit is None:
                    return None
                if any(i not in boundary for i in indices):
                    continue
                key = frozenset(boundary[i] for i in indices)
                value = orbit if intrinsic else (1, None)
                if key in sections and sections[key] != value:
                    return None
                sections[key] = value
    ids, vertices, cells = {}, [], []
    order = labels.index
    for key in sorted(sections, key=lambda k: (len(k), sorted(map(order, k)))):
        count, signature = sections[key]
        names = sorted(key, key=order)
        ids[key] = []
        for i in range(count):
            if len(key) == 1:
                name = names[0] + (f".{i+1}" if count > 1 else "")
                vertices.append({"id": name, "dimension": dimension-1-int(intrinsic)})
            else:
                def restriction(face):
                    face = frozenset(face)
                    choices = ids.get(face, [])
                    if len(choices) == 1:
                        return choices[0]
                    if len(choices) == count and signature is not None and sections[face][1] == signature:
                        return choices[i]
                    raise ValueError("Unmatched primitive characters on intersecting torus strata.")
                try:
                    ends = [restriction((v,)) for v in names]
                    faces = [restriction(face) for face in combinations(names, len(names)-1)]
                except ValueError:
                    return None
                name = "C" + "-".join(names) + f".{i+1}"
                cells.append({"id": name, "dimension": len(names)-1, "vertices": ends, "faces": faces})
            ids[key].append(name)
    return {"status": "complete", "vertices": vertices, "cells": cells,
            "convention": ("Exceptional divisor on the smooth strict transform" if intrinsic else
                           "Ambient exceptional divisor") + " in the germ over the origin over Qbar."}
