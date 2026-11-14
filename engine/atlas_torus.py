"""Incidence of divisor sections on an atlas with a common dense torus.

Each regular cone supplies the lattice basis of its chart and a common
linear map from torus coordinates to the original input. The maps and
strict equations are checked. For internal rays their intersections lie
over the origin. For rays mapping onto positive-dimensional centres, an
orbit lies over the origin precisely when every base monomial vanishes.
Smooth transverse sections and their character restrictions identify
the closures of the components meeting those orbits.

An orbit section of Newton dimension at least two is geometrically
connected by the toric Lefschetz theorem, hence integral by smoothness.
For a segment it is the zero set of a squarefree polynomial in a primitive
character. That character is represented in the common lattice. Its root
index propagates to a containing split section only when the character
and polynomial agree. Single-component faces need no root matching.
See native/KERNEL.md for the hypotheses and reference.
"""
from functools import lru_cache

from .fan_geometry import canonical_laurent_terms, complete_fan as _complete_fan, _determinant
from .kernel_json import kernel_json
from .polynomial import integer_from_decimal, integer_text
from .newton_faces import certify_face, edge_polynomial, exhaustive_faces


def _normalized(terms):
    first = terms[min(terms)]
    if first == 1:
        return tuple(sorted(terms.items()))
    from .native_library import affine_request
    from .polynomial import format_poly, rational_text, sparse_row
    names = tuple(f"x{i}" for i in range(len(next(iter(terms)))))
    row, = affine_request(["function", " ".join(names), f"({format_poly(terms,names)})/({rational_text(first)})"])
    return tuple(sorted(sparse_row(row).items()))


@lru_cache(maxsize=4096)
def _orbit(rays, terms, exhaustive=False):
    return _section(_normalized(canonical_laurent_terms(rays, dict(terms))), exhaustive)


@lru_cache(maxsize=4096)
def _section(terms, exhaustive):
    common = dict(terms)
    proof = certify_face(common, exhaustive=exhaustive)
    edge = proof.get("edge")
    count = kernel_json("atlas_torus_sections", "section", nondegenerate=proof["status"] == "nondegenerate",
                        rank=proof["face_dimension"], degree=integer_text(edge["degree"]) if edge else None)
    if count is None:
        return None
    signature = (tuple(edge["direction"]), _normalized(edge_polynomial(edge))) if edge else None
    return integer_from_decimal(count), signature


@lru_cache(maxsize=4096)
def _absolute_torus_section(terms):
    """Prove geometric irreducibility on the open torus by factoring its monomial-free equation."""
    from .affine_kernel import _polynomial
    from .chart_expressions import runtime_expressions
    from .geometric_components import _certify, _factor_records
    from .reference import QQ
    variables = runtime_expressions().symbols(f"orbit_var0:{len(terms[0][0])}")
    expression = _polynomial(dict(terms), variables)
    factors = _factor_records(variables, expression, QQ)
    if len(factors) != 1:
        return False
    factor, multiplicity, data = factors[0]
    return multiplicity == 1 and _certify(variables, factor, data, QQ, 1) is True


@lru_cache(maxsize=4096)
def _affine_orbit(rays, terms, exhaustive):
    common = _normalized(canonical_laurent_terms(rays, dict(terms)))
    section = _section(common, exhaustive)
    if section == (1, None) and not _absolute_torus_section(common):
        return None
    return section


def _chart_terms(chart, dimension, common_linear):
    from .affine_kernel import sparse_terms
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
    actual = tuple(g for _, g in chart.coordinates)
    from .point_geometry import encode_terms
    if not kernel_json("atlas_torus_sections", "projection", rays=[list(map(integer_text, r)) for r in rays],
                       linear=[list(map(integer_text, r)) for r in linear],
                       images=[encode_terms(dict(sparse_terms(axes, image))) for image in actual]):
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
        if at in boundary:
            return None
        boundary[at] = label
    return rays, boundary, sparse_terms(axes, strict[0][0])


def torus_complex(charts, dimension, intrinsic):
    """Return a certified common-torus complex, or defer to general gluing."""
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
    if not _determinant(linear):
        return None
    return section_complex((_chart_terms(c, dimension, linear) for c in charts), dimension, intrinsic)


def section_complex(chart_terms, dimension, intrinsic, *, common_source=False):
    """Orbit sections of a verified regular cover with its maps in the common torus.

    With a common source, the caller has derived every strict equation from
    one polynomial in the same verified frame by dividing its pullback by
    a monomial. Those equations already agree up to a unit on the dense
    torus. Independently supplied charts require this equality test.
    """
    chart_terms = tuple(chart_terms)
    if any(details is None for details in chart_terms):
        return None
    ray_ids, equations, charts = {}, {}, []
    for rays, boundary, terms in chart_terms:
        chart = {"rays": [ray_ids.setdefault(ray, len(ray_ids)) for ray in rays],
                 "boundary": list(boundary.items())}
        if intrinsic:
            chart["powers"] = [list(map(integer_text, e)) for e, _ in terms]
        if not common_source:
            try:
                equation = _normalized(canonical_laurent_terms(rays, dict(terms)))
            except ArithmeticError:
                return None
            chart["equation"] = equations.setdefault(equation, len(equations))
        charts.append(chart)
    plan = kernel_json("atlas_torus_sections", "plan", dimension=dimension, intrinsic=intrinsic,
                       common_source=common_source, rays=[list(map(integer_text, ray)) for ray in ray_ids],
                       charts=charts)
    if plan is None:
        return None
    if "complex" in plan:
        return plan["complex"]

    exhaustive = exhaustive_faces()
    sections, characters = [], {None: 0}
    for index, affine, terms in plan["checks"]:
        rays, _, polynomial = chart_terms[index]
        restricted = tuple(polynomial[j] for j in terms)
        prove_section = _affine_orbit if affine else _orbit
        section = prove_section(rays, restricted, exhaustive)
        if section is None:
            return None
        count, character = section
        sections.append((integer_text(count), characters.setdefault(character, len(characters))))
    return kernel_json("atlas_torus_sections", "glue", incidence=plan["incidence"], sections=sections)
