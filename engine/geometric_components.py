"""Exact geometric components of a smooth affine stratum over a number field.

The shape calculation localizes in independent projection parameters. Every
candidate returned to the caller is contracted by saturation and its union
is checked against the *original* ideal. This catches components lying over
special parameter values. None means a field extension or more work is needed.
"""

from .kernel_json import kernel_json
from .component_ring import ComponentRing
from .polynomial import integer_from_decimal, integer_text, rational_text
from .reference import QQ
from .affine_kernel import _sparse_expression, rational_factor_rows
from .blowup_ideal import ideal_basis, saturate
from . import field_factors


def _component_operation(operation, **operands):
    return kernel_json("atlas_component_geometry", operation, **operands)


def _encoded_support(powers):
    return [[integer_text(k) for k in row] for row in powers]


def _factor_records(variables, f, domain):
    """Retain the ordered native factors, powers, and rational support certificates."""
    if domain == QQ:
        return [(_sparse_expression(variables, row), m, (row, primitive))
                for row, m, primitive in rational_factor_rows(variables, f)[1]]
    from .field_polynomial import polynomial
    return [(p.as_expr(), m, None) for p, m in field_factors.factors(polynomial(f, variables, domain))]


def _certify(variables, expression, data, domain, effort):
    ring = ComponentRing(variables, domain)
    row, primitive = (ring.row(expression), False) if data is None else data
    return ring.request("absolute_factor", polynomial=row, primitive=primitive, support_checked=data is not None,
                        effort=integer_text(effort))


def _absolute_factor(poly, effort):
    return _certify(poly.gens, poly.as_expr(), None, poly.domain, effort)


def _factor_equations(poly, effort):
    ring = ComponentRing(poly.gens, poly.domain)
    return ring.request("coefficient_test", polynomial=ring.row(poly.as_expr()), effort=integer_text(effort))


def _certified_factors(variables, polynomial, domain, effort, reduced_support, axis=None):
    ring = ComponentRing(variables, domain)
    factors = _factor_records(variables, polynomial, domain)
    indices = ring.request("eligible_factors", factors=[row[0] if row else ring.row(f) for f, _, row in factors],
                           powers=[integer_text(m) for _, m, _ in factors], reduced=reduced_support, axis=axis)
    if indices is None:
        return None
    primes = []
    for i in indices:
        factor, _, data = factors[i]
        if _certify(variables, factor, data, domain, effort) is not True:
            return None
        primes.append(factor)
    return primes


def _disjoint(variables, primes, domain):
    from .graph_record import _graph_request, _graph_ring
    names, text, relation = _graph_ring(variables, domain)
    return _graph_request("disjoint_components", names, relation=relation,
                          pieces=[list(map(text, prime)) for prime in primes])


def _intersection(variables, a, b, domain):
    from .graph_record import _graph_request, _graph_ring
    from .field_polynomial import _line_expression
    names, text, relation = _graph_ring(variables, domain)
    rows = _graph_request("intersection", names, relation=relation, left=list(map(text, a)), right=list(map(text, b)))
    if domain == QQ:
        return tuple(_sparse_expression(tuple(variables), row) for row in rows)
    return tuple(_line_expression(row, variables, domain) for row in rows)


def _coverage(variables, components, ideal, domain, reduced_support):
    from .graph_record import _graph_request, _graph_ring
    names, text, relation = _graph_ring(variables, domain)
    return _graph_request("component_coverage", names, relation=relation, reduced=reduced_support,
                          ambient=list(map(text, ideal)), pieces=[list(map(text, p)) for p in components])


def _projections(variables, effort):
    """Transport the ordered native projection schedule lazily, stopping at the first complete proof."""
    count = _component_operation("projection_count", width=len(variables), effort=integer_text(effort))
    for index in range(integer_from_decimal(count)):
        pair = _component_operation("projection_rows", width=len(variables), index=integer_text(index))
        yield tuple(tuple(_sparse_expression(variables, row) for row in images) for images in pair)


def _coordinate_images(variables, projection):
    if not hasattr(projection, "todok"):
        return tuple(projection)
    if projection.shape != (len(variables), len(variables)):
        raise ValueError("The projection matrix must be square in the given coordinates.")
    matrix = [[] for _ in variables]
    for (i, j), c in projection.todok().items():
        matrix[i].append([j, rational_text(c)])
    rows, = _component_operation("linear_images", width=len(variables), matrix=matrix)
    return tuple(_sparse_expression(variables, row) for row in rows)


def _shape(variables, generators, dimension, domain, effort, matrix, inverse=None, *, reduced_support=False):
    images = _coordinate_images(variables, matrix)
    if inverse is None:
        from .affine_kernel import inverse_map
        # The denominator of a supplied invertible linear matrix is a constant.
        from .affine_kernel import _ring
        from .native_library import affine_request
        wire = _ring(variables)
        images_back = []
        for a, b in inverse_map(variables, images, variables):
            row, = affine_request(["function", wire.header, "("+wire.text(a)+")/("+wire.text(b)+")"])
            images_back.append(_sparse_expression(variables, row))
    else:
        images_back = _coordinate_images(variables, inverse)
    ideal = field_factors.compose(variables, images, variables, generators, domain)
    ring = ComponentRing(variables, domain)
    shape = ring.request("projection_shape", generators=[ring.row(f) for f in ideal], dimension=dimension)
    if shape is None:
        return None
    parameters = (*variables[:dimension], variables[-1])
    factors = _certified_factors(parameters, ring.expression(shape["eliminant"]), domain, effort, reduced_support,
                                 axis=len(parameters)-1)
    if factors is None:
        return None
    graph = tuple(map(ring.expression, shape["graph"]))
    denominator = ring.expression(shape["denominator"])
    primes = [saturate(variables, (*graph, factor), denominator, domain=domain) for factor in factors]
    if not _coverage(variables, primes, ideal, domain, reduced_support):
        return None
    return [ideal_basis(variables, field_factors.compose(variables, images_back, variables, prime, domain),
                        domain=domain) for prime in primes]


def geometric_components(variables, ideal, dimension, *, domain=QQ, effort=1, reduced_support=False):
    """List certified geometric prime ideals, [] for empty, or None.

    By default the primes must be disjoint and give the original ideal.
    A reduced-support decomposition allows meeting branches and nilpotents.
    Absolute primality and radical containment of the intersection ideal
    prove that every geometric component is accounted for on the entire chart.

    Designed for smooth equidimensional strata. Returned coordinates and ideals
    are over the supplied field K. A geometric component that is not individually
    definable over K requires a new call over a larger number field. Higher
    integer effort considers more projections and larger exact factor tests.
    """
    variables = tuple(variables)
    if not isinstance(effort,int) or effort<1:
        raise ValueError("effort must be a positive integer")
    generators = ideal_basis(variables,ideal,domain=domain)
    ring = ComponentRing(variables, domain)
    step = ring.request("decomposition_step", generators=[ring.row(g) for g in generators],
                        dimension=integer_text(dimension))
    action = step["action"]
    if action == "empty":
        return []
    if action == "ambient":
        return [()]
    if action == "undecided":
        return None
    if action == "prime":
        return [generators]
    if action == "graph":
        axis, row = step["axis"], step["row"]
        others = (*variables[:axis], *variables[axis+1:])
        mapping = (*variables[:axis], ring.expression(step["value"]), *variables[axis+1:])
        equations = field_factors.compose(others, mapping, variables, (*generators[:row], *generators[row+1:]), domain)
        smaller = geometric_components(others, equations, dimension, domain=domain, effort=effort,
                                       reduced_support=reduced_support)
        if smaller is None:
            return None
        return [ideal_basis(variables, (*prime, generators[row]), domain=domain) for prime in smaller]
    if action == "hypersurface":
        factors = _certified_factors(variables, generators[0], domain, effort, reduced_support)
        if factors is None:
            return None
        candidates = [(p,) for p in factors]
        return candidates if reduced_support or _disjoint(variables, candidates, domain) else None
    if action != "project":
        raise RuntimeError("Unknown native geometric decomposition step.")
    for projection, inverse in _projections(variables, effort):
        found = _shape(variables, generators, dimension, domain, effort, projection, inverse,
                       reduced_support=reduced_support)
        if found is not None:
            return found
    return None
