"""Exact graded and cumulative lengths of the maximal-ideal filtration over QQ."""

from math import comb, inf, prod

from . import native_polynomial as np
from .algebra_execution import cpu_stage
from .local_algebra import _initial_monomials
from .point_germs import _large_translation, centered_ideal
from .polynomial import InputError
from .quotient_monomials import _components, _standard_boxes
from .rational_point import coordinate_fraction, normalize_point


def _boxes(width, heads):
    """Disjoint boxes in the standard monomials, with unbounded upper ends allowed."""
    bounds = [inf] * width
    mixed = []
    for row in heads:
        axes = [i for i, k in enumerate(row) if k]
        if len(axes) == 1:
            i, = axes
            bounds[i] = min(bounds[i], row[i])
        else:
            mixed.append(row)
    mixed = sorted(set(row for row in mixed if all(k < b for k, b in zip(row, bounds))))
    if width == 2:
        # The planar staircase splits into strips at its successive downward steps.
        left, height = 0, bounds[1]
        for x, y in mixed:
            if y < height:
                if left < x:
                    yield (left, 0), (x, height)
                left, height = x, y
        yield (left, 0), (bounds[0], height)
    else:
        yield from _standard_boxes(tuple(bounds), mixed)


def _series_components(width, heads, store_boxes):
    """Dimension and multiplicity are additive and multiplicative across tensor factors."""
    dimension, multiplicity = 0, 1
    components = []
    for axes, rows in _components(width, heads):
        if not rows:
            dimension += len(axes)
            continue  # Its numerator is 1, in either power series expansion below.
        boxes = []
        dim, degree = -1, 0
        for lower, upper in _boxes(len(axes), rows):
            lengths = tuple(b-a for a, b in zip(lower, upper) if b != inf)
            free = len(axes) - len(lengths)
            if free > dim:
                dim, degree = free, 0
            if free == dim:
                degree += prod(lengths)
            if store_boxes:
                boxes.append((sum(lower), lengths, free))
        dimension += dim
        multiplicity *= degree
        if store_boxes:
            components.append((dim, boxes))
    return dimension, multiplicity, components


def _add(poly, degree, coefficient):
    coefficient += poly.get(degree, 0)
    if coefficient:
        poly[degree] = coefficient
    else:
        poly.pop(degree, None)


def _multiply(left, right, limit):
    product = {}
    for i, a in left.items():
        for j, b in right.items():
            if i+j <= limit:
                _add(product, i+j, a*b)
    return product


def _numerator(components, limit, *, at_one):
    """Sparse truncated numerator at t=0 or z=1-t, factored by coordinate components."""
    result = {0: 1}
    for dim, boxes in components:
        component = {}
        for offset, lengths, free in boxes:
            if at_one:
                # Write the box series over the component denominator z^dim.
                # t^offset=(1-z)^offset and each bounded coordinate contributes
                # (1-(1-z)^length)/z. Only the first limit+1 terms are needed.
                shift = dim-free
                if shift > limit:
                    continue
                box = {k+shift: (-1)**k * comb(offset, k)
                       for k in range(min(offset, limit-shift)+1)}
                for length in lengths:
                    factor = {k: (-1)**k * comb(length, k+1)
                              for k in range(min(length-1, limit-shift)+1)}
                    box = _multiply(box, factor, limit)
            else:
                # Over the full polynomial ring denominator (1-t)^width the
                # box numerator is t^offset times the product of 1-t^length.
                if offset > limit:
                    continue
                box = {offset: 1}
                for length in lengths:
                    if offset+length <= limit:
                        box = _multiply(box, {0: 1, length: -1}, limit)
            for k, coefficient in box.items():
                _add(component, k, coefficient)
        result = _multiply(result, component, limit)
    return result


def _coefficient(degree, poles):
    """Coefficient of t^degree in 1/(1-t)^poles, including a scalar ring."""
    if poles == 0:
        return int(degree == 0)
    return comb(degree+poles-1, poles-1)


def _invariants(width, heads, degrees):
    if (0,)*width in heads:
        return -1, 0, {n: (0, 0) for n in degrees}

    dimension, multiplicity, components = _series_components(width, heads, bool(degrees))
    if not degrees:
        return dimension, multiplicity, {}

    # For a box, offset+sum(length-1) is at most the sum of the coordinate
    # maxima among the heads. Beyond this bound every polynomial part of
    # its rational series vanishes, so the Taylor coefficients at one suffice.
    bound = sum(map(max, zip(*heads))) if heads else 0
    early = {n for n in degrees if n <= bound}
    late = set(degrees) - early
    values = {}
    if early:
        numerator = _numerator(components, max(early), at_one=False)
        for n in early:
            values[n] = tuple(sum(c * _coefficient(n-k, poles) for k, c in numerator.items() if k <= n)
                              for poles in (width+1, width))
    if late:
        numerator = _numerator(components, dimension, at_one=True)
        for n in late:
            values[n] = tuple(sum(c * _coefficient(n, poles-k) for k, c in numerator.items() if k < poles)
                              for poles in (dimension+1, dimension))
    return dimension, multiplicity, values


def handle(operation, ring, operands):
    if ring["field"] != "QQ" or ring["order"] not in ("lex", "grevlex"):
        raise InputError("Hilbert–Samuel calculations require QQ and lex or grevlex order.")
    if operation not in ("hilbert-samuel-multiplicity", "hilbert-samuel-function"):
        raise InputError(f"Unknown Hilbert–Samuel operation {operation!r}.")
    function = operation == "hilbert-samuel-function"
    degrees = operands.get("degrees", list(range(6))) if function else []
    if not isinstance(degrees, list) or any(type(n) is not int or n < 0 for n in degrees):
        raise InputError("Degrees must be a list of nonnegative integers.")

    variables = tuple(map(np.Symbol, ring["variables"]))
    generators = tuple(np.coerce(f) for f in operands["ideal"] if f)
    heads = None
    result = {"ring": ring, "scope": "origin"}
    if "point" in operands:
        point = normalize_point(operands["point"], len(variables))
        result.update(scope="point", point=point)
        shifts = tuple(map(coordinate_fraction, point))
        if any(shifts):
            with cpu_stage("center_at_point", variables=len(variables), generators=len(generators)):
                if len(generators) == 1 and _large_translation(variables, shifts, generators[0]):
                    # Initial ideals of principal germs have one generator.
                    # Its total degree alone determines their Hilbert series.
                    from .point_order import order_at_rational_point
                    order = order_at_rational_point(variables, shifts, generators[0])
                    heads = [(order,) + (0,)*(len(variables)-1)]
                else:
                    generators = centered_ideal(variables, shifts, generators)

    if heads is None:
        heads = _initial_monomials(variables, generators, ring["order"], local=True)
    with cpu_stage("hilbert_samuel_staircase", generators=len(heads), degrees=len(degrees)):
        dimension, multiplicity, values = _invariants(len(variables), heads, degrees)
    result["dimension"] = dimension
    if function:
        result.update(type=operation, degrees=degrees, values=[values[n][0] for n in degrees],
                      graded_values=[values[n][1] for n in degrees], multiplicity=multiplicity)
    else:
        result.update(type="integer", value=multiplicity)
    return result
