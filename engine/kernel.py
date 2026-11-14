"""Ideal computations over Q carried out whole in the native kernel.

Each function answers like its namesake in engine/blowup_ideal.py, generator
for generator (native/ideal.cpp follows that file step by step). It returns
None when the kernel cannot take the problem: the library is absent or
switched off with ATLAS_GROEBNER=sympy, the ring has too many variables, an
exponent exceeds the packed range, or the input is not a polynomial over Q.
Callers can retry in the unpacked native kernel.
"""

from . import native_polynomial as np
from .reference import sp

from .chart_kernel import crossings_rows, strata_rows
from .native_library import Unavailable
from .groebner import _MAX_VARIABLES, _MODE, _kernel, _line


class _Unsuitable(Exception):
    pass


def _lines(gens, exprs):
    if np.native_ring(gens):
        return [_line(gens, expr) for expr in exprs]
    lines = []
    for expr in exprs:
        if not isinstance(expr, sp.Basic):
            if isinstance(expr, str):
                raise _Unsuitable
            try:
                expr = sp.sympify(expr)
            except sp.SympifyError:
                raise _Unsuitable from None
        try:
            lines.append(_line(gens, expr))
        except sp.polys.polyerrors.BasePolynomialError:
            raise _Unsuitable from None
    return lines


def _require(variables):
    """Stop unless the kernel is in use and has room for that many variables."""
    if _MODE == "sympy" or _kernel() is None or not 1 <= variables <= _MAX_VARIABLES:
        raise _Unsuitable


def _ask(header, lines):
    """The lines of the kernel's answer."""
    from .native_library import Unavailable, ideal_request
    try:
        return ideal_request(header, lines)
    except Unavailable as exc:
        raise _Unsuitable from exc


def _exprs(gens, rows):
    from .affine_kernel import _sparse_expression
    return tuple(_sparse_expression(tuple(gens), row) for row in rows)


def _optional(function):
    def wrapper(*arguments):
        try:
            return function(*arguments)
        except (_Unsuitable, Unavailable):
            return None
    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper


@_optional
def dimension(variables, ideal):
    """Krull dimension, with -1 for the empty scheme."""
    try:
        _require(len(variables))
        rows = _ask(f"dimension {len(variables)}", _lines(variables, ideal))
    except _Unsuitable:
        if _MODE == "sympy" or not variables or not hasattr(_kernel(), "atlas_wide_dimension_abi"):
            raise
        from .affine_kernel import _request, _ring
        ring = _ring(tuple(variables))
        rows = _request(["wide_dimension", ring.header, "grevlex", *map(ring.text, ideal)])
    return int(rows[0])


@_optional
def strict_parts(variables, ambient, divisors, functions):
    """The exceptional orders and the strict ideal of each pulled-back factor."""
    _require(len(variables) + 1)
    rows = _ask(f"strict_parts {len(variables)} {len(ambient)} {len(divisors)}" +
                "".join(f" {len(ideal)}" for ideal in divisors),
                _lines(variables, (*ambient, *(g for ideal in divisors for g in ideal), *functions)))
    result, offset = [], 0
    for _ in functions:
        *orders, size = map(int, rows[offset].split())
        offset += 1
        result.append((tuple(orders), _exprs(variables, rows[offset:offset + size])))
        offset += size
    return result


@_optional
def graph_strict_part(variables, ambient, pivot, function):
    """Divide a principal divisor by its exceptional pivot on a graph chart."""
    _require(len(variables) + 1)
    rows = _ask(f"graph_strict_part {len(variables)} {len(ambient)} 1 1",
                _lines(variables, (*ambient, pivot, function)))
    order, size = map(int, rows[0].split())
    if size != 1 or len(rows) != 2:
        raise ArithmeticError("Expected a principal residual ideal on the chart.")
    return order, _exprs(variables, rows[1:])


@_optional
def residual_dimension_below(variables, base, residual, threshold):
    """Whether the dimension after adding the residual ideal is below the threshold."""
    _require(len(variables))
    rows = _ask(f"residual_dimension_below {len(variables)} {threshold} {len(base)}",
                _lines(variables, (*base, *residual)))
    return bool(int(rows[0]))


@_optional
def graph_orders(variables, dimension, ambient, prime, equation):
    """Orders (N, nu) on a graph projecting to its first dimension coordinates."""
    _require(len(variables))
    rows = _ask(f"graph_orders {len(variables)} {dimension} {len(ambient)} {len(prime)}",
                _lines(variables, (*ambient, *prime, equation)))
    return tuple(map(int, rows[0].split()))


@_optional
def saturate(variables, ideal, by):
    """Generators of I : by^infinity: its reduced lexicographic basis."""
    _require(len(variables) + 1)
    lines = _lines(variables, (*ideal, by))
    return _exprs(variables, _ask(f"saturate {len(variables)} {len(lines) - 1}", lines))


@_optional
def finite_radical(variables, ideal):
    """Reduced grevlex basis of the radical of a finite rational ideal."""
    _require(len(variables))
    rows = _ask(f"finite_radical {len(variables)}", _lines(variables, ideal))
    if rows == ["positive_dimension"]:
        from .centres import GiveUp
        raise GiveUp("A centre expected to be finite is not.")
    return _exprs(variables, rows)


@_optional
def radical_subset(variables, smaller, larger):
    _require(len(variables) + 1)
    small = _lines(variables, smaller)
    rows = _ask(f"radical {len(variables)} {len(small)}", small + _lines(variables, larger))
    return rows == ["1"]


def _witness(variables, expected_dimension, rows, at):
    empty, smooth, size = map(int, rows[at].split())
    record = {"expected_dimension": expected_dimension, "empty": bool(empty), "smooth": bool(smooth),
              "bad_ideal": _exprs(variables, rows[at + 1:at + 1 + size])}
    return record, at + 1 + size


@_optional
def smooth_locus(variables, ideal, expected_dimension):
    _require(len(variables) + 1)
    rows = _ask(f"smooth {len(variables)} {int(expected_dimension)}", _lines(variables, ideal))
    return _witness(variables, expected_dimension, rows, 0)[0]


def _divisors(variables, ambient, strict, exceptional):
    """The header fields and the lines of an ambient ideal, a strict transform and divisors."""
    ambient = _lines(variables, ambient)
    strict = None if strict is None else _lines(variables, strict)
    pieces = [_lines(variables, ideal) for ideal in exceptional]
    sizes = [len(ambient), -1 if strict is None else len(strict), len(pieces), *map(len, pieces)]
    return " ".join(map(str, sizes)), ambient + (strict or []) + [line for piece in pieces for line in piece]


@_optional
def verify_snc(variables, ambient, strict, exceptional, ambient_dimension):
    _require(len(variables) + 1)
    if not isinstance(ambient_dimension, int):
        raise _Unsuitable
    sizes, lines = _divisors(variables, ambient, strict, exceptional)
    rows = _ask(f"snc {len(variables)} {int(ambient_dimension)} {sizes}", lines)
    labels = ["strict"] if strict is not None else []
    labels += [f"E{index}" for index in range(1, len(exceptional) + 1)]
    count = int(rows[0].split()[1])
    strata, at = [], 1
    for _ in range(count):
        parts = [int(word) for word in rows[at].split()[1:]]
        witness, at = _witness(variables, ambient_dimension - len(parts), rows, at + 1)
        strata.append({"components": [labels[i] for i in parts], **witness})
    failures = [stratum for stratum in strata if not stratum["smooth"]]
    return {"ok": not failures, "strata": strata, "bad_strata": failures}


@_optional
def blowup_charts(variables, ambient, strict, center, exceptional, coordinates):
    """For each generator of the centre: the ambient ideal, the strict
    transform (or None) and the transforms of the old exceptional ideals,
    in the chart coordinates supplied."""
    _require(len(variables) + len(center))
    sizes, lines = _divisors(variables, ambient, strict, exceptional)
    rows = _ask(f"charts {len(variables)} {sizes}", lines + _lines(variables, center))
    charts, at = [], 0
    for gens in coordinates:
        a, s, *pieces = map(int, rows[at].split())
        at += 1
        chart = [_exprs(gens, rows[at:at + a])]
        at += a
        chart.append(None if s < 0 else _exprs(gens, rows[at:at + s]))
        at += max(s, 0)
        transforms = []
        for size in pieces:
            transforms.append(_exprs(gens, rows[at:at + size]))
            at += size
        chart.append(tuple(transforms))
        charts.append(tuple(chart))
    return charts


def _verify_rows(variables, lines, center_size, boundary_sizes):
    header = " ".join(map(str, ("verify", variables, center_size, len(boundary_sizes), *boundary_sizes)))
    rows = _ask(header, lines)
    if rows[0].startswith("rejected"):
        return None
    charts, at = [], 1
    while at < len(rows):
        a, s = map(int, rows[at].split())
        charts.append((rows[at + 1:at + 1 + a], rows[at + 1 + a:at + 1 + a + s]))
        at += 1 + a + s
    return int(rows[0].split()[1]), charts


@_optional
def verify(variables, f, center, projective, boundary):
    """Whether blowing up the centre resolves the hypersurface f with the
    boundary ideals as its exceptional divisor: None for the kernel's
    inability to say, False for a rejection, otherwise the number of strata
    proved smooth and, for each chart, the lines of its ambient ideal and of
    its strict transform (to be read in the chart's coordinates)."""
    _require(len(variables) + len(center))
    wide = (*variables, *projective)
    lines = _lines(variables, (f, *center))
    for ideal in boundary:
        lines += _lines(wide, ideal)
    answer = _verify_rows(len(variables), lines, len(center), [len(ideal) for ideal in boundary])
    return False if answer is None else answer


@_optional
def verify_lines(variables, lines, center_size, boundary_sizes):
    """verify, for polynomials already written as kernel lines."""
    _require(variables + center_size)
    answer = _verify_rows(variables, lines, center_size, boundary_sizes)
    return False if answer is None else answer


def exprs(gens, rows):
    """The expressions of kernel lines in the given generators."""
    return _exprs(tuple(gens), rows)


@_optional
def surface_step(variables, equations, functions, working, units=(), threefold=0):
    """A step of the procedure for a smooth surface (engine/surface.py), on
    the surface given by the equations: for each open set used, the step
    found there as (kind, top, flag, positions of the labels, generators of
    the closure of the locus). `functions` are the equations of the
    divisors in the order of their labels, `working` the ideals of the points
    considered, `units` functions that are invertible on the part of the
    surface considered; the first `threefold` equations are those of the
    threefold in which the others cut the surface. Returns "nomodel" when no
    two variables are coordinates near some of the points."""
    _require(len(variables) + 1)
    working = [tuple(w) for w in working]
    units = tuple(units)
    lines = _lines(variables, (*equations, *functions, *units, *(g for w in working for g in w)))
    sizes = "".join(f" {len(w)}" for w in working)
    rows = _ask(f"surface {len(variables)} {threefold} {len(equations)} {len(functions)} {len(units)} "
                f"{len(working)}{sizes}", lines)
    if rows == ["nomodel"]:
        return "nomodel"
    found, at = [], 0
    while at < len(rows) and rows[at] != "none":
        fields = rows[at].split()
        labels = int(fields[4])
        count = int(fields[5 + labels])
        found.append((int(fields[1]), int(fields[2]), int(fields[3]), tuple(map(int, fields[5:5 + labels])),
                      _exprs(variables, rows[at + 1:at + 1 + count])))
        at += 1 + count
    return found



@_optional
def crossings(variables, dimension, ambient, divisors, fibre, region):
    """The sets of divisors of the chart V(ambient) whose intersection is not
    smooth of the expected codimension at some point over the fibre, as
    (positions of the divisors, whether such a point lies in the region,
    generators of the ideal of those points)."""
    _require(len(variables))
    found = crossings_rows(len(variables), dimension, _lines(variables, ambient),
                           [_lines(variables, ideal) for ideal in divisors],
                           _lines(variables, fibre), _lines(variables, region))
    return [(indices, inside, _exprs(variables, rows)) for indices, inside, rows in found]


@_optional
def blowup_linear(variables, ambient, places, shifts, divisors, fibre, region, functions):
    """The blowup of the centre x_p = shift (for the positions p of `places`),
    written out in the kernel. For each chart of the blowup, in the order of
    the places: {"divisors": [(order along the new divisor, whether it meets
    the chart, rows)], "fibre": rows, "region": rows, "functions": rows}. The
    rows are polynomials in the variables of the chart, where the new
    coordinate and the ratios take the places of the old ones; read them
    with `exprs`."""
    _require(len(variables) + 1)
    divisors = [tuple(ideal) for ideal in divisors]
    ambient, fibre, region, functions = tuple(ambient), tuple(fibre), tuple(region), tuple(functions)
    lines = _lines(variables, (*ambient, *shifts, *(g for ideal in divisors for g in ideal), *fibre, *region,
                               *functions))
    header = (f"blowup {len(variables)} {len(ambient)} {len(places)} " + " ".join(map(str, places))
              + f" {len(divisors)}" + "".join(f" {len(ideal)}" for ideal in divisors)
              + f" {len(fibre)} {len(region)} {len(functions)}")
    rows = _ask(header, lines)
    charts, at = [], 0
    while at < len(rows):
        at += 1  # "chart j"
        chart = {"divisors": []}
        for _ in divisors:
            _, order, here, count = rows[at].split()
            chart["divisors"].append((int(order), here == "1", rows[at + 1:at + 1 + int(count)]))
            at += 1 + int(count)
        for name in ("fibre", "region", "functions"):
            count = int(rows[at].split()[1])
            chart[name] = rows[at + 1:at + 1 + count]
            at += 1 + count
        charts.append(chart)
    return charts


@_optional
def strata(variables, base, ideals, removed=()):
    """For closed sets V(I), I in `ideals`, of the part V(base) of a chart:
    {positions of some of them: the Euler characteristic with compact
    supports of the points of the part on exactly those}, the nonzero ones.
    The points of the closed sets V(Z), Z in `removed`, are left out."""
    _require(len(variables) + 1)
    return strata_rows(len(variables), _lines(variables, base), [_lines(variables, ideal) for ideal in ideals],
                       [_lines(variables, ideal) for ideal in removed])


@_optional
def plane_components(variables, equations, region, removed):
    """Owned geometric and rational component counts of a coordinate plane curve."""
    if _MODE in ("sympy", "check"):
        from tests.plane_components_oracle import plane_components as reference
    if _MODE == "sympy":
        return reference(variables, equations, region, removed)
    _require(len(variables))
    sizes = [len(equations), len(region), *map(len, removed)]
    rows = _ask("plane_components " + " ".join(map(str, (len(variables), *sizes))),
                _lines(variables, (*equations, *region, *(g for z in removed for g in z))))
    if rows == ["unsupported"]:
        return None
    counts = tuple(map(int, rows[0].split()))
    if _MODE == "check" and counts != reference(variables, equations, region, removed):
        raise AssertionError("The plane component counts differ from the reference.")
    return counts
