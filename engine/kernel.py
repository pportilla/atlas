"""Ideal computations over Q carried out whole in the native kernel.

Each function answers like its namesake in engine/blowup_ideal.py, generator
for generator (native/ideal.cpp follows that file step by step). It returns
None when the kernel cannot take the problem: the library is absent or
switched off with ATLAS_GROEBNER=sympy, the ring has too many variables, an
exponent exceeds the packed range, or the input is not a polynomial over Q.
The caller then computes with SymPy, which also reports malformed input.
"""
import ctypes

import sympy as sp
from sympy.polys.polyerrors import BasePolynomialError

from .groebner import _MAX_VARIABLES, _MODE, _kernel, _line, _poly


class _Unsuitable(Exception):
    pass


def _lines(gens, exprs):
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
        except BasePolynomialError:
            raise _Unsuitable from None
    return lines


def _require(variables):
    """Stop unless the kernel is in use and has room for that many variables."""
    if _MODE == "sympy" or _kernel() is None or not 1 <= variables <= _MAX_VARIABLES:
        raise _Unsuitable


def _ask(header, lines):
    """The lines of the kernel's answer."""
    library = _kernel()
    payload = (header + "\n" + "".join(line + "\n" for line in lines)).encode("ascii")
    answer = library.atlas_ideal(payload, len(payload))
    if not answer:
        raise _Unsuitable
    try:
        return ctypes.string_at(answer).decode("ascii").split("\n")[:-1]
    finally:
        library.atlas_groebner_release(answer)


def _exprs(gens, rows):
    return tuple(_poly(gens, row)[1] for row in rows)


def _optional(function):
    def wrapper(*arguments):
        try:
            return function(*arguments)
        except _Unsuitable:
            return None
    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper


@_optional
def saturate(variables, ideal, by):
    """Generators of I : by^infinity: its reduced lexicographic basis."""
    _require(len(variables) + 1)
    lines = _lines(variables, (*ideal, by))
    return _exprs(variables, _ask(f"saturate {len(variables)} {len(lines) - 1}", lines))


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
    divisors = [tuple(ideal) for ideal in divisors]
    lines = _lines(variables, (*ambient, *(g for ideal in divisors for g in ideal), *fibre, *region))
    sizes = "".join(f" {len(ideal)}" for ideal in divisors)
    rows = _ask(f"crossings {len(variables)} {dimension} {len(tuple(ambient))} {len(divisors)}{sizes} "
                f"{len(tuple(fibre))} {len(tuple(region))}", lines)
    found, at = [], 0
    while at < len(rows) and rows[at] != "none":
        fields = rows[at].split()
        size = int(fields[1])
        count = int(fields[3 + size])
        found.append((tuple(map(int, fields[2:2 + size])), fields[2 + size] == "1",
                      _exprs(variables, rows[at + 1:at + 1 + count])))
        at += 1 + count
    return found


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
    base = tuple(base)
    ideals = [tuple(ideal) for ideal in ideals]
    removed = [tuple(ideal) for ideal in removed]
    lines = _lines(variables, (*base, *(g for ideal in (*ideals, *removed) for g in ideal)))
    rows = _ask(f"strata {len(variables)} {len(base)} {len(ideals)}" + "".join(f" {len(ideal)}" for ideal in ideals)
                + f" {len(removed)}" + "".join(f" {len(ideal)}" for ideal in removed), lines)
    found = {}
    for row in rows:
        if row == "none":
            break
        fields = row.split()
        size = int(fields[1])
        found[tuple(map(int, fields[2:2 + size]))] = int(fields[2 + size])
    return found
