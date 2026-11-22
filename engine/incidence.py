"""Attach oriented faces to the connected exceptional intersections of a chart atlas."""

import ctypes
from functools import lru_cache

from .kernel_json import kernel_json
from .polynomial import integer_text


def dual_cells(counts, order, fixed=None):
    """The cells of dimension at least one of the dual complex with the
    labels as vertices: for a set J of labels whose intersection has m
    components, m cells with the vertices J. `order` sorts the labels and
    `fixed` maps the sets that are cells already known (the vertices and
    cells of a fan) to their names: these are not listed again, and serve
    as faces. Returns (cells, None), or (None, reason) when the counts do
    not determine the complex: a number is unknown, a divisor has several
    components, or an intersection with several components is a face."""
    labels = set().union(*counts, *(fixed or {}))
    rows = _intersections(counts)
    result = incidence("cells", labels=sorted(labels, key=order), counts=rows,
                       fixed=[[list(key), name] for key,name in (fixed or {}).items()])
    return result["edges"], result["reason"]


def _intersections(counts):
    return [[list(key), None if n is None else integer_text(n)] for key, n in counts.items()]


def incidence(operation, **fields):
    return kernel_json("atlas_incidence", operation, **fields)


def counted_boundary(counts, orbits, dimension, intrinsic, marked, *, charts=()):
    labels = [next(iter(key)) for key in counts if len(key) == 1]
    from .chart_gluing_kernel import chart_input
    return incidence("boundary", labels=labels, counts=_intersections(counts), dimension=dimension,
                     intrinsic=intrinsic, marked=marked, charts=[chart_input(c) for c in charts],
                     orbits={next(iter(k)): v == 1 for k,v in orbits.items() if len(k) == 1})


@lru_cache(maxsize=128)
def stratum_dimension(ambient, codimension):
    return incidence("dimension", ambient=ambient, removed=codimension)


_ComparePieces = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t)


def component_classes(sections, compare):
    """Ask the native partition to evaluate only the necessary chart overlaps."""
    from .native_library import _library
    if not sections:
        return ()
    labels = {}
    array = ctypes.c_size_t * len(sections)
    charts = array(*(chart for chart, _ in sections))
    strata = array(*(labels.setdefault(label, len(labels)) for _, label in sections))
    classes = array()
    errors = []

    @_ComparePieces
    def compare_pair(i, j):
        # ctypes cannot propagate an exception across a C++ stack frame.
        try:
            return compare(i, j)
        except BaseException as exc:
            errors.append(exc)
            return -2

    partition = _library().atlas_partition_components
    indices = ctypes.POINTER(ctypes.c_size_t)
    partition.argtypes = [ctypes.c_size_t, indices, indices, _ComparePieces, indices]
    partition.restype = ctypes.c_int
    status = partition(len(sections), charts, strata, compare_pair, classes)
    if errors:
        raise errors[0]
    if status:
        raise ArithmeticError("Unable to partition the geometric components.")
    return tuple(classes)
