"""Adapter for the independent C++/GMP A-series surface resolution."""
import ctypes
from functools import lru_cache
from pathlib import Path
import sys

from .polynomial import integer_text, rational_text

_COORDS = ("a", "b", "c")
_LIBRARY = Path(__file__).resolve().parents[1] / "native" / "build" / (
    "libchain.dylib" if sys.platform == "darwin" else "libchain.so"
)


@lru_cache(maxsize=1)
def _load():
    library = ctypes.CDLL(str(_LIBRARY))
    library.chain_resolve.argtypes = [ctypes.c_char_p, ctypes.c_int]
    library.chain_resolve.restype = ctypes.c_void_p
    library.chain_free.argtypes = [ctypes.c_void_p]
    library.chain_free.restype = None
    return library


def _read_atlas(lines, variables, unit, mode):
    atlas, centers, cells, vertices = {}, {}, [], {}
    for line in lines:
        kind, *args = line.split("\t")
        if kind == "chart":
            name, parent, step, leaf = args
            chart = {"id": name, "coordinates": list(_COORDS), "map": {}, "map_to_parent": {},
                     "parent": parent, "step": step, "final": leaf == "1",
                     "exceptional": [], "ambient_components": []}
            atlas[name] = chart
        elif kind == "center":
            centers[name] = args
        elif kind == "cell":
            dimension, *corners = args
            cells.append({"dimension": int(dimension), "vertices": corners})
        elif kind == "map":
            chart["map"][variables[len(chart["map"])]] = args[0]
        elif kind == "ray":
            chart.setdefault("split_cone", []).append(list(map(int, args)))
        elif kind == "parentmap":
            chart["map_to_parent"][_COORDS[len(chart["map_to_parent"])]] = args[0]
        elif kind == "strict":
            chart["strict_transform"] = args[0]
        elif kind == "factor":
            prefix = "" if unit == "1" else f"({unit})*"
            chart["ambient_total_transform"] = f"{prefix}{args[0]}*({chart['strict_transform']})"
        elif kind == "ambient":
            label, eq, order = args
            component = {"id": label, "equation": eq, "ideal": [eq], "multiplicity": int(order)}
            chart["ambient_components"].append(component)
            if mode == "embedded":
                chart["exceptional"].append(component)
                if chart["final"]:
                    vertices.setdefault(label, {"id": label, "label": label, "multiplicity": int(order)})
        elif kind == "curve":
            label, genus, intersection, *eqs = args
            chart["exceptional"].append({"id": label, "equation": ", ".join(eqs), "ideal": eqs})
            if chart["final"]:
                vertices.setdefault(label, {"id": label, "label": label, "genus": int(genus),
                                             "self_intersection": int(intersection)})
    return atlas, centers, vertices, cells


def _steps(atlas, centers, variables):
    steps = {}
    for chart in atlas.values():
        if not chart["parent"]:
            continue
        name = chart["step"]
        step = steps.setdefault(name, {"id": name, "center_charts": []})
        is_line = name == "L"
        step["type"] = "line blowup" if is_line else "point blowup"
        if is_line:
            step["center"] = "The line u = 0 in the projective directions of the preceding point blowup."
        elif name == "P1":
            step["center"] = "origin"
            step["center_ideal"] = list(variables)
        else:
            step["center"] = "The origin of the preceding z chart."
        chart_id = chart["parent"]
        if any(c["chart"] == chart_id for c in step["center_charts"]):
            continue
        parent = atlas[chart_id]
        step["center_charts"].append({
            "chart": chart_id, "coordinates": list(_COORDS), "ideal": centers[chart["id"]],
            "strict_transform": parent["strict_transform"], "map": parent["map"],
        })
    return list(steps.values())


def _complex(vertices, cells, mode):
    convention = ("Reduced ambient exceptional divisor over the origin."
                  if mode == "embedded" else "Reduced exceptional curves on the resolved surface over the origin.")
    return {"status": "complete", "convention": convention+" Components are geometric.",
            "vertices": list(vertices.values()),
            "cells": cells}


def resolve_chain(terms, variables, mode="embedded"):
    """Return a surface resolution or None if this native family does not apply."""
    if mode not in ("embedded", "intrinsic"):
        raise ValueError("mode must be embedded or intrinsic")
    if len(variables) != 3 or not 2 <= len(terms) <= 4 or not _LIBRARY.is_file():
        return None
    entries = terms.items() if isinstance(terms, dict) else terms
    payload = [str(len(terms))]
    for exp, coeff in entries:
        if len(exp) != 3:
            raise ValueError("monomial dimension differs from the number of variables")
        value = coeff if isinstance(coeff, str) else rational_text(coeff)
        payload.append(f"{value} {' '.join(map(integer_text, exp))}")
    lib = _load()
    pointer = lib.chain_resolve("\n".join(payload).encode("ascii"), mode == "intrinsic")
    if not pointer:
        raise RuntimeError("The native surface chain computation could not finish.")
    try:
        lines = ctypes.string_at(pointer).decode("ascii").splitlines()
    finally:
        lib.chain_free(pointer)
    if lines == ["defer"]:
        return None
    _, exponent, unit, discriminant, rank = lines[0].split("\t")
    atlas, centers, vertices, cells = _read_atlas(lines[1:], variables, unit, mode)
    charts = [c for c in atlas.values() if c["final"]]
    steps = _steps(atlas, centers, variables)
    field = {"name": "Q"} if discriminant == "0" else {
        "name": "Q(theta)", "generator": "theta", "minimal_polynomial": f"theta^2-({discriminant})",
        "description": f"theta is either root of theta^2 = {discriminant}.",
    }
    return {
        "status": "resolved", "engine": "native-a-series", "multiplicity": 2,
        "summary": f"A{rank} surface singularity resolved in {len(steps)} blowup"
                   + ("." if len(steps) == 1 else "s."),
        "coefficient_field": field, "steps": steps, "charts": charts,
        "dual_complex": _complex(vertices, cells, mode),
        "certificate": {
            "method": "split binary quadratic and monomial point blowups",
            "normalized_chart": {"id": "N", "coordinates": list(_COORDS),
                                 "map": atlas["N"]["map"], "equation": f"a*b+c^{exponent}", "unit": unit},
            "description": "At each point blowup the z chart lowers the exponent by two. "
                           "Each side chart has a strict equation linear in a direction coordinate.",
        },
        "notes": ["The three affine coordinates are local to each chart. The map gives the images of the original variables.",
                  "Each listed exceptional ideal in intrinsic mode is taken in the strict transform coordinate ring."]
                 if mode == "intrinsic" else
                 ["Every final chart of the ambient blowups is included, with its map to the input coordinates."],
    }
