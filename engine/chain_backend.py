"""Adapter for the independent C++/GMP A-series surface resolution."""
import ctypes
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import sys

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
    atlas = {}
    for line in lines:
        kind, *args = line.split("\t")
        if kind == "chart":
            name, parent, step, leaf = args
            chart = {"id": name, "coordinates": list(_COORDS), "map": {}, "map_to_parent": {},
                     "parent": parent, "step": step, "final": leaf == "1",
                     "exceptional": [], "ambient_components": []}
            atlas[name] = chart
        elif kind == "map":
            chart["map"][variables[len(chart["map"])]] = args[0]
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
        elif kind == "curve":
            label, *eqs = args
            chart["exceptional"].append({"id": label, "equation": ", ".join(eqs), "ideal": eqs})
    return atlas


def _steps(atlas, variables):
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
        generators = set()
        for expr in chart["map_to_parent"].values():
            if "*" in expr:
                generators.update(expr.split("*"))
        parent = atlas[chart_id]
        step["center_charts"].append({
            "chart": chart_id, "coordinates": list(_COORDS), "ideal": sorted(generators),
            "strict_transform": parent["strict_transform"], "map": parent["map"],
        })
    return list(steps.values())


def _complex(charts, k, mode):
    vertices = {}
    simplices = set()
    for chart in charts:
        ids = []
        for component in chart["exceptional"]:
            label = component["id"]
            ids.append(label)
            vertex = vertices.setdefault(label, {"id": label, "label": label})
            if mode == "embedded":
                vertex["multiplicity"] = component["multiplicity"]
            else:
                vertex.update({"genus": 0, "self_intersection": -2})
        if mode == "embedded":
            # Each set of exceptional rays in one maximal cone gives a single
            # irreducible toric stratum. Its closure contains exactly its faces.
            for size in range(2, len(ids)+1):
                simplices.update(tuple(sorted(face)) for face in combinations(ids, size))
    if mode == "intrinsic":
        both_sides = range(1, (k+1)//2)
        chain = [f"E{i}u" for i in both_sides]
        if k % 2 == 0:
            chain.append(f"E{k//2}")
        chain.extend(f"E{i}v" for i in reversed(both_sides))
        if set(chain) != set(vertices):
            raise RuntimeError("Curve labels in the affine charts do not cover the exceptional chain.")
        simplices.update(tuple(sorted(pair)) for pair in zip(chain, chain[1:]))
    convention = ("Reduced ambient exceptional divisor over the origin."
                  if mode == "embedded" else "Reduced exceptional curves on the resolved surface over the origin.")
    return {"status": "complete", "convention": convention+" Components are geometric.",
            "vertices": list(vertices.values()),
            "cells": [{"dimension": len(s)-1, "vertices": list(s)}
                      for s in sorted(simplices, key=lambda s: (len(s), s))]}


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
        payload.append(f"{coeff} {' '.join(map(str, exp))}")
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
    _, exponent, unit, discriminant = lines[0].split("\t")
    k = int(exponent)
    atlas = _read_atlas(lines[1:], variables, unit, mode)
    charts = [c for c in atlas.values() if c["final"]]
    steps = _steps(atlas, variables)
    field = {"name": "Q"} if discriminant == "0" else {
        "name": "Q(theta)", "generator": "theta", "minimal_polynomial": f"theta^2-({discriminant})",
        "description": f"theta is either root of theta^2 = {discriminant}.",
    }
    return {
        "status": "resolved", "engine": "native-a-series", "multiplicity": 2,
        "summary": f"A{ k-1 } surface singularity resolved in {len(steps)} blowup"
                   + ("." if len(steps) == 1 else "s."),
        "coefficient_field": field, "steps": steps, "charts": charts,
        "dual_complex": _complex(charts, k, mode),
        "certificate": {
            "method": "split binary quadratic and monomial point blowups",
            "normalized_chart": {"id": "N", "coordinates": list(_COORDS),
                                 "map": atlas["N"]["map"], "equation": f"a*b+c^{k}", "unit": unit},
            "description": "At each point blowup the z chart lowers the exponent by two. "
                           "Each side chart has a strict equation linear in a direction coordinate.",
        },
        "notes": ["The three affine coordinates are local to each chart. The map gives the images of the original variables.",
                  "Each listed exceptional ideal in intrinsic mode is taken in the strict transform coordinate ring."]
                 if mode == "intrinsic" else
                 ["Every final chart of the ambient blowups is included, with its map to the input coordinates."],
    }
