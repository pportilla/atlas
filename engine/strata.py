"""Numerical data and Euler characteristics on a certified resolution atlas."""
import re

from .native_library import CHECK, _MODE
from .polynomial import InputError, parse


class InvalidChart(ValueError):
    """The chart equations contradict the claimed resolution."""


def rational_chart_outputs(record, polynomial, *, complex=True, intrinsic=False, integrals=True):
    from .identity_atlas import identity_outputs
    options = dict(complex=complex, intrinsic=intrinsic, integrals=integrals)
    identity = identity_outputs(record, polynomial, **options)
    if identity is not None:
        return identity
    if len(record["variables"]) == 2:
        from .plane_atlas import plane_outputs
        plane = plane_outputs(record, polynomial, **options)
        if plane is not None:
            return plane
    if record.get("engine") == "verified-ideal-blowup" and ("charts" in record or not record.get("atlas")):
        from .graph_record import graph_outputs
        graph = graph_outputs(record, polynomial, **options)
        if graph is not None:
            return graph
    from .point_atlas import point_outputs
    point = point_outputs(record, polynomial, **options)
    if point is not None:
        return point
    from .monomial_atlas import fan_outputs
    return fan_outputs(record, polynomial, **options)


def resolution_strata(record, polynomial=None, *, computed=None):
    if not isinstance(record, dict) or record.get("status") != "resolved":
        return None
    variables = record.get("variables")
    if not isinstance(variables, list) or not variables:
        raise InputError("The resolution needs its ordered list of variables.")
    if polynomial is None:
        polynomial = parse(record.get("polynomial"), variables, bounded=False)
    if computed is None and record.get("mode") == "intrinsic" and "embedded_companion" in record:
        # Both sets of charts refer to the same original polynomial. The
        # companion must pass the ordinary embedded verification on re-read.
        companion = record["embedded_companion"]
        if not isinstance(companion, dict) or not isinstance(companion.get("atlas"), list) or not companion["atlas"]:
            raise InputError("The embedded companion needs its resolution atlas.")
        record = dict(status="resolved", mode="embedded", variables=variables, atlas=companion["atlas"])
    try:
        if computed is not None:
            found = computed
        else:
            try:
                certified = rational_chart_outputs(record, polynomial, complex=False)
            except InvalidChart:
                return None
            if certified is not None:
                _, found = certified
            else:
                from . import atlas
                from .local_blowups import GiveUp
                try:
                    if (charts := atlas.from_record(record, polynomial, native=_MODE not in ("check", "sympy"))) is None:
                        # Archived resolutions from before chart ownership was serialized.
                        proof = record.get("certificate", {})
                        if proof.get("numerical_data") and proof.get("strata") and not record.get("input_divisor"):
                            return proof["numerical_data"], proof["strata"], "Certified strata of the archived resolution"
                        return None
                    found = atlas.outputs(charts, len(variables), complex=False)
                except GiveUp:
                    return None
    except (IndexError, KeyError, TypeError) as exc:
        raise InputError("Incomplete chart or divisor data in the resolution record.") from exc
    data, strata = found["numerical_data"], found["strata"]
    if CHECK:
        from .strata_reference import IDENTITY_METHOD, resolution_strata as reference
        old = reference(record, polynomial)
        if old is not None and not _same_integrals(data, strata, *old[:2]):
            # Blowing up a crossing changes the strata. If the reference
            # used the identity, compare the two resolution invariants.
            if old[2] != IDENTITY_METHOD:
                raise AssertionError("The atlas and the reference resolution strata give different zeta integrals.")
            if _zeta_integrals(data, strata) != _zeta_integrals(*old[:2]):
                raise AssertionError("The atlas and the identity give different zeta functions.")
    return data, strata, "Euler characteristics on the disjoint parts of the resolution atlas"


def _zeta_integrals(data, strata):
    from .topological import _acampo, _reference
    numbers = {r["component"]: (r["multiplicity"], r["nu"]) for r in data}
    value = _reference(numbers, strata)
    if value is None:
        raise AssertionError("The topological zeta integral fails Z(0) = 1.")
    return _acampo(data, strata), value[0]


def _same_integrals(data, strata, old_data, old_strata):
    # The reference algorithms can group disjoint branches of equal weight.
    # Relabel by numerical data before comparing the sums of their strata.
    def group(numbers, rows):
        by_name = {r["component"]: (r["multiplicity"], r["nu"]) for r in numbers}
        totals = {}
        for row in rows:
            key = tuple(sorted(by_name[c] for c in row["components"]))
            totals[key] = totals.get(key, 0) + row["euler_characteristic"]
        return {k:v for k,v in totals.items() if v}
    return group(data, strata) == group(old_data, old_strata)


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


def _vertex_aliases(found, old):
    """Match named branches to an independently computed graph. Preserve the
    parent divisor, loops and every edge multiplicity under the relabelling."""
    from collections import Counter
    if any(c.get("dimension", len(c["vertices"])-1) != 1 for g in (found, old) for c in g["cells"]):
        return None
    names = {v["id"] for v in found["vertices"]}
    candidates = {}
    for v in old["vertices"]:
        name = v["id"]
        if name in names:
            candidates[name] = [name]
            continue
        parents = v.get("ambient_components") or ([v["ambient_divisor"]] if "ambient_divisor" in v else [])
        if not parents and name.endswith(("u", "v")):
            parents = [name[:-1]]
        candidates[name] = sorted(k for k in names if any(k == p or
                                  (k.startswith(p + ".") and k[len(p)+1:].isdigit()) for p in parents))
    if len(candidates) != len(names) or not all(candidates.values()):
        return None
    if not old["cells"] and not found["cells"]:
        # Intrinsic curves have only points. Within each disjoint group of
        # indistinguishable points the order has no incidence information.
        parts = {}
        for k, values in candidates.items():
            parts.setdefault(tuple(values), []).append(k)
        pairs = {a: b for values, keys in parts.items() if len(values) == len(keys)
                 for a, b in zip(values, keys)}
        return pairs if len(pairs) == len(names) and len(set(pairs.values())) == len(names) else None
    def adjacency(g):
        a = {v["id"]: Counter() for v in g["vertices"]}
        for cell in g["cells"]:
            u, v = cell["vertices"]
            a[u][v] += 1
            a[v][u] += 1
        return a
    source, target = adjacency(old), adjacency(found)
    chosen = {}
    def match():
        if len(chosen) == len(names):
            return {b: a for a, b in chosen.items()}
        key = min((k for k in candidates if k not in chosen),
                  key=lambda k: (len(candidates[k]), -sum(n in chosen for n in source[k])))
        for v in candidates[key]:
            if (v in chosen.values() or sorted(source[key].values()) != sorted(target[v].values()) or
                    source[key][key] != target[v][v] or
                    any(source[key][k] != target[v][w] for k, w in chosen.items())):
                continue
            chosen[key] = v
            answer = match()
            if answer is not None:
                return answer
            del chosen[key]
        return None
    return match()


def annotate_complex(old, found):
    """Preserve compatible names and annotations on a computed complex."""
    from collections import Counter
    if old["status"] != "complete" or found["status"] != "complete":
        return found
    if CHECK:
        def dimensions(g):
            return len(g["vertices"]), Counter(len(c["vertices"])-1 for c in g["cells"])
        if dimensions(found) != dimensions(old):
            raise AssertionError("The atlas and the reference have different cell counts.")
    old_vertices = {v["id"]: v for v in old["vertices"]}
    aliases = {}
    if set(old_vertices) != {v["id"] for v in found["vertices"]}:
        aliases = _vertex_aliases(found, old)
        if aliases is None:
            return found
        found = {**found, "cells": [{**cell, "vertices": [aliases[u] for u in cell["vertices"]],
                                     "faces": [aliases.get(f, f) for f in cell["faces"]]} for cell in found["cells"]]}
    proved = {aliases.get(v["id"], v["id"]): v for v in found["vertices"]}
    found = {**found, "vertices": [{**old_vertex, **proved[name], "id": name}
                                    for name, old_vertex in old_vertices.items()]}
    if CHECK and Counter(frozenset(c["vertices"]) for c in old["cells"]) != Counter(
            frozenset(c["vertices"]) for c in found["cells"]):
        raise AssertionError("The atlas and the reference have different cell incidences.")
    old_cells = {}
    for cell in old["cells"]:
        old_cells.setdefault(frozenset(cell["vertices"]), []).append(cell)
    for cell in found["cells"]:
        matches = old_cells.get(frozenset(cell["vertices"]))
        if matches:
            annotation = matches.pop(0)
            for k, v in annotation.items():
                if k not in ("id", "faces", "vertices", "dimension"):
                    cell[k] = v
    return {**old, **found}
