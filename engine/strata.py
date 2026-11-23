"""Numerical data and Euler characteristics on a certified resolution atlas."""
import re

from .polynomial import InputError, format_poly, parse, saved_polynomial_input


class InvalidZetaCertificate(InputError):
    """The proposed geometric or numerical certificate contradicts the source germ."""


class InvalidChart(InvalidZetaCertificate):
    """The chart equations contradict the claimed resolution."""


@saved_polynomial_input()
def replay_record_zetas(record, kinds, polynomial=None):
    """Replay or complete a source record through the common zeta entry point."""
    from .resolution_zetas import replay_zetas
    if polynomial is not None:
        names = record.get("variables")
        if not isinstance(names, list) or not names:
            raise InputError("The resolution needs its ordered list of variables.")
        if "polynomial" in record:
            if parse(record["polynomial"], names, bounded=False) != polynomial:
                raise InvalidChart("The supplied polynomial disagrees with the resolution record.")
        else:
            record = dict(record, polynomial=format_poly(polynomial, names))
    return replay_zetas({"record": record, "zetas": list(kinds)})


def rational_chart_outputs(record, polynomial, *, complex=True, intrinsic=False, integrals=True,
                           replay_canonical=False):
    from .identity_atlas import identity_outputs
    options = dict(complex=complex, intrinsic=intrinsic, integrals=integrals)
    if replay_canonical and record.get("engine") == "canonical-blowups":
        from .canonical_replay import canonical_outputs
        return canonical_outputs(record, polynomial, **options)
    if record.get("engine") == "central-plane-arrangement" or any(
            isinstance(chart, dict) and "arrangement_chart" in chart for chart in (record.get("atlas") or ())):
        from .plane_arrangement import arrangement_outputs
        return arrangement_outputs(record, polynomial, integrals)
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


def proposal_chart_outputs(record, polynomial, *, integrals):
    """Verify the owned chart bundle of an ordinary solver proposal."""
    from .local_blowups import GiveUp
    intrinsic = record["mode"] == "intrinsic"
    certified = rational_chart_outputs(record, polynomial, intrinsic=intrinsic, integrals=integrals,
                                       replay_canonical=True)
    if certified is not None:
        return certified
    if record.get("engine") == "canonical-blowups":
        raise GiveUp("The proposal lacks the canonical centre rule needed to reconstruct its chart cover.")
    from . import atlas
    charts, reason = atlas.verified_modification_charts(record, polynomial)
    if charts is None:
        raise GiveUp(reason)
    factor_fibre = any(step.get("rule") == "repeated tangent" for step in record.get("steps", ()))
    width = len(record["variables"])
    try:
        output = atlas.outputs(charts, width, intrinsic=intrinsic, integrals=integrals, factor_fibre=factor_fibre)
    except GiveUp:
        if not intrinsic or not integrals:
            raise
        # A smooth strict transform can need a further embedded resolution.
        output = atlas.outputs(charts, width, intrinsic=True, integrals=False, factor_fibre=factor_fibre)
    return atlas.to_json(charts), output


def resolution_strata(record, polynomial=None, *, computed=None, strict=False):
    """Read embedded strata; strict replay distinguishes invalid proof from missing evidence."""
    if not isinstance(record, dict) or record.get("status") != "resolved":
        return None
    variables = record.get("variables")
    if not isinstance(variables, list) or not variables:
        raise InputError("The resolution needs its ordered list of variables.")
    if polynomial is None:
        polynomial = parse(record.get("polynomial"), variables, bounded=False)
    if computed is None and record.get("engine") == "toric-graph":
        from .toric_graph import graph_strata
        return graph_strata(record, polynomial)
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
            if strict:
                saved = record.get("atlas") or []
                if not isinstance(saved, list) or any(not isinstance(c, dict) for c in saved):
                    raise InvalidChart("The saved atlas must be a list of charts.")
                if any(c.get("repair_projection") or c.get("removed") and
                       (c.get("torus") or record.get("fan")) for c in saved):
                    # A deleted origin-fibre centre and its replacement are
                    # a positive coverage claim. Check it before accepting
                    # any stratum or falling back to the source polynomial.
                    from .atlas import _INCOMPLETE_REPAIR, from_json
                    from .repair_replay import restore_repair_maps
                    if restore_repair_maps(from_json(saved, native=True)) is None:
                        raise InvalidChart(_INCOMPLETE_REPAIR)
            try:
                certified = rational_chart_outputs(record, polynomial, complex=False, replay_canonical=strict,
                                                   intrinsic=strict and record.get("mode") == "intrinsic")
            except InvalidChart:
                if strict:
                    raise
                return None
            if certified is not None:
                _, found = certified
                if "strata" not in found:
                    return None  # A certified intrinsic resolution may still require an embedded companion.
            else:
                from . import atlas
                from .local_blowups import GiveUp
                try:
                    if strict:
                        # Older sources may need a new calculation. A claimed
                        # modification that contradicts its equation is invalid.
                        if record.get("engine") == "canonical-blowups" or not record.get("atlas"):
                            return None
                        charts, reason = atlas.verified_modification_charts(record, polynomial)
                        if charts is None:
                            if reason == atlas._INCOMPLETE_REPAIR:
                                raise InvalidChart(reason)
                            return None
                    else:
                        charts = atlas.from_record(record, polynomial, native=True)
                    if charts is None:
                        # Archived resolutions from before chart ownership was serialized.
                        proof = record.get("certificate", {})
                        if proof.get("numerical_data") and proof.get("strata") and not record.get("input_divisor"):
                            # Archived numerical certificates predate component equations.
                            # Only an owned chart can identify a branch in the original variables.
                            data = [{k: v for k, v in row.items() if k != "equation"} for row in proof["numerical_data"]]
                            return data, proof["strata"], "Certified strata of the archived resolution"
                        return None
                    factor_fibre = any(step.get("rule") == "repeated tangent" for step in record.get("steps", ()))
                    found = atlas.outputs(charts, len(variables), complex=False, factor_fibre=factor_fibre,
                                          intrinsic=strict and record.get("mode") == "intrinsic")
                except GiveUp:
                    return None
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise InvalidChart("Incomplete chart or divisor data in the resolution record.") from exc
    data, strata = found["numerical_data"], found["strata"]
    return data, strata, "Euler characteristics on the disjoint parts of the resolution atlas"


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
    from .incidence import incidence
    # Only the vertex, parent and endpoint labels participate in matching.
    def graph(value):
        return {"vertices": [{k: v[k] for k in ("id", "ambient_components", "ambient_divisor") if k in v}
                             for v in value["vertices"]],
                "cells": [{k: c[k] for k in ("dimension", "vertices") if k in c} for c in value["cells"]]}
    return incidence("aliases", old=graph(old), found=graph(found))


def annotate_complex(old, found):
    """Preserve compatible names and annotations on a computed complex."""
    if old["status"] != "complete" or found["status"] != "complete":
        return found
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
