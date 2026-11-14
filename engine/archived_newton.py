"""Recover the chart partition of a Newton repair saved before ownership was serialized.

Those records contain the centre in the Newton chart, the printed maps of
each final patch, its ambient equations and divisors. Replay the local
blowups starting from that centre. Accept their partition only after
matching all recorded patches bijectively, by equations in their coordinate
rings. The replay uses the finite parameter and its relation from the
archive, which need not be the parameter a new Newton run would choose.
"""
from collections import defaultdict
from dataclasses import replace
import re

from .atlas import _expression, _field_relation
from .blowup_ideal import empty_locus, ideal_basis, radical_subset
from .chart_expressions import chart_expressions
from .constructible_euler import euler_characteristic
from .groebner import groebner
from .local_blowups import GiveUp, _dimension, acampo_over
from .repair_cover import chart_signature, projected_patch
from .repair_parent_cover import certified_parent_cover


def _saved_signature(row, api, axes=None):
    variables = tuple(api.Symbol(name) for name in row["coordinates"])
    if len(variables) != len(set(variables)):
        return None
    symbols = dict(zip(row["coordinates"], axes or variables))

    def read(texts):
        return tuple(_expression(text, symbols) for text in texts)

    components = [(d["id"], d["total_multiplicity"], read(d["ideal"])) for d in row["exceptional"]]
    components.append(("strict", 1, read(row["strict_transform"])))
    return chart_signature(axes or variables, read(row["ambient_ideal"]),
                           [(k, _expression(s, symbols)) for k, s in row["map"].items()], components)


def _replay_centre(parent_index, parent, centre, step, rows):
    if not rows or not step.get("centers"):
        return None
    api = chart_expressions(parent.variables)
    dimension = len(parent.variables)
    # The archived centre may be the graph of a separating parameter on
    # the cover. Eliminate that parameter to recover the closed set in the
    # parent. Reuse its recorded value on each lift in the local replay.
    names = {v.name for v in parent.variables}
    extra = {s for text in centre["ideal"] for s in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)} - names
    if len(extra) > 1:
        return None
    parameters = tuple(api.Symbol(name) for name in extra)
    variables = (*parent.variables, *parameters)
    symbols = {v.name: v for v in variables}
    equations = tuple(_expression(text, symbols) for text in centre["ideal"])
    if ideal_basis(variables, equations) != ideal_basis(variables, tuple(_expression(s, symbols) for s in step["ideal"])):
        return None
    count = step.get("points")
    if type(count) != int or count < 1 or bool(parameters) != (count > 1):
        return None
    ambient = tuple(g for g in equations if parameters and g.free_symbols == set(parameters))
    if parameters and (len(ambient) != 1 or not _field_relation(parameters[0], ambient[0], count)):
        return None
    basis = groebner(equations, (*parameters, *parent.variables), order="lex")
    closed = tuple(g for g in basis.exprs if not g.free_symbols.intersection(parameters))
    if (_dimension(parent.variables, closed) != 0 or _dimension(variables, equations) != 0
            or euler_characteristic(parent.variables, closed) != count
            or euler_characteristic(variables, equations) != count
            or not radical_subset(parent.variables, (*parent.fibre, *parent.region), closed)):
        return None
    try:
        _, _, charts, history = acampo_over(
            variables, parent.components, tuple(g for g in equations if g not in ambient),
            ambient=ambient, dimension=dimension, count_strata=False)
    except GiveUp:
        return None
    if len(step["centers"]) != len(history) or len(charts) != len(rows):
        return None
    labels = {}
    for claimed, proved in zip(step["centers"], history):
        if (claimed.get("stage") != proved["stage"] or claimed.get("dimension") != proved["dimension"]
                or claimed["id"] in labels.values() or claimed["id"] in {c[0] for c in parent.components}):
            return None
        labels[proved["id"]] = claimed["id"]
        if set(claimed.get("components", ())) != {labels.get(k, k) for k in proved["components"]}:
            return None

    # Each expected patch must match exactly one supplied patch. In
    # particular, deleting or duplicating a pivot chart cannot pass just
    # because the other pivots still map onto the same centre.
    expected = {}
    for chart in charts:
        patch = projected_patch(parent_index, parent, chart, labels)
        signature = _signature(patch)
        if signature in expected:
            return None
        expected[signature] = patch
    patches = []
    for row in rows:
        signature = _saved_signature(row, api)
        if signature not in expected:
            # Early files could rename patch variables. The coordinate list
            # proposes an identification by position, then quotient-ring
            # equality tests every equation in that presentation.
            axes = tuple(api.Dummy("patch_coordinate") for _ in row["coordinates"])
            generic = _saved_signature(row, api, axes)
            matches = [key for key, p in expected.items() if len(p.variables) == len(axes)
                       and _signature(p, axes) == generic]
            if len(matches) != 1:
                return None
            signature, = matches
        patch = expected.pop(signature)
        patches.append(patch)
    return closed, patches


def _signature(patch, axes=None):
    variables = axes or patch.variables
    rename = dict(zip(patch.variables, variables)) if axes else {}
    def move(polynomial):
        return polynomial.xreplace(rename) if rename else polynomial
    return chart_signature(variables, tuple(move(g) for g in patch.ambient),
                           [(name, move(f)) for name, f in patch.coordinates],
                           [(name, n, tuple(move(g) for g in ideal)) for name, n, _, ideal in patch.components])


def archived_repair_charts(record, toric_charts):
    """Restore and certify the missing ownership of the early raw Newton format."""
    parents = [c for c in record["charts"] if "ray_indices" in c]
    raw = [c for c in record["charts"] if "ray_indices" not in c]
    if (not toric_charts or len(parents) != len(toric_charts) or not raw
            or len({c["id"] for c in parents}) != len(parents)
            or not certified_parent_cover(toric_charts)):
        return None
    if any(
            key in c for c in raw for key in ("components", "region", "fibre", "removed", "repair_projection")):
        return None
    # Names select candidates from the archive. Their coordinate equations
    # and the independently reproduced divisors below decide every match.
    repair_steps = [s for s in record.get("steps", ()) if "centers" in s]
    steps = {(s["chart"], s["id"]): s for s in repair_steps}
    if len(repair_steps) != len(steps):
        return None
    groups = defaultdict(list)
    for row in raw:
        if not all(k in row for k in ("parent", "coordinates", "ambient_ideal",
                                      "map", "exceptional", "strict_transform")):
            return None
        groups[row["parent"]].append(row)
    root_charts, patches = [], []
    for i, (item, parent) in enumerate(zip(parents, toric_charts)):
        removed = []
        for centre in item.get("blown_up", ()):
            key = item["id"], centre["id"]
            step = steps.pop(key, None)
            if step is None or item["id"] not in groups:
                return None
            # A new exceptional divisor is local to this centre. Reading
            # its name narrows the candidates independently of chart IDs.
            # The complete divisor ideals and maps are verified on replay.
            labels = {s["id"] for s in step["centers"]}
            rows = [row for row in groups[item["id"]] if any(d["id"] in labels for d in row["exceptional"])]
            result = _replay_centre(i, parent, centre, step, rows)
            if result is None:
                return None
            closed, local = result
            if any(not empty_locus(parent.variables, (*ideal, *closed)) for ideal in removed):
                return None
            for row in rows:
                groups[item["id"]].remove(row)
            removed.append(closed)
            patches.extend(local)
        root_charts.append(replace(parent, removed=(*parent.removed, *removed)))
    return None if any(groups.values()) or steps else [*root_charts, *patches]
