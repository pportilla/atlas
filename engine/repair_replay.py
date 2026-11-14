"""Reconstruct missing Newton repair projections using only the chart equations."""
from dataclasses import replace
from functools import lru_cache

from .affine_kernel import pullbacks
from .blowup_ideal import radical_subset
from .local_blowups import _dimension, _present
from .repair_parent_cover import _cone, certified_parent_cover
from .repair_projection import recover_projection


def _lift_charts(charts, group):
    """Assign each supplied repair to the unique parent owning its whole closed image."""
    from .repair_cover import verify_repair_cover
    parents = set(group)
    owners = {j for j in parents if charts[j].removed}
    if any(_dimension(charts[j].variables, hole) != 0 or
           not radical_subset(charts[j].variables, (*charts[j].region, *charts[j].fibre), hole)
           for j in owners for hole in charts[j].removed):
        return None
    restored = list(charts)
    used = set()
    for i, child in enumerate(charts):
        if i in parents:
            continue
        owned = (*child.ambient, *child.region, *child.fibre)
        if not _present(child.variables, owned, child.removed):
            return None
        options = []
        for j in owners:
            parent = charts[j]
            if child.repair_projection and child.repair_projection[0] != j:
                continue
            mapping = recover_projection(child, parent)
            if mapping is None:
                continue
            if child.repair_projection:
                claimed = child.repair_projection[1]
                if len(mapping) != len(claimed) or not radical_subset(
                        child.variables, [a-b for a,b in zip(claimed, mapping)], child.ambient):
                    continue
            holes = tuple(pullbacks(child.variables, mapping, parent.variables, h) for h in parent.removed)
            if not _present(child.variables, owned, (*child.removed, *holes)):
                options.append((j, mapping))
        if len(options) != 1:
            return None
        link, = options
        used.add(link[0])
        restored[i] = replace(child, repair_projection=link)
    if used != owners or not verify_repair_cover(restored):
        return None
    return tuple(restored)


@lru_cache(maxsize=128)
def _restore(charts):
    # New charts already carry the maps. A finite marked chart, an ordinary
    # plane blowup, or a single unmarked cover does not imply a Newton repair.
    if not any(c.removed for c in charts) or any(c.point_blowup or c.coefficient_field for c in charts):
        return charts
    known_parents = {c.repair_projection[0] for c in charts if c.repair_projection}
    ordinary = [i for i,c in enumerate(charts) if not c.repair_projection]
    if all(charts[i].torus or i in known_parents for i in ordinary):
        return charts if all(not charts[i].removed or i in known_parents for i in ordinary) else None

    # Monomial exponent columns recover each candidate regular fan, even
    # when the record omitted the lattice metadata or changed chart order.
    # Charts of a local blowup may also have monomial maps. A candidate
    # must cover the *whole* original fan with its saved orbit ownership.
    names = tuple(sorted(dict(charts[0].coordinates)))
    candidates = {}
    for i in ordinary:
        cone = _cone(charts[i], names)
        if cone is not None:
            candidates.setdefault(cone[0], []).append(i)
    declared = known_parents | {i for i in ordinary if charts[i].torus}
    solutions = []
    found_parent_cover = False
    for group in candidates.values():
        if not declared.issubset(group) or not any(charts[i].removed for i in group):
            continue
        if not certified_parent_cover(charts[i] for i in group):
            continue
        found_parent_cover = True
        if len(group) == len(charts):
            continue
        if (restored := _lift_charts(charts, group)) is not None:
            solutions.append(restored)
    if len(solutions) == 1:
        return solutions[0]
    # If there is a Newton parent or an unmarked finite repair above a
    # deleted parent, an unsuccessful certificate cannot support incidence.
    if declared or found_parent_cover:
        return None
    return charts


def restore_repair_maps(charts):
    """Return charts with certified polynomial parent maps, or None on an unresolved repair."""
    return _restore(tuple(charts))
