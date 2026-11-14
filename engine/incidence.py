"""Attach oriented faces to the connected exceptional intersections of a chart atlas."""

def dual_cells(counts, order, fixed=None):
    """The cells of dimension at least one of the dual complex with the
    labels as vertices: for a set J of labels whose intersection has m
    components, m cells with the vertices J. `order` sorts the labels and
    `fixed` maps the sets that are cells already known (the vertices and
    cells of a fan) to their names: these are not listed again, and serve
    as faces. Returns (cells, None), or (None, reason) when the counts do
    not determine the complex: a number is unknown, a divisor has several
    components, or an intersection with several components is a face."""
    fixed = fixed or {}

    def name(key):
        if key in fixed:
            return fixed[key]
        return next(iter(key)) if len(key) == 1 else "C" + "-".join(sorted(key, key=order)) + ".1"

    out = []
    for key in sorted(counts, key=lambda k: (len(k), sorted(map(order, k)))):
        if key in fixed:
            continue
        count, labels = counts[key], sorted(key, key=order)
        if count is None:
            return None, "The number of components of " + " ∩ ".join(labels) + " was not determined."
        if len(key) == 1:
            if count != 1:
                return None, f"{labels[0]} has {count} components; the cells are not attached to each of them."
            continue
        faces = [key - {label} for label in labels]
        if any(face not in fixed and counts.get(face) != 1 for face in faces):
            return None, "An intersection with several components is a face of " + " ∩ ".join(labels) + "."
        out.extend({"id": "C" + "-".join(labels) + f".{k}", "dimension": len(key) - 1, "vertices": labels,
                    "faces": [name(face) for face in faces]} for k in range(1, count + 1))
    return out, None

