"""Draw the computed dual tree around the path between the two strict transforms."""


def _path(neighbors, first, last):
    """The exceptional vertices on the unique path between the attachments."""
    parents = {first: None}
    pending = [first]
    while last not in parents:
        node = pending.pop()
        for child in neighbors[node]:
            if child not in parents:
                parents[child] = node
                pending.append(child)

    path = [last]
    while path[-1] != first:
        path.append(parents[path[-1]])
    return path[::-1]


def _arm_length(neighbors, node, parent):
    # In the welcome family each component off the attachment path is a chain.
    length = 1
    while following := neighbors[node] - {parent}:
        child, = following
        parent, node = node, child
        length += 1
    return length


def graph_rows(result, width):
    """Return the ASCII drawing; o marks an exceptional curve, an arrow a branch."""
    dual = result["dual_complex"]
    neighbors = {v["id"]: set() for v in dual["vertices"]}
    for cell in dual["cells"]:
        a, b = cell["vertices"]
        neighbors[a].add(b)
        neighbors[b].add(a)

    first, last = (b["ambient_components"][0] for b in result["branches"])
    path = _path(neighbors, first, last)
    path_nodes = set(path)
    # Leave the horizontal row for the two strict transforms. If both attach
    # to one vertex, one arrow turns upward and the longer chain extends left.
    if first == last:
        lengths = sorted((_arm_length(neighbors, n, first) for n in neighbors[first]), reverse=True)
        left, below = lengths + [0] * (2 - len(lengths))
        # On a very narrow page send the long chain down to make room for both tips.
        if 2 * left + 3 > width:
            left, below = below, left
        spacing = min(5, (width - 3) // max(1, left))
        arrow = min(3, width - left * spacing - 1)
        spine = ("o" + "-" * (spacing - 1)) * left + "o" + "-" * (arrow - 1) + ">"
        column = left * spacing
        rows = [" " * column + "^", " " * column + "|", spine]
        for _ in range(below):
            rows.extend((" " * column + "|", " " * column + "o"))
        return tuple(rows)

    gaps = len(path) - 1
    # Up to four dashes between exceptional circles on a wide sheet. Reduce
    # that spacing first; transpose the diagram when the path still won't fit.
    spacing = min(5, max(2, (width - 5) // gaps))
    arrow = min(3, max(2, (width - 1 - spacing * gaps) // 2))
    spine = ("<" + "-" * (arrow - 1) + ("-" * (spacing - 1)).join("o" for _ in path)
             + "-" * (arrow - 1) + ">")
    # At distinct attachments all sampled off-path arms extend singly from
    # their own vertex. Each pair of vertical cells is one edge and its end.
    heights = []
    for node in path:
        others = neighbors[node] - path_nodes
        if others:
            neighbor, = others
            heights.append(_arm_length(neighbors, neighbor, node))
        else:
            heights.append(0)
    rows = []
    for level in range(2 * max(heights), 0, -1):
        row = [" "] * len(spine)
        for index, height in enumerate(heights):
            if level <= 2 * height:
                row[arrow + index * spacing] = "o" if level % 2 == 0 else "|"
        rows.append("".join(row).rstrip())
    rows.append(spine)
    if len(spine) <= width:
        return tuple(rows)
    if len(rows) > width:
        raise ValueError("The resolution diagram needs a wider sheet.")
    rotate = str.maketrans("-|<>", "|-^v")
    return tuple("".join(col).translate(rotate).rstrip()
                 for col in zip(*(row.ljust(len(spine)) for row in rows)))
