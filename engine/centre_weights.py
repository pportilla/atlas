"""Exact residual exponents, candidate values and jet orders of one BM presentation.

For a collection (e_i, g_i, w_i), extracting its common exceptional part
subtracts w_i * min_j(e_jh / w_j) in each divisor column h. On a stratum
I, the remaining order is the minimum of (ord(g_i) + sum_I e_ih) / w_i.
The order condition at threshold v is therefore an ordinary jet condition
of order ceil(v*w_i - sum_I e_ih) for every row. The chart decides which
of these conditions has a point on its owned open set.
"""
from .kernel_json import kernel_json
from .rational_point import coordinate_fraction

from . import native_library
from .polynomial import checked_native, integer_from_decimal, integer_text, rational_text


def level_weights(operation, weights, **fields):
    return kernel_json("atlas_centre_level", operation, weights=list(map(rational_text, weights)), **fields)


def centre_search(operation_name, query, **fields):
    from .chart_gluing_kernel import operation
    return operation(dict(operation=operation_name, **fields), query, symbol="atlas_centre_search")


def jet_order(initial, live):
    from .centres import GiveUp
    order = centre_search("jet_order", lambda task: live(task["i"]), initial=initial)
    if order is None:
        raise GiveUp("An order is beyond the limit.")
    return order


def divisor_subsets(labels, present, *, largest=False):
    def query(task):
        return present(tuple(labels[j] for j in task["indices"]))
    selected = centre_search("divisor_subsets", query, count=len(labels), largest=largest)
    return [tuple(labels[j] for j in row) for row in selected]


def invariant_row(values):
    return ["inf" if x == float("inf") else rational_text(x) for x in values]


def maximum_indices(values, birth_width=0):
    rows = [None if item is None else [invariant_row(item[0]), invariant_row(item[1])] for item in values]
    return level_weights("maximum", (), values=rows, birth_width=birth_width)


def surface_maxima(steps):
    rows = [None if step is None else [invariant_row(step[0][:3]), step[0][3]] for step in steps]
    return level_weights("surface_maximum", (), values=rows)


def centre_rows(variables, generators):
    from .affine_kernel import frame_function
    from .kernel import _lines
    return _lines(variables, (frame_function(variables, g) for g in generators))


def cover_products(variables, holes):
    from .kernel import exprs
    rows = level_weights("cover_products", (), width=len(variables), holes=[centre_rows(variables, z) for z in holes])
    return list(exprs(variables, rows))


def order_fields(variables, parts):
    """Align chart coordinate identities with the order of expression presentation."""
    from .chart_expressions import chart_expressions
    from .native_polynomial import native_ring
    if native_ring(variables):
        keys = [v._coords[0] for v in variables]
        width = len(str(len(keys)))
        positions = {v: str(i).zfill(width) for i, v in enumerate(sorted(keys))}
        return {"coordinate_keys": [positions[v] for v in keys]}
    # Callers also accept SymPy objects with their own display convention.
    # Only their presentation order is passed to the arithmetic kernel.
    indices = sorted(range(len(parts)), key=lambda i: chart_expressions(variables).default_sort_key(parts[i]))
    rank = dict(zip(indices,range(len(indices))))
    return {"coordinate_keys": [v.name for v in variables], "display_rank": [rank[i] for i in range(len(parts))]}


def ordered_parts(variables, parts, degrees=()):
    fields = order_fields(variables, parts)
    return level_weights("ordered_parts", (), **fields, parts=[centre_rows(variables, part) for part in parts],
                         degrees=[rational_text(d) for d in degrees])


class Presentation:
    def __init__(self, labels, collection):
        self.labels = tuple(labels)
        self.weights = tuple(w for _, _, w in collection)
        self.exponents = tuple(tuple(e[h] for h in labels) for e, _, _ in collection)
        self.rows = (f"{len(labels)} {len(collection)}", *(" ".join(map(rational_text, (w, *row)))
                     for w, row in zip(self.weights, self.exponents)))

    def request(self, operation, extra=()):
        if not hasattr(native_library._library(), "atlas_centre_weights_abi"):
            raise native_library.Unavailable("The installed kernel predates residual weight arithmetic.")
        return native_library.affine_request([operation, *self.rows, *extra])

    def residual(self):
        def reference():
            from tests.centre_weight_oracle import residual
            return residual(self)
        def native():
            return tuple(tuple(map(coordinate_fraction, s.split())) for s in self.request("bm_residual"))
        return tuple(dict(zip(self.labels, row)) for row in checked_native(native, reference))

    def candidate_values(self, strata, bounds):
        def reference():
            from tests.centre_weight_oracle import values
            return values(self, strata, bounds)
        def native():
            extra = [" ".join(map(integer_text, (len(ids), *ids, *orders))) for ids, orders in zip(strata, bounds)]
            return list(map(coordinate_fraction, self.request("bm_values", extra)))
        return checked_native(native, reference)

    def point_maximum(self, point_orders):
        rows = [" ".join(map(integer_text, (len(ids), *ids, *orders))) for ids, orders in point_orders]
        value, = self.request("bm_point_maximum", rows)
        return coordinate_fraction(value)

    def requirements(self, strata, value):
        def reference():
            from tests.centre_weight_oracle import requirements
            return requirements(self, strata, value)
        def native():
            extra = [rational_text(value), *(" ".join(map(str, (len(ids), *ids))) for ids in strata)]
            return [tuple(map(integer_from_decimal, row.split())) for row in self.request("bm_requirements", extra)]
        return checked_native(native, reference)
