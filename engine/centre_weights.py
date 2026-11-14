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


def invariant_row(values):
    return ["inf" if x == float("inf") else rational_text(x) for x in values]


def maximum_indices(values):
    return level_weights("maximum", (), values=[[invariant_row(value),invariant_row(key)] for value,key in values])


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

    def requirements(self, strata, value):
        def reference():
            from tests.centre_weight_oracle import requirements
            return requirements(self, strata, value)
        def native():
            extra = [rational_text(value), *(" ".join(map(str, (len(ids), *ids))) for ids in strata)]
            return [tuple(map(integer_from_decimal, row.split())) for row in self.request("bm_requirements", extra)]
        return checked_native(native, reference)
