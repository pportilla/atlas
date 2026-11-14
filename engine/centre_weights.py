"""Exact residual exponents, candidate values and jet orders of one BM presentation.

For a collection (e_i, g_i, w_i), extracting its common exceptional part
subtracts w_i * min_j(e_jh / w_j) in each divisor column h. On a stratum
I, the remaining order is the minimum of (ord(g_i) + sum_I e_ih) / w_i.
The order condition at threshold v is therefore an ordinary jet condition
of order ceil(v*w_i - sum_I e_ih) for every row. The chart decides which
of these conditions has a point on its owned open set.
"""
from fractions import Fraction
from math import ceil

from . import native_library
from .polynomial import checked_native


class Presentation:
    def __init__(self, labels, collection):
        self.labels = tuple(labels)
        self.weights = tuple(w for _, _, w in collection)
        self.exponents = tuple(tuple(e[h] for h in labels) for e, _, _ in collection)
        self.rows = (f"{len(labels)} {len(collection)}", *(" ".join(map(str, (w, *row)))
                     for w, row in zip(self.weights, self.exponents)))

    def request(self, operation, extra=()):
        if not hasattr(native_library._library(), "atlas_centre_weights_abi"):
            raise native_library.Unavailable("The installed kernel predates residual weight arithmetic.")
        return native_library.affine_request([operation, *self.rows, *extra])

    def residual(self):
        def reference():
            alpha = tuple(min(row[j]/w for row, w in zip(self.exponents, self.weights))
                          for j in range(len(self.labels)))
            return (alpha, *(tuple(x-a*w for x, a in zip(row, alpha))
                             for row, w in zip(self.exponents, self.weights)))
        def native():
            return tuple(tuple(map(Fraction, s.split())) for s in self.request("bm_residual"))
        return tuple(dict(zip(self.labels, row)) for row in checked_native(native, reference))

    def candidate_values(self, strata, bounds):
        def reference():
            values = {Fraction(0)}
            for chosen, orders in zip(strata, bounds):
                for row, w, order in zip(self.exponents, self.weights, orders):
                    c = sum(row[j] for j in chosen)
                    values.update((k+c)/w for k in range(order+1))
            return sorted(values)
        def native():
            extra = [" ".join(map(str, (len(ids), *ids, *orders))) for ids, orders in zip(strata, bounds)]
            return list(map(Fraction, self.request("bm_values", extra)))
        return checked_native(native, reference)

    def requirements(self, strata, value):
        def reference():
            return [tuple(ceil(value*w-sum(row[j] for j in ids)) for w, row in zip(self.weights, self.exponents))
                    for ids in strata]
        def native():
            extra = [str(value), *(" ".join(map(str, (len(ids), *ids))) for ids in strata)]
            return [tuple(map(int, row.split())) for row in self.request("bm_requirements", extra)]
        return checked_native(native, reference)
