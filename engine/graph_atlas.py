"""Atlas charts of a projective ideal blowup and of the identity map."""
from .reference import sp
from . import native_polynomial as np

from . import kernel
from .affine_kernel import decode
from .general_resolution import expression
from .groebner import CHECK, _MODE
from .local_blowups import Chart, GiveUp
from .polynomial import InputError


def identity_atlas(record, polynomial, *, native=False):
    names = record["variables"]
    base = tuple(map(np.Symbol if native else sp.Symbol, names))
    factors = record["certificate"]["components"]
    components = []
    for i, factor in enumerate(factors, 1):
        ideal = (decode(base, factor["equation"]),)
        label = "strict" if len(factors) == 1 else f"S{i}"
        components.append((label, factor["multiplicity"], 1, ideal))
    return [Chart(base, (), tuple(components), base, (), tuple(zip(names, base)), ())]


def _orders(coords, base, ambient, prime, f):
    answer = kernel.graph_orders(coords, len(base), ambient, prime, f)
    if answer is None or CHECK:
        from .blowup_ideal import ideal_subset, empty_locus
        from tests.graph_monodromy_oracle import _jacobian_order
        from tests.nonreduced_oracle import _valuation
        from .reference import QQ
        if (ideal_subset(coords, (f,), ambient) or ideal_subset(coords, prime, ambient)
                or empty_locus(coords, (*ambient, *prime))):
            raise GiveUp("The graph needs a nonzero pullback and a nonempty proper divisor.")
        expected = (_valuation(coords, ambient, prime, f, QQ),
                    1 + _jacobian_order(coords, base, ambient, prime, QQ))
        if answer is None:
            return expected
        if answer != expected:
            raise AssertionError("The graph orders differ from their reference values.")
    return answer


def graph_charts(record, polynomial):
    """The first nonzero homogeneous coordinate partitions the fibre. Every
    equation lives over Q. Adjoining a root of the coefficient field's
    minimal polynomial represents all embeddings, with one selected sheet."""
    from .blowup_ideal import saturate
    base_names = list(record["variables"])
    base = tuple(map(sp.Symbol, base_names))
    f = expression(polynomial, base)
    field = record.get("coefficient_field", {})
    coefficient_field = ()
    if field.get("generator"):
        from tests.graph_monodromy_oracle import _field_coefficients
        coeffs = _field_coefficients(field)
        theta = sp.Symbol(field["generator"])
        relation = sp.Add(*(sp.Rational(c)*theta**i for i, c in enumerate(reversed(coeffs))))
        coefficient_field = (theta, relation, len(coeffs) - 1, field.get("exact_root", ""))
    charts = []
    factors = record.get("input_divisor", {}).get("factors", [])
    for item in record["charts"]:
        if not set(base_names).issubset(item["coordinates"]):
            raise InputError("The graph chart does not retain the original coordinates.")
        names = base_names + [v for v in item["coordinates"] if v not in base_names]
        if coefficient_field:
            names.append(theta.name)
        coords = tuple(map(sp.Symbol, names))

        def read(source):
            return decode(coords, source)

        ambient = tuple(map(read, item["ambient_ideal"]))
        if coefficient_field:
            ambient += (relation,)
        projective = tuple(map(read, item["projective_coordinates"]))
        earlier = projective[:projective.index(1)]
        if not set(earlier).issubset(coords):
            raise InputError("Expected coordinate ratios in a standard graph chart.")
        components = []
        for divisor in item.get("ambient_components", item["exceptional"]):
            prime = tuple(map(read, divisor["ideal"]))
            n, nu = _orders(coords, base, ambient, prime, f)
            components.append((divisor["id"], n, nu, prime))
        if len(factors) > 1:
            for k, factor in enumerate(factors, 1):
                g = decode(base, factor["equation"])
                ideal = tuple(saturate(coords, (*ambient, g), read(item["pivot_generator"])))
                if ideal != (1,):
                    components.append((f"S{k}", factor["multiplicity"], 1, ideal))
        else:
            strict = tuple(map(read, item.get("reduced_strict_transform", item["strict_transform"])))
            components.append(("strict", factors[0]["multiplicity"] if factors else 1, 1, strict))
        chart = Chart(coords, ambient, tuple(components), base, earlier, tuple(zip(base_names, base)), ())
        if coefficient_field:
            from dataclasses import replace
            chart = replace(chart, coefficient_field=coefficient_field)
        charts.append(chart)
    return charts
