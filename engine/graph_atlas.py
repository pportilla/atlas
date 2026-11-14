"""Atlas charts of a projective ideal blowup and of the identity map."""
from .reference import sp
from . import native_polynomial as np

from .affine_kernel import decode
from .local_blowups import Chart


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


def graph_charts(record, polynomial):
    """Use the same checked chart record when a caller requests symbolic display."""
    from .atlas import from_json
    from .graph_record import graph_record_charts
    return from_json(graph_record_charts(record, polynomial), native=False)
