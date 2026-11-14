"""Materialize the compact fan charts and their owned atlas from one native proof."""
from . import native_library
from .chart_kernel import strata_totals
from .kernel_json import kernel_json
from .point_geometry import encode_terms
from .strata import records


def construct(polynomial, names, fan, integrals):
    n = len(names)
    if (native_library._MODE != "native" or not 3 <= n <= 4 or len(polynomial) > 3
            or len(fan["cones"]) > 4 or len(fan["rays"]) > 4*n
            or any(max(e) > 3 for e in polynomial)
            or any(c.numerator.bit_length() > 64 or c.denominator.bit_length() > 64 for c in polynomial.values())
            or not hasattr(native_library._library(), "atlas_monomial_resolution")):
        return None
    checked = kernel_json("atlas_monomial_resolution", "construct", variables=names, terms=encode_terms(polynomial),
                          fan={"rays":fan["rays"], "cones":fan["cones"]}, integrals=integrals)
    if checked is None:
        return None
    coords = [f"u{i+1}" for i in range(n)]
    frame = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    centres = [[names[i] for i in axes] for axes in checked["centres"]]
    charts, atlas, data, contributions = [], [], {}, []
    for k, (cone, row) in enumerate(zip(fan["cones"], checked["charts"], strict=True), 1):
        components, exceptional = [], []
        for j, order, nu in row["exceptional"]:
            label = f"E{cone[j]-n+1}"
            components.append(dict(id=label, multiplicity=order, nu=nu, ideal=[coords[j]]))
            exceptional.append(dict(id=label, equation=coords[j], total_multiplicity=order))
            data[label] = (order, nu)
        if row["strict"] is not None:
            components.append(dict(id="strict", multiplicity=1, nu=1, ideal=[row["strict"]]))
            data["strict"] = (1, 1)
        if integrals:
            for indices, chi in row["strata"]:
                labels = frozenset(components[i]["id"] for i in indices)
                contributions.append((labels, 1, int(chi)))
        projection = row["projection"]
        mapping = dict(zip(names, projection, strict=True))
        charts.append(dict(id=f"N{k}", coordinates=coords, ray_indices=cone, map=mapping,
                           strict_transform=row["strict"] or "1", exceptional=exceptional))
        atlas.append(dict(coordinates=list(coords), ambient_ideal=[], components=components, fibre=projection,
                          region=row["region"], removed=[], map=dict(mapping),
                          torus=dict(rays=tuple(tuple(fan["rays"][i]) for i in cone), linear_map=frame)))
    boundary = checked["dual_complex"]
    found = dict(reason=None, cells=boundary["cells"], dual_complex=boundary)
    if integrals:
        data, strata = records(dict(data=data, strata=strata_totals(contributions)))
        found.update(numerical_data=data, strata=strata)
    return centres, charts, atlas, found
