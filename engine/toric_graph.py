"""Marshal polynomial graph data to the common toric chart verifier."""
from .kernel_json import kernel_json
from .polynomial import InputError, LimitError, format_poly


STRATA_METHOD = "Torus orbit strata of the embedded polynomial graph"


def _request(operation, variables, equation, graph, **extra):
    return kernel_json("atlas_toric_graph", operation, variables=variables, polynomial=equation,
                       auxiliary=graph["auxiliary"], graph=graph["equation"], divisor=graph["divisor"],
                       multiplicity=graph.get("multiplicity","1"), unit=graph.get("unit","1"), **extra)


def resolve_graph(variables, equation, graph, *, verified=None, integrals=False, mode="embedded", fan=None):
    from .newton_fan import build_fan
    if mode not in ("embedded", "intrinsic"):
        raise InputError("Unknown toric graph resolution mode.")
    if fan is None:
        support = _request("support", variables, equation, graph)
        fan = build_fan([tuple(map(int, row)) for row in support], exhaustive=True)
    proof = _request("charts", variables, equation, graph, fan=fan, integrals=integrals, intrinsic=mode=="intrinsic")
    charts = proof.pop("charts")
    complex_ = proof.pop("dual_complex")
    output = {"dual_complex": complex_}
    if integrals:
        output.update(numerical_data=proof.pop("numerical_data"), strata=proof.pop("strata", []))
    record = {"status": "resolved", "engine": "toric-graph", "mode": mode, "schema_version": 1,
            "variables": variables, "polynomial": equation, "origin": [0] * len(variables),
            "base_field": "Q", "geometric_field": "algebraic closure of Q",
            "graph": graph, "fan": fan, "charts": charts,
            "steps": [{"id": "Graph", "type": "polynomial graph isomorphism"},
                      {"id": "Newton", "type": "regular toric modification of the graph"}],
            "summary": f"{mode.capitalize()} resolution on {len(charts)} graph charts meeting the origin fibre.",
            "certificate": {"method": "Jacobian-minor tests on the graph and complete intersection in every torus orbit",
                            "chart_ownership": "Each orbit belongs to the first maximal cone containing it.",
                            "atlas_verification": "Embedded normal crossings on the owned origin fibre", **proof},
            "dual_complex": complex_}
    if verified is not None:
        verified.append((record, ([], output)))
    return record


def try_graphs(variables, equation, faces, verified, integrals, mode):
    """Verify exact graph lifts proposed from degenerate Newton faces."""
    if not faces:
        return None
    proposals = kernel_json("atlas_toric_graph", "candidates", variables=variables,
                            polynomial=equation, faces=faces)
    for graph in proposals:
        try:
            return resolve_graph(variables, equation, graph, verified=verified, integrals=integrals, mode=mode)
        except (InputError, LimitError):
            continue
    return None


def _replay(operation, record, polynomial, **options):
    if record.get("mode") not in ("embedded", "intrinsic"):
        raise InputError("Toric graph records require embedded or intrinsic mode.")
    variables = record["variables"]
    origin = record.get("origin")
    if origin is not None and origin != [0] * len(variables):
        raise InputError("Toric graph records require the coordinate origin.")
    equation = record["polynomial"] if polynomial is None else format_poly(polynomial, variables)
    return _request(operation, variables, equation, record["graph"], fan=record["fan"],
                    saved_charts=record.get("charts"), intrinsic=record["mode"]=="intrinsic", **options)


def graph_outputs(record, polynomial, integrals):
    """Rebuild the graph charts, incidence and requested integrals together."""
    output = _replay("outputs", record, polynomial, integrals=integrals)
    return [], output


def graph_complex(record, polynomial=None):
    """Reconstruct the geometric dual complex from the certified fan and graph."""
    return _replay("incidence", record, polynomial)


def graph_strata(record, polynomial):
    """Rebuild the maps and validate the saved fan, equations and all orbit Jacobians."""
    strata = _replay("strata", record, polynomial)
    return strata["numerical_data"], strata.get("strata", []), STRATA_METHOD
