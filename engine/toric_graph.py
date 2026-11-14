"""Marshal polynomial graph data to the common toric chart verifier."""
from .kernel_json import kernel_json
from .polynomial import InputError, format_poly


def _request(operation, variables, equation, graph, **extra):
    return kernel_json("atlas_toric_graph", operation, variables=variables, polynomial=equation,
                       auxiliary=graph["auxiliary"], graph=graph["equation"], divisor=graph["divisor"], **extra)


def resolve_graph(variables, equation, graph):
    from .newton_fan import build_fan
    support = _request("support", variables, equation, graph)
    fan = build_fan([tuple(map(int, row)) for row in support], exhaustive=True)
    proof = _request("charts", variables, equation, graph, fan=fan)
    charts = proof.pop("charts")
    complex_ = proof.pop("dual_complex")
    return {"status": "resolved", "engine": "toric-graph", "mode": "embedded", "schema_version": 1,
            "variables": variables, "polynomial": equation, "origin": [0] * len(variables),
            "base_field": "Q", "geometric_field": "algebraic closure of Q",
            "graph": graph, "fan": fan, "charts": charts,
            "steps": [{"id": "Graph", "type": "polynomial graph isomorphism"},
                      {"id": "Newton", "type": "regular toric modification of the graph"}],
            "summary": f"Embedded resolution on {len(charts)} graph charts meeting the origin fibre.",
            "certificate": {"method": "Jacobian-minor tests on the graph and complete intersection in every torus orbit",
                            "chart_ownership": "Each orbit belongs to the first maximal cone containing it.",
                            "atlas_verification": "Embedded normal crossings on the owned origin fibre", **proof},
            "dual_complex": complex_}


def _replay(operation, record, polynomial):
    if record.get("mode") != "embedded":
        raise InputError("Toric graph records require embedded mode.")
    variables = record["variables"]
    origin = record.get("origin")
    if origin is not None and origin != [0] * len(variables):
        raise InputError("Toric graph records require the coordinate origin.")
    equation = record["polynomial"] if polynomial is None else format_poly(polynomial, variables)
    return _request(operation, variables, equation, record["graph"], fan=record["fan"],
                    saved_charts=record.get("charts"))


def graph_complex(record, polynomial=None):
    """Reconstruct the geometric dual complex from the certified fan and graph."""
    return _replay("incidence", record, polynomial)


def graph_strata(record, polynomial):
    """Rebuild the maps and validate the saved fan, equations and all orbit Jacobians."""
    strata = _replay("strata", record, polynomial)
    return strata["numerical_data"], strata["strata"], "Torus orbit strata of the embedded polynomial graph"
