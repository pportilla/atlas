"""Transport and presentation of local zeta certificates from plane discriminant resolutions."""
import json
import subprocess

from . import native_process
from .exact_json import result_json
from .monodromy import _decode_factors, _quotient, _request
from .polynomial import InputError, LimitError, format_poly, integer_from_decimal, integer_text, parse, rational_text
from .resolution_zetas import with_pole_check
from .topological import reduce_terms


CONTACT_METHODS = {
    "three-graph-contact": "Integration over the embedded plane resolution of the three pairwise root differences",
    "four-graph-contact": "Integration over the embedded plane resolution of the six pairwise root differences",
    "quadratic-translate-contact": "Integration of six conjugate roots over the resolved plane discriminant",
}


def _engine(proof):
    if proof.get("contact_model") == "quadratic-translates":
        return "quadratic-translate-contact"
    return "four-graph-contact" if proof.get("graph_count") == 4 else "three-graph-contact"


def contact_data(poly, names, *, saved=None):
    if len(names) != 3:
        return None
    payload = {"contact_zetas": True, "variables": names,
               "terms": [[*map(integer_text, powers), rational_text(c)] for powers, c in poly.items()]}
    if saved is not None:
        payload["saved_contact"] = saved
    # This is an optional fast path. A bounded decline leaves the main
    # resolution search available even when the overall search is exhaustive.
    try:
        done = native_process.run("plane", input=result_json(payload), timeout=5 if saved is None else None)
    except FileNotFoundError:
        if saved is not None:
            raise InputError("Build the native plane worker to replay this contact certificate.") from None
        return None
    except subprocess.TimeoutExpired:
        return None
    if done.returncode:
        if saved is not None:
            raise LimitError("The plane worker stopped during contact certificate replay.")
        return None
    answer = json.loads(done.stdout, parse_int=integer_from_decimal)
    if isinstance(answer, dict) and "error" in answer:
        if saved is not None:
            raise InputError(answer["error"])
        return None
    return answer


def _outputs(proof, integration, kinds):
    results = {}
    engine = _engine(proof)
    method = CONTACT_METHODS[engine]
    if "monodromy" in kinds:
        rows = integration["monodromy"]
        product = _request("acampo", rows=[[r["multiplicity"], r["euler_characteristic"]] for r in rows])
        factors = _decode_factors(product["factors"])
        nearby = []
        for row in proof["nearby_strata"]:
            local = _decode_factors(row["factors"])
            nearby.append(dict(row, factors=local, expression=_quotient(local)))
        results["monodromy_zeta"] = {
            "status": "computed", "variable": "T", "method": method, "resolution_engine": engine,
            "convention": "prod_q det(I - T*h^* | H^q(F_0; Q))^((-1)^(q+1))",
            "formula": "Product of transverse and crossing zetas, corrected for blowups of the plane base",
            "expression": _quotient(factors), "factors": factors, "nearby_strata": nearby,
            "contributions": [dict(r, multiplicity=integer_from_decimal(r["multiplicity"]),
                                   euler_characteristic=integer_from_decimal(r["euler_characteristic"])) for r in rows],
            "milnor_fiber_euler_characteristic": integer_from_decimal(product["chi"])}
    if "topological" in kinds:
        data = [{"component": r["component"], "multiplicity": integer_from_decimal(r["n"]),
                 "nu": integer_from_decimal(r["nu"]), **({"auxiliary": True} if r.get("auxiliary") else {})}
                for r in integration["data"]]
        strata = [{"components": names, "euler_characteristic": integer_from_decimal(chi)}
                  for names, chi in integration["strata"]]
        results["topological_zeta"] = reduce_terms(
            data, strata, method, results.get("monodromy_zeta", {}),
            "Sum of signed root-contact integration terms over the plane resolution strata")
    return with_pole_check(results)


def contact_zetas(poly, names, kinds):
    proof = contact_data(poly, names)
    if proof is None:
        return None
    integration = proof.pop("integration")
    from .strata import InvalidZetaCertificate
    try:
        outputs = _outputs(proof, integration, kinds)
    except (InvalidZetaCertificate, LimitError):
        return None
    if any(outputs[kind + "_zeta"]["status"] != "computed" for kind in kinds):
        return None
    return {"contact_certificate": proof, **outputs}


def contact_record(poly, names, kinds):
    result = contact_zetas(poly, names, kinds)
    if result is None:
        return None
    engine = _engine(result["contact_certificate"])
    summary = ("Computed local zetas from the resolved discriminant of three quadratic translates."
               if engine == "quadratic-translate-contact" else
               "Computed local zetas from the plane resolution of the graph contacts.")
    return {"schema_version": 1, "engine": engine, "status": "zetas_computed",
            "summary": summary,
            "polynomial": format_poly(poly, names), "variables": names, "origin": [0] * len(names),
            "mode": "embedded", "base_field": "Q", "geometric_field": "algebraic closure of Q",
            "charts": [], "steps": [],
            "dual_complex": {"status": "unavailable", "convention": "Exceptional divisor over the origin.",
                             "reason": "The contact computation supplies a plane resolution for integration.",
                             "vertices": [], "cells": []}, **result}


def replay_contact(record, kinds):
    names = record.get("variables")
    if (not isinstance(names, list) or len(names) != 3 or record.get("origin") != [0, 0, 0]
            or record.get("mode") != "embedded" or not isinstance(record.get("contact_certificate"), dict)):
        raise InputError("Expected a plane contact certificate at the coordinate origin.")
    poly = parse(record.get("polynomial"), names, bounded=False)
    proof = contact_data(poly, names, saved=record["contact_certificate"])
    if proof is None:
        raise InputError("The polynomial does not certify the saved contacts.")
    integration = proof.pop("integration")
    return _outputs(proof, integration, kinds)
