"""Exact witnesses for the local topological monodromy conjecture of one germ."""

from copy import deepcopy

from .monodromy import _request, factor_rows
from .polynomial import integer_from_decimal, integer_text
from .session_algebra import milnor_number, polynomial


def unavailable_check(reason, resolution=None, topological=None, monodromy=None):
    missing = {"status": "unavailable", "reason": reason}
    return {"status": "unavailable", "reason": reason, "resolution": resolution,
            "topological_zeta": topological or missing, "monodromy_zeta": monodromy or missing,
            "poles": [], "candidate_poles": None, "milnor_number": None}


def _pole_witnesses(topological, monodromy):
    from .local_blowups import _is_strict

    # A strict branch on a nonempty stratum supplies every root of unity
    # whose order divides its multiplicity at a generic smooth point.
    present = {name for row in topological["strata"] if row["euler_characteristic"]
               for name in row["components"]}
    numbers = topological["numerical_data"]
    data = [{"n": integer_text(row["multiplicity"]), "nu": integer_text(row["nu"]),
             "component": row["component"], "branch": _is_strict(row["component"]) and row["component"] in present}
            for row in numbers]
    evidence = _request("poles", data=data, poles=topological["poles"], factors=factor_rows(monodromy["factors"]))
    poles = []
    for pole, row in zip(topological["poles"], evidence["poles"]):
        q, exponent = integer_from_decimal(row["q"]), integer_from_decimal(row["exponent"])
        branch = numbers[row["branch"]] if row["branch"] is not None else {}
        witness = None
        if exponent:
            witness = "origin zeta"
        elif q == 1:
            witness = "origin H^0"
        elif branch:
            witness = "nearby smooth component"
        poles.append({**pole, "eigenvalue_order": q, "cyclotomic_exponent": exponent,
                      "witness": witness, "witness_component": branch.get("component"),
                      "witness_multiplicity": branch.get("multiplicity"), "transverse": None})
    return poles, evidence["candidates"]


def check_zetas(record, topological, monodromy):
    """Find eigenvalue witnesses for the actual poles of a computed zeta pair."""
    poles, candidates = _pole_witnesses(topological, monodromy)
    missing = [pole for pole in poles if pole["witness"] is None]
    if missing:
        from .monodromy_binomial_witness import nearby_binomial_witnesses
        nearby = nearby_binomial_witnesses(record["polynomial"], record["variables"],
                                           {p["eigenvalue_order"] for p in missing})
        for pole in missing:
            if evidence := nearby.get(pole["eigenvalue_order"]):
                pole.update(witness="nearby transverse binomial", transverse=evidence)
        morse_poles = [pole for pole in missing if pole["witness"] is None and pole["eigenvalue_order"] == 2]
        if morse_poles:
            from .monodromy_morse_witness import nearby_morse_witness
            evidence = nearby_morse_witness(record["polynomial"], record["variables"])
            if evidence:
                for pole in morse_poles:
                    pole.update(witness="nearby transverse Morse", transverse=evidence)
        missing = [pole for pole in missing if pole["witness"] is None]
    result = {"status": "verified", "reason": "Every actual pole has a local monodromy eigenvalue witness.",
              "poles": poles, "candidate_poles": candidates}
    if missing:
        result.update(status="unresolved", reason="A nontrivial order is invisible in the origin zeta "
                      "and its nearby monodromy is undetermined.")
    return result


def check_resolution(resolution, *, check_isolation, compute):
    record = resolution._record
    topological, monodromy = deepcopy(resolution._local_zetas)
    if record["status"] != "resolved":
        reason = (record.get("diagnostic") or {}).get("description") or record["summary"]
        return unavailable_check(reason, resolution, topological, monodromy)
    for zeta in (topological, monodromy):
        if zeta["status"] != "computed":
            return unavailable_check(zeta["reason"], resolution, topological, monodromy)

    result = check_zetas(record, topological, monodromy)
    result.update(resolution=resolution, topological_zeta=topological, monodromy_zeta=monodromy, milnor_number=None)
    if check_isolation and result["status"] == "unresolved":
        try:
            f = polynomial(record["polynomial"], variables=resolution.coordinates)
            mu = milnor_number(f, compute=compute or "auto")
        except (ValueError, ArithmeticError, RuntimeError, OSError) as exc:
            result["reason"] += f" The isolation check was unavailable ({exc})."
            return result
        result["milnor_number"] = mu
        if mu.finite:
            # The reduced cohomology of an isolated hypersurface Milnor
            # fibre is concentrated in degree n-1. A nontrivial order
            # cannot cancel there, and nearby smooth fibres only give 1.
            expected_chi = integer_from_decimal(_request("isolated_euler", dimension=len(record["variables"]),
                                                         mu=integer_text(mu.value)))
            if expected_chi == monodromy["milnor_fiber_euler_characteristic"]:
                result.update(status="isolated_candidate", reason="A nontrivial order is missing for a "
                              "finite local Milnor number. Independently check the resolution and zetas.")
            else:
                result["reason"] = "The finite local Milnor number disagrees with the zeta's fibre Euler characteristic."
    return result
