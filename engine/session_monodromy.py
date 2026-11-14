"""Exact witnesses for the local topological monodromy conjecture of one germ."""

from copy import deepcopy
from fractions import Fraction

from .monodromy import cyclotomic_exponent
from .polynomial import rational_text
from .session_algebra import milnor_number, polynomial


def unavailable_check(reason, resolution=None, topological=None, monodromy=None):
    missing = {"status": "unavailable", "reason": reason}
    return {"status": "unavailable", "reason": reason, "resolution": resolution,
            "topological_zeta": topological or missing, "monodromy_zeta": monodromy or missing,
            "poles": [], "candidate_poles": None, "milnor_number": None}


def _candidate_poles(topological):
    components = {}
    for row in topological["numerical_data"]:
        value = Fraction(-row["nu"], row["multiplicity"])
        components.setdefault(value, []).append(row["component"])
    orders = {Fraction(p["value"]): p["order"] for p in topological["poles"]}
    return [{"value": rational_text(value), "components": components[value], "pole_order": orders.get(value, 0)}
            for value in sorted(components)]


def _pole_witnesses(topological, monodromy):
    from .local_blowups import _is_strict

    # The numerical data used by a nonempty stratum over 0 identify branches
    # through 0. At a generic smooth point of such a branch, f = unit * u^m.
    # Its fibre has m sheets permuted cyclically, so every m-th root occurs.
    present = {name for row in topological["strata"] if row["euler_characteristic"]
               for name in row["components"]}
    branches = [row for row in topological["numerical_data"]
                if _is_strict(row["component"]) and row["component"] in present]
    poles = []
    for pole in topological["poles"]:  # Already reduced by exact cancellation.
        q = Fraction(pole["value"]).denominator
        exponent = cyclotomic_exponent(monodromy["factors"], q)
        component = multiplicity = None
        if exponent:
            witness = "origin zeta"
        elif q == 1:
            witness = "origin H^0"  # Constants on the nonempty Milnor fibre.
        else:
            branch = next((b for b in branches if b["multiplicity"] % q == 0), None)
            witness = None
            if branch:
                witness = "nearby smooth component"
                component, multiplicity = branch["component"], branch["multiplicity"]
        poles.append({**pole, "eigenvalue_order": q, "cyclotomic_exponent": exponent,
                      "witness": witness, "witness_component": component,
                      "witness_multiplicity": multiplicity, "transverse": None})
    return poles


def check_resolution(resolution, *, check_isolation, compute):
    record = resolution._record
    topological, monodromy = deepcopy(resolution._local_zetas)
    if record["status"] != "resolved":
        reason = (record.get("diagnostic") or {}).get("description") or record["summary"]
        return unavailable_check(reason, resolution, topological, monodromy)
    for zeta in (topological, monodromy):
        if zeta["status"] != "computed":
            return unavailable_check(zeta["reason"], resolution, topological, monodromy)

    poles = _pole_witnesses(topological, monodromy)
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
              "resolution": resolution, "topological_zeta": topological, "monodromy_zeta": monodromy,
              "poles": poles, "candidate_poles": _candidate_poles(topological), "milnor_number": None}
    if missing:
        result.update(status="unresolved", reason="A nontrivial order is invisible in the origin zeta "
                      "and its nearby monodromy is undetermined.")
        if check_isolation:
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
                expected_chi = 1 + (-1)**(len(record["variables"]) - 1) * mu.value
                if expected_chi == monodromy["milnor_fiber_euler_characteristic"]:
                    result.update(status="isolated_candidate", reason="A nontrivial order is missing for a "
                                  "finite local Milnor number. Independently check the resolution and zetas.")
                else:
                    result["reason"] = "The finite local Milnor number disagrees with the zeta's fibre Euler characteristic."
    return result
