"""Exact witnesses for the local topological monodromy conjecture of one germ."""

from copy import deepcopy

from .monodromy import _request, factor_rows, numerical_rows, stratum_rows
from .polynomial import integer_from_decimal, integer_text, kernel_fraction


def inconclusive_check(reason, resolution=None, topological=None, monodromy=None):
    def outcome(value):
        return value if value and value.get("status") == "computed" else {"status": "inconclusive", "reason": reason}
    return {"status": "inconclusive", "reason": reason, "resolution": resolution,
            "topological_zeta": outcome(topological), "monodromy_zeta": outcome(monodromy),
            "poles": [], "candidate_poles": None, "milnor_number": None}


def _nearby_witness(stratum, exponent):
    """Present a certified local factor in the shared session's witness format."""
    model = stratum["model"]
    if model == "smooth component":
        return {"witness": "nearby smooth component", "witness_component": stratum["component"],
                "witness_multiplicity": stratum["multiplicity"]}
    transverse = {"model": model, "coordinates": tuple(stratum["coordinates"]), "cyclotomic_exponent": exponent}
    if model == "sum of squares":
        transverse.update(parameters=tuple(stratum["parameters"]), parameter_values=tuple(stratum["parameter_values"]),
                          transverse_hessian_determinant=kernel_fraction(stratum["transverse_hessian_determinant"]))
        return {"witness": "nearby transverse Morse", "transverse": transverse}
    if model == "constant transverse type":
        transverse["parameters"] = tuple(stratum["parameters"])
        return {"witness": "nearby constant transverse type", "transverse": transverse}
    if model == "transverse composition":
        transverse.update(transverse_coordinates=tuple(stratum["transverse_coordinates"]),
                          transverse_germ=stratum["transverse_germ"])
        return {"witness": "nearby transverse composition", "transverse": transverse}
    transverse["powers"] = tuple(stratum["powers"])
    return {"witness": "nearby transverse binomial", "transverse": transverse}


def check_zetas(topological, monodromy):
    """Find eigenvalue witnesses for the actual poles of a computed zeta pair."""
    from .contact_zetas import CONTACT_METHODS
    numbers = [r for r in topological["numerical_data"] if not r.get("auxiliary")]
    nearby = monodromy.get("nearby_strata", [])
    evidence = _request("poles", data=numerical_rows(numbers), poles=topological["poles"],
                        factors=factor_rows(monodromy["factors"]), strata=stratum_rows(topological["strata"]),
                        candidate_evidence=monodromy.get("resolution_engine") in CONTACT_METHODS,
                        nearby_factors=[factor_rows(stratum["factors"]) for stratum in nearby])
    poles = []
    for pole, row in zip(topological["poles"], evidence["poles"]):
        q, exponent = integer_from_decimal(row["q"]), integer_from_decimal(row["exponent"])
        branch = numbers[row["branch"]] if row["branch"] is not None else {}
        matches = [{"stratum_index": m["index"], "cyclotomic_exponent": integer_from_decimal(m["exponent"])}
                   for m in row["nearby_matches"]]
        item = {**pole, "eigenvalue_order": q, "cyclotomic_exponent": exponent, "nearby_matches": matches,
                "witness": row["witness"], "witness_component": branch.get("component"),
                "witness_multiplicity": branch.get("multiplicity"), "transverse": None}
        if item["witness"] is None and matches:
            first = matches[0]
            item.update(_nearby_witness(nearby[first["stratum_index"]], first["cyclotomic_exponent"]))
        poles.append(item)
    for candidate in evidence["candidates"]:
        if "cyclotomic_exponent" in candidate:
            for field in ("eigenvalue_order", "cyclotomic_exponent"):
                candidate[field] = integer_from_decimal(candidate[field])
    result = {"status": "verified", "reason": "Every actual pole has a local monodromy eigenvalue witness.",
              "poles": poles, "candidate_poles": evidence["candidates"]}
    if any(pole["witness"] is None for pole in poles):
        result.update(status="unresolved", reason="A nontrivial order is invisible in the origin zeta "
                      "and its nearby monodromy is undetermined.")
    return result


def check_resolution(resolution, *, check_isolation, compute, search="exhaustive", zetas=None):
    record = resolution._record
    topological, monodromy = deepcopy(zetas if zetas is not None else resolution._zeta_pair(search, compute))
    for zeta in (topological, monodromy):
        if zeta.get("status") != "computed":
            reason = (record.get("diagnostic") or {}).get("description") or zeta.get("reason") or record["summary"]
            if search == "quick":
                return inconclusive_check(reason, resolution, topological, monodromy)
            raise RuntimeError(f"The exhaustive zeta calculation ended without a certificate. {reason}")

    result = check_zetas(topological, monodromy)
    result.update(resolution=resolution, topological_zeta=topological, monodromy_zeta=monodromy, milnor_number=None)
    if check_isolation and result["status"] == "unresolved":
        from .session_algebra import milnor_number, polynomial
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
