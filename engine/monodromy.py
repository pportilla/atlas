"""Local Lefschetz zeta function of the complex Milnor fiber at the origin."""
from collections import Counter

from .polynomial import InputError, parse


def _quotient(factors):
    numerator, denominator = [], []
    for row in factors:
        order, power = row["multiplicity"], row["exponent"]
        term = "(1-T)" if order == 1 else f"(1-T^{order})"
        if abs(power) > 1:
            term += f"^{abs(power)}"
        (numerator if power > 0 else denominator).append(term)
    top = "*".join(numerator) or "1"
    if not denominator:
        return top
    bottom = "*".join(denominator)
    return f"{top}/{bottom}" if len(denominator) == 1 else f"{top}/({bottom})"


def monodromy_zeta(record, *, polynomial=None):
    """Compute from an Atlas record. Intrinsic-only data may need an embedded resolution."""
    if not isinstance(record, dict) or "status" not in record:
        raise InputError("Expected an Atlas resolution record.")
    if record["status"] != "resolved":
        return {"status":"unavailable", "reason":"The resolution has not been certified."}
    variables = record.get("variables")
    if not isinstance(variables, list) or not variables:
        raise InputError("The resolution needs its ordered list of variables.")
    if record.get("origin", [0]*len(variables)) != [0]*len(variables):
        raise InputError("The resolution must be at the origin.")

    engine = record.get("engine")
    if not engine or record.get("mode") not in ("embedded", "intrinsic") or not record.get("charts"):
        raise InputError("The record needs a resolution mode, engine, and charts.")
    try:
        toric = engine in ("binomial-coordinate-blowups", "newton-polyhedron", "native-toric")
        graph = engine == "verified-ideal-blowup" and record["mode"] == "embedded"
        if (toric or graph) and polynomial is None:
            polynomial = parse(record.get("polynomial"), variables, bounded=False)
        if record.get("certificate", {}).get("strata"):
            # The resolution was completed by blowups after a toric
            # modification. A'Campo's formula uses the points of its strata
            # that lie on a single component.
            from .strata import resolution_strata
            found = resolution_strata(record, polynomial)
            rows = None
            if found is not None:
                orders = {row["component"]: row["multiplicity"] for row in found[0]}
                rows = [{"component": row["components"][0], "multiplicity": orders[row["components"][0]],
                         "euler_characteristic": row["euler_characteristic"]}
                        for row in found[1] if len(row["components"]) == 1]
            method = found[2] if found is not None else "Strata computed with the resolution"
        elif toric:
            from .monodromy_toric import toric_contributions
            rows = toric_contributions(record, polynomial, variables)
            method = "Euler characteristics on torus orbits of the regular fan"
        elif graph:
            from .monodromy_graph import graph_contributions
            rows = graph_contributions(record, polynomial)
            method = "Euler integration on disjoint pieces of the projective origin fiber"
        else:
            from .monodromy_blowups import blowup_contributions
            rows = blowup_contributions(record)
            method = "Open exceptional strata of the point blowups"
        if rows is None and not record.get("certificate", {}).get("strata"):
            from .monodromy_special import special_contributions
            rows = special_contributions(record)
            method = ("Open strata of the embedded A-series model" if engine == "native-a-series"
                      else "Open strata of the normal crossing support at the origin")
    except (KeyError, IndexError, TypeError) as exc:
        raise InputError("Incomplete or inconsistent chart or divisor data in the resolution record.") from exc
    if rows is None:
        return {"status":"unavailable", "reason":
                "This record does not certify the open ambient divisor strata needed by A'Campo's formula. "
                "Run atlas resolve with --mode embedded --monodromy."}

    powers = Counter()
    for row in rows:
        n, chi = row["multiplicity"], row["euler_characteristic"]
        if type(n) is not int or n <= 0 or type(chi) is not int:
            raise InputError("Expected a positive divisor multiplicity and an integral Euler characteristic.")
        powers[n] -= chi
    factors = [{"multiplicity":n, "exponent":e} for n,e in sorted(powers.items()) if e]
    return {"status":"computed", "variable":"T", "convention":"prod_q det(I - T*h^* | H^q(F_0; Q))^((-1)^(q+1))",
            "formula":"prod_i (1-T^N_i)^(-chi_c(E_i^open intersect pi^(-1)(0)))",
            "method":method, "resolution_engine":engine,
            "expression":_quotient(factors), "factors":factors, "contributions":rows,
            "milnor_fiber_euler_characteristic":-sum(n*e for n,e in powers.items())}
