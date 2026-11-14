"""Local Lefschetz zeta function of the complex Milnor fiber at the origin."""
from .kernel_json import kernel_json
from .polynomial import InputError, integer_from_decimal, integer_text


def _request(operation, **fields):
    return kernel_json("atlas_monodromy", operation, **fields)


def factor_rows(factors):
    return [[integer_text(row["multiplicity"]), integer_text(row["exponent"])] for row in factors]


def acampo_factors(rows):
    inputs = []
    for row in rows:
        n, chi = row["multiplicity"], row["euler_characteristic"]
        if type(n) is not int or n <= 0 or type(chi) is not int:
            raise InputError("Expected a positive divisor multiplicity and an integral Euler characteristic.")
        inputs.append([integer_text(n), integer_text(chi)])
    result = _request("acampo", rows=inputs)
    factors = [{"multiplicity": integer_from_decimal(n), "exponent": integer_from_decimal(e)}
               for n, e in result["factors"]]
    return factors, integer_from_decimal(result["chi"])


def cyclotomic_exponent(factors, order):
    """Valuation at a primitive root of this order of the product of (1 - T^N)^e."""
    return integer_from_decimal(_request("cyclotomic", factors=factor_rows(factors), order=integer_text(order)))


def _quotient(factors):
    numerator, denominator = [], []
    for row in factors:
        order, power = row["multiplicity"], row["exponent"]
        term = "(1-T)" if order == 1 else f"(1-T^{integer_text(order)})"
        if abs(power) > 1:
            term += f"^{integer_text(abs(power))}"
        (numerator if power > 0 else denominator).append(term)
    top = "*".join(numerator) or "1"
    if not denominator:
        return top
    bottom = "*".join(denominator)
    return f"{top}/{bottom}" if len(denominator) == 1 else f"{top}/({bottom})"


def monodromy_zeta(record, *, polynomial=None, strata=None):
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

    saved = isinstance(record.get("atlas"), list) and bool(record["atlas"])
    if record.get("mode") not in ("embedded", "intrinsic") or not (
            saved or (record.get("engine") and record.get("charts"))):
        raise InputError("The record needs its resolution mode and charts.")
    from .strata import resolution_strata
    found = resolution_strata(record, polynomial) if strata is None else strata
    if found is None:
        return {"status": "unavailable", "reason": "The atlas does not certify the embedded strata at the origin."}
    data, strata, method = found
    orders = {r["component"]: r["multiplicity"] for r in data}
    rows = [{"component": row["components"][0], "multiplicity": orders[row["components"][0]],
             "euler_characteristic": row["euler_characteristic"]}
            for row in strata if len(row["components"]) == 1]
    factors, chi = acampo_factors(rows)
    return {"status":"computed", "variable":"T", "convention":"prod_q det(I - T*h^* | H^q(F_0; Q))^((-1)^(q+1))",
            "formula":"prod_i (1-T^N_i)^(-chi_c(E_i^open intersect pi^(-1)(0)))",
            "method":method, "resolution_engine":record.get("engine"),
            "expression":_quotient(factors), "factors":factors, "contributions":rows,
            "milnor_fiber_euler_characteristic":chi}
