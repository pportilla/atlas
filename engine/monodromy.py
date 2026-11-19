"""Local Lefschetz zeta functions at the origin and on certified nearby strata."""
from .kernel_json import kernel_json
from .point_geometry import decode_terms, encode_terms
from .polynomial import InputError, LimitError, format_poly, integer_from_decimal, integer_text, parse


def _request(operation, **fields):
    return kernel_json("atlas_monodromy", operation, **fields)


def factor_rows(factors):
    return [[integer_text(row["multiplicity"]), integer_text(row["exponent"])] for row in factors]


def stratum_rows(strata):
    return [[row["components"], integer_text(row["euler_characteristic"])] for row in strata]


def numerical_rows(data):
    return [{"component": row["component"], "n": integer_text(row["multiplicity"]), "nu": integer_text(row["nu"])}
            for row in data]


def _decode_factors(pairs):
    return [{"multiplicity": integer_from_decimal(n), "exponent": integer_from_decimal(e)} for n, e in pairs]


def acampo_strata(data, strata):
    result = _request("acampo_strata", data=[[r["component"], integer_text(r["multiplicity"])] for r in data],
                      strata=stratum_rows(strata))
    contributions = [{"component": r["component"], "multiplicity": integer_from_decimal(r["multiplicity"]),
                      "euler_characteristic": integer_from_decimal(r["euler_characteristic"])}
                     for r in result["contributions"]]
    return contributions, _decode_factors(result["factors"]), integer_from_decimal(result["chi"])


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


def _nearby_strata(polynomial, variables, data, strata, origin_factors):
    rows = _request("nearby_strata", width=len(variables), terms=encode_terms(polynomial),
                    data=numerical_rows(data), strata=stratum_rows(strata), origin_factors=factor_rows(origin_factors))
    equations = {row["component"]: row["equation"] for row in data if "equation" in row}
    result = []
    for row in rows:
        factors = _decode_factors(row["factors"])
        stratum = {"model": row["model"], "factors": factors, "expression": _quotient(factors)}
        if "component" in row:
            stratum.update(component=row["component"], multiplicity=integer_from_decimal(row["multiplicity"]))
            if row["component"] in equations:
                stratum["equation"] = equations[row["component"]]
        if "axes" in row:
            axes = row["axes"]
            stratum["coordinates"] = [variables[i] for i in axes]
            stratum["parameters"] = [name for i, name in enumerate(variables) if i not in axes]
        if "powers" in row:
            stratum["powers"] = list(map(integer_from_decimal, row["powers"]))
        if "determinant" in row:
            stratum["transverse_hessian_determinant"] = row["determinant"]
            stratum["parameter_values"] = [1] * len(stratum["parameters"])
        result.append(stratum)
    return result + _transverse_compositions(polynomial, variables)


def _transverse_names(count, variables):
    for letter in "uvwabc":
        names = [f"{letter}{i}" for i in range(1, count + 1)]
        if not set(names) & set(variables):
            return names
    return [f"_u{i}" for i in range(1, count + 1)]


def _origin_factors(polynomial, variables):
    """The origin zeta of a germ in fewer variables, from its own certified resolution."""
    from .resolve import resolve
    try:
        record = resolve({"polynomial": polynomial, "variables": variables, "mode": "embedded",
                          "search": "quick", "zetas": ["monodromy"]})
    except (InputError, LimitError, ArithmeticError, RuntimeError, OSError):
        return None
    zeta = record.get("monodromy_zeta") or {}
    return zeta.get("factors") if zeta.get("status") == "computed" else None


def _transverse_compositions(polynomial, variables):
    # The native search certifies f = G(h) along a critical component. The
    # local zeta at its generic points near 0 is the zeta of G at the origin.
    rows = _request("transverse_compositions", width=len(variables), terms=encode_terms(polynomial))
    result = []
    for row in rows:
        equations = [format_poly(decode_terms(terms), variables) for terms in row["equations"]]
        names = _transverse_names(len(equations), variables)
        germ = format_poly(decode_terms(row["germ"]), names)
        # The native search returns the zeta of a Brieskorn-Pham germ directly.
        factors = _decode_factors(row["factors"]) if "factors" in row else _origin_factors(germ, names)
        if factors is None:
            continue
        result.append({"model": "transverse composition", "factors": factors, "expression": _quotient(factors),
                       "coordinates": equations, "dimension": row["dimension"],
                       "transverse_coordinates": names, "transverse_germ": germ})
    return result


def monodromy_zeta(record, *, polynomial=None, strata=None):
    """Return the origin zeta and certified nearby zetas from an embedded resolution."""
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
    if polynomial is None:
        polynomial = parse(record.get("polynomial"), variables, bounded=False)
    found = resolution_strata(record, polynomial) if strata is None else strata
    if found is None:
        return {"status": "unavailable", "reason": "The atlas does not certify the embedded strata at the origin."}
    data, strata, method = found
    rows, factors, chi = acampo_strata(data, strata)
    return {"status":"computed", "variable":"T", "convention":"prod_q det(I - T*h^* | H^q(F_0; Q))^((-1)^(q+1))",
            "formula":"prod_i (1-T^N_i)^(-chi_c(E_i^open intersect pi^(-1)(0)))",
            "method":method, "resolution_engine":record.get("engine"),
            "expression":_quotient(factors), "factors":factors, "contributions":rows,
            "milnor_fiber_euler_characteristic":chi,
            "nearby_strata":_nearby_strata(polynomial, variables, data, strata, factors)}
