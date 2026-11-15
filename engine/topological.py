"""The local topological zeta function of Denef and Loeser at the origin.

For an embedded resolution of f whose total transform has the components
E_i, with f vanishing to order N_i along E_i and the pulled-back volume form
to order nu_i - 1,

    Z_top,0(s) = sum over sets I of chi(E_I^o meet the fibre over 0) * prod_{i in I} 1/(N_i*s + nu_i),

where E_I^o is the set of points lying on the components of I and on no
other. The strict transform is one of the components, with N = nu = 1.
(Denef and Loeser, J. Amer. Math. Soc. 5 (1992).)
"""
import ctypes

from . import native_library as gb
from .monodromy import acampo_strata, cyclotomic_exponent
from .polynomial import InputError, integer_from_decimal


def _acampo(data, strata):
    """The exponents {N: -chi} of A'Campo's formula, from the strata on one component."""
    _, factors, _ = acampo_strata(data, strata)
    return {row["multiplicity"]: row["exponent"] for row in factors}


def _polynomial_text(coefficients):
    parts = []
    for degree in reversed(range(len(coefficients))):
        coefficient = coefficients[degree]
        if not coefficient:
            continue
        absolute = abs(coefficient)
        power = "s" if degree == 1 else f"s^{degree}"
        term = str(absolute) if degree == 0 else power if absolute == 1 else f"{absolute}*{power}"
        prefix = (" - " if coefficient < 0 else " + ") if parts else ("-" if coefficient < 0 else "")
        parts.append(prefix + term)
    return "".join(parts) or "0"


def _factor_display(lines):
    content, count = map(int, next(lines).split())
    factors = []
    for _ in range(count):
        exponent, *coefficients = map(int, next(lines).split())
        terms = [(i, a) for i, a in reversed(list(enumerate(coefficients))) if a]
        value = _polynomial_text(coefficients)
        base = f"({value})" if len(terms) > 1 else value
        key = (len(terms), terms)
        factors.append((key, value, base, exponent))
    return content, sorted(factors)


def _product(content, factors, grouped=False):
    parts = [str(content)] if content != 1 or not factors else []
    for _, value, base, exponent in factors:
        if exponent == 1:
            parts.append(base if grouped or len(parts) or len(factors) > 1 else value)
        else:
            parts.append(f"{base}^{exponent}")
    return "*".join(parts)


def _fraction_display(nc, nf, dc, df):
    sign = "-" if nc < 0 else ""
    numerator = _product(abs(nc), nf, grouped=bool(df or dc != 1 or sign))
    if dc == 1 and not df:
        return sign + numerator
    if not nf and nc == 1 and dc == 1 and len(df) == 1 and df[0][3] > 1:
        return f"{df[0][2]}^(-{df[0][3]})"
    denominator = _product(dc, df, grouped=True)
    if len(df) + int(dc != 1) > 1:
        denominator = f"({denominator})"
    return f"{sign}{numerator}/{denominator}"


def _reduce(numbers, strata):
    library = gb._kernel()
    if library is None or not hasattr(library, "atlas_topological"):
        raise ArithmeticError("Build the native topological zeta kernel.")
    operation = library.atlas_topological
    operation.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    operation.restype = ctypes.c_void_p
    positions = {name: i for i, name in enumerate(numbers)}
    rows = [f"{len(numbers)} {len(strata)}", *(f"{n} {nu}" for n, nu in numbers.values())]
    for row in strata:
        indices = [str(positions[name]) for name in row["components"]]
        rows.append(" ".join([str(row["euler_characteristic"]), str(len(indices)), *indices]))
    payload = ("\n".join(rows) + "\n").encode("ascii")
    answer = operation(payload, len(payload))
    if not answer:
        library.atlas_groebner_error.restype = ctypes.c_char_p
        raise InputError(library.atlas_groebner_error().decode())
    try:
        lines = iter(ctypes.string_at(answer).decode("ascii").splitlines())
    finally:
        library.atlas_groebner_release(answer)
    coefficients = next(lines)
    result = None
    if coefficients != "bad origin":
        expanded = _polynomial_text(list(map(int, coefficients.split())))
        nc, nf = _factor_display(lines)
        dc, df = _factor_display(lines)
        value = _fraction_display(nc, nf, dc, df)
        denominator = _product(dc, df)
        poles = []
        names = list(numbers)
        for line in lines:
            root, order, *indices = line.split()
            poles.append({"value": root, "order": int(order), "components": [names[int(i)] for i in indices]})
        result = (value, expanded, denominator, poles)
    return result


def topological_zeta(record, *, polynomial=None, monodromy=None, strata=None):
    """Compute from an Atlas record of an embedded resolution.

    `monodromy` is the monodromy zeta function of the same germ, when known
    (by default the one stored in the record). It is used to mark the poles
    and to check the strata: they must give the same zeta function.
    """
    if not isinstance(record, dict) or "status" not in record:
        raise InputError("Expected an Atlas resolution record.")
    if record["status"] != "resolved":
        return {"status": "unavailable", "reason": "The resolution has not been certified."}
    if strata is None:
        from .strata import resolution_strata
        strata = resolution_strata(record, polynomial)
    found = strata
    if found is None:
        return {"status": "unavailable", "reason":
                "The strata of the total transform over the origin are not determined by this record. "
                "Run atlas resolve with --mode embedded --monodromy."}
    data, strata, method = found
    numbers = {row["component"]: (row["multiplicity"], row["nu"]) for row in data}
    reduced = _reduce(numbers, strata)
    if reduced is None:
        return {"status": "unavailable", "reason": "The strata of this record fail the identity Z(0) = 1."}
    value, numerator, denominator, poles = reduced
    monodromy = monodromy or record.get("monodromy_zeta") or {}
    if monodromy.get("status") == "computed":
        known = {factor["multiplicity"]: factor["exponent"] for factor in monodromy["factors"]}
        if known != _acampo(data, strata):
            return {"status": "unavailable", "reason":
                    "The strata of this record do not give its monodromy zeta function."}
        # exp(2*pi*i*s0) is a zero or pole of the monodromy zeta function at
        # the origin exactly when the cyclotomic polynomial of its order
        # divides an unbalanced product of the factors 1 - T^N.
        for pole in poles:
            # The native kernel returns the canonical rational as a/b or a.
            q = integer_from_decimal(pole["value"].partition("/")[2] or "1")
            balance = cyclotomic_exponent(monodromy["factors"], q)
            pole["in_monodromy_zeta_at_origin"] = balance != 0
    return {"status": "computed", "variable": "s",
            "formula": "sum_I chi(E_I^open intersect pi^(-1)(0)) * prod_{i in I} 1/(N_i*s + nu_i)",
            "expression": value, "numerator": numerator, "denominator": denominator,
            "method": method, "poles": poles, "numerical_data": data, "strata": strata}
