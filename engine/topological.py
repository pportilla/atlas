"""The local topological zeta function of Denef and Loeser at the origin.

For an embedded resolution of f whose total transform has the components
E_i, with f vanishing to order N_i along E_i and the pulled-back volume form
to order nu_i - 1,

    Z_top,0(s) = sum over sets I of chi(E_I^o meet the fibre over 0) * prod_{i in I} 1/(N_i*s + nu_i),

where E_I^o is the set of points lying on the components of I and on no
other. Each strict-transform component has nu = 1 and N equal to the
multiplicity of the corresponding original branch.
(Denef and Loeser, J. Amer. Math. Soc. 5 (1992).)
"""
import ctypes

from . import native_library as gb
from .monodromy import acampo_strata, cyclotomic_exponent
from .polynomial import InputError, integer_from_decimal, integer_text


def _acampo(data, strata):
    """The exponents {N: -chi} of A'Campo's formula, from the strata on one component."""
    _, factors, _ = acampo_strata(data, strata)
    return {row["multiplicity"]: row["exponent"] for row in factors}


def _factor_powers(monodromy):
    return {row["multiplicity"]: row["exponent"] for row in monodromy["factors"]}


def _polynomial_text(coefficients):
    parts = []
    for degree in reversed(range(len(coefficients))):
        coefficient = coefficients[degree]
        if not coefficient:
            continue
        absolute = abs(coefficient)
        power = "s" if degree == 1 else f"s^{degree}"
        term = integer_text(absolute) if degree == 0 else power if absolute == 1 else f"{integer_text(absolute)}*{power}"
        prefix = (" - " if coefficient < 0 else " + ") if parts else ("-" if coefficient < 0 else "")
        parts.append(prefix + term)
    return "".join(parts) or "0"


def _factor_display(lines):
    content, count = map(integer_from_decimal, next(lines).split())
    factors = []
    for _ in range(count):
        exponent, *coefficients = map(integer_from_decimal, next(lines).split())
        terms = [(i, a) for i, a in reversed(list(enumerate(coefficients))) if a]
        value = _polynomial_text(coefficients)
        base = f"({value})" if len(terms) > 1 else value
        key = (len(terms), terms)
        factors.append((key, value, base, exponent))
    return content, sorted(factors)


def _product(content, factors, grouped=False):
    parts = [integer_text(content)] if content != 1 or not factors else []
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
    rows = [f"{len(numbers)} {len(strata)}",
            *(f"{integer_text(n)} {integer_text(nu)}" for n, nu in numbers.values())]
    for row in strata:
        indices = [str(positions[name]) for name in row["components"]]
        rows.append(" ".join([integer_text(row["euler_characteristic"]), str(len(indices)), *indices]))
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
        expanded = _polynomial_text(list(map(integer_from_decimal, coefficients.split())))
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

    An explicitly supplied monodromy result is checked against the same
    germ. Stored results are recomputed from their certificate on replay.
    """
    if not isinstance(record, dict) or "status" not in record:
        raise InputError("Expected an Atlas resolution record.")
    from .strata import InvalidZetaCertificate, replay_record_zetas
    if strata is None:
        paired = (monodromy or {}).get("status") == "computed"
        result = replay_record_zetas(record, ("monodromy", "topological") if paired else ("topological",), polynomial)
        if paired and _factor_powers(monodromy) != _factor_powers(result["monodromy_zeta"]):
            raise InvalidZetaCertificate("The supplied monodromy factors disagree with the recomputed origin zeta.")
        return result["topological_zeta"]
    if record["status"] != "resolved":
        raise InputError("Explicit resolution strata require a certified embedded resolution.")
    data, strata, method = strata
    monodromy = monodromy or {}
    if monodromy.get("status") == "computed":
        if _factor_powers(monodromy) != _acampo(data, strata):
            raise InvalidZetaCertificate("The embedded strata disagree with the supplied monodromy factors.")
    return reduce_terms(data, strata, method, monodromy,
                        "sum_I chi(E_I^open intersect pi^(-1)(0)) * prod_{i in I} 1/(N_i*s + nu_i)")


def reduce_terms(data, strata, method, monodromy, formula):
    """Reduce the certified sum of products of linear denominators."""
    numbers = {row["component"]: (row["multiplicity"], row["nu"]) for row in data}
    reduced = _reduce(numbers, strata)
    if reduced is None:
        from .strata import InvalidZetaCertificate
        raise InvalidZetaCertificate("The strata of this record fail the identity Z(0) = 1.")
    value, numerator, denominator, poles = reduced
    if monodromy.get("status") == "computed":
        # exp(2*pi*i*s0) is a zero or pole of the monodromy zeta function at
        # the origin exactly when the cyclotomic polynomial of its order
        # divides an unbalanced product of the factors 1 - T^N.
        for pole in poles:
            # The native kernel returns the canonical rational as a/b or a.
            q = integer_from_decimal(pole["value"].partition("/")[2] or "1")
            balance = cyclotomic_exponent(monodromy["factors"], q)
            pole["in_monodromy_zeta_at_origin"] = balance != 0
    return {"status": "computed", "variable": "s",
            "formula": formula,
            "expression": value, "numerator": numerator, "denominator": denominator,
            "method": method, "poles": poles, "numerical_data": data, "strata": strata}
