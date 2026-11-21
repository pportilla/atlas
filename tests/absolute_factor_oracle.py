"""Independent exact comparison of absolute factors presented in different number fields."""
from collections import Counter
import json

import sympy as sp


X, Y, Z, A = sp.symbols("x y z a")
XYZ = (X, Y, Z)


def expression(value, generator="a"):
    return sp.sympify(value.replace("^", "**"), locals={"x": X, "y": Y, "z": Z, generator: A})


def monic(expr):
    return sp.Poly(expr, *XYZ, domain=sp.QQ).monic().as_expr()


def norm(factor, modulus):
    return sp.resultant(sp.Poly(modulus, A).monic().as_expr(), factor, A)


def check_orbit(value, multiplicity, known):
    # In a common splitting field the norm of one factor is a power of
    # its rational orbit product. All calculations here take place over QQ.
    value = monic(value)
    for polynomial, _, power in known:
        if power != multiplicity:
            continue
        n, rem = divmod(sp.Poly(value, *XYZ).total_degree(), sp.Poly(polynomial, *XYZ).total_degree())
        if not rem and n >= 1 and sp.expand(value - polynomial**n) == 0:
            return polynomial
    raise ValueError(f"Unmatched factor of multiplicity {multiplicity}, with norm {value}")


def check(case, solver, output):
    known = [(monic(expression(f)), size, power) for f, size, power in case.orbits]
    expected = {f: size for f, size, _ in known}
    counted = Counter()
    p = expression(case.expression)
    if solver == "Singular":
        unit = next(line.removeprefix("@unit ") for line in output.splitlines() if line.startswith("@unit "))
        product = sp.Rational(unit)
        rows = [line.split("|") for line in output.splitlines() if line.startswith("@orbit|")]
        for _, power, minimal, representative in rows:
            modulus, value = expression(minimal), expression(representative)
            orbit = norm(value, modulus)
            product *= orbit**int(power)
            counted[check_orbit(orbit, int(power), known)] += int(sp.degree(modulus, A))
        if sp.expand(product - p) != 0:
            raise ValueError("Singular orbit products do not reconstruct the input")
    else:
        r = json.loads(output)["result"]
        field = r["ring"]["field"]
        generator = "a" if field == "QQ" else field["generator"]
        modulus = A if field == "QQ" else expression(field["minimal_polynomial"], generator)
        product = sp.Rational(r["unit"])
        different = set()
        for row in r["factors"]:
            value, power = expression(row["polynomial"], generator), int(row["multiplicity"])
            orbit = norm(value, modulus)
            counted[check_orbit(orbit, power, known)] += 1
            product = sp.rem(product * value**power, modulus, A).expand()
            # Normalize by the first coordinate-monomial coefficient inside
            # QQ[a]/(modulus), independently of the solver's displayed unit.
            lc = sp.Poly(value, *XYZ).LC()
            normalized = sp.rem(value * sp.invert(lc, modulus, A), modulus, A).expand()
            if normalized in different:
                raise ValueError("Atlas repeated a factor up to multiplication by a field unit")
            different.add(normalized)
        if sp.rem(product - p, modulus, A).expand() != 0:
            raise ValueError("Atlas factors do not reconstruct the input")
    if counted != expected:
        raise ValueError(f"{solver} geometric orbit sizes {counted} differ from {expected}")
