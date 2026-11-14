"""Trace SymPy polynomial and number-field operations called by selected Atlas routes."""
import argparse
from collections import Counter
from pathlib import Path
import sys


GROEBNER = {"groebner", "reduced"}
ALGEBRA = {"factor_list", "sqf_part", "gcd", "gcdex", "exquo", "det", "rref", "rank",
           "primitive_element", "minimal_polynomial", "field_isomorphism", "all_roots",
           "real_roots", "count_roots", "resultant", "subresultants", "ground_roots",
           "expand", "diff", "cancel", "simplify"}
REPRESENTATION = {"from_dict", "as_expr", "count_ops"}

GERMS = {
    "ordinary": ("x^2+y^2+z^2", "xyz"),
    "cusp": ("y^2-x^3", "xy"),
    "marked-plane": ("(y^2-2*x^4)^2-x^11", "xy"),
    "chain": ("x^7+y^2-z^2", "xyz"),
    "marked-chain": ("x^2+y^2+z^5", "xyz"),
    "repair": ("z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2", "xyz"),
}
ADDITIONAL = ("graph", "marked-graph", "archived-graph", "centre", "wide",
              "algebraic-literal", "grlex")


def workload(name, symbolic_inputs=False):
    if name in GERMS:
        from engine.resolve import resolve
        source, names = GERMS[name]
        return lambda: resolve(dict(polynomial=source, variables=list(names), mode="intrinsic",
                                    compute="cpu", search="quick", monodromy=True))
    if symbolic_inputs:
        import sympy as api
        domain = api.QQ
    else:
        from engine import native_polynomial as api
        from engine.reference import QQ as domain
    x, y, z, u, v = map(api.Symbol, ("x", "y", "z", "U", "V"))
    if name == "centre":
        from engine.centres import maximum
        f = (y+x**2)**4 + (x-z**2)**6 + z**11
        return lambda: maximum((x,y,z), (), 3, (f,), [], [(x,y,z)], {}, [])
    if name == "wide":
        from engine.local_blowups import _radical_points
        variables = tuple(api.Symbol(f"w{i}") for i in range(33))
        return lambda: _radical_points(variables, (variables[0]**2, *variables[1:]))
    if name == "algebraic-literal":
        if not symbolic_inputs:
            from engine.marked_field import description
            return lambda: description("tan(pi/8)")
        from engine.field_polynomial import lift
        field = api.QQ.algebraic_field(api.sqrt(2))
        value = api.tan(api.pi/8, evaluate=False)
        return lambda: lift(value, (), field, u)
    if name == "grlex":
        from engine.groebner import groebner
        return lambda: groebner((x*y-1, y*y-x), (x,y), order="grlex", domain=domain)

    from engine.general_resolution import verify_candidate
    from engine.monodromy import monodromy_zeta
    from engine.resolution_search import ResolutionCandidate
    if name == "graph":
        center = (y-x**2,z)
        candidate = ResolutionCandidate(center, (u,v), (center,), domain, None, "audit")
        return lambda: verify_candidate((x,y,z), (y-x**2)**2-x*z*z, candidate, "embedded")

    if symbolic_inputs:
        alpha = api.sqrt(2)
        field = api.QQ.algebraic_field(alpha)
    else:
        from engine.candidate_field import indexed_field
        field, alpha = indexed_field((1, 0, -2), 1)
    candidate = ResolutionCandidate((x,y), (u,v), ((x+alpha*y,y),), field, alpha, "audit")

    def solve():
        record = verify_candidate((x,y,z), x*x-y*y*z, candidate, "embedded", effort=3)
        record.update(variables=list("xyz"), polynomial="x^2-y^2*z", mode="embedded")
        if name == "archived-graph":
            del record["coefficient_field"]["minimal_polynomial_coefficients"]
        return monodromy_zeta(record)
    return solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=(*GERMS, *ADDITIONAL))
    parser.add_argument("--atlas-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--symbolic-inputs", action="store_true",
                        help="Exercise compatibility with callers supplying SymPy expressions")
    args = parser.parse_args()
    sys.path.insert(0, str(args.atlas_root.resolve()))
    calculate = workload(args.case, args.symbolic_inputs)
    calls = Counter()
    first_import = []
    watched = GROEBNER | ALGEBRA | REPRESENTATION

    def observe(frame, event, unused):
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        func = frame.f_code.co_name
        imported = module == "sympy" and func == "<module>" and not first_import
        if not imported and (func not in watched or not module.startswith("sympy.")):
            return
        p = frame.f_back
        while p and not p.f_globals.get("__name__", "").startswith("engine."):
            p = p.f_back
        if p:
            site = f"{Path(p.f_code.co_filename).name}:{p.f_lineno}"
            if imported:
                first_import.append(site)
            elif p.f_globals.get("__file__"):
                calls[site, module, func] += 1

    sys.setprofile(observe)
    try:
        result = calculate()
    finally:
        sys.setprofile(None)
    if isinstance(result, dict):
        print("result", result.get("engine"), result.get("status"), result.get("expression", ""))
    print("sympy loaded after calculation", "sympy" in sys.modules,
          "first import in calculation", first_import)
    for (site, module, func), n in sorted(calls.items()):
        label = "representation/ordering" if func in REPRESENTATION else "operation"
        print(label, site, f"{module}.{func}", n)


if __name__ == "__main__":
    main()
