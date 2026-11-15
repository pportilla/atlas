"""Compare ideal operations in quotient rings using the Atlas and Singular script interpreters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.ideal_operators import atlas_sample, run_singular, script_loop_sample
from benchmarks.normalization import ideal as singular_ideal
from benchmarks.primary_decomposition import cpu_name
from engine.native_library import _LIBRARY, affine_request
from engine.polynomial import parse_expression
from engine.session import new_session

CASES = {
    "cusp": dict(relations=["x^5+y^2"], left=["y", "x^5"], right=["x", "x*y"], probe="x^5"),
    "nilpotents": dict(relations=["x^4", "y^3"], left=["(18446744073709551629/7)*x*y", "y^2"],
                       right=["x+y", "x^2"], probe="x+x*y"),
}
# Singular qring declarations do not automatically reduce the representatives of ideal generators.
OPERATIONS = {
    "declare": ("ideal((a,b),ring=K)", "ideal", "reduce(ideal(a,b),Z)"),
    "sum": ("ideal_sum(I,J)", "ideal", "reduce(I+J,Z)"),
    "product": ("ideal_product(I,J)", "ideal", "reduce(I*J,Z)"),
    "square": ("ideal_power(I,2)", "ideal", "std(I^2)"),
    "groebner": ("groebner(I)", "ideal", "std(I)"),
    "membership": ("membership(f,I)", "int", "(reduce(f,std(I))==0)"),
    "normal_form": ("normal_form(f,I)", "poly", "reduce(f,std(I))"),
    "further_quotient": ("quotient_ring(I,fresh=True)", "def", "further(I)"),
}


def prepare(case):
    a, b = case["left"]
    c, d = case["right"]
    probe = case["probe"]
    relations = ",".join(case["relations"])
    session = new_session(None)
    source = f'''x,y = coordinates("x","y")
K = ideal({relations}).quotient_ring
a = {a}
b = {b}
c = {c}
d = {d}
f = K({probe})
I = ideal((a,b),ring=K)
J = ideal((c,d),ring=K)'''
    for _ in session.run(source):
        pass
    singular = f'''ring P=0,(x,y),dp; option(redSB); short=0;
qring K=std(ideal({relations})); short=0;
ideal Z=std(ideal(0)); poly a={a}; poly b={b}; poly c={c}; poly d={d};
poly f=reduce({probe},Z); ideal I=reduce(ideal(a,b),Z); ideal J=reduce(ideal(c,d),Z);
proc further(ideal generators) {{qring T=std(generators); return(T);}}'''
    return session, singular


def singular_sample(program, kind, expression, batch):
    code = f'''{program}
{kind} answer={expression}; int i; int start=rtimer;
for(i=1;i<={batch};i++) {{answer={expression};}}
print(rtimer-start);'''
    return int(run_singular(code)) / batch


def check(program, kind, expression, atlas_value, *, expect_basis):
    code = f"{program}\n{kind} expected={expression};\n"
    if kind in ("ideal", "def"):
        if kind == "def":
            code += "setring expected; ideal defining=ideal(basering); setring P; ideal H=fetch(expected,defining);\n"
            generators = atlas_value.relations
        else:
            code += "ideal H=expected;\n"
            generators = atlas_value.generators
        received = singular_ideal([str(g) for g in generators])
        code += (f"ideal G={received}; int agrees=(size(reduce(G,std(H)))==0 "
                 "&& size(reduce(H,std(G)))==0);\n")
        if expect_basis:
            # Every true standard-basis row must reduce to zero by the supplied
            # generators in their given form, including the qring's relations.
            code += 'attrib(G,"isSB",1); agrees=(agrees && size(reduce(std(H),G))==0);\n'
        code += "print(agrees);"
    elif kind == "poly":
        code += f"print(reduce(({atlas_value.representative})-expected,Z)==0);"
    else:
        code += f"print(expected=={int(atlas_value.value)});"
    if run_singular(code) != "1":
        raise AssertionError(f"Atlas and Singular disagree for {expression}.")


def native_wire(session, operation):
    variables = session.variables
    parent = variables["K"]
    relations = parent.ring["relations"]
    if operation == "declare":
        left = [str(variables["a"]), str(variables["b"])]
    else:
        left = [str(g) for g in variables["I"].generators]
    right = [str(g) for g in variables["J"].generators] if operation in ("sum", "product") else []
    query = [str(variables["f"])] if operation in ("normal_form", "membership") else []
    name = {"declare": "ideal", "sum": "ideal-sum", "product": "ideal-product",
            "square": "ideal-power", "normal_form": "normal-form",
            "further_quotient": "quotient-ring"}.get(operation, operation)
    rows = [parse_expression(f, parent.variables, as_row=True) for f in (*relations, *left, *right, *query)]
    return ["quotient_ideal_prepared", " ".join(parent.variables), parent.order, name,
            "2" if operation == "square" else "0", f"{len(relations)} {len(left)} {len(right)}", *rows]


def measure(session, program, operation, batch, trials):
    command, kind, expression = OPERATIONS[operation]
    wire = native_wire(session, operation)
    atlas_times, singular_times, wire_times = [], [], []
    for trial in range(trials):
        if trial % 2 == 0:
            singular_times.append(singular_sample(program, kind, expression, batch))
        elapsed, result = script_loop_sample(session, "answer = " + command, batch)
        atlas_times.append(elapsed)
        if trial % 2:
            singular_times.append(singular_sample(program, kind, expression, batch))
        elapsed, native = atlas_sample(lambda: affine_request(wire), batch)
        wire_times.append(elapsed)
        if json.loads(native[0]) != result.record["result"]:
            raise AssertionError(f"Native wire result differs from {command}.")
        check(program, kind, expression, result, expect_basis=operation in ("groebner", "square"))
    atlas = statistics.median(atlas_times)
    singular = statistics.median(singular_times)
    return dict(atlas_us=atlas_times, singular_us=singular_times, atlas_native_wire_us=wire_times,
                atlas_native_wire_median_us=statistics.median(wire_times), atlas_median_us=atlas,
                singular_median_us=singular, atlas_over_singular=round(atlas / singular, 2), agreement=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 2 or args.batch < 1:
        parser.error("Use at least two trials and one operation per batch.")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    report = dict(cpu=cpu_name(), cpu_affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
                  singular=subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
                  atlas_library_sha256=hashlib.sha256(_LIBRARY.read_bytes()).hexdigest(), trials=args.trials,
                  operations_per_trial=args.batch, ring="QQ[x,y], grevlex, modulo the recorded relations",
                  conditions=("Time one named script assignment per iteration in a running interpreter. "
                              "Both use the same fixed parent, polynomials and prepared input ideals. Exclude process "
                              "startup, input setup, one warm-up, display and all agreement checks. Alternate timing "
                              "order across trials. Singular uses one thread, rtimer in microseconds, short=0, "
                              "option(redSB), and explicit quotient reduction for raw ideal generators. Each trial "
                              "checks both ideal containments, equality of quotient classes, or equality of booleans "
                              "in Singular. Atlas native-wire timings call the shared library directly with "
                              "prepared sparse inputs and include sparse decoding, mathematics and JSON output."), cases={})
    for name, case in CASES.items():
        session, singular = prepare(case)
        measurements = {op: measure(session, singular, op, args.batch, args.trials) for op in OPERATIONS}
        report["cases"][name] = dict(**case, operations=measurements)
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
