"""Compare canonical quotient-ring arithmetic in Atlas and Singular over QQ."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.primary_decomposition import cpu_name, singular, summary
from engine import native_library
from engine import native_polynomial as np
from engine.native_library import affine_request
from engine.polynomial import parse_expression


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple
    relations: tuple
    action: str
    operands: tuple = ()
    exponent: int = 0
    order: str = "grevlex"


def corpus():
    xy, xyz, xyzw = ("x", "y"), ("x", "y", "z"), ("x", "y", "z", "w")
    circle = ("x^2+y^2-1", "x*y-1/3")
    thick = ("x^3", "y^4")
    torus = ("x*y-1",)
    twisted = ("x*z-y^2", "x*w-y*z", "y*w-z^2")
    a, b = "x^2+2*x*y+3*y^2-1/2", "x^3*y-x*y^2+5"
    return (
        Case("declare_hyperbola", xy, torus, "declare"),
        Case("declare_curve", xyzw, twisted + ("x^2*w-x*y*z",), "declare", order="lex"),
        Case("circle_reduce_lex", xy, circle, "reduce", ("x^7+3*x^3*y^2-5*y^9",), order="lex"),
        Case("circle_reduce_grevlex", xy, circle, "reduce", ("x^7+3*x^3*y^2-5*y^9",)),
        Case("nilpotent_add", xy, thick, "add", (a, b)),
        Case("torus_subtract", xyz, torus, "subtract", ("x*y*z^2+x^2*y+2/3", "x+z^2-1/7")),
        Case("wide_rational_negate", xy, (), "negate", ("(18446744073709551629/11)*x^2*y+2/7",)),
        Case("grid_multiply", xy, ("x^3-x", "y^3-y"), "multiply", (a, b)),
        Case("curve_multiply", xyzw, twisted, "multiply",
             ("x^2+w^2+3*y*z-2*z^2+1/2", "x*z+w^2-x*w+y^2")),
        Case("torus_equal", xyz, torus, "equal", ("x^2*y^2*z^3", "z^3")),
        Case("curve_unequal", xyzw, twisted, "equal", ("x*z^2+w", "y^2*z")),
        Case("field_power", ("x",), ("x^4+x+1",), "power", ("2*x^3-x^2+1",), 41),
        Case("nilpotent_power", xy, thick, "power", ("x+y",), 37),
        Case("dual_negative_power", xy, ("x^3",), "power", ("1+x*y",), -7),
        Case("constant_power_zero", xy, thick, "power", ("0",), 0),
        Case("torus_unit", xyz, torus, "unit", ("x^2",)),
        Case("domain_nonunit", xy, ("y-x^2",), "unit", ("x+1",)),
        Case("nilpotent_nonunit", xy, thick, "unit", ("x+y",)),
        Case("zero_ring_unit", xy, ("1",), "unit", ("0",)),
        Case("torus_inverse", xyz, torus, "inverse", ("x^2",)),
        Case("thick_axis_inverse", xyz, ("x^3", "y*z"), "inverse", ("1+x*y+x^2*z",)),
        Case("grid_inverse", xy, thick, "inverse", ("2+x+y+x^2*y",)),
        Case("reduced_grid_inverse", xy, ("x^3-x", "y^3-y"), "inverse", ("5+x+2*y",)),
        Case("mixed_torus_inverse", xyz, ("x*y-2","x*z-3"), "inverse", ("y+z",)),
        Case("sum_inverse_on_hypersurface", xyzw, ("(x+y)*(z+w)-1",), "inverse", ("x+y",)),
        Case("field_division", ("x",), ("x^4+x+1",), "divide", ("x^6-2*x+3", "x^2+x+3")),
        Case("torus_division", xyz, torus, "divide", ("x*z+y+1/2", "y^3")),
    )


def native_rows(c):
    names = tuple(map(np.Symbol, c.variables))
    return [np.row(names, np.from_terms(names, parse_expression(expr, c.variables)))
            for expr in (*c.relations, *c.operands)]


def native_wire(c):
    return ["quotient_ring", " ".join(c.variables), c.order, c.action, str(c.exponent),
            str(len(c.relations)), *native_rows(c)]


def atlas_cli(c):
    cmd = [str(ROOT / "atlas"), "algebra", "quotient-ring", "--vars", ",".join(c.variables),
           "--order", c.order, "--action", c.action, "--compute", "cpu", "--json"]
    for g in c.relations:
        cmd.extend(("--gen", g))
    for flag, operand in zip(("--poly", "--other-poly"), c.operands):
        cmd.extend((flag, operand))
    if c.action == "power":
        cmd.extend(("--exponent", str(c.exponent)))
    return cmd


def atlas_kernel(c, scope, batch, timeout):
    cmd = [str(ROOT / "native/build/bench_quotient_ring"), c.action, c.order,
           str(c.exponent), str(batch), scope]
    source = "\n".join([" ".join(c.variables), str(len(c.relations)), *native_rows(c)]) + "\n"
    output = subprocess.run(cmd, input=source, text=True, capture_output=True, timeout=timeout, check=True)
    ns, result = output.stdout.splitlines()
    return int(ns) / (batch * 1e6), json.loads(result)


# Singular division returns a Bezout witness and the remainder in the qring. Power
# by repeated squaring keeps every intermediate in normal form on both sides.
PROCEDURES = """
proc qinverse(poly f, ideal Z) {
    poly one=reduce(1,Z); if(one==0){return(poly(0));}
    poly d=reduce(f,Z); if(d!=0 && deg(d)==0){return(1/leadcoef(d));}
    list D=division(ideal(one),ideal(d));
    if(size(D[2])){ERROR("nonunit");}
    return(reduce(D[1][1,1],Z));
}
proc qpower(poly a, int n, ideal Z) {
    poly h=reduce(1,Z);
    if(n<0){a=qinverse(a,Z);n=-n;} else {a=reduce(a,Z);}
    while(n>0){
        if(n%2){h=reduce(h*a,Z);}
        n=n div 2; if(n){a=reduce(a*a,Z);}
    }
    return(h);
}
"""


def ambient(c):
    order = {"lex": "lp", "grevlex": "dp"}[c.order]
    source = f'option(redSB);ring R=0,({",".join(c.variables)}),{order};short=0;'
    source += f'ideal I=ideal({",".join(c.relations) or "0"});'
    if c.action in ("equal", "unit"):
        source += "int answer;"
    return source + "".join(f"poly {name}={operand};" for name, operand in zip(("p", "q"), c.operands))


def setup(c):
    source = "qring Q=std(I);short=0;"
    if c.action != "declare":
        source += "ideal Z=std(0);"
        for name, ambient_name in zip(("a", "b")[:len(c.operands)], ("p", "q")):
            source += f"poly {name}=fetch(R,{ambient_name});"
        if c.action not in ("equal", "unit"):
            source += "poly answer;"
        if c.action in ("inverse", "divide"):
            source += "list D;"
    return source


def methods(c):
    if c.action == "power" and c.exponent != 0:
        return ("modular_power", "builtin_power")
    return ("builtin",)


def statement(c, method):
    if c.action == "declare":
        return ""
    if c.action in ("inverse", "divide"):
        denominator = "a" if c.action == "inverse" else "b"
        numerator = "" if c.action == "inverse" else "a*"
        return (f"D=division(ideal(1),ideal({denominator}));"
                f'if(size(D[2])){{ERROR("nonunit");}}answer=reduce({numerator}D[1][1,1],Z);')
    if method == "builtin_power":
        base = "qinverse(a,Z)" if c.exponent < 0 else "a"
        return f"answer=reduce({base}^{abs(c.exponent)},Z);"
    if c.action == "power" and c.exponent == 0:
        return "answer=reduce(1,Z);"
    expressions = {
        "reduce": "reduce(a,Z)", "add": "reduce(a+b,Z)", "subtract": "reduce(a-b,Z)",
        "multiply": "reduce(a*b,Z)", "negate": "reduce(-a,Z)", "equal": "(reduce(a-b,Z)==0)",
        "unit": "(reduce(1,std(ideal(a)))==0)", "power": f"qpower(a,{c.exponent},Z)",
    }
    return f"answer={expressions[c.action]};"


def singular_json(c):
    kind = "quotient-ring" if c.action == "declare" else (
        "boolean" if c.action in ("equal", "unit") else "quotient-element")
    prefix = json.dumps({"type":kind, "ring":{"field":"QQ", "variables":list(c.variables),
                                             "order":c.order}}, separators=(",",":"))[:-1] + ',"relations":['
    source = (
        'ideal H=ideal(basering); string payload=' + json.dumps(prefix) + '; int j;'
        'for(j=size(H);j>0;j--){if(j<size(H)){payload=payload+",";}'
        r'payload=payload+"\"" + string(H[j]/leadcoef(H[j]))+"\"";}payload=payload+"]";'
    )
    if kind == "quotient-element":
        source += r'payload=payload+",\"representative\":\""+string(answer)+"\"";'
    if kind == "boolean":
        source += r'payload=payload+",\"value\":";if(answer){payload=payload+"true";}else{payload=payload+"false";}'
    return source + 'print(payload+"}");'


def compare_json(c, result, source):
    item, = [json.loads(line) for line in source.splitlines() if line.startswith("{")]
    def terms(poly):
        return tuple(sorted(parse_expression(poly, c.variables).items()))
    if item["type"] != result["type"] or item["ring"] != result["ring"]:
        raise ValueError(f"Singular returned different JSON metadata for {c.name}")
    if sorted(map(terms, item["relations"])) != sorted(map(terms, result["relations"])):
        raise ValueError(f"Singular returned different relations for {c.name}")
    if "representative" in result and terms(item["representative"]) != terms(result["representative"]):
        raise ValueError(f"Singular returned a different representative for {c.name}")
    if "value" in result and item["value"] != result["value"]:
        raise ValueError(f"Singular returned a different boolean for {c.name}")


def singular_source(c, scope, batch, method):
    needs_procedure = method == "modular_power" or method == "builtin_power" and c.exponent < 0
    prefix = (PROCEDURES if needs_procedure else "") + ambient(c)
    prepare, op = setup(c), statement(c, method)
    if scope == "cold":
        return prefix + prepare + op + singular_json(c)
    if scope == "prepared":
        return prefix + prepare + op + "int start=rtimer;\n" + (op + "\n") * batch + 'print("TIME "+string(rtimer-start));'
    step = prepare + op + "setring R;kill Q;\n"
    return prefix + step + "int start=rtimer;\n" + step * batch + 'print("TIME "+string(rtimer-start));'


def validate(c, result, timeout, method=None):
    expected_type = "quotient-ring" if c.action == "declare" else (
        "boolean" if c.action in ("equal","unit") else "quotient-element")
    if result["type"] != expected_type or result["ring"] != {
        "field":"QQ", "variables":list(c.variables), "order":c.order
    }:
        raise ValueError(f"{c.name}: unexpected result schema {result}")
    verify = PROCEDURES + ambient(c) + setup(c) + statement(c, method or methods(c)[0])
    if expected_type == "boolean":
        verify += f'if(answer!={int(result["value"])}){{ERROR("wrong boolean");}}'
    if c.action in ("inverse","divide"):
        identity = "answer*a-1" if c.action == "inverse" else "answer*b-a"
        verify += f'if(reduce({identity},Z)!=0){{ERROR("failed inverse identity");}}'
    verify += "setring R;"
    if expected_type == "quotient-element":
        # Compare the ambient polynomials exactly; equivalence alone would accept
        # an unreduced representative.
        verify += f'poly actual=fetch(Q,answer); if(actual!=({result["representative"]})){{ERROR("wrong representative");}}'
    verify += f'ideal H=ideal({",".join(result["relations"]) or "0"});ideal B=std(I);'
    verify += """
int j,k,found;
for(j=1;j<=size(B);j++){
    found=0;
    for(k=1;k<=size(H);k++){if(H[k]*leadcoef(B[j])==B[j]){found++;}}
    if(found!=1){ERROR("different reduced basis");}
}
if(size(B)!=size(H)){ERROR("different basis size");}
print("VERIFIED");
"""
    output = singular(verify, seed=31, timeout=timeout)
    if "VERIFIED" not in output:
        raise ValueError(f"{c.name}: incomplete Singular verification")


def elapsed(call):
    start = perf_counter_ns()
    value = call()
    return (perf_counter_ns() - start) / 1e6, value


def paired(atlas, reference):
    ratios = [s/a for a, s in zip(atlas, reference)]
    return {"atlas":summary(atlas), "singular":summary(reference),
            "median_speedup":statistics.median(reference)/statistics.median(atlas),
            "wins":sum(r > 1 for r in ratios), "worst_paired_speedup":min(ratios)}


def run(c, args):
    result, = affine_request(native_wire(c))
    result = json.loads(result)
    for method in methods(c):
        validate(c, result, args.timeout, method)
    record = {"input":asdict(c), "verified":True, "result":result}
    for scope in ("prepared","setup","cold"):
        if c.action == "declare" and scope == "prepared":
            continue
        if scope not in args.scopes:
            continue
        systems = ["atlas", *methods(c)]
        samples = {name: [] for name in systems}
        sources = {m: singular_source(c, scope, args.batch, m) for m in methods(c)}
        for trial in range(args.cold_trials if scope == "cold" else args.trials):
            offset = trial % len(systems)
            for system in systems[offset:] + systems[:offset]:
                if system != "atlas":
                    clock, out = elapsed(lambda: singular(sources[system], seed=31, timeout=args.timeout))
                    if "// **" in out:
                        raise ValueError(f"Unexpected Singular warning on {c.name}: {out[:500]}")
                    if scope == "cold":
                        compare_json(c, result, out)
                    else:
                        ticks, = [int(line.split()[1]) for line in out.splitlines() if line.startswith("TIME ")]
                        clock = ticks / (args.batch * 1000)
                elif scope == "cold":
                    cmd = atlas_cli(c)
                    clock, proc = elapsed(lambda: subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout, check=True))
                    if json.loads(proc.stdout)["result"] != result:
                        raise AssertionError(f"{c.name}: CLI changed the result")
                else:
                    clock, answer = atlas_kernel(c, scope, args.batch, args.timeout)
                    if answer != result:
                        raise AssertionError(f"{c.name}: typed kernel changed the result")
                samples[system].append(clock)
        winner = min(methods(c), key=lambda m: statistics.median(samples[m]))
        record[scope] = paired(samples["atlas"], samples[winner])
        record[scope]["singular_method"] = winner
        if len(methods(c)) > 1:
            record[scope]["singular_alternatives"] = {m:summary(samples[m]) for m in methods(c)}
    return record


def binary_hashes():
    binary_dir = ROOT / "native/build"
    files = [binary_dir / name for name in ("bench_quotient_ring", "local_algebra", "quotient_ring")]
    files += [native_library._LIBRARY, ROOT / "atlas", Path(shutil.which("Singular"))]
    return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--cold-trials", type=int, default=7)
    parser.add_argument("--batch", type=int, default=80)
    parser.add_argument("--scope", dest="scopes", action="append", choices=("prepared","setup","cold"))
    parser.add_argument("--case", dest="cases", action="append", choices=[c.name for c in corpus()])
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.scopes = args.scopes or ["prepared","setup","cold"]
    if min(args.batch,args.trials,args.cold_trials) < 1:
        parser.error("trial counts and batch sizes must be positive")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    cases = [c for c in corpus() if not args.cases or c.name in args.cases]
    report = {
        "cpu":cpu_name(), "affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        "singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
        "trials":args.trials, "cold_trials":args.cold_trials, "batch":args.batch,
        "binary_sha256":binary_hashes(),
        "scopes":{
            "prepared":"Pre-parsed operands, each system prepares a quotient and its basis once; timed result is canonical.",
            "setup":"Pre-parsed ambient inputs, each system rebuilds the quotient and computes a canonical result on every iteration.",
            "cold":"Fresh public ./atlas algebra quotient-ring --json and Singular process; includes expression parsing, quotient, operation and output."
        },
        "cases":[],
    }
    for c in cases:
        item = run(c, args)
        report["cases"].append(item)
        print(c.name, *(f"{s}={item[s]['median_speedup']:.2f}x" for s in args.scopes if s in item), file=sys.stderr, flush=True)
    if binary_hashes() != report["binary_sha256"]:
        raise RuntimeError("The benchmark executables changed during this run.")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
