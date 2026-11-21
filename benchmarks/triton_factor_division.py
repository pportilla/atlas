"""Time complete CPU/GPU Atlas computations, and check new operations independently in Singular."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.algebra import dispatch
from engine.gpu_devices import triton_device


def workloads():
    common = "(x-1)^3*(x+3)^2*(2*x-1)*(3*x+2)*(x+2)*(x^2+1)"
    a, b = f"({common})*(x^2+x+1)", f"({common})*(x^3+2)"
    for op in ("factor", "absolute-factor", "squarefree-factor", "gcd", "extended-gcd", "cancel-fraction", "polynomial-division", "normal-form"):
        binary = op in ("gcd","extended-gcd","cancel-fraction")
        divide = op in ("polynomial-division", "normal-form")
        names = ["x", "y"] if divide else ["x"]
        operands = {"polynomial": "(x+y+2)^16*(2^219+3)/7" if divide else a}
        if binary:
            operands["other_polynomial"] = b
        elif op == "normal-form":
            operands["ideal"] = ["2*x*y^2-3*y^4"]
        elif divide:
            operands["other_polynomial"] = "2*x*y^2-3*y^4"
        yield op, {"operation": op, "ring": {"variables": names, "order": "grevlex"}, "operands": operands}

    roots = "*".join(f"(x-{i})" for i in range(1, 91))
    yield "factor-90-lines", {"operation": "factor", "ring": {"variables": ["x"], "order": "grevlex"},
                              "operands": {"polynomial": roots}}


def singular_absolute(case, reference, trials, command):
    from benchmarks.absolute_factor import Case, SINGULAR_HEADER
    from tests.absolute_factor_oracle import check
    f = case["operands"]["polynomial"]
    witness = Case(f, (("x-1", 1, 3), ("x+3", 1, 2), ("2*x-1", 1, 1), ("3*x+2", 1, 1),
                       ("x+2", 1, 1), ("x^2+1", 2, 1), ("x^2+x+1", 2, 1)))
    check(witness, "Atlas", json.dumps({"result": reference}))
    header = SINGULAR_HEADER.replace("x,y,z", "x").replace(",lp;", ",dp;")
    program = header + f"""
int i,start; poly f;
writeAbsolute({f});
for (i=1;i<={trials};i++) {{
 start=rtimer; f={f}; writeAbsolute(f); print(rtimer-start);
}}
quit;
"""
    run = subprocess.run(command, input=program, text=True, capture_output=True, check=True, timeout=60)
    rows = run.stdout.splitlines()
    if len(rows) != 2*trials+1:
        raise RuntimeError(run.stdout + run.stderr)
    for row in [rows[0], *rows[1::2]]:
        check(witness, "Singular", row.removeprefix("ATLASBENCH ").replace(";", "\n"))
    return [int(elapsed)/1000 for elapsed in rows[2::2]]


def singular(case, reference, trials):
    # Each solve parses the same input formula and materializes its result.
    # Compare monic Euclidean data and every square-free group algebraically.
    op = case["operation"]
    f = case["operands"]["polynomial"].replace("2^219", "number(2)^219")
    args = case["operands"]
    g = args.get("other_polynomial", args.get("ideal", ["1"])[0])
    cmd = ["Singular","-q","--no-rc","--threads=1","--ticks-per-sec=1000000","--random=1"]
    if op == "absolute-factor":
        return singular_absolute(case, reference, trials, cmd)
    declaration, statement = "list A;", "A"

    if op == "cancel-fraction":
        declaration = ("proc cancelPair(poly a,poly b) { poly h=gcd(a,b); a=a/h; b=b/h; "
                       "return(list(a/leadcoef(b),b/leadcoef(b))); } list A;")
        call = "cancelPair(f,g)"
        check = f"ok=(A[1]==({reference['numerator']})) && (A[2]==({reference['denominator']}));"
    elif op == "polynomial-division":
        call = "division(ideal(f),ideal(g))"
        check = f"ok=(A[1][1,1]==({reference['quotient']})) && (A[2][1]==({reference['remainder']}));"
    elif op == "extended-gcd":
        call = "extgcd(f,g)"
        check = (f"ok=(A[1]/leadcoef(A[1])==({reference['gcd']}))"
                 f" && (({reference['left_coefficient']})*f+({reference['right_coefficient']})*g==({reference['gcd']}))"
                 " && (A[2]*f+A[3]*g==A[1]);")
    elif op in ("gcd", "normal-form"):
        declaration, statement = "poly p;", "p"
        call = "gcd(f,g)" if op == "gcd" else "reduce(f,std(ideal(g)))"
        p = "p/leadcoef(p)" if op=="gcd" else "p"
        check = f"ok=({p}==({reference['polynomial']}));"
    else:
        call = "sqrfree(f)" if op=="squarefree-factor" else "factorize(f)"
        check = ("unit=A[1][1]; for (j=2;j<=size(A[1]);j++){unit=unit*leadcoef(A[1][j])^A[2][j];} "
                 f"ok=(unit==({reference['unit']})) && (size(A[1])=={1+len(reference['factors'])});")
        for factor in reference["factors"]:
            check += ("found=0; for (j=2;j<=size(A[1]);j++) {"
                      f" if (A[2][j]=={factor['multiplicity']} && A[1][j]/leadcoef(A[1][j])==({factor['polynomial']}))"
                      " { found=1; } } ok=ok && found;")
    program = f"""
ring R=0,({','.join(case['ring']['variables'])}),dp;
poly f,g,unit; {declaration} int i,start,ok,j,found;
f={f};g={g};{statement}={call};
for (i=1;i<={trials};i++) {{
  start=rtimer; f={f};g={g};{statement}={call}; print(rtimer-start);
  {check} print(ok);
}}
quit;
"""
    run = subprocess.run(cmd, input=program, text=True, capture_output=True, check=True, timeout=60)
    lines = run.stdout.splitlines()
    if len(lines)!=trials*2 or lines[1::2] != ["1"]*trials:
        raise RuntimeError(run.stdout + run.stderr)
    return [int(s)/1000 for s in lines[::2]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Use at least one trial.")
    os.environ["RESOLUTION_GPU_BACKEND"] = "triton"
    device = triton_device(0)
    if not device:
        parser.error("A physical Triton device at index zero is required.")
    import torch
    import triton
    singular_version = subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0]
    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "triton": triton.__version__,
                "Singular": singular_version}
    out = {"device": device, "versions": versions, "trials": args.trials, "timing_unit": "ms", "workloads": {}}
    for label, case in workloads():
        ref = dispatch(dict(case, compute="cpu"))["result"]
        methods = ["cpu","gpu"]
        times = {m: [] for m in methods}
        for trial in range(args.trials + 1):
            for mode in methods[trial%2:] + methods[:trial%2]:
                start = perf_counter_ns()
                answer = dispatch(dict(case, compute=mode))
                elapsed = (perf_counter_ns()-start)/1e6
                assert ref == answer["result"]
                if mode=="gpu" and not any(s["backend"]=="triton" and s["operation"].startswith("poly_")
                                            for s in answer["execution"]["stages"]):
                    raise AssertionError(answer["execution"])
                if trial:
                    times[mode].append(elapsed)
        times["Singular"] = singular(case, ref, args.trials)
        out["workloads"][label] = {"request": case, "times_ms": times,
                                                "medians_ms": {m: median(values) for m,values in times.items()}}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
