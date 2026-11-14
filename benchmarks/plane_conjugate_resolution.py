"""Compare fresh Atlas and Singular resolutions of x^p+y^p at the origin over QQ.

Atlas returns both zeta functions, all charts, and individually indexed complex
branches. Singular resolve.lib returns checked charts over QQ. For individual
branches the general Singular run finds a splitting field with hnoether.lib.
The cyclotomic run uses the known splitting field for this family. Each
branch run checks the product of all reported linear factors. Singular runs
omit zeta functions and FLINT's canonical complex root indices.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
from statistics import median
import subprocess
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
CASES = (13, 31, 61, 97)
CHARTS = '''
LIB "resolve.lib";
ring src=0,(x,y),dp;
poly f=x^d+y^d;
list re=resolve(ideal(f),1,"L","A");
int mu=vdim(std(jacob(f)));
int chartMask=0;
int ix;
for(ix=1;ix<=size(re[1]);ix++){
    def sc=re[1][ix]; setring sc;
    poly e=BO[4][1][1]; poly h=BO[2][1]; poly a0=BO[5][1]; poly a1=BO[5][2];
    int ok=(a0^d+a1^d==e^d*h) && (vdim(std(ideal(e,h)))==d);
    ok=ok && (std(ideal(h,diff(h,var(1)),diff(h,var(2))))[1]==1);
    if(ok && a0==e && a1==e*var(2)){chartMask=chartMask+1;}
    if(ok && a1==e && a0==e*var(2)){chartMask=chartMask+2;}
    print("CHART "+string(a0)+" "+string(a1)+" "+string(e)+" "+string(h));
    setring src; kill sc;
}
print("CHART_CHECK "+string(size(re[1]))+" "+string(chartMask)+" "+string(mu));
'''
BRANCHES = '''
LIB "hnoether.lib";
ring loc=0,(x,y),ls;
poly g=x^d+y^d;
list H=hnexpansion(g);
def K=H[1]; setring K;
poly tangentProduct=1;
for(ix=1;ix<=size(hne);ix++){
    poly slope=hne[ix][1][1,1];
    tangentProduct=tangentProduct*(var(2)-slope);
    print("SLOPE "+string(slope)); kill slope;
}
print("BRANCH_CHECK "+string(size(hne))+" "+string(tangentProduct==var(2)^d+1));
'''

# For odd prime d, the roots of v^d+1 are the odd powers of a primitive
# (2*d)-th root. This is an explicit, input-specific field presentation.
CYCLOTOMIC = '''
ring cycloRing=0,(a),dp;
poly phi=(a^d+1)/(a+1);
string modulus=string(phi);
ring ext=(0,a),(u,v),dp;
execute("minpoly="+modulus+";");
number step=a^2; number slope=a;
poly tangentProduct=1;
for(ix=1;ix<=d;ix++){
    print("SLOPE "+string(slope));
    tangentProduct=tangentProduct*(v-slope);
    slope=slope*step;
}
print("BRANCH_CHECK "+string(d)+" "+string(tangentProduct==v^d+1));
'''
SINGULAR_VARIANTS = {"singular_charts": "", "singular_charts_and_branches": BRANCHES,
                     "singular_charts_and_cyclotomic_branches": CYCLOTOMIC}


def cpu_seconds():
    used = resource.getrusage(resource.RUSAGE_CHILDREN)
    return used.ru_utime + used.ru_stime


def measure(command, cwd, timeout, source=None):
    start_cpu = cpu_seconds()
    start = perf_counter()
    try:
        result = subprocess.run(command, cwd=cwd, input=source, text=True,
                                capture_output=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": perf_counter() - start,
                "cpu_s": cpu_seconds() - start_cpu}, ""
    row = {"status": "complete", "wall_s": perf_counter() - start,
           "cpu_s": cpu_seconds() - start_cpu}
    return row, result.stdout


def atlas_record(record, degree):
    if record["status"] != "resolved" or record["certificate"].get("atlas_verification") != (
            "Embedded normal crossings on the owned origin fibre"):
        raise ValueError("Expected a certified embedded resolution")
    numbers = record["algebraic_numbers"]
    minimal = ", ".join(str((-1)**i) for i in range(degree))
    labels = {f"PolynomialRootIndexed(List({minimal}), {i})" for i in range(1, degree)}
    assert {value["exact"] for value in numbers.values()} == labels
    branches = record["branches"]
    assert len(branches) == degree and len(numbers) == degree - 1
    assert all(row["point"][0] == "0" and row["chart"] == "B1.u" for row in branches)
    assert {row["point"][1] for row in branches} == {"-1", *numbers}
    assert len(record["charts"]) == len(record["atlas"]) == 2
    for chart, radial, strict, projection in zip(
            record["atlas"], ("u", "v"), (f"v^{degree}+1", f"u^{degree}+1"),
            ({"x": "u", "y": "u*v"}, {"x": "u*v", "y": "v"}), strict=True):
        assert chart["map"] == projection
        assert chart["components"] == [
            dict(id="E1", multiplicity=degree, nu=2, ideal=[radial]),
            dict(id="strict", multiplicity=1, nu=1, ideal=[strict])]
    mono, topo = record["monodromy_zeta"], record["topological_zeta"]
    assert mono["status"] == topo["status"] == "computed"
    assert mono["factors"] == [dict(multiplicity=degree, exponent=degree-2)]
    assert mono["milnor_fiber_euler_characteristic"] == 1 - (degree-1)**2
    assert {tuple(s["components"]): s["euler_characteristic"] for s in topo["strata"]} == {
        ("E1",): 2-degree, ("E1", "strict"): degree}
    # All fields including approximate roots and their ordering remain in the comparison.
    del record["elapsed_ms"]
    return record


def singular_record(output, degree, with_branches):
    lines = output.splitlines()
    if ("result is o.k." not in output or any(line.lstrip().startswith("?") for line in lines)
            or f"CHART_CHECK 2 3 {(degree-1)**2}" not in lines):
        raise ValueError(output[-2000:])
    if with_branches:
        slopes = [line for line in lines if line.startswith("SLOPE ")]
        if f"BRANCH_CHECK {degree} 1" not in lines or len(set(slopes)) != degree:
            raise ValueError(output[-2000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--degrees", type=int, choices=CASES, nargs="+", default=CASES)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--branch-timeout", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.trials < 1 or min(args.timeout, args.branch_timeout) <= 0:
        parser.error("Trial counts and time limits must be positive")
    singular = shutil.which("Singular")
    if not singular:
        parser.error("Install Singular with resolve.lib and hnoether.lib")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    roots = {"atlas": args.atlas_root.resolve()}
    if args.baseline:
        roots["baseline"] = args.baseline.resolve()
    report = {"conditions": __doc__, "trials": args.trials, "cpu_affinity": args.cpu,
              "command": "./atlas resolve 'x^p+y^p' --vars x,y --mode embedded --compute cpu --quick --monodromy --json",
              "atlas_and_chart_timeout_s": args.timeout, "singular_branch_timeout_s": args.branch_timeout,
              "singular": subprocess.check_output([singular, "--version"], text=True).splitlines()[0],
              "platform": platform.platform(), "plane_binary_sha256": {
                  k: hashlib.sha256((r/"native/build/plane").read_bytes()).hexdigest() for k, r in roots.items()},
              "cases": {}}
    for degree in args.degrees:
        results = report["cases"][str(degree)] = {"runs": [], "medians_s": {}}
        expected = None
        for trial in range(args.trials):
            labels = [*reversed(roots), *SINGULAR_VARIANTS]
            for label in (reversed(labels) if trial % 2 else labels):
                if label in roots:
                    root = roots[label]
                    command = ["./atlas", "resolve", f"x^{degree}+y^{degree}", "--vars", "x,y", "--mode",
                               "embedded", "--compute", "cpu", "--quick", "--monodromy", "--json"]
                    row, out = measure(command, root, args.timeout)
                    current = atlas_record(json.loads(out), degree)
                    if expected is None:
                        expected = current
                    if current != expected:
                        raise ValueError(f"Full Atlas record differs on {label}, degree {degree}, trial {trial+1}")
                    row["record_sha256"] = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
                else:
                    branch_code = SINGULAR_VARIANTS[label]
                    source = f"int d={degree};\n" + CHARTS + branch_code + "quit;\n"
                    row, out = measure([singular, "-q", "--no-rc"], ROOT,
                                       args.branch_timeout if branch_code else args.timeout, source)
                    if row["status"] == "complete":
                        singular_record(out, degree, bool(branch_code))
                row.update(solver=label, trial=trial+1)
                results["runs"].append(row)
                print(degree, label, trial+1, row["status"], round(row["wall_s"], 3), flush=True)
                args.output.write_text(json.dumps(report, indent=2)+"\n")
        for label in labels:
            rows = [r for r in results["runs"] if r["solver"] == label]
            if all(r["status"] == "complete" for r in rows):
                results["medians_s"][label] = {k: median(r[k] for r in rows) for k in ("wall_s", "cpu_s")}
        args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
