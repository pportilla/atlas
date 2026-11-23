"""Time fresh local resolutions and cross-check the Milnor fibre Euler characteristics."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
from statistics import median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.zoo_data import catalogue_file

CASES = """chain-3-3-4 chain-3-2-4 loop2-2-4-4 loop-2-3-3 loop-2-3-4
cone-four-lines cone-braid-arrangement hidden-e12 hidden-bp-2-5-5 hidden-bp-2-5-5-quadric
threefold-e7 laufer-1 laufer-2 threefold-chain-2-3-2-2 threefold-loop2-loop2 threefold-chain-loop
threefold-e13 cusp-curve-a4 cone-d4-quartic quartic-face-plus hesse-triangle-quartic""".split()


def run(command, timeout, source=None):
    start = time.perf_counter()
    try:
        process = subprocess.run(
            command, input=source, text=True, capture_output=True, cwd=ROOT, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": time.perf_counter() - start}, ""
    return {
        "status": "complete" if process.returncode == 0 else "failed",
        "wall_s": time.perf_counter() - start,
    }, process.stdout


def oracle(record):
    # The central complement is the decone times C*, so its Poincaré polynomial
    # divided by (1+t), evaluated at -1, gives the Euler characteristic of the decone.
    # For the cusp suspension use Thom–Sebastiani and the two fibres of h^2.
    head = 'LIB "arr.lib";' if "Hyperplane arrangement" in record["labels"] else ""
    head += f"ring r=0,({','.join(record['variables'])}),ds;poly f={record['equation']};"
    source = "int mu=vdim(std(jacob(f)));int tau=vdim(std(jacob(f)+ideal(f)));int chi;"
    if "Hyperplane arrangement" in record["labels"]:
        source += """ideal factors=factorize(f,1);arr A=factors;int i;int sign=1;intvec b=arrPoincare(A);
for(i=2;i<=size(b);i++){chi=chi+sign*(i-1)*b[i];sign=-sign;}chi=deg(f)*chi;"""
    elif record["id"] == "cusp-curve-a4":
        source += "ring P=0,(y,z),ds;poly h=y^2-z^3;int mf=1-vdim(std(jacob(h)));setring r;chi=1-4*(2*mf-1);"
    elif record["id"] == "quartic-face-plus":
        # The homogeneous Milnor fibre is the four-sheeted covering of the
        # projective complement of two smooth conics. Their intersection has
        # length four, supported at two points, with coincident tangent lines.
        source += """
ring P=(0,a),(x,y,z),dp; minpoly=a^2+1;
poly F=imap(r,f);poly g=x^2+y^2+a*z^2;poly h=x^2+y^2-a*z^2;
ideal J=std(ideal(g,h));ideal A=std(J+ideal(y-1));ideal pts=std(ideal(y-1,z,x^2+1));
int n=vdim(pts);
if(F!=g*h || dim(std(jacob(g)))!=0 || dim(std(jacob(h)))!=0 || dim(std(J+ideal(y)))!=0
   || n!=2 || vdim(A)!=4 || size(reduce(J,pts)) || reduce(z^2,A)!=0 || reduce(x^2+1,A)!=0
   || size(reduce(minor(jacob(ideal(g,h)),2),pts))){ERROR("Expected two smooth tangent conics");}
int ec=deg(F)*(3-2*2+n);setring r;chi=ec;"""
    elif record["id"] == "hesse-triangle-quartic":
        # Three independent linear forms give an ordinary projective
        # triangle. The quartic term is nonzero at all its nodes. The
        # superisolated Milnor number is (d-1)^3 plus the three node numbers.
        source += """
ring P=(0,a),(x,y,z),dp;minpoly=a^2+a+1;
poly F=imap(r,f);poly h=jet(F,3);ideal L=x+y+z,x+a*y+a^2*z,x+a^2*y+a*z;
int n=vdim(std(jacob(h)+ideal(x-1)));
if(h!=L[1]*L[2]*L[3] || det(jacob(L))==0 || F-h!=x^4 || n!=3 || dim(std(jacob(h)+ideal(x)))!=0)
{ERROR("Expected a projective triangle with three nodes away from x=0");}
int smu=(deg(h)-1)^3+n;setring r;
if(mu!=smu){ERROR("Superisolated Milnor number differs from the local Jacobian length");}
chi=1+smu;"""
    else:
        source += f"chi=1+({(-1) ** record['dimension']})*mu;"
    # Exact comparisons of the full rational zetas for these three germs.
    # The homogeneous cones have a factor 1/(d*s+3) from the vertex.
    # Blow up the triple point of the four lines, or resolve each conic
    # tangency by two blowups on the projective plane. For the Hesse
    # triangle, over each node the point blowup has (N,nu)=(4,5) and the
    # blowup along its intersection with the first exceptional has (8,8).
    zetas = {
        "cone-d4-quartic": ("1", "(-1/(s+1)-1/(3*s+2)+3/(s+1)^2+3/((s+1)*(3*s+2)))/(4*s+3)"),
        "quartic-face-plus": ("1/(1-T^4)",
                             "(1+2*(1/(2*s+2)-1/(4*s+3)+1/((2*s+2)*(4*s+3))+2/((s+1)*(4*s+3))))/(4*s+3)"),
        "hesse-triangle-quartic": ("1/(1-T^4)^3", "3/(4*s+5)+6/((4*s+5)*(8*s+8))+6/((3*s+3)*(8*s+8)*(s+1))"),
    }
    if record["id"] in zetas:
        mono, top = zetas[record["id"]]
        recorded = record["zeta"]
        source += (f'ring Z=(0,s,T),(w),dp;number mono={mono};number top={top};'
                   f'if(mono!=({recorded["monodromy"]["expression"]}) || top!=({recorded["topological"]["expression"]}))'
                   '{ERROR("Resolution zeta differs from the independent strata");}setring r;')
    _, out = run(
        ["Singular", "-q", "--no-rc"],
        30,
        head + source + 'print("ORACLE "+string(mu)+" "+string(tau)+" "+string(chi));quit;',
    )
    match = re.search(r"^ORACLE (-?\d+) (-?\d+) (-?\d+)$", out, re.M)
    if not match or re.search(r"^\s*\?", out, re.M):
        raise RuntimeError(out[-2000:])
    return list(map(int, match.groups()))


def atlas_trial(record, mode, timeout):
    names = ",".join(record["variables"])
    flags = ["--monodromy"] if mode == "embedded" else []
    command = [
        "./atlas",
        "resolve",
        record["equation"],
        "--vars",
        names,
        "--mode",
        mode,
        "--compute",
        "cpu",
        "--quick",
        *flags,
        "--json",
    ]
    timing, output = run(command, timeout)
    if timing["status"] == "complete":
        answer = json.loads(output)
        if (
            answer["status"] != "resolved"
            or answer["dual_complex"]["status"] != "complete"
            or answer["dual_complex"] != record[mode]["dual_complex"]
        ):
            raise RuntimeError(f"Uncertified {mode} Atlas trial for {record['id']}")
        if mode == "embedded" and any(
            answer[kind + "_zeta"] != record["zeta"][kind] for kind in ("monodromy", "topological")
        ):
            raise RuntimeError(f"Zeta mismatch for {record['id']}")
        timing["charts"] = len(answer["charts"])
    return timing


def compare(record, trials, timeout):
    ident, eq = record["id"], record["equation"]
    names = ",".join(record["variables"])
    expected = oracle(record)
    invariants = record["invariants"]
    local = [
        invariants[key]["value"] if invariants[key]["finite"] else -1
        for key in ("milnor_number", "tjurina_number")
    ]
    if expected != [*local, record["zeta"]["monodromy"]["milnor_fiber_euler_characteristic"]]:
        raise RuntimeError(f"Independent local invariants disagree for {ident}")
    program = (
        f'LIB "resolve.lib";ring r=0,({names}),dp;poly f={eq};'
        'list re=resolve(ideal(f),1,"L","A");print("CHARTS "+string(size(re[1])));quit;'
    )
    rows = {"atlas_embedded": [], "singular": []}
    for trial in range(trials):
        for tool in ("atlas_embedded", "singular") if trial % 2 == 0 else ("singular", "atlas_embedded"):
            if tool == "atlas_embedded":
                row = atlas_trial(record, "embedded", timeout)
            else:
                row, out = run(["Singular", "-q", "--no-rc"], timeout, program)
                if row["status"] == "complete":
                    count = re.search(r"^CHARTS (\d+)$", out, re.M)
                    if not count or "result is o.k." not in out or re.search(r"^\s*\?", out, re.M):
                        row["status"] = "failed"
                        row["reason"] = next(
                            (s.strip() for s in out.splitlines() if s.lstrip().startswith("?")),
                            "No checked resolution returned",
                        )
                    else:
                        row["charts"] = int(count[1])
            rows[tool].append(row)
    rows["atlas_intrinsic"] = [atlas_trial(record, "intrinsic", timeout) for _ in range(trials)]
    return {
        "id": ident,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "equation": eq,
        "variables": record["variables"],
        "mu_tau_fibre_euler": expected,
        "trials": rows,
        "median_wall_s": {
            tool: median(r["wall_s"] for r in runs)
            for tool, runs in rows.items()
            if all(r["status"] == "complete" for r in runs)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 2 or args.timeout <= 0:
        parser.error("Use at least two trials and a positive time limit")
    report = {
        "conditions": "Fresh processes, local germ over QQ, wall time includes startup. Atlas quick CPU embedded "
        "resolution requests its dual complex and both zetas. Separate intrinsic trials request its dual complex. Singular resolve.lib uses checks and keeps all "
        "charts (1,L,A). Outside the timer Singular computes local Milnor and Tjurina numbers and an independent "
        "Milnor fibre Euler characteristic. Distinct valid resolutions can have different chart counts.",
        "platform": platform.platform(),
        "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
        "timeout_s": args.timeout,
        "trial_count": args.trials,
        "cases": [],
    }
    for ident in args.only:
        record = json.loads(catalogue_file(f"records/{ident}.json").read_text())
        row = compare(record, args.trials, args.timeout)
        report["cases"].append(row)
        print(ident, row["median_wall_s"], flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
