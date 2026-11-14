"""Time and verify embedded resolution of x^2-y^2*(z+t) at the origin over QQ.

Each fresh Atlas process returns its complete JSON record. Singular resolve.lib
checks the embedded resolution. The driver also verifies each Singular pullback
and matches both blowup projections. For a toric Atlas resolution, Singular
independently checks the reported factorization and Jacobian of each chart map
outside the timer. Optional Atlas zeta runs are timed separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess

from plane_conjugate_resolution import measure

ROOT = Path(__file__).resolve().parents[1]
POLYNOMIAL = "x^2-y^2*(z+t)"
SINGULAR = '''
LIB "resolve.lib";
ring src=0,(x,y,z,t),dp;
poly f=x^2-y^2*(z+t);
list re=resolve(ideal(f),1,"L","A");
int ix; int chartMask=0;
for(ix=1;ix<=size(re[1]);ix++){
    def sc=re[1][ix]; setring sc;
    poly e=BO[4][1][1]; poly h=BO[2][1];
    poly a0=BO[5][1]; poly a1=BO[5][2]; poly b=BO[5][3]; poly c=BO[5][4];
    poly pull=a0^2-a1^2*(b+c);
    int ok=(pull==e^2*h || pull==-e^2*h);
    // The new coordinate in these two resolve.lib charts is var(4).
    if(ok && a0==e && a1==e*var(4)){chartMask=chartMask+1;}
    if(ok && a1==e && a0==e*var(4)){chartMask=chartMask+2;}
    print("MAP "+string(a0)+" "+string(a1)+" "+string(e)+" "+string(h));
    setring src; kill sc;
}
print("CHECK "+string(size(re[1]))+" "+string(chartMask));
quit;
'''
SINGULAR_DEFAULT = '''
LIB "resolve.lib";
ring src=0,(x,y,z,t),dp;
list re=resolve(ideal(x^2-y^2*(z+t)),"L","A");
print("CHARTS "+string(size(re[1])));
quit;
'''


def checked_record(raw, zetas):
    r = json.loads(raw)
    assert r["status"] == "resolved" and r["dual_complex"]["status"] == "complete"
    assert len(r["atlas"]) == len(r["charts"])
    assert r["certificate"]["atlas_verification"] == "Embedded normal crossings on the owned origin fibre"
    if r["engine"] == "verified-ideal-blowup":
        assert len(r["charts"]) == 2
        assert r["steps"] == [{"id": "B", "type": "blowup of an ideal", "generators": ["x", "y"]}]
        for chart in r["atlas"]:
            assert [(c["id"], c["multiplicity"], c["nu"]) for c in chart["components"]] == [
                ("E1", 2, 2), ("strict", 1, 1)]
    else:
        assert r["engine"] == "verified-monomial-modification" and len(r["charts"]) == 3
        assert r["certificate"]["boundary_image_coordinate_ideals"] == [["x", "y"]]
    if zetas:
        assert r["topological_zeta"]["expression"] == "(s + 2)/(2*(s + 1)^2)"
        assert r["monodromy_zeta"]["factors"] == [{"multiplicity": 2, "exponent": -1}]
    elapsed = r.pop("elapsed_ms")
    return r, elapsed


def check_toric_pullbacks(singular, record, timeout):
    """Reconstruct f after substitution, and the determinant of each projection."""
    lines = ["ring charts=0,(u1,u2,u3,u4),dp;"]

    def expression(text):
        if not re.fullmatch(r"[0-9u+*^() /-]+", text):
            raise ValueError(f"Unexpected chart polynomial {text!r}")
        return text

    for i, chart in enumerate(record["atlas"], 1):
        assert chart["coordinates"] == ["u1", "u2", "u3", "u4"]
        lines.append(f"matrix M{i}[4][4];")
        for j, source in enumerate(record["variables"], 1):
            lines.append(f"poly m{i}_{j}={expression(chart['map'][source])};")
            for k in range(1, 5):
                lines.append(f"M{i}[{j},{k}]=diff(m{i}_{j},var({k}));")
        components = {c["id"]: c for c in chart["components"]}
        strict = components.pop("strict")
        h, = strict["ideal"]
        total, volume = [], []
        for divisor in components.values():
            e, = divisor["ideal"]
            total.append(f"({expression(e)})^{int(divisor['multiplicity'])}")
            volume.append(f"({expression(e)})^{int(divisor['nu'])-1}")
        jac = "*".join(volume)
        lines.append(f"poly pull{i}=m{i}_1^2-m{i}_2^2*(m{i}_3+m{i}_4);")
        lines.append(f'print("PULLBACK {i} "+string(pull{i}==({"*".join(total)})*({expression(h)})));')
        lines.append(f'print("JACOBIAN {i} "+string(det(M{i})==({jac}) || det(M{i})==-({jac})));')
    _, output = measure([singular, "-q", "--no-rc"], ROOT, timeout, "\n".join([*lines, "quit;"]))
    checks = [f"{kind} {i} 1" for i in range(1, len(record["atlas"]) + 1)
              for kind in ("PULLBACK", "JACOBIAN")]
    if output.splitlines() != checks:
        raise ValueError(output[-1500:])
    return checks


def check_singular_fibre(singular, timeout):
    # The two standard opens glue the exceptional fibre to P^1. Verify that
    # the reduced strict divisor hits it at exactly one point, transversely.
    extra = '''
    matrix M[4][4]; int j,k; ideal image=a0,a1,b,c;
    for(j=1;j<=4;j++){for(k=1;k<=4;k++){M[j,k]=diff(image[j],var(k));}}
    int jac=(det(M)==e || det(M)==-e);
    poly hf=reduce(h,std(image)); poly r=var(4);
    int fibre=(b==var(2) && c==var(3) && e==var(1));
    int xchart=(a0==e && a1==e*r && (hf==1 || hf==-1));
    int ychart=(a1==e && a0==e*r && (hf==r^2 || hf==-r^2)
                && (diff(h,b)==1 || diff(h,b)==-1));
    print("FIBRE "+string(ok && jac && fibre && (xchart || ychart)));
'''
    script = SINGULAR.replace('    print("MAP ', extra + '    print("MAP ')
    _, output = measure([singular, "-q", "--no-rc"], ROOT, timeout, script)
    if output.splitlines().count("FIBRE 1") != 2 or "CHECK 2 3" not in output.splitlines():
        raise ValueError(output[-2000:])
    # chi(P^1 minus one point) = chi(one point) = 1. A'Campo uses the first
    # stratum. The topological zeta also uses the point on the strict divisor.
    return {"origin_fibre": "P^1", "exceptional": {"N": 2, "nu": 2},
            "strict": {"N": 1, "nu": 1}, "euler_E_only": 1, "euler_E_strict": 1,
            "monodromy": "1/(1-T^2)", "topological": "(s + 2)/(2*(s + 1)^2)"}


def run(args):
    roots = {"before": args.baseline.resolve(), "after": args.atlas_root.resolve()}
    singular = shutil.which("Singular")
    if not singular:
        raise SystemExit("Install Singular with resolve.lib")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    binaries = ("atlas", "native/build/libgroebner.so", "native/build/newton_fan",
                "native/build/libsmall_newton_fan.so", "native/build/resolve_supervisor")
    report = {"input": POLYNOMIAL, "variables": ["x", "y", "z", "t"], "mode": "embedded",
              "search": args.search,
              "platform": platform.platform(), "python": platform.python_version(),
              "comparison": "One warm-up per variant, alternating fresh processes on the same CPU. "
                            "All Atlas fields except elapsed_ms are compared on every run. "
                            "Singular_default times the standard local resolution and final chart count. "
                            "Singular also enables resolve.lib correctness checks and verifies its two pullbacks.",
              "file_sha256": {k: {f: hashlib.sha256((p/f).read_bytes()).hexdigest()
                                  for f in binaries if (p/f).is_file()} for k, p in roots.items()},
              "cpu_affinity": args.cpu, "trials": args.trials, "scope": __doc__,
              "singular": subprocess.check_output([singular, "--version"], text=True).splitlines()[0],
              "face_binary_sha256": {k: hashlib.sha256((p/"native/build/face_gb").read_bytes()).hexdigest()
                                     for k, p in roots.items()}, "results": {}}
    report["singular_fibre"] = check_singular_fibre(singular, args.timeout)
    for zetas in (False, True):
        cmd = ["./atlas", "resolve", POLYNOMIAL, "--vars", "x,y,z,t", "--mode", "embedded",
               "--compute", "cpu", "--json"]
        if args.search != "default":
            cmd.append("--" + args.search)
        if zetas:
            cmd.append("--monodromy")
        order = list(roots) if zetas else [*roots, "Singular_default", "Singular"]
        result = {"command": cmd, "runs": []}
        report["results"]["resolution_and_zetas" if zetas else "resolution"] = result
        expected = None
        for trial in range(-1, args.trials):
            for solver in (order if trial % 2 == 0 else reversed(order)):
                if solver in ("Singular", "Singular_default"):
                    checked = solver == "Singular"
                    script = SINGULAR if checked else SINGULAR_DEFAULT
                    row, text = measure([singular, "-q", "--no-rc"], ROOT, args.timeout, script)
                    marker = "CHECK 2 3" if checked else "CHARTS 2"
                    if (marker not in text.splitlines() or (checked and "result is o.k." not in text)
                            or any(line.lstrip().startswith("?") for line in text.splitlines())):
                        raise ValueError(text[-1200:])
                else:
                    row, raw = measure(cmd, roots[solver], args.timeout)
                    current, row["reported_elapsed_ms"] = checked_record(raw, zetas)
                    if zetas:
                        for field in ("monodromy", "topological"):
                            assert current[field + "_zeta"]["expression"] == report["singular_fibre"][field]
                    if expected is None:
                        expected = current
                        result["engine"] = current["engine"]
                        result["charts"] = len(current["atlas"])
                        result["agreed_record"] = current
                        if current["engine"] == "verified-monomial-modification":
                            result["singular_checks_on_atlas_maps"] = check_toric_pullbacks(
                                singular, current, args.timeout)
                    if current != expected:
                        raise ValueError(f"Full resolution differs for {solver} in trial {trial+1}")
                    row["record_sha256"] = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
                    if trial == -1 and zetas:
                        archived = {k: current[k] for k in ("status", "variables", "polynomial", "mode", "atlas", "engine")}
                        _, saved = measure(["./atlas", "monodromy", "-", "--json"], roots[solver],
                                           args.timeout, json.dumps(archived))
                        if json.loads(saved) != current["monodromy_zeta"]:
                            raise ValueError(f"Saved chart replay differs for {solver}")
                if trial >= 0:
                    result["runs"].append(dict(trial=trial+1, solver=solver, **row))
        result["median_s"] = {solver: {k: statistics.median(r[k] for r in result["runs"] if r["solver"] == solver)
                                       for k in ("wall_s", "cpu_s")} for solver in order}
        paired = {solver: [r["wall_s"] for r in result["runs"] if r["solver"] == solver]
                  for solver in order}
        result["median_paired_speedup"] = statistics.median(a/b for a, b in zip(paired["before"], paired["after"]))
        result["faster_pairs"] = sum(a > b for a, b in zip(paired["before"], paired["after"]))
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(cmd, result["median_s"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--atlas-root", type=Path, default=ROOT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--search", choices=("quick", "exhaustive", "default"), default="quick")
    args = parser.parse_args()
    if args.trials < 1 or args.timeout <= 0:
        parser.error("Use a positive trial count and time limit")
    run(args)
