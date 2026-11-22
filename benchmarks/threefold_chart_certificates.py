"""Compare named four-variable divisor and positive-dimensional canonical cover certificates with Singular."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from zoo_resolution import oracle, run

EQUATIONS = {
    "triple-flat": "x*y*z*t*(x+2*y)",
    "five-general": "x*y*z*t*(x+y+z+t)",
    "two-pencils": "x*y*z*t*(x+y)*(z+t)",
    "primitive-rational-pencils": "(-2/13)*x*y*z*t*(2*x-9*t)*(3*y+5*z)",
    "doubled-quadric": "x^2+(y*z-t^2)^2+y^5+z^5",
}
NAMES = list("xyzt")


def resolution(ident, equation, trials, timeout):
    embedded = ["./atlas", "resolve", equation, "--vars", "x,y,z,t", "--mode", "embedded",
                "--monodromy", "--compute", "cpu", "--quick", "--json"]
    program = (f'LIB "resolve.lib";ring r=0,(x,y,z,t),dp;poly f={equation};'
               'list result=resolve(ideal(f),1,"L","A");print("CHARTS "+string(size(result[1])));quit;')
    numbers = oracle(dict(id=ident, variables=NAMES, equation=equation, dimension=3,
                          labels=[] if ident == "doubled-quadric" else ["Hyperplane arrangement"]))
    rows = {"atlas_embedded": [], "singular": []}
    for trial in range(trials):
        for system in (rows if trial % 2 == 0 else reversed(rows)):
            if system == "atlas_embedded":
                row, out = run(embedded, timeout)
                if row["status"] == "complete":
                    record = json.loads(out)
                    if (record["status"] != "resolved" or record["dual_complex"]["status"] != "complete"
                            or record["topological_zeta"]["status"] != "computed"
                            or record["monodromy_zeta"]["milnor_fiber_euler_characteristic"] != numbers[2]):
                        raise ValueError(f"Atlas resolution or independent Euler check failed on {ident}")
                    row["charts"] = len(record["charts"])
            else:
                row, out = run(["Singular", "-q", "--no-rc"], timeout, program)
                if row["status"] == "complete":
                    count = re.search(r"^CHARTS (\d+)$", out, re.M)
                    if not count or "result is o.k." not in out or re.search(r"^\s*\?", out, re.M):
                        raise ValueError(f"Singular did not certify {ident}: {out[-500:]}")
                    row["charts"] = int(count[1])
            rows[system].append(row)
    return dict(equation=equation, mu_tau_fibre_euler=numbers, trials=rows)


def marked_traces(trials):
    rows = {}
    for polynomial in ("(x*(z^2-y))^2+1", "x^2+(z^2-y)^2"):
        for d in (-1, 2):
            name = f"{polynomial} over a^2={d}"
            expected = 2 if d < 0 else 1
            singular = (f"ring R=(0,a),(x,y,z),lp;minpoly=a^2-({d});poly f={polynomial};"
                        'ideal B=factorize(f,1);print("COUNT "+string(size(B)));quit;')
            # Fresh processes; direct API timings include Python imports.
            atlas = ("from engine.candidate_field import indexed_field\n"
                     "from engine.field_polynomial import polynomial\n"
                     "from engine.field_factors import factors\n"
                     "from engine import native_polynomial as np\n"
                     "v=np.symbols('x y z');x,y,z=v\n"
                     f"k,_=indexed_field((1,0,{-d}),0)\n"
                     f"print('COUNT',len(factors(polynomial({polynomial.replace('^','**')},v,k))))\n")
            result = {"atlas": [], "singular": []}
            for trial in range(trials):
                for system in (result if trial % 2 == 0 else reversed(result)):
                    row, out = (run([sys.executable, "-c", atlas], 30) if system == "atlas" else
                                run(["Singular", "-q", "--no-rc"], 30, singular))
                    if row["status"] != "complete" or f"COUNT {expected}" not in out or re.search(r"^\s*\?", out, re.M):
                        raise ValueError(f"Marked-field factorization mismatch for {name}: {out[-500:]}")
                    result[system].append(row["wall_s"])
            rows[name] = dict(expected_factors=expected, wall_s=result)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.trials < 2 or a.timeout <= 0:
        p.error("Use at least two trials and a positive timeout.")
    report = dict(conditions="Fresh processes, exact QQ at zero. Atlas embedded quick CPU includes chart "
                  "verification, dual complex and both zetas. Singular resolve.lib checks and retains all charts "
                  "(1,L,A). Independent Singular Jacobian lengths / projective arrangement complement verify "
                  "Milnor fibre Euler characteristics. Separate marked-field trials time exact factorization with startup.",
                  cpu_affinity=sorted(os.sched_getaffinity(0)), timeout_s=a.timeout,
                  singular=subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
                  marked_factors=marked_traces(a.trials), resolutions={})
    for ident, equation in EQUATIONS.items():
        report["resolutions"][ident] = resolution(ident, equation, a.trials, a.timeout)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(ident, flush=True)
