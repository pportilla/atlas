"""Compare every truncated implicit-jet coefficient against Singular at the same exact point."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "rational_graph": dict(variables=["x", "y", "z"], dependent=[1, 2], rows=["z-x*y", "y-x^2"],
                           f="z", mu=[0, 1], point=[[2], [4], [8]], degree=4),
    "algebraic_taylor": dict(variables=["x", "y"], dependent=[], rows=[], f="x^3*y^3-y+x^2",
                             mu=[-2, 0, 1], point=[[2], [0, 1]], degree=3),
    "quadratic_implicit": dict(variables=["x", "y"], dependent=[1], rows=["y^2-x"], f="y",
                               mu=[-2, 0, 1], point=[[2], [0, 1]], degree=4),
    "cubic_implicit": dict(variables=["x", "y"], dependent=[1], rows=["y^3+y-x"], f="y",
                           mu=[1, 1, 0, 1], point=[[-1], [0, 1]], degree=3),
    "two_free_parameters": dict(variables=["x", "y", "z"], dependent=[1], rows=["y^2-x-z^2"],
                                f="y*z+x", mu=[-2, 0, 1], point=[[2], [0, 1], []], degree=5),
}


def element(coefficients):
    return "(" + ("+".join(f"({q})" + (f"*a^{i}" if i else "") for i, q in enumerate(coefficients) if str(q) != "0")
                  or "0") + ")"


def singular_data(case, candidate, repeats):
    variables, dependent = case["variables"], case["dependent"]
    free = [i for i in range(len(variables)) if i not in dependent]
    order = [*dependent, *free]
    field = "0" if len(case["mu"]) == 2 else "(0,a)"
    shifts = [f"{variables[i]}+{element(case['point'][i])}" for i in order]
    lines = ['system("--ticks-per-sec",1000000);', f'ring R={field},({",".join(variables[i] for i in order)}),lp;']
    if field != "0":
        lines.append(f'minpoly={element(case["mu"])};')
    lines += [f'map shift=R,{",".join(shifts)};', f'poly f={case["f"]};', f'int d={case["degree"]};']
    if case["rows"]:
        lines.append(f'ideal frame={",".join(case["rows"])};')
        compute = 'ideal I=shift(frame)+maxideal(d+1); return(reduce(shift(f),std(I)));'
    else:
        compute = 'return(jet(shift(f),d));'
    polynomial = []
    for term in candidate:
        factors = [element(term["coefficient"])]
        factors.extend(variables[i]+"^"+str(e) for i, e in zip(free, term["powers"]) if e)
        polynomial.append("*".join(factors))
    lines += [f'proc solve(){{{compute}}}', 'poly result=solve();', f'poly expected={"+".join(polynomial) or "0"};',
              'int i; int t=rtimer;', f'for(i=1;i<={repeats};i++){{result=solve();}}',
              'int elapsed=rtimer-t;', 'print("JET "+string(result==expected)+" "+string(elapsed));', 'quit;']
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if min(args.trials, args.repeats) < 1:
        parser.error("trials and repeats must be positive")
    os.sched_setaffinity(0, {args.cpu})
    report = {"cpu": args.cpu, "trials": args.trials, "repeats": args.repeats, "cases": {}}
    for name, case in CASES.items():
        payload = json.dumps(dict(case, repeats=args.repeats))
        def atlas():
            return json.loads(subprocess.run([ROOT / "native/build/bench_point_jet"], input=payload,
                                             text=True, capture_output=True, timeout=120, check=True).stdout)
        jet = atlas()["terms"]
        source = singular_data(case, jet, args.repeats)
        samples = {"atlas_us": [], "singular_us": []}
        for i in range(args.trials):
            for system in (("atlas", "singular") if i % 2 == 0 else ("singular", "atlas")):
                if system == "atlas":
                    result = atlas()
                    if result["terms"] != jet:
                        raise ValueError(f"{name}: changing Atlas coefficients")
                    samples["atlas_us"].append(result["us"])
                else:
                    output = subprocess.run(["Singular", "-q", "--no-rc"], input=source, text=True,
                                            capture_output=True, timeout=120, check=True).stdout
                    if any(line.lstrip().startswith("?") for line in output.splitlines()):
                        raise ValueError(output)
                    line, = [s for s in output.splitlines() if s.startswith("JET ")]
                    _, agrees, elapsed = line.split()
                    if agrees != "1":
                        raise ValueError(f"{name}: different coefficient on the jet")
                    samples["singular_us"].append(int(elapsed) / args.repeats)
        report["cases"][name] = dict(samples=samples, medians={k: median(v) for k, v in samples.items()})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
