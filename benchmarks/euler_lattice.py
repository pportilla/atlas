"""Compare affine Euler characteristics with Singular's chern.lib on the same input."""
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import monotonic, process_time

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "fermat_curve": ("x,y", "x^11+y^11-1", -99),
    "free_torus_fibres": ("x,y,z", "x^2*y^2+x^2*z^2+y^2*z^2-1", 16),
    "double_quadric_chart": ("z,t,u", "(z*t-u^2)^5+1+z^5+t^5+u^5", 145),
    "quadratic_affine_fibres": ("x,y", "x*y^2+y+1", 0),
    "contact_quadratic": ("z,t,u,h", "(2*(z*t-u^2)^6+3*z^6+5*t^6+7*u^6+1)*h^2"
                          "-((z*t-u^2)^3+u^3+1)*h+1", -622),
}


def atlas_case(name):
    sys.path.insert(0, str(ROOT))
    from engine import native_polynomial as polynomial
    from engine.constructible_euler import euler_characteristic
    from engine.polynomial import parse_expression

    coordinates, text, _ = CASES[name]
    variables = tuple(polynomial.Symbol(v) for v in coordinates.split(","))
    equation = polynomial.from_terms(variables, parse_expression(text, coordinates.split(",")))
    start = process_time()
    value = euler_characteristic(variables, [equation])
    print("CHI", value, "MS", round((process_time() - start)*1000, 3))


def measure(solver, name, seed, timeout):
    variables, f, expected = CASES[name]
    if solver == "Atlas":
        cmd = [sys.executable, __file__, "--atlas-case", name]
        script = ""
    else:
        cmd = ["Singular", "-q", "--no-rc", "--random=" + str(seed), "--ticks-per-sec=1000"]
        script = (f'LIB "chern.lib"; ring r=0,({variables}),dp; ideal I={f}; int start=timer; '
                  'print("CHI "+string(EulerAff(I))+" MS "+string(timer-start)); quit;')
    start = monotonic()
    try:
        result = subprocess.run(cmd, input=script, capture_output=True, cwd=ROOT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_seconds": round(monotonic() - start, 3)}
    wall = round(monotonic() - start, 4)
    match = re.search(r"^CHI (-?\d+) MS ([\d.]+)$", result.stdout, re.M)
    if result.returncode or not match:
        raise RuntimeError(result.stdout + result.stderr)
    chi = int(match.group(1))
    if chi != expected:
        raise AssertionError((solver, name, chi, expected))
    return {"status": "verified", "chi": chi, "wall_seconds": wall,
            "calculation_cpu_ms": float(match.group(2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, help="Run only the named cases.")
    parser.add_argument("--atlas-case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.atlas_case:
        return atlas_case(args.atlas_case)
    if not args.output or args.timeout <= 0 or args.trials < 1:
        parser.error("Supply an output path, a positive timeout and at least one trial.")
    report = {"platform": platform.platform(), "solvers": {
        "Singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
        "Atlas": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "method": ("Each solver starts in a new process for each input. Wall time includes process "
                   "startup. Calculation CPU time excludes input parsing and library loading. "
                   "Singular uses chern.lib::EulerAff with the trial number as the seed."),
        "timeout_seconds": args.timeout, "cases": {name: CASES[name] for name in args.cases or CASES},
        "runs": []}
    for trial in range(1, args.trials + 1):
        for name in report["cases"]:
            for solver in ("Atlas", "Singular") if trial % 2 else ("Singular", "Atlas"):
                measured = measure(solver, name, trial, args.timeout)
                report["runs"].append(dict(trial=trial, case=name, solver=solver, **measured))
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
