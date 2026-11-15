"""Compare chart multiplication modulo its relation and algebraic coefficient division with Singular."""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.app import calculate_resolution
from engine.session import new_session
from benchmarks.published_comparison import cpu_name


def atlas_operations():
    session = new_session(calculate_resolution)
    list(session.run("""
x, y, z = coordinates("x", "y", "z")
C = resolve((x+y)^2-z*(x-y)^2, compute="cpu").chart("G1")
X, Y, Z, t = C.coordinates
D = resolve(x^4*y^2-2*z^6, mode="intrinsic", compute="cpu").chart("B4")
u, v, w = D.coordinates
theta = polynomial("theta", ring=u.parent)
"""))
    values = session.variables
    x, y, t = (values[key] for key in ("X", "Y", "t"))
    theta = values["theta"]
    numerator = values["u"] + values["v"]
    if theta**2 != 2 or t*x != y:
        raise ValueError("The sample chart presentation changed.")

    def multiply(indices):
        inputs = [x + i for i in indices]
        start = perf_counter_ns()
        answers = [t * value for value in inputs]
        elapsed = perf_counter_ns() - start
        if any(answer != y + i * t for i, answer in zip(indices, answers)):
            raise ValueError("Atlas returned a wrong chart normal form.")
        return elapsed / (1000 * len(inputs))

    def divide(indices):
        inputs = [theta + i for i in indices]
        start = perf_counter_ns()
        answers = [numerator / value for value in inputs]
        elapsed = perf_counter_ns() - start
        if any(answer * d != numerator for d, answer in zip(inputs, answers)):
            raise ValueError("Atlas returned a wrong coefficient quotient.")
        return elapsed / (1000 * len(inputs))

    return {"chart_product": multiply, "coefficient_division": divide}


SINGULAR = {
    "chart_product": (
        "ring R=0,(x,y,z,t),lp; ideal G=std(t*x-y);",
        "x+k", "reduce(t*inputs[i],G)", "answers[i] == y+k*t",
    ),
    "coefficient_division": (
        "ring R=(0,a),(u,v,w),lp; minpoly=a^2-2; poly dividend=u+v;",
        "number(a+k)", "dividend/inputs[i]", "answers[i]*inputs[i] == dividend",
    ),
}


def singular_sample(command, case, indices):
    ring, operand, operation, check = SINGULAR[case]
    program = f"""
{ring}
int i,k,elapsed;
list inputs,answers;
for(i=1;i<={len(indices)};i++) {{ k={indices.start}+i-1; inputs[i]={operand}; }}
elapsed=rtimer;
for(i=1;i<={len(indices)};i++) {{ answers[i]={operation}; }}
elapsed=rtimer-elapsed;
for(i=1;i<={len(indices)};i++) {{
    k={indices.start}+i-1;
    if(!({check})) {{ ERROR("incorrect result"); }}
}}
print("ELAPSED "+string(elapsed));
quit;
"""
    result = subprocess.run(command, input=program, capture_output=True, text=True, check=True, timeout=60)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or not lines[0].startswith("ELAPSED ") or result.stderr.strip():
        raise RuntimeError(result.stdout + result.stderr)
    return int(lines[0].split()[1]) / len(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if args.trials < 2 or args.batch < 1:
        parser.error("Use at least two trials and a positive batch size.")
    executable = shutil.which("Singular")
    if not executable:
        parser.error("Install Singular to run this comparison.")
    command = [executable, "-q", "--no-rc", "--random=47", "--threads=1", "--ticks-per-sec=1000000"]
    report = {
        "cpu": cpu_name(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "singular": subprocess.check_output([executable, "--version"], text=True).splitlines()[0],
        "trials": args.trials, "operations_per_trial": args.batch,
        "scope": (
            "Microseconds per operation on prepared chart objects and input values, including returning each "
            "polynomial. Atlas calls the same typed operators as the prompt. Singular calls reduce with a "
            "prepared lex basis or divides by a coefficient in QQ(a), a^2=2. Excludes process startup, "
            "resolution, ring declarations, operand construction and agreement checks. One discarded warmup "
            "batch, disjoint consecutive k starting at 2 even across trials, and alternating system order."
        ),
        "cases": {},
    }
    for case, measure in atlas_operations().items():
        times = {"atlas": [], "singular": []}
        for trial in range(args.trials + 1):
            start = 2 + trial * args.batch
            ks = range(start, start + args.batch)
            for name in (("atlas", "singular") if trial % 2 == 0 else ("singular", "atlas")):
                elapsed = measure(ks) if name == "atlas" else singular_sample(command, case, ks)
                if trial:
                    times[name].append(round(elapsed, 3))
        ring, operand, operation, check = SINGULAR[case]
        report["cases"][case] = {
            "ring": ring, "operand": operand, "operation": operation, "agreement": check,
            "samples_us": times,
            "median_us": {name: statistics.median(rows) for name, rows in times.items()},
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
