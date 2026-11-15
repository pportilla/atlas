"""Compare Atlas homomorphism application with Singular map application over QQ."""
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
from benchmarks.primary_decomposition import cpu_name
from engine.session_algebra import polynomial
from engine.session_maps import hom
from engine.session_polynomial import coordinates


# For input f+i*x, the expected answer is expected+i*images[0].
CASES = (
    ("inclusion", "(x+y)^5 + (1/7)*x^3", ("u", "v"), "(u+v)^5 + (1/7)*u^3"),
    ("nonlinear", "(x-y)^2 + x^12*y", ("u+v^2", "u"), "v^4 + (u+v^2)^12*u"),
    ("collision", "x^512 + y^512", ("2*u/3", "u"), "((2/3)^512+1)*u^512"),
)


def atlas_trial(case, count):
    _, source, images, expected = case
    x, y = coordinates("x", "y")
    u, v, w = coordinates("u", "v", "w")
    images = tuple(polynomial(s, ring=u.parent) for s in images)
    mapping = hom(x.parent, u.parent, images=images)
    base = polynomial(source, ring=x.parent)
    inputs = [base + i*x for i in range(1, count+1)]
    expected_base = polynomial(expected, ring=u.parent)
    mapping(base)
    start = perf_counter_ns()
    results = [mapping(f) for f in inputs]
    elapsed = (perf_counter_ns() - start) / 1000 / count
    if any(f != expected_base + i*images[0] for i, f in enumerate(results, 1)):
        raise AssertionError(f"Atlas disagrees for {case[0]}")
    return elapsed


def singular_trial(case, count):
    _, source, images, expected = case
    indices = range(1, count+1)
    lines = ["ring src=0,(x,y),dp;", f"poly f={source};"]
    lines += [f"poly p{i}=f+{i}*x;" for i in indices]
    lines += ["ring dst=0,(u,v,w),dp;", f"map h=src,{','.join(images)};",
              "poly warm=h(f);", f"poly base={expected};", f"poly firstImage={images[0]};"]
    lines += [f"poly g{i};" for i in indices]
    lines += ['system("--ticks-per-sec",1000000);', "int start=rtimer;"]
    lines += [f"g{i}=h(p{i});" for i in indices]
    lines += ["int elapsed=rtimer-start;", "int ok=1;"]
    lines += [f"ok=ok && (g{i}==base+{i}*firstImage);" for i in indices]
    lines += ['print("RESULT "+string(elapsed)+" "+string(ok));', "quit;"]
    run = subprocess.run(["Singular", "-q", "--no-rc", "--threads=1"], input="\n".join(lines), text=True,
                         capture_output=True, timeout=30, check=True)
    parts = run.stdout.split()
    if len(parts) != 3 or parts[0] != "RESULT" or parts[2] != "1":
        raise AssertionError(f"Singular disagrees for {case[0]}: {run.stdout[:600]} {run.stderr[:600]}")
    return int(parts[1]) / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 2 or args.count < 1:
        parser.error("Use at least two trials and one input per trial.")
    report = {"conditions": "CPU, QQ[x,y] to QQ[u,v,w], prepared maps and distinct inputs f+i*x. "
                            "Times include each public map application and materializing its image. "
                            "Map construction, operand setup, one warmup, and agreement checks are untimed. "
                            "System order alternates between trials.",
              "cpu": cpu_name(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "unit": "microseconds per polynomial", "trials": args.trials, "inputs_per_trial": args.count,
              "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "cases": []}
    for case in CASES:
        methods = (("atlas", atlas_trial), ("singular", singular_trial))
        samples = {name: [] for name, _ in methods}
        for trial in range(args.trials):
            for name, method in methods if trial % 2 == 0 else reversed(methods):
                samples[name].append(method(case, args.count))
        report["cases"].append({"name": case[0], "polynomial": case[1], "images": case[2],
                                "expected": case[3], "samples": samples,
                                "median": {name: median(times) for name, times in samples.items()}})
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
