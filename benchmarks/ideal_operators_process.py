"""Compare a standalone Atlas algebra command with a fresh single-threaded Singular process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.ideal_operators import CASES, OPERATIONS, check, singular_ideal
from benchmarks.primary_decomposition import cpu_name


def timed(argv, source=None):
    start = perf_counter_ns()
    run = subprocess.run(argv, input=source, capture_output=True, text=True, check=True, timeout=60)
    elapsed = (perf_counter_ns() - start) / 1e6
    if run.stderr.strip() or any(row.lstrip().startswith("?") for row in run.stdout.splitlines()):
        raise RuntimeError(run.stdout + run.stderr)
    return elapsed, run.stdout


def compare(left, right, operation, trials):
    name = {"sum": "ideal-sum", "product": "ideal-product", "square": "ideal-power"}[operation]
    atlas = [str(ROOT / "atlas"), "algebra", name, "--vars", "x,y", "--compute", "cpu", "--json"]
    for generator in left:
        atlas.extend(["--gen", generator])
    if operation == "square":
        atlas.extend(["--exponent", "2"])
    else:
        for generator in right:
            atlas.extend(["--other-gen", generator])
    declaration = f"ring R=0,(x,y),dp; option(redSB); short=0; ideal I={singular_ideal(left)};"
    if operation != "square":
        declaration += f" ideal J={singular_ideal(right)};"
    expression = OPERATIONS[operation][1]
    source = f"{declaration}\nideal answer={expression}; print(answer); quit;\n"
    singular = ["Singular", "-q", "--no-rc", "--threads=1"]
    methods = {"atlas": (atlas, None), "singular": (singular, source)}
    samples = {method: [] for method in methods}
    for i in range(trials + 1):
        for method in (list(methods) if i % 2 else list(reversed(methods))):
            elapsed, output = timed(*methods[method])
            if i:
                samples[method].append(round(elapsed, 3))
            if method == "atlas":
                result = json.loads(output)["result"]["generators"]
        check(declaration, expression, [result])
    return {"samples_ms": samples, "median_ms": {k: statistics.median(v) for k, v in samples.items()},
            "mutual_containment_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("Use at least two trials.")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    report = {
        "cpu": cpu_name(), "trials": args.trials,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "native_cli_sha256": hashlib.sha256((ROOT / "native/build/local_algebra").read_bytes()).hexdigest(),
        "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
        "conditions": "Wall time includes starting each executable, parsing inputs, computing the ideal, "
        "writing all results to pipes, and process exit. Atlas emits its full JSON record including replay. "
        "Singular prints the ideal in the same ring on one thread. One warm-up per executable is discarded. "
        "Process order alternates across trials. Mathematical verification runs outside the timer.",
        "cases": {},
    }
    for name, (left, right) in CASES.items():
        report["cases"][name] = {op: compare(left, right, op, args.trials) for op in OPERATIONS}
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
