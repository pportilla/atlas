"""Time the regular Newton fan of hypersurface germs in five and six variables.

Usage, from the repository root:

    python3 benchmarks/newton_fan_subdivision.py [--binary native/build/newton_fan] [--trials 3]
                                                 [--seconds 120] [--check-below 60000] [--output report.json]

Each germ is sent to the fan program as the worker sends it. The report has
the number of rays and maximal cones, the wall time of every trial and the
largest resident memory of the program. A fan with fewer cones than
--check-below is checked here with independent integer arithmetic: every
cone has determinant one, one Newton vertex is minimal on all its rays,
interior facets are shared by exactly two cones, and the cone sections fill
the standard simplex. To compare with another build of the program, compile
its source (`git show REVISION:native/newton_fan.cpp`) with the flags of
`native/newton_fan.mk` and pass it as --binary.
"""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import combinations
import json
from math import gcd, prod
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GERMS = [
    ("x^3*y+y^3*z+z^3*t+t^3*u+u^4", "x,y,z,t,u"),
    ("x^3+y^5+z^8+t^9+u^9+x*y*z*t*u", "x,y,z,t,u"),
    ("x^3+y^6+z^7+t^7+u^8+x*y*z*t*u", "x,y,z,t,u"),
    ("x^3+y^8+z^8+t^9+u^9+x*y*z*t*u", "x,y,z,t,u"),
    ("x^3+y^6+z^7+t^7+u^9+v^11+x*y*z*t*u*v", "x,y,z,t,u,v"),
    ("x^3+y^5+z^7+t^9+u^10+v^10+x*y*z*t*u*v", "x,y,z,t,u,v"),
    ("x^4*y+y^3*z+z^3*t+t^4*u+u^3*v+v^3", "x,y,z,t,u,v"),
    ("x^2*y+y^2*z+z^4*t+t^4*u+u^2*v+v^4*x", "x,y,z,t,u,v"),
    ("x^3*y+y^2*z+z^4*t+t^4*u+u^4*v+v^2*x", "x,y,z,t,u,v"),
    ("x^2*y+y^6+z^4+z^2*t^2+t^5+u^3+v^7+u*v^5", "x,y,z,t,u,v"),
    ("x*y-z*t+u^3+v^4", "x,y,z,t,u,v"),
]
# Reports the largest resident set of the program it runs, in kilobytes.
WRAPPER = ("import resource,subprocess,sys\n"
           "p=subprocess.run([sys.argv[1]],stdin=sys.stdin,stdout=sys.stdout)\n"
           "sys.stderr.write('MAXRSS %d\\n'%resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)\n")


def exponents(equation, names):
    rows = []
    for term in re.split(r"[+-]", equation):
        row = [0] * len(names)
        for factor in term.strip().split("*"):
            match = re.fullmatch(r"([a-z])(?:\^(\d+))?", factor)
            if match:
                row[names.index(match.group(1))] += int(match.group(2) or 1)
        if any(row):
            rows.append(row)
    return rows


def determinant(matrix):
    a = [list(row) for row in matrix]
    last, sign = 1, 1
    for j in range(len(a) - 1):
        pivot = next((i for i in range(j, len(a)) if a[i][j]), None)
        if pivot is None:
            return 0
        if j != pivot:
            a[j], a[pivot] = a[pivot], a[j]
            sign = -sign
        for i in range(j + 1, len(a)):
            for k in range(j + 1, len(a)):
                a[i][k] = (a[i][k] * a[j][j] - a[i][j] * a[j][k]) // last
        last = a[j][j]
    return sign * a[-1][-1]


def regular_and_complete(fan, support):
    rays, cones = fan["rays"], fan["cones"]
    dim = len(support[0])
    if len({tuple(r) for r in rays}) != len(rays) or any(min(r) < 0 or gcd(*r) != 1 for r in rays):
        return False
    minima = []
    for ray in rays:
        weights = [sum(a * b for a, b in zip(ray, e)) for e in support]
        minima.append({j for j, v in enumerate(weights) if v == min(weights)})
    volume, facets = Fraction(0), Counter()
    for cone in cones:
        generators = [rays[i] for i in cone]
        if determinant(generators) != 1 or not set.intersection(*(minima[i] for i in cone)):
            return False
        volume += Fraction(1, prod(sum(r) for r in generators))
        for facet in combinations(cone, dim - 1):
            facets[tuple(sorted(facet))] += 1
    for facet, count in facets.items():
        boundary = any(all(rays[i][j] == 0 for i in facet) for j in range(dim))
        if count != (1 if boundary else 2):
            return False
    return volume == 1


def measure(binary, equation, names, trials, seconds, check_below):
    support = exponents(equation, names)
    request = json.dumps({"exponents": support, "exhaustive": False})
    row = {"germ": equation, "variables": names, "seconds": []}
    for _ in range(trials):
        start = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-c", WRAPPER, str(binary)], input=request, capture_output=True,
                                  text=True, timeout=seconds)
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-9", "-f", str(binary)])
            row["outcome"] = f"no answer in {seconds} s"
            return row
        row["seconds"].append(round(time.perf_counter() - start, 3))
        fan = json.loads(proc.stdout)
        if "error" in fan:
            row["outcome"] = fan["error"]
            return row
        sizes = (len(fan["rays"]), len(fan["cones"]))
        if row.setdefault("sizes", sizes) != sizes:
            raise AssertionError(f"The fan of {equation} differs between trials.")
        peak = re.search(r"MAXRSS (\d+)", proc.stderr)
        row["peak_resident_mb"] = max(row.get("peak_resident_mb", 0), int(peak.group(1)) // 1024 if peak else 0)
    rays, cones = row.pop("sizes")
    row.update(outcome="regular fan", rays=rays, cones=cones, median_seconds=statistics.median(row["seconds"]),
               largest_ray_coordinate=max(max(r) for r in fan["rays"]))
    if cones < check_below:
        row["regular_and_complete"] = regular_and_complete(fan, support)
        if not row["regular_and_complete"]:
            raise AssertionError(f"The fan of {equation} failed its independent check.")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", default=str(ROOT / "native" / "build" / "newton_fan"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--check-below", type=int, default=60000)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"program": args.binary, "host": platform.processor() or platform.machine(), "trials": args.trials,
              "rows": []}
    for equation, names in GERMS:
        row = measure(Path(args.binary).resolve(), equation, names.split(","), args.trials, args.seconds,
                      args.check_below)
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
