"""Atlas before and after the native canonical steps: full records and alternating timings.

Two measurements on the same inputs, in alternating order between the two
builds, after one warm-up round, pinned to one logical CPU:

- `cli`: fresh `atlas resolve` processes, with and without --monodromy. The
  whole JSON record except timing fields (keys ending in `_ms`) must be the
  same in every run of both builds.
- `lane`: the canonical (origin) lane alone, in a fresh interpreter, split
  into the centre choice, the blowups and the presence and crossing checks.
  `cold` is the first call of the interpreter, as in a fresh resolution
  worker; `warm` follows one call on an unrelated input, so that lazy imports
  and caches of the step-by-step route are already in place.

Singular is not run: the comparison is between Atlas builds.
"""
import argparse
import gzip
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

CASES = ["(x+y)^2+z^3+x^3", "(x*y-z^2)^2+x^5+y^5", "x^2+(y+z)^3+y^4", "(x+2*y)^2+(y+z)^3+z^4", "x^2*y+y^4+z^3"]

LANE = r'''
import json, sys, time, collections
sys.path.insert(0, sys.argv[1])
from engine.polynomial import parse
from engine import local_blowups as lb
from engine.origin_backend import _resolve
times = collections.Counter()
def wrap(owner, name, key):
    f = getattr(owner, name, None)
    if f is None:
        return
    def timed(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            times[key] += time.perf_counter() - t
    setattr(owner, name, timed)
wrap(lb._Canonical, "choose", "choose"); wrap(lb._Direct, "choose", "choose"); wrap(lb, "_blown_up", "blowups")
wrap(lb, "_checked_charts", "checks"); wrap(lb, "_own_failures", "checks")
if sys.argv[3] == "warm":
    _resolve(parse("x^2+y^3+z^5", ["x", "y", "z"]), ("x", "y", "z"), "embedded", None, defer_outputs=True)
times.clear()
t = time.perf_counter()
_resolve(parse(sys.argv[2], ["x", "y", "z"]), ("x", "y", "z"), "embedded", None, defer_outputs=True)
times["total"] = time.perf_counter() - t
print(json.dumps({k: 1000 * v for k, v in times.items()}))
'''


def without_timings(value):
    if isinstance(value, list):
        return [without_timings(v) for v in value]
    if isinstance(value, dict):
        return {k: without_timings(v) for k, v in value.items() if not k.endswith("_ms")}
    return value


def cli(tree, case, zetas):
    command = [str(tree / "atlas"), "resolve", case, "--vars", "x,y,z", "--mode", "embedded", "--compute", "cpu",
               "--json", *(["--monodromy"] if zetas else [])]
    start = time.perf_counter()
    done = subprocess.run(command, cwd=tree, capture_output=True, text=True, check=True)
    return 1000 * (time.perf_counter() - start), without_timings(json.loads(done.stdout))


def lane(tree, case, start):
    done = subprocess.run([sys.executable, "-c", LANE, str(tree), case, start], cwd=tree, capture_output=True,
                          text=True, check=True)
    return json.loads(done.stdout)


def alternate(trees, trials, measure):
    samples = {name: [] for name in trees}
    for trial in range(trials + 1):
        for name in (list(trees) if trial % 2 else list(trees)[::-1]):
            value = measure(trees[name])
            if trial:
                samples[name].append(value)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    trees = dict(before=args.before.resolve(), after=args.after.resolve())
    report = dict(cpu=args.cpu, trials=args.trials, machine=platform.machine(), processor=platform.processor(),
                  python=platform.python_version(), cli={}, lane={})
    for case in CASES:
        for zetas in (False, True):
            records = {}

            def measure(tree):
                wall, record = cli(tree, case, zetas)
                if records.setdefault("record", record) != record:
                    raise ValueError(f"The records of {case} differ between runs")
                return wall

            samples = alternate(trees, args.trials, measure)
            key = case + (" --monodromy" if zetas else "")
            medians = {name: statistics.median(values) for name, values in samples.items()}
            report["cli"][key] = dict(engine=records["record"]["engine"], samples=samples, median_ms=medians)
            print(f"cli  {key:40} before {medians['before']:7.1f} ms  after {medians['after']:7.1f} ms", flush=True)
        report["lane"][case] = {}
        for start in ("cold", "warm"):
            samples = alternate(trees, args.trials, lambda tree: lane(tree, case, start))
            medians = {name: {part: statistics.median(run.get(part, 0) for run in runs) for part in
                              ("total", "choose", "blowups", "checks")} for name, runs in samples.items()}
            report["lane"][case][start] = dict(samples=samples, median_ms=medians)
            print(f"lane {start} {case:35} " + "  ".join(
                f"{name} " + " ".join(f"{part} {value:6.1f}" for part, value in parts.items())
                for name, parts in medians.items()), flush=True)
        args.output.write_bytes(gzip.compress(json.dumps(report, indent=1).encode(), mtime=0))


if __name__ == "__main__":
    main()
