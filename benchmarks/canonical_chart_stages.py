"""Compare centre selection and exceptional incidence inside the canonical CLI's solver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import process_time_ns
from unittest.mock import patch

from centre_selection import CASES


def sample(directory, record):
    # Import the solver from the selected installation, including its native library.
    sys.path.insert(0, str(directory))
    from engine import atlas, local_blowups, origin_backend
    from engine.native_library import replay_scope
    from engine.polynomial import parse

    variables = record["variables"]
    poly = parse(record["polynomial"], variables, bounded=False)
    choose = local_blowups._Canonical.choose
    decisions = []

    def timed_choice(*args, **kwargs):
        start = process_time_ns()
        try:
            return choose(*args, **kwargs)
        finally:
            decisions.append(process_time_ns() - start)

    with replay_scope(compute="cpu"), patch.object(local_blowups._Canonical, "choose", timed_choice):
        start = process_time_ns()
        result = origin_backend._resolve(poly, variables, record["mode"], None, defer_outputs=True)
        producer = process_time_ns() - start
        if (result is None or json.loads(json.dumps(result["steps"])) != record["steps"]
                or result["atlas"] != record["atlas"]):
            raise ValueError("Isolated centre selection produced different blowups or owned charts")
        charts = atlas.from_record(record, poly, native=True)
        start = process_time_ns()
        rational = {}
        counts = local_blowups.cells(charts, len(variables), rational_counts=rational)
        incidence = process_time_ns() - start

    def ordered(counts):
        return sorted((sorted(labels), value) for labels, value in counts.items())

    return dict(choose_ms=sum(decisions)/1e6, producer_ms=producer/1e6, incidence_ms=incidence/1e6,
                blowups=len(decisions), intersections=ordered(counts), rational=ordered(rational))


def compare(equation, trees, trials):
    args = ["resolve", equation, "--vars", "x,y,z", "--mode", "embedded", "--compute", "cpu", "--json"]
    record = json.loads(subprocess.check_output([trees["before"] / "atlas", *args], cwd=trees["before"]))
    if record["engine"] != "canonical-blowups" or record["status"] != "resolved":
        raise ValueError(f"Expected a complete canonical resolution for {equation}")
    saved = json.dumps(record)
    samples = {key: [] for key in trees}
    expected = None
    for i in range(trials + 1):
        for key in (list(trees) if i % 2 else reversed(trees)):
            result = json.loads(subprocess.check_output([sys.executable, __file__, "--worker", trees[key]],
                                                        cwd=trees[key], input=saved, text=True))
            geometry = {field: result.pop(field) for field in ("blowups", "intersections", "rational")}
            if expected is None:
                expected = geometry
            if geometry != expected:
                raise ValueError(f"Incidence or centre count differs for {equation}")
            if i:
                samples[key].append(result)
    timings = {k: {field: round(median(row[field] for row in rows), 3) for field in rows[0]}
               for k, rows in samples.items()}
    return dict(polynomial=equation, charts=len(record["charts"]), measurements=samples,
                median_ms=timings, verified=expected)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(sample(Path(sys.argv[2]), json.load(sys.stdin))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("At least one timed pair is required")
    os.sched_setaffinity(0, {args.cpu})
    trees = {"before": args.before.resolve(), "after": args.after.resolve()}
    report = dict(cpu=args.cpu, trials=args.trials,
                  conditions="Fresh interpreter per sample, one discarded warmup per build and input, alternating order. "
                  "Process CPU time excludes interpreter startup. Producer includes centre selection. Incidence times "
                  "the first count on already decoded charts, including rational component counts.",
                  native_sha256={k: hashlib.sha256((p/"native/build/libgroebner.so").read_bytes()).hexdigest()
                                 for k, p in trees.items()}, cases=[])
    for equation, mode in CASES:
        if mode == "embedded":
            case = compare(equation, trees, args.trials)
            report["cases"].append(case)
            print(equation, case["median_ms"], flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
