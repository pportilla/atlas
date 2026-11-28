"""Time complete centre invariant requests in fresh processes and check the exact native replies."""
import argparse
import ctypes
import gzip
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("canonical-s15-maximum.jsonl.gz")
REPAIR_INVARIANTS = {"s15": ["4", "0", "1", "1", "1", "0", "inf"],
                     "loop": ["2", "0", "2", "1", "1", "0", "inf"],
                     "chain": ["3", "0", "4/3", "1", "1", "0", "inf"],
                     "chain-gcd": ["3", "0", "1", "1", "1", "0", "inf"]}


def worker(root, repair=None, corpus=FIXTURE):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    kernel = ctypes.CDLL(str(root / "native/build" / ("libgroebner" + suffix)))
    maximum = kernel.atlas_centre_maximum
    maximum.argtypes = [ctypes.c_char_p]
    maximum.restype = ctypes.c_char_p
    if repair:
        request = Path(__file__).with_name(f"canonical-{repair}-repair.json").read_bytes()
        start = perf_counter_ns()
        payload = maximum(request)
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        result = json.loads(payload)["result"]
        expected = dict(centre=["1 1 0 0", "1 0 1 0", "1 0 0 1"], key=[],
                        value=REPAIR_INVARIANTS[repair])
        deadline = result == {"raise": "GiveUp", "message": "The time allowed for the blowups is over."}
        if result != {"maximum": expected} and not deadline:
            raise ValueError(f"Unexpected Newton repair reply: {result}")
        return dict(ms=elapsed_ms, deadline=deadline)
    times = []
    with gzip.open(corpus, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            request = row["request"].encode()
            start = perf_counter_ns()
            result = maximum(request)
            times.append((perf_counter_ns() - start) / 1e6)
            if result != row["response"].encode():
                raise ValueError(f"Centre request {len(times)} returned a different result: {result!r}")
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--reference", type=Path, default=ROOT, help="Pre-change build or same build with proposals off")
    parser.add_argument("--output", type=Path)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--repair", nargs="?", choices=REPAIR_INVARIANTS, const="s15",
                        help="Replay a captured Newton repair with a 15 s deadline (default repair: s15)")
    inputs.add_argument("--corpus", type=Path, default=FIXTURE,
                        help="Gzipped JSONL of completed maximum calls with their expected native replies")
    parser.add_argument("--reference-switch", choices=("0", "1"), default="0",
                        help="ATLAS_LINEAR_ALGEBRA setting for the reference checkout")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        print(json.dumps(worker(options.worker, options.repair, options.corpus)))
        return
    if options.trials < 1:
        parser.error("at least one trial is required")
    os.sched_setaffinity(0, {options.cpu})
    trials = {"reference": [], "candidate": []}
    roots = {"reference": options.reference.resolve(), "candidate": ROOT}
    for i in range(options.trials):
        for name in list(roots)[::1 if i % 2 == 0 else -1]:
            env = dict(os.environ, ATLAS_LINEAR_ALGEBRA=options.reference_switch if name == "reference" else "1")
            env.pop("ATLAS_RECORD_REQUESTS", None)
            command = [sys.executable, __file__, "--cpu", str(options.cpu), "--worker", roots[name]]
            if options.repair:
                command.extend(["--repair", options.repair])
            else:
                command.extend(["--corpus", str(options.corpus)])
            result = subprocess.check_output(command, env=env, text=True, timeout=365)
            sample = json.loads(result)
            if options.repair and name == "candidate" and sample["deadline"]:
                raise ValueError(f"The candidate exceeded the deadline for {options.repair}")
            trials[name].append(sample)
    report = dict(cpu=options.cpu, kernel=platform.release(), clock="native ABI wall ms",
                  reference_linear_algebra=options.reference_switch)
    if options.repair:
        report.update(repair=options.repair,
                      conditions="One captured Newton repair in each fresh process. Alternating solver order. "
                                 "The 15 s reference deadline is censored.", trials=trials,
                      median_ms={name: None if any(t["deadline"] for t in values) else median(t["ms"] for t in values)
                                 for name, values in trials.items()})
    else:
        report.update(corpus=options.corpus.name,
                      conditions="One fresh process per mode and trial. Calls in archive order. "
                                 "Alternating solver order. Native deadlines are stored in the requests.",
                      trials_ms=trials, median_ms={k: list(map(median, zip(*v))) for k, v in trials.items()})
    text = json.dumps(report, indent=2)
    if options.output:
        options.output.write_text(text + "\n")
    print(json.dumps(report["median_ms"]))


if __name__ == "__main__":
    main()
