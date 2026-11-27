"""Time complete S15 invariant requests in fresh processes and check the exact native replies."""
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


def worker(root):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    kernel = ctypes.CDLL(str(root / "native/build" / ("libgroebner" + suffix)))
    maximum = kernel.atlas_centre_maximum
    maximum.argtypes = [ctypes.c_char_p]
    maximum.restype = ctypes.c_char_p
    times = []
    with gzip.open(FIXTURE, "rt") as stream:
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
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        print(json.dumps(worker(options.worker)))
        return
    if options.trials < 1:
        parser.error("at least one trial is required")
    os.sched_setaffinity(0, {options.cpu})
    trials = {"reference": [], "candidate": []}
    roots = {"reference": options.reference.resolve(), "candidate": ROOT}
    for i in range(options.trials):
        for name in list(roots)[::1 if i % 2 == 0 else -1]:
            env = dict(os.environ, ATLAS_LINEAR_ALGEBRA="0" if name == "reference" else "1")
            env.pop("ATLAS_RECORD_REQUESTS", None)
            result = subprocess.check_output(
                [sys.executable, __file__, "--cpu", str(options.cpu), "--worker", roots[name]],
                env=env, text=True, timeout=365)
            trials[name].append(json.loads(result))
    report = dict(cpu=options.cpu, kernel=platform.release(), clock="native ABI wall ms",
                  conditions="One fresh process per mode and trial. Three calls in original order. "
                             "Alternating solver order. Each call has a 120 s native deadline.",
                  trials_ms=trials, median_ms={k: list(map(median, zip(*v))) for k, v in trials.items()})
    text = json.dumps(report, indent=2)
    if options.output:
        options.output.write_text(text + "\n")
    print(json.dumps(report["median_ms"]))


if __name__ == "__main__":
    main()
