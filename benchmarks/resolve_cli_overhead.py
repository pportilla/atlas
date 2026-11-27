"""Measure fresh resolve CLI processes with and without a preloaded engine."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

from centre_selection import without_timings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import fork_server

CASES = {"odp": "x^2+y^2+z^2", "inclined_pinch": "(x+y+z)^2-z*(x-y)^2", "pinch": "x^2-z*y^2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    os.sched_setaffinity(0, {args.cpu})
    report = {"cpu": args.cpu, "python": sys.version.split()[0], "cases": {}}
    with fork_server.serving(prepare_now=True):
        deadline = time.monotonic() + 30
        while not fork_server.available():
            if time.monotonic() >= deadline:
                raise RuntimeError("The fork server did not finish its preload.")
            time.sleep(.01)
        for index, (name, equation) in enumerate(CASES.items()):
            command = [str(ROOT / "atlas"), "resolve", equation, "--vars", "x,y,z", "--compute", "cpu", "--json"]
            samples = {"standalone": [], "preloaded": []}
            digest = None
            for trial in range(args.trials):
                order = list(samples)
                if (index + trial) % 2:
                    order.reverse()
                for mode in order:
                    env = dict(os.environ, ATLAS_PRELOADED_CLI="trace" if mode == "preloaded" else "0")
                    started = time.perf_counter()
                    proc = subprocess.run(command, env=env, capture_output=True, check=True, timeout=120)
                    wall = (time.perf_counter() - started) * 1000
                    if mode == "preloaded" and proc.stderr.strip() != b"Atlas CLI worker: preloaded":
                        raise RuntimeError(f"Preloaded worker unavailable for {name}: {proc.stderr!r}")
                    record = json.loads(proc.stdout)
                    if record["status"] != "resolved" or record["dual_complex"]["status"] != "complete":
                        raise ValueError(f"Incomplete resolution of {name}")
                    current = sha256(json.dumps(without_timings(record), separators=(",", ":")).encode()).hexdigest()
                    if digest is not None and digest != current:
                        raise ValueError(f"Different record for {name} in {mode} trial {trial + 1}")
                    digest = current
                    computation = record["elapsed_ms"]
                    samples[mode].append(dict(wall_ms=round(wall, 3), compute_ms=computation,
                                              fixed_ms=round(wall - computation, 3)))
            report["cases"][name] = dict(equation=equation, record_sha256=digest, trials=samples,
                                         median_ms={k: {metric: median(row[metric] for row in rows)
                                                        for metric in rows[0]} for k, rows in samples.items()})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
