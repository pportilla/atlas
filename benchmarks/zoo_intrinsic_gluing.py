"""Time intrinsic incidence on the same canonical double-quadric charts.

python3 benchmarks/zoo_intrinsic_gluing.py --trials 3 --output report.json
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.zoo_data import catalogue_file
from engine.general_atlas import native_general_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("Use at least two trials.")
    case = json.loads(catalogue_file("records/double-quadric-contact.json").read_text())
    start = perf_counter()
    record = json.loads(subprocess.check_output([
        ROOT / "atlas", "resolve", case["equation"], "--vars", ",".join(case["variables"]),
        "--mode", "intrinsic", "--compute", "cpu", "--exhaustive", "--json"], cwd=ROOT))
    cli_seconds = perf_counter() - start
    expected = case["intrinsic"]["dual_complex"]
    if record["status"] != "resolved" or record["dual_complex"] != expected:
        raise AssertionError("The live intrinsic solve differs from the catalogue.")
    times = {"all_pairs": [], "global_divisor_labels": []}
    for trial in range(args.trials):
        for use_labels in ((False, True) if trial % 2 == 0 else (True, False)):
            start = perf_counter()
            proof = native_general_outputs(record, record["atlas"], intrinsic=True, integrals=False,
                                           global_boundary_labels=use_labels)
            elapsed = perf_counter() - start
            if proof["dual_complex"] != expected:
                raise AssertionError("Mismatch in the complete intrinsic complex.")
            times["global_divisor_labels" if use_labels else "all_pairs"].append(elapsed)
    cpuinfo = Path("/proc/cpuinfo")
    cpu_lines = cpuinfo.read_text().splitlines() if cpuinfo.exists() else ()
    cpu = next((line.partition(":")[2].strip() for line in cpu_lines if line.startswith("model name")), platform.machine())
    report = {"equation": case["equation"], "variables": case["variables"], "mode": "intrinsic", "compute": "cpu",
              "conditions": "One complete fresh CLI solve followed by alternating verification on its fixed "
                            "serialized charts within one Python process. Each trial checks the full labelled "
                            "complex against the published Zoo record. Other system load is uncontrolled.",
              "charts": len(record["atlas"]), "steps": len(record["steps"]), "vertices": len(expected["vertices"]),
              "cells": len(expected["cells"]), "cpu": cpu,
              "trials": args.trials, "cli_wall_seconds": cli_seconds, "verification_seconds": times,
              "complex_sha256": sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
