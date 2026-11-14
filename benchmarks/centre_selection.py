"""Compare complete canonical resolutions from two built Atlas checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from plane_conjugate_resolution import measure


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("(x+y)^2+z^3+x^3", "embedded"),
    ("(x*y-z^2)^2+x^5+y^5", "embedded"),
    ("x^2+(y+z)^3+y^4", "embedded"),
    ("(x+y)^2+z^3+x^3", "intrinsic"),
    ("x^2+(y+z)^3+y^4", "intrinsic"),
)
TIMING_FIELDS = {"elapsed_ms", "native_elapsed_ms", "rank_elapsed_ms", "lift_ms", "verify_ms"}


def without_timings(value):
    if isinstance(value, dict):
        return {k: without_timings(v) for k, v in value.items() if k not in TIMING_FIELDS}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


def compare(command, roots, trials, case_index):
    reference = None
    samples = []
    for trial in range(trials):
        order = list(roots)
        if (case_index + trial) % 2:
            order.reverse()
        times = {}
        for version in order:
            timing, output = measure(command, roots[version], timeout=None)
            times[version] = timing["wall_s"]
            record = without_timings(json.loads(output))
            if (record["engine"] != "canonical-blowups" or record["status"] != "resolved"
                    or record["dual_complex"]["status"] != "complete"):
                raise ValueError(f"{version}: expected a complete canonical resolution for {command}")
            if reference is None:
                reference = record
            elif record != reference:
                fields = [k for k in record.keys() | reference.keys() if record.get(k) != reference.get(k)]
                raise ValueError(f"{version}: differing record fields {sorted(fields)} for {command}")
        samples.append({"order": order, "wall_s": times})
    agreed = json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
    return {
        "command": command, "samples": samples,
        "median_wall_s": {v: median(s["wall_s"][v] for s in samples) for v in roots},
        "median_paired_speedup": median(s["wall_s"]["baseline"] / s["wall_s"]["candidate"] for s in samples),
        "agreed_record_sha256": hashlib.sha256(agreed).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="built checkout before the change")
    parser.add_argument("--candidate", default=ROOT, type=Path, help="built checkout with the change")
    parser.add_argument("--trials", type=int, default=5, help="pairs per input (default 5)")
    parser.add_argument("--quick", action="store_true", help="use the public bounded search")
    parser.add_argument("--output", required=True, type=Path, help="destination for the JSON report")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    policy = "--quick" if args.quick else "--exhaustive"
    report = {
        "checkouts": {v: str(root) for v, root in roots.items()}, "trials": args.trials,
        "scope": "Fresh public CLI wall time including startup and JSON output. Alternating execution order.",
        "ignored_timing_fields": sorted(TIMING_FIELDS), "cases": [],
    }
    for i, (polynomial, mode) in enumerate(CASES):
        command = ["./atlas", "resolve", polynomial, "--vars", "x,y,z", "--mode", mode, policy, "--json"]
        report["cases"].append(compare(command, roots, args.trials, i))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
