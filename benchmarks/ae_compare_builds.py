"""Compare extended left-right codimension records from two native builds.

Record the inputs that the AE test modules send to the solver, together with
the benchmark maps and pairwise-coprime line powers:

    python3 benchmarks/ae_compare_builds.py record --output /tmp/ae-inputs.json

Then run every input through both builds' `local_algebra` command and compare
the complete JSON records, without the elapsed-time fields:

    python3 benchmarks/ae_compare_builds.py compare /path/to/old/native/build native/build /tmp/ae-inputs.json

A pair whose finiteness and value agree but whose proof fields differ, such
as `method` or `last_weight`, is reported separately from a disagreement.
An input that times out in either build is reported and not compared.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from math import gcd
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = ("tests.test_ae_graded", "tests.test_ae_special", "tests.test_ae_literature",
                "tests.test_ae_codimension_interface", "tests.test_ae_coordinate_changes",
                "tests.test_ae_coprime_reference")


def record(output, triples):
    sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
    from ae_codimension import CASES
    import engine.local_algebra as local_algebra

    seen = []
    handle = local_algebra.handle

    def recording(operation, ring, operands, **options):
        if operation == "ae-codimension":
            seen.append({"variables": list(ring["variables"]), "order": ring["order"],
                         "components": list(operands["components"])})
        return handle(operation, ring, operands, **options)

    local_algebra.handle = recording
    try:
        suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
        if not unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful():
            raise SystemExit("The AE test modules failed; no inputs were recorded.")
    finally:
        local_algebra.handle = handle
    for _, variables, components, _ in CASES:
        seen.append({"variables": variables.split(","), "order": "grevlex", "components": list(components)})
    # Pairwise-coprime 2 <= a < b < c, smallest a+b+c first.
    powers = sorted((sum(t), t) for t in combinations(range(2, 16), 3)
                    if all(gcd(u, v) == 1 for u, v in combinations(t, 2)))
    for _, (a, b, c) in powers[:triples]:
        seen.append({"variables": ["x", "y"], "order": "grevlex",
                     "components": [f"x^{a}", f"y^{b}", f"(x+y)^{c}"]})
    unique = {json.dumps(item, sort_keys=True): item for item in seen}
    Path(output).write_text(json.dumps(list(unique.values()), indent=1) + "\n")
    print(f"{len(seen)} requests, {len(unique)} distinct inputs", file=sys.stderr)


def run(build, item, timeout):
    command = [str(Path(build) / "local_algebra"), "./atlas", "algebra", "ae-codimension",
               "--vars", ",".join(item["variables"]), "--order", item["order"], "--compute", "cpu", "--json"]
    for component in item["components"]:
        command += ["--component", component]
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"timeout": timeout}
    except (UnicodeEncodeError, ValueError) as error:  # inputs that only the Python interfaces accept
        return {"unrunnable": str(error)}
    try:
        output = json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"exit": done.returncode, "stdout": done.stdout, "stderr": done.stderr}
    execution = output.get("execution")
    if isinstance(execution, dict):
        execution.pop("elapsed_ms", None)
        for stage in execution.get("stages", []):
            stage.pop("elapsed_ns", None)
    return output


def compare(old, new, inputs, timeout, jobs):
    items = json.loads(Path(inputs).read_text())

    def both(item):
        start = time.perf_counter()
        before = run(old, item, timeout)
        middle = time.perf_counter()
        after = run(new, item, timeout)
        return item, before, after, middle - start, time.perf_counter() - middle

    counts = dict(identical=0, same_value_other_proof=0, different=0, timeouts=0)
    with ThreadPoolExecutor(jobs) as pool:
        for item, before, after, old_s, new_s in pool.map(both, items):
            label = json.dumps(item["components"])
            if "timeout" in before or "timeout" in after:
                counts["timeouts"] += 1
                print(f"TIMEOUT {label} old={'timeout' if 'timeout' in before else f'{old_s:.3f}s'}"
                      f" new={'timeout' if 'timeout' in after else f'{new_s:.3f}s'} {after.get('result', '')}")
            elif before == after:
                counts["identical"] += 1
            elif "result" in before and "result" in after and \
                    (before["result"].get("finite"), before["result"].get("value")) == \
                    (after["result"].get("finite"), after["result"].get("value")):
                counts["same_value_other_proof"] += 1
                print(f"SAME VALUE, OTHER PROOF {label}\n  old: {json.dumps(before['result'])}\n"
                      f"  new: {json.dumps(after['result'])}")
            else:
                counts["different"] += 1
                print(f"DIFFERENT {label}\n  old: {json.dumps(before)[:400]}\n  new: {json.dumps(after)[:400]}")
    print(" ".join(f"{k}={v}" for k, v in counts.items()), f"total={len(items)}")
    return 1 if counts["different"] else 0


def main():
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record")
    recorder.add_argument("--output", required=True)
    recorder.add_argument("--triples", type=int, default=60, help="pairwise-coprime line powers to add")
    comparer = commands.add_parser("compare")
    comparer.add_argument("old_build")
    comparer.add_argument("new_build")
    comparer.add_argument("inputs")
    comparer.add_argument("--timeout", type=float, default=30, help="seconds per process")
    comparer.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.command == "record":
        record(args.output, args.triples)
    else:
        raise SystemExit(compare(args.old_build, args.new_build, args.inputs, args.timeout, args.jobs))


if __name__ == "__main__":
    main()
