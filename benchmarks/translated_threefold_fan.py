"""Compare exact chart and zeta computations for a hidden square and its diagonal model."""
import argparse
import json
from pathlib import Path
import subprocess

from zoo_resolution import ROOT, compare, run


NAMES = ["x", "y", "z", "t"]
CASES = {
    "diagonal-2347": "x^2+y^3+z^4+t^7",
    "shifted-2347": "(x-y*z-t^2)^2+y^3+z^4+t^7",
}


def reference(ident, polynomial):
    record = dict(id=ident, equation=polynomial, variables=NAMES, dimension=3, labels=[],
                  invariants={key: dict(value=36, finite=True) for key in ("milnor_number", "tjurina_number")})
    for mode in ("embedded", "intrinsic"):
        command = ["./atlas", "resolve", polynomial, "--vars", ",".join(NAMES), "--mode", mode,
                   "--quick", "--compute", "cpu", "--json"]
        if mode == "embedded":
            command.append("--monodromy")
        timing, output = run(command, 30)
        if timing["status"] != "complete":
            raise RuntimeError(f"{ident} lacks a completed {mode} reference")
        actual = json.loads(output)
        record[mode] = {"dual_complex": actual["dual_complex"]}
        if mode == "embedded":
            record["zeta"] = {k: actual[k + "_zeta"] for k in ("monodromy", "topological")}
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 2 or args.timeout <= 0:
        parser.error("Use at least two trials and a positive timeout.")
    records = [reference(ident, eq) for ident, eq in CASES.items()]
    for kind in ("monodromy", "topological"):
        if records[0]["zeta"][kind]["expression"] != records[1]["zeta"][kind]["expression"]:
            raise RuntimeError(f"The {kind} zeta differs under the polynomial automorphism")
    result = dict(conditions="QQ at the origin. Fresh CPU processes including startup and serialization. "
                  "Atlas embedded trials produce verified charts, incidence and both zetas. Intrinsic trials "
                  "include incidence. Singular resolve.lib checks the embedded resolution and retains all "
                  "charts, with independent Jacobian and Tjurina lengths and Milnor-fibre Euler agreement.",
                  singular=subprocess.check_output(["Singular", "--version"], text=True, cwd=ROOT).splitlines()[0],
                  trials=args.trials, timeout_s=args.timeout, cases=[])
    for record in records:
        summary = compare(record, args.trials, args.timeout)
        result["cases"].append(summary)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(record["id"], summary["median_wall_s"], flush=True)
