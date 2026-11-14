"""Time fresh Atlas and Singular processes computing extended left-right codimension.

From the repository root, run `python3 benchmarks/ae_codimension.py --cpu 13 --trials 7`.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from statistics import median
import subprocess
from time import perf_counter_ns


ROOT = Path(__file__).resolve().parents[1]
# Match the existing single-threaded Singular comparison policy.
SINGULAR = ("Singular", "-q", "--no-rc", "--threads=1", "--flint-threads=1", "--random=47")
U, V = "(x+y+x*y+y^2)", "(y+y^2)"
# The last field is the expected dimension; Singular uses -1 for infinity.
CASES = (
    ("immersion", "x,y", ("x+y^2", "y", "x^3"), 0),
    ("univariate", "t", ("t^12+t^17",), 10),
    ("scalar-separated", "x,y", ("x^4+y^5",), 11),
    ("scalar-mixed", "x,y", ("x^4+y^5+x^2*y^3",), 10),
    ("monomial-curve", "t", ("t^3", "t^5", "t^7"), 7),
    ("crosscap", "x,y", ("x", "y^2", "x*y"), 0),
    ("higher-order-double-cover", "x,y", ("x", "y^2", "y^3+x^12*y"), 11),
    ("plane-graded", "x,y", ("x", "x*y+y^5"), 3),
    ("plane-coordinate-change", "x,y", ("x+y^2", "(x+y^2)*y+y^5"), 3),
    ("space-graded", "x,y,z", ("x", "y", "z^5+x*z+y*z^2"), 1),
    ("positive-dimensional-contact", "x,y", ("x^2+x^3*y", "x*y+x*y^4"), -1),
    ("dense-plane-coordinate-change", "x,y", (U, f"{U}*{V}+{V}^7"), 10),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(name, variables, components, expected, trials, timeout, root):
    command = [str(root / "atlas"), "algebra", "ae-codimension",
               "--vars", variables, "--compute", "cpu", "--json"]
    for component in components:
        command.extend(("--component", component))
    # The no-extra-argument call asks for the extended tangent space. Leave
    # Singular's local standard-basis reduction settings at their defaults.
    script = f'''LIB "classifyMapGerms.lib";
ring r=0,({variables}),(c,ds);
ideal f={",".join(components)};
list d=coDimMap(f);
print("AE "+string(d[2]));
quit;
'''
    samples = {"atlas": [], "singular": []}
    method = None
    # One discarded startup/execution of each program warms filesystem caches.
    for trial in range(trials + 1):
        for impl in (("atlas", "singular") if trial % 2 else ("singular", "atlas")):
            start = perf_counter_ns()
            done = subprocess.run(command if impl == "atlas" else SINGULAR, cwd=root,
                                  input=None if impl == "atlas" else script, text=True,
                                  capture_output=True, check=True, timeout=timeout)
            ms = (perf_counter_ns() - start) / 1e6
            if impl == "atlas":
                rec = json.loads(done.stdout)
                result = rec["result"]
                if rec["status"] != "ok" or result["finite"] is None or result["finite"] != (expected >= 0):
                    raise ValueError(f"{name}: invalid certification {result}")
                if result["value"] != (expected if expected >= 0 else None):
                    raise ValueError(f"{name}: wrong Atlas value {result}")
                if method is not None and result["method"] != method:
                    raise ValueError(f"{name}: changed solver method")
                method = result["method"]
            else:
                m = re.fullmatch(r"AE (-?\d+)", done.stdout.strip())
                if m is None or int(m[1]) != expected or done.stderr:
                    raise ValueError(f"{name}: Singular reported {done.stdout!r} {done.stderr!r}")
            if trial:
                samples[impl].append(ms)
    times = {k: median(v) for k, v in samples.items()}
    return {"input": {"variables": variables.split(","), "components": components},
            "finite": expected >= 0, "value": None if expected < 0 else expected, "method": method,
            "fresh_wall_ms": samples, "median_ms": times,
            "singular_over_atlas": times["singular"] / times["atlas"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=ROOT)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 2 or args.timeout <= 0:
        parser.error("use at least two trials and a positive timeout")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    root = args.atlas_root.resolve()
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").is_file() else ""
    model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "cpu": model[1] if model else platform.processor(),
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "atlas_native_sha256": sha256(root / "native/build/libgroebner.so"),
              "atlas_cli_binary_sha256": sha256(root / "native/build/local_algebra"),
              "singular_version": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "trials": args.trials, "timeout_seconds": args.timeout,
              "conditions": "Fresh processes, one discarded run of each solver, alternating measured order, "
                            "CPU, QQ, single germ at the origin. Atlas prints the complete CLI JSON. "
                            "Singular loads classifyMapGerms.lib, uses (c,ds), calls coDimMap(f), "
                            "and prints entry 2. All runs check finite status and codimension.",
              "cases": {}}
    for name, variables, components, expected in CASES:
        r = run_case(name, variables, components, expected, args.trials, args.timeout, root)
        report["cases"][name] = r
        print(f"{name}: {r['median_ms']['atlas']:.3f} vs {r['median_ms']['singular']:.3f} ms "
              f"({r['singular_over_atlas']:.2f}x)", flush=True)
    if (sha256(root / "native/build/libgroebner.so") != report["atlas_native_sha256"]
            or sha256(root / "native/build/local_algebra") != report["atlas_cli_binary_sha256"]):
        raise RuntimeError("Atlas binaries changed during the comparison. Rerun against one fixed build.")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
