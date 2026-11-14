"""Time three implementations of the extended tangent quotient on three line powers."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter_ns

from ae_codimension import ROOT, SINGULAR, sha256


CASES = (
    ((1, 2, 3), 1), ((1, 3, 5), 20), ((2, 3, 5), 115), ((2, 3, 7), 242),
    ((3, 4, 5), 529), ((2, 5, 7), 729), ((3, 5, 7), 1710),
)
MODULE = Path(__file__).with_name("ae_coprime_module.sing")


def measure(root, degrees, expected, trials, timeout, direct_timeout):
    a, b, c = degrees
    components = [f"x^{a}", f"y^{b}", f"(x+y)^{c}"]
    cli = [str(root / "atlas"), "algebra", "ae-codimension", "--vars", "x,y", "--compute", "cpu", "--json"]
    for f in components:
        cli.extend(("--component", f))
    direct = (f'LIB "classifyMapGerms.lib"; ring r=0,(x,y),(c,ds); ideal f={",".join(components)}; '
              'list d=coDimMap(f); print("AE "+string(d[2])); quit;\n')
    module = f"ring R=0,(X,Y),dp; int a={a}; int b={b}; int c={c};\n" + MODULE.read_text()
    commands = {"atlas": (cli, None, timeout), "singular_codimmap": (SINGULAR, direct, direct_timeout),
                "singular_module": (SINGULAR, module, timeout)}
    results = {k: {"samples_ms": []} for k in commands}
    for trial in range(trials+1):
        for name in (commands if trial%2 else reversed(commands)):
            samples = results[name]
            if "timeout_s" in samples:
                continue
            command, source, limit = commands[name]
            start = perf_counter_ns()
            try:
                run = subprocess.run(command, input=source, text=True, capture_output=True,
                                     timeout=limit, cwd=root, check=True)
            except subprocess.TimeoutExpired:
                samples.update(timeout_s=limit, timed_out_during="warmup" if trial == 0 else f"trial {trial}")
                continue
            wall_ms = (perf_counter_ns()-start)/1e6
            if name == "atlas":
                result = json.loads(run.stdout)["result"]
                if result["value"] != expected or result["finite"] is not True:
                    raise AssertionError(result)
                if "method" in samples and samples["method"] != result["method"]:
                    raise AssertionError("Native solver method changed between trials.")
                samples["method"] = result["method"]
            elif run.stdout.strip() != f"AE {expected}" or run.stderr:
                raise AssertionError((degrees, name, run.stdout, run.stderr))
            if trial:
                samples["samples_ms"].append(wall_ms)
        if all("timeout_s" in s for s in results.values()):
            raise RuntimeError(f"No computation completed for {degrees}")
    for s in results.values():
        if len(s["samples_ms"]) == trials:
            s["median_ms"] = median(s["samples_ms"])
    return {"degrees": degrees, "value": expected, "timings": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas-root", type=Path, default=ROOT)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--timeout", type=float, default=15, help="seconds per process")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.trials < 2 or args.timeout <= 0:
        p.error("Use at least two trials and a positive timeout.")
    # This Singular build rebinds itself to logical CPU 0 in single-thread mode.
    os.sched_setaffinity(0, {0})
    root = args.atlas_root.resolve()
    binaries = [root/"native/build/libgroebner.so", root/"native/build/local_algebra"]
    fingerprints = {p.name: sha256(p) for p in binaries}
    report = {"cpu": next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), "cpu_affinity": [0], "trials": args.trials,
              "conditions": "QQ at the origin. Fresh processes, compulsory single-threaded Singular. "
                            "One discarded warmup per solver, then alternating order. A timeout omits the median "
                            "for that solver and triple. The coDimMap limits are at least 45 s for (2,3,5) "
                            "and 90 s for (2,3,7).",
              "singular": subprocess.check_output(["Singular","--version"], text=True).splitlines()[0],
              "binaries_sha256": fingerprints, "cases": []}
    for degrees, expected in CASES:
        direct_timeout = args.timeout
        if degrees == (2, 3, 5):
            direct_timeout = max(direct_timeout, 45)
        if degrees == (2, 3, 7):
            direct_timeout = max(direct_timeout, 90)
        row = measure(root, degrees, expected, args.trials, args.timeout, direct_timeout)
        report["cases"].append(row)
        print(degrees, expected, {k:v.get("median_ms", "timeout") for k,v in row["timings"].items()}, flush=True)
    if {p.name: sha256(p) for p in binaries} != fingerprints:
        raise RuntimeError("Atlas binaries changed during timing.")
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
