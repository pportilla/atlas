"""Screen each distinct canonical Zoo equation once in Singular, then time its Atlas modes.

Example:
    python3 benchmarks/screen_fallback_resolution.py --zoo ../resolution-zoo \
        --cpus 4 6 --output build/fallback-screen
    python3 benchmarks/screen_fallback_resolution.py --zoo ../resolution-zoo \
        --cpus 4 6 --confirm --prior benchmarks/fallback-resolution-ryzen-6800h.json.gz \
        --output build/fallback-confirmations

Each worker uses one logical CPU. A fresh Singular process has 1.75 seconds of
wall time, including startup. Each corresponding fresh Atlas CLI has 12 seconds.
These single observations rank cases for fallback_resolution.py. They are never
included in its five-trial medians. --prior can reuse earlier raw screening
observations and completed comparison medians from its compressed report.
With the checked-in report it reuses the original 88 equation observations
and reruns the 318 new probes. Use the same CPU list to resume an output directory.
With --confirm, choose a separate output directory for one checkpoint per mode.
Completed comparison cases and explicitly recorded failures in --prior are reused;
remaining modes with completed Singular diagnostics receive the full trials.
If --prior already contains a confirmation batch, its original requested mode
list is replayed instead. Existing files in the output directory are reused.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace
import subprocess
import sys

from fallback_resolution import ROOT, cases, run, run_case, singular_answer, summarize
from resolution_zeta import atlas_fingerprint


def workloads(zoo, prior):
    measured = prior.get("comparison", {})
    probes = prior.get("singular_short_screen", {})
    # A completed screen records the exact previous medians supplied to
    # each job. Later repeated trials must not change its diagnostic inputs.
    archived = {(row["equation"], tuple(row["variables"])): row for row in prior.get("whole_zoo_screen", [])}
    if archived:
        measured = {name: {"median_ms": {"atlas_fresh": row}} for sample in archived.values()
                    for name, row in sample["modes"].items() if row["source"] == "previous full comparison"}
    grouped = defaultdict(list)
    for name, (equation, names, mode) in cases(zoo).items():
        if name.startswith("zoo/"):
            grouped[(equation, tuple(names))].append((name, mode))
    jobs = []
    for (equation, names), modes in grouped.items():
        old = archived.get((equation, names))
        if old is not None:
            ss = old["singular"]
            jobs.append(dict(equation=equation, variables=names, modes=modes,
                             reference=None if ss["source"] == "fresh screen" else ss))
            continue
        reference = None
        for name, _ in modes:
            if name in measured:
                row = measured[name]
                med = row["median_ms"].get("singular_fresh")
                sample = {"compute_ms": med["compute_ms"]} if med else row["samples"]["singular_fresh"][-1]
                reference = dict(sample, source="previous full comparison", name=name)
                break
            if reference is None and name in probes and "error" not in probes[name]:
                reference = dict(probes[name], source="previous short screen", name=name)
        jobs.append(dict(equation=equation, variables=names, modes=modes, reference=reference))
    return jobs, measured


def evaluate(job, root, singular, timeout, atlas_timeout, measured):
    eq, axes = job["equation"], job["variables"]
    reference = job["reference"]
    if reference is None:
        source = ('LIB "resolve.lib";system("--ticks-per-sec",1000000);'
                  f'ring R=0,({",".join(axes)}),dp;poly f={eq};'
                  'int t=rtimer;list rr=resolve(ideal(f));int elapsed=rtimer-t;'
                  'print("ATLASBENCH "+string(elapsed)+" "+string(size(rr[1])));exit;')
        try:
            wall, out = run([singular, "-q", "--no-rc"], source, root, timeout)
            answers = [v.removeprefix("ATLASBENCH ") for v in out.splitlines() if v.startswith("ATLASBENCH ")]
            if len(answers) != 1 or any(line.lstrip().startswith("?") for line in out.splitlines()):
                error = next((v.strip() for v in out.splitlines() if v.lstrip().startswith("?")), "missing chart output")
                raise ValueError(f"Singular did not return one resolved chart count: {error}")
            answer, = answers
            reference = dict(singular_answer(wall, answer), source="fresh screen")
        except subprocess.TimeoutExpired:
            reference = dict(timeout_s=timeout, source="fresh screen")
        except (ValueError, subprocess.CalledProcessError) as error:
            reference = dict(error=str(error), source="fresh screen")
    result = dict(equation=eq, variables=axes, singular=reference, modes={})
    for name, mode in job["modes"]:
        med = measured.get(name)
        if med and "atlas_fresh" in med["median_ms"]:
            row = dict(med["median_ms"]["atlas_fresh"], source="previous full comparison")
        elif "compute_ms" not in reference:
            row = dict(source="not timed", reason="no completed Singular timing")
        else:
            try:
                wall, out = run([root / "atlas", "resolve", eq, "--vars", ",".join(axes), "--compute", "cpu",
                                 "--mode", mode, "--json"], None, root, atlas_timeout)
                row = dict(summarize(json.loads(out), wall), source="fresh screen")
            except subprocess.TimeoutExpired:
                row = dict(timeout_s=atlas_timeout, source="fresh screen")
            except (ValueError, KeyError, subprocess.CalledProcessError) as error:
                row = dict(error=str(error), source="fresh screen")
        result["modes"][name] = row
    return result


def worker(cpu, jobs, args, measured):
    sys.path.insert(0, str(ROOT))
    os.sched_setaffinity(0, {cpu})
    path = args.output / f"cpu-{cpu}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["equation"], tuple(row["variables"])))
    with path.open("a") as stream:
        for job in jobs:
            if (job["equation"], tuple(job["variables"])) in done:
                continue
            row = evaluate(job, args.atlas_root, args.singular, args.timeout, args.atlas_timeout, measured)
            row["cpu"] = cpu
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()


def confirmation_worker(cpu, jobs, args, metadata):
    sys.path.insert(0, str(ROOT))
    os.sched_setaffinity(0, {cpu})
    options = SimpleNamespace(**{**vars(args), "timeout": args.confirm_timeout, "replay": False})
    for name, case, path in jobs:
        if path.exists():
            continue
        report = dict(metadata, cpu=cpu, cases={})
        try:
            report["cases"][name] = run_case(name, case, options, os.environ.copy())
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, TimeoutError) as e:
            report["failure"] = {"name": name, "type": type(e).__name__, "message": str(e)}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, separators=(",", ":")))
        tmp.replace(path)


def confirm(prior, args):
    if not prior.get("whole_zoo_screen"):
        raise ValueError("--confirm requires a prior report containing the completed diagnostic screen.")
    if os.environ.get("ATLAS_RECORD_REQUESTS"):
        raise ValueError("Disable native request recording before timing.")
    replayed = prior.get("confirmation_batch")
    if replayed is None:
        wanted = None
        existing = set(prior.get("comparison", {}))
        existing.update(row["name"] for row in prior.get("five_trial_errors", []))
    else:
        # Reproduce the recorded wide batch even after some of its inputs
        # have received additional measurements on a newer implementation.
        wanted = set(replayed["requested_modes"])
        existing = set()
    jobs = []
    for row in prior["whole_zoo_screen"]:
        if "compute_ms" not in row["singular"]:
            continue
        for name, atlas in row["modes"].items():
            path = args.output / (name.replace("/", "__") + ".json")
            if name in existing or (wanted is not None and name not in wanted) or path.exists():
                continue
            case = row["equation"], row["variables"], name.rsplit("/", 1)[1]
            estimate = atlas.get("compute_ms", args.atlas_timeout * 1000) + row["singular"]["compute_ms"]
            jobs.append((estimate, name, case, path))
    # Distribute the predicted long jobs first. Predictions only assign
    # the CPU. Every requested case keeps its full independent deadline.
    groups = {cpu: [] for cpu in args.cpus}
    loads = dict.fromkeys(args.cpus, 0)
    for cost, name, case, path in sorted(jobs, reverse=True):
        cpu = min(args.cpus, key=loads.get)
        groups[cpu].append((name, case, path))
        loads[cpu] += cost
    metadata = dict(python=platform.python_version(), kernel=platform.release(), processor=platform.processor(),
                    atlas_sha256=atlas_fingerprint(args.atlas_root), trials=args.trials, timeout_s=args.confirm_timeout,
                    singular=subprocess.check_output([args.singular, "--version"], text=True).splitlines()[0],
                    conditions="Alternating fresh CLIs and alternating warm in-process solves after an untimed warmup. "
                               "Each process inherits one pinned CPU. A timed-out series has no complete median.")
    print(f"Confirming {len(jobs)} remaining mode/input pairs on CPUs {args.cpus}", flush=True)
    with ProcessPoolExecutor(max_workers=len(args.cpus)) as pool:
        futures = [pool.submit(confirmation_worker, cpu, assigned, args, metadata) for cpu, assigned in groups.items() if assigned]
        for future in futures:
            future.result()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo", type=Path, required=True)
    p.add_argument("--cpus", type=int, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior", type=Path)
    p.add_argument("--atlas-root", type=Path, default=ROOT)
    p.add_argument("--singular", default="Singular")
    p.add_argument("--timeout", type=float, default=1.75)
    p.add_argument("--atlas-timeout", type=float, default=12)
    p.add_argument("--confirm", action="store_true", help="Run full trials on reachable modes still missing medians")
    p.add_argument("--trials", type=int, default=5, help="Trials per solver and phase in confirmation mode")
    p.add_argument("--confirm-timeout", type=float, default=120, help="Wall seconds per full trial")
    args = p.parse_args()
    if (len(args.cpus) != len(set(args.cpus)) or not set(args.cpus) <= os.sched_getaffinity(0)
            or args.trials < 1 or min(args.timeout, args.atlas_timeout, args.confirm_timeout) <= 0):
        p.error("The CPUs must be distinct and available and the timeouts positive.")
    args.atlas_root = args.atlas_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    prior = {}
    if args.prior:
        opener = gzip.open if args.prior.suffix == ".gz" else open
        with opener(args.prior, "rt") as stream:
            prior = json.load(stream)
    if args.confirm:
        confirm(prior, args)
        return
    jobs, measured = workloads(args.zoo, prior)
    print(f"{len(jobs)} distinct equations, {sum(len(x['modes']) for x in jobs)} modes, "
          f"{sum(x['reference'] is None for x in jobs)} new Singular observations", flush=True)
    with ProcessPoolExecutor(max_workers=len(args.cpus)) as pool:
        futures = [pool.submit(worker, cpu, jobs[i::len(args.cpus)], args, measured)
                   for i,cpu in enumerate(args.cpus)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
