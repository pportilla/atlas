"""Compare fresh and warm Atlas CPU resolution against Singular resolve(ideal(f)).

Build Atlas in this checkout and in a separate checkout of the reference revision.
With $before pointing to the reference, e.g. a built git worktree, run

    python3 benchmarks/fallback_resolution.py --cpu 4 --output comparison.json
    python3 benchmarks/fallback_resolution.py --cpu 4 --paired-fresh "$before" --output paired.json
    python3 benchmarks/fallback_resolution.py --cpu 4 --record-corpus "$before" --replay --output records.json
    python3 tools/replay_requests.py benchmarks/canonical-requests.jsonl.gz

Add --zoo ../resolution-zoo to include all recorded fallback modes on its screens.
--cases selects names such as zoo/s2b-newton-17/embedded. Default runs alternate
five trials of each solver and stop a series at the first 120-second timeout.
JSON includes raw compute and wall samples, hashes, and independently verified
embedded canonical charts. For a saved input acquired during resolution, set
ATLAS_RECORD_REQUESTS to a directory and run the replay tool on that directory.
Disable recording during timing. Reported compute timings are Atlas CPU elapsed_ms and
Singular resolve.lib rtimer around resolve(ideal(f)) in characteristic zero.
The checked-in fallback-resolution-ryzen-6800h.json.gz records the Singular 4.4.1
runs, the reference revision, raw paired samples, checks and per-mode Zoo coverage.
Read it with gzip.open(..., "rt") and json.load().
"""

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
from statistics import median
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

from centre_selection import without_timings
from native_resolution_boundary import CASES as BOUNDARY, singular_chart_checks
from resolution_zeta import Session, atlas_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).with_name("fallback-resolution-baseline.csv")
MARKER = "ATLASBENCH "
SESSION_CODE = """
import json, sys, time
sys.path.insert(0,sys.argv[1])
from engine.resolve import resolve
print('ATLASBENCH ready', flush=True)
for line in sys.stdin:
    start = time.perf_counter()
    record = resolve(json.loads(line))
    wall = (time.perf_counter()-start)*1000
    print('ATLASBENCH '+json.dumps([wall, record], separators=(',',':')), flush=True)
"""


def cases(zoo):
    with SPEC.open() as stream:
        result = {r["name"]: (r["equation"], r["variables"].split(), "embedded")
                  for r in csv.DictReader(stream)}
    for name, (eq, axes, mode, *_) in BOUNDARY.items():
        if (eq, list(axes), mode) not in result.values():
            result["boundary_" + name] = eq, list(axes), mode
    # Three controls from benchmarks/resolve.py include plane branches and a chain.
    for name, eq, axes in (("plane_cusp", "y^2-x^3", "xy"),
                           ("curve_contact", "(y^2-2*x^4)^2-x^11", "xy"),
                           ("A20", "x*y-z^21", "xyz")):
        result[name] = eq, list(axes), "intrinsic"
    if zoo:
        ids = {c["id"] for path in (zoo / "screens").glob("*.json")
               for c in json.loads(path.read_text())["cases"]}
        if not ids:
            raise ValueError(f"No screen cases found in {zoo}")
        for ident in sorted(ids):
            path = zoo / "records" / (ident + ".json")
            if path.is_file():
                record = json.loads(path.read_text())
                for mode in ("embedded", "intrinsic"):
                    if record[mode]["engine"] == "canonical-blowups":
                        result[f"zoo/{ident}/{mode}"] = record["equation"], record["variables"], mode
    return result


def digest(value):
    return hashlib.sha256(json.dumps(without_timings(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def summarize(record, wall):
    if record["status"] != "resolved" or record["dual_complex"]["status"] != "complete":
        raise ValueError(f"Uncertified record: {record['status']}, {record['dual_complex']['status']}")
    from engine.resolution_progress import recorded_blowups
    return dict(compute_ms=record["elapsed_ms"], wall_ms=round(wall, 3), engine=record["engine"],
                charts=len(record["charts"]), blowups=recorded_blowups(record), record_sha256=digest(record))


def singular_answer(wall, answer):
    us, charts = map(int, answer.split())
    if charts <= 0:
        raise ValueError("Singular returned no final charts")
    return dict(compute_ms=us / 1000, wall_ms=round(wall, 3), charts=charts)


def run(command, payload, root, timeout):
    start = perf_counter()
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          cwd=root, text=True, start_new_session=True) as p:
        try:
            stdout, stderr = p.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            raise
        if p.returncode:
            raise subprocess.CalledProcessError(p.returncode, command, stdout, stderr)
    return 1000 * (perf_counter() - start), stdout


def replay_record(root, record, timeout):
    # Rebuild both outputs from the published charts, without a cached answer.
    fields = {"dual_complex", "monodromy_zeta", "topological_zeta", "monodromy_check"}
    data = json.dumps({k: v for k, v in record.items() if k not in fields})
    _, dual = run([root / "atlas", "recompute-dual-complex", "-"], data, root, timeout)
    code = ("from engine.resolution_zetas import replay_zetas\nimport json,sys\n"
            "print(json.dumps(replay_zetas(dict(record=json.load(sys.stdin),zetas=['monodromy','topological']))))")
    _, zetas = run([sys.executable, "-c", code], data, root, timeout)
    return digest(dict(dual=json.loads(dual), zetas=json.loads(zetas)))


def check_record_corpus(reference, candidate, args):
    report = {}
    with Path(__file__).with_name("canonical-record-corpus.csv").open() as f:
        for case in csv.DictReader(f):
            name = case["name"]
            command = ["resolve", case["equation"], "--vars", ",".join(case["variables"].split()),
                       "--mode", case["mode"], "--" + case["policy"], "--compute", "cpu", "--json"]
            before, after = (json.loads(run([tree / "atlas", *command], None, tree, args.timeout)[1])
                             for tree in (reference, candidate))
            if before["status"] != "resolved" or digest(before) != digest(after):
                raise ValueError(f"{name}: complete records differ or reference did not resolve")
            item = report[name] = dict(engine=after["engine"], record_sha256=digest(after))
            if args.replay:
                # Feed the *old* record to both builds, also checking backwards compatibility.
                old, new = (replay_record(root, before, args.timeout) for root in (reference, candidate))
                if old != new:
                    raise ValueError(f"{name}: saved-chart replays differ")
                item["saved_replay_sha256"] = new
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(name, item, flush=True)


def paired_fresh(reference, candidate, cases, args):
    """Interleave old and new fresh processes on the same CPU to measure regressions."""
    report = {}
    for name, (equation, variables, mode) in cases.items():
        command = ["resolve", equation, "--vars", ",".join(variables), "--mode", mode,
                   "--compute", "cpu", "--json"]
        samples = {"before": [], "after": []}
        agreed = None
        for trial in range(args.trials):
            builds = [("before", reference), ("after", candidate)]
            for label, build in (builds if trial % 2 == 0 else reversed(builds)):
                wall, output = run([build / "atlas", *command], None, build, args.timeout)
                row = summarize(json.loads(output), wall)
                if agreed is not None and row["record_sha256"] != agreed:
                    raise ValueError(f"{name}: complete record differs in {label} trial {trial}")
                agreed = row["record_sha256"]
                samples[label].append(row)
        medians = {key: median(r["compute_ms"] for r in value) for key, value in samples.items()}
        report[name] = dict(record_sha256=agreed, samples=samples, median_compute_ms=medians,
                            ratio_after_before=medians["after"] / medians["before"])
        print(name, medians, flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def run_case(name, case, args, env):
    eq, axes, mode = case
    request = dict(polynomial=eq, variables=axes, mode=mode, compute="cpu")
    host = args.atlas_root
    atlas_cli = [host / "atlas", "resolve", eq, "--vars", ",".join(axes), "--compute", "cpu",
                 "--mode", mode, "--json"]
    singular_base = ('LIB "resolve.lib";system("--ticks-per-sec",1000000);\n'
                     f'ring R=0,({",".join(axes)}),dp;poly f={eq};\n'
                     'proc trial(poly f){int t=rtimer;list rr=resolve(ideal(f));int elapsed=rtimer-t;'
                     'print("ATLASBENCH "+string(elapsed)+" "+string(size(rr[1])));}\n')
    atlas = SimpleNamespace(label="atlas", command=[sys.executable, "-I", "-u", "-c", SESSION_CODE, str(host)],
                            setup="", job=lambda _: json.dumps(request) + "\n")
    singular = SimpleNamespace(label="singular", command=[args.singular, "-q", "--no-rc"],
                               setup=singular_base + 'print("ATLASBENCH ready");\n', job=lambda _: "trial(f);\n")
    samples = {f"{a}_{b}": [] for a in ("atlas", "singular") for b in ("fresh", "warm")}
    solved = None
    warm_only = getattr(args, "atlas_warm_only", False)
    for phase in (("warm",) if warm_only else ("fresh", "warm")):
        with ExitStack() as stack:
            active = {}
            for trial in range(-1 if phase == "warm" else 0, args.trials):
                order = ("atlas", "singular") if trial % 2 == 0 else ("singular", "atlas")
                for tool in (("atlas",) if warm_only else order):
                    key = f"{tool}_{phase}"
                    if samples[key] and "timeout_s" in samples[key][-1]:
                        # A censored solve has no finite median. Leave each subsequent
                        # unmeasured trial visible, rather than imputing its duration.
                        continue
                    try:
                        if phase == "warm":
                            if tool not in active:
                                active[tool] = stack.enter_context(Session(
                                    atlas if tool == "atlas" else singular, env, args.timeout, isolated=True))
                            wall, answer = active[tool].warm(eq)
                            if tool == "atlas":
                                wall, record = json.loads(answer)
                        elif tool == "atlas":
                            wall, raw = run(atlas_cli, None, host, args.timeout)
                            record = json.loads(raw)
                        else:
                            wall, raw = run(singular.command, singular_base + "trial(f);quit;\n", host, args.timeout)
                            answers = [l[len(MARKER):] for l in raw.splitlines() if l.startswith(MARKER)]
                            if len(answers) != 1 or any(l.lstrip().startswith("?") for l in raw.splitlines()):
                                raise ValueError(f"Singular failed for {name}: {raw[-2000:]}")
                            answer = answers[0]
                        if tool == "atlas":
                            row = summarize(record, wall)
                            if solved is None:
                                solved = record
                            elif row["record_sha256"] != digest(solved):
                                raise ValueError(f"{name} complete records differ across {key} trials: "
                                                 f"{solved['engine']} {digest(solved)} and "
                                                 f"{record['engine']} {row['record_sha256']}")
                        else:
                            row = singular_answer(wall, answer)
                    except (TimeoutError, subprocess.TimeoutExpired):
                        if phase == "warm" and tool in active:
                            active[tool].stop()
                        samples[key].append(dict(timeout_s=args.timeout, warmup=(trial < 0)))
                        continue
                    if trial >= 0:
                        samples[key].append(row)
    medians = {key: {field: round(median(s[field] for s in runs), 3) for field in ("compute_ms", "wall_ms")}
               for key, runs in samples.items() if len(runs) == args.trials and all("compute_ms" in s for s in runs)}
    result = dict(equation=eq, variables=axes, mode=mode, samples=samples, median_ms=medians)
    if solved:
        result["record_sha256"] = digest(solved)
        if args.replay:
            result["saved_replay_sha256"] = replay_record(host, solved, args.timeout)
        if solved["engine"] == "canonical-blowups" and len(axes) == 3 and mode == "embedded":
            checks = singular_chart_checks(solved, singular.command, args.timeout)
            result["singular_verified_charts"] = len(checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--zoo", type=Path, help="Add all recorded fallback cases in these Resolution Zoo screens")
    parser.add_argument("--atlas-root", type=Path, default=ROOT, help="Built Atlas checkout to measure")
    parser.add_argument("--reference", type=Path, help="Check complete records and saved replay against an earlier report")
    parser.add_argument("--record-corpus", type=Path,
                        help="Compare the 24 historical CLI records against this clean built checkout")
    parser.add_argument("--paired-fresh", type=Path,
                        help="Alternate fresh CLI trials against this built checkout, checking every record")
    parser.add_argument("--singular", default="Singular")
    parser.add_argument("--atlas-warm-only", action="store_true",
                        help="Repeat only the warm Atlas series when a Singular comparison was censored.")
    parser.add_argument("--replay", action="store_true", help="Also recompute the dual complex and zetas from saved charts")
    args = parser.parse_args()
    args.atlas_root = args.atlas_root.resolve()
    if args.trials < 1 or args.timeout <= 0:
        parser.error("trials and timeout must be positive")
    os.sched_setaffinity(0, {args.cpu})
    sys.path.insert(0, str(ROOT))
    if args.record_corpus:
        check_record_corpus(args.record_corpus.resolve(), args.atlas_root, args)
        return
    env = os.environ.copy()
    if env.get("ATLAS_RECORD_REQUESTS"):
        parser.error("Disable corpus recording during timing measurements")
    suite = cases(args.zoo)
    selected = args.cases or list(suite)
    unknown = set(selected) - suite.keys()
    if unknown:
        parser.error(f"Unknown cases {sorted(unknown)}")
    if args.paired_fresh:
        paired_fresh(args.paired_fresh.resolve(), args.atlas_root, {k: suite[k] for k in selected}, args)
        return
    reference = json.loads(args.reference.read_text())["cases"] if args.reference else {}
    report = dict(cpu=args.cpu, processor=platform.processor(), kernel=platform.release(),
                  python=platform.python_version(), atlas_sha256=atlas_fingerprint(args.atlas_root),
                  singular=subprocess.check_output([args.singular, "--version"], text=True).splitlines()[0],
                  trials=args.trials, timeout_s=args.timeout, conditions="Alternating fresh CLI and fresh Singular, "
                  "then alternating calls in one warm process per solver and germ after one untimed repeat. "
                  "Atlas elapsed_ms and Singular microsecond rtimer around resolve(ideal(f)). All processes "
                  "pinned to one logical CPU. Timed-out series are censored. Atlas wall time for warm runs "
                  "covers resolve(request). Fresh wall time includes JSON output and process exit.", cases={})
    if args.atlas_warm_only:
        report["conditions"] = ("Five Atlas in-process measurements by default after an untimed warm-up. "
                                "One pinned CPU, no concurrent Singular process.")
    for name in selected:
        result = run_case(name, suite[name], args, env)
        if args.reference:
            for key in ("record_sha256", "saved_replay_sha256") if args.replay else ("record_sha256",):
                if result.get(key) is None or result[key] != reference[name][key]:
                    raise ValueError(f"{name} {key} disagrees with {args.reference}")
        report["cases"][name] = result
        print(name, result["median_ms"], flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
