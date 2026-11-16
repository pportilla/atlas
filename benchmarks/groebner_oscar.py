"""Compare Atlas's native QQ Gröbner solver with OSCAR on the identical reduced-basis workload."""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import signal
import socket
from statistics import median
import subprocess
from time import monotonic

from groebner import table
from groebner_native_comparison import CORPUS, DRIVER, ROOT, canonical, canonical_output, case_groups, request_text
from groebner_singular import singular_round

OSCAR_DRIVER = ROOT / "benchmarks/bench_oscar_basis.jl"


class SolverNotice(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Worker:
    """A sequential line-protocol worker. Kill the process group on timeout before timing another solver."""

    def __init__(self, command, env=None):
        self.command = command
        self.env = env
        self.process = None

    def start(self):
        self.socket, child = socket.socketpair()
        try:
            with child:
                self.process = subprocess.Popen(self.command, stdin=child, stdout=child,
                                                env=self.env, start_new_session=True)
        except BaseException:
            self.socket.close()
            raise
        self.reader = self.socket.makefile("rb")

    def close(self):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()
        self.reader.close()
        self.socket.close()
        self.process = None

    def readline(self, deadline):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError
        self.socket.settimeout(remaining)
        line = self.reader.readline()
        if not line:
            raise RuntimeError(f"{self.command[0]} stopped before completing the request")
        return line.decode().rstrip("\r\n")

    def solve(self, request, timeout):
        deadline = monotonic() + timeout
        self.socket.settimeout(timeout)
        self.socket.sendall(request)
        header = self.readline(deadline)
        if header.startswith("SKIP BACKEND_ERROR: "):
            raise SolverNotice("error", header.removeprefix("SKIP BACKEND_ERROR: "))
        if header.startswith("ERROR "):
            raise SolverNotice("error", header[6:])
        if header.startswith("SKIP "):
            raise SolverNotice("skipped", header[5:])
        if not header.startswith("TIME "):
            raise RuntimeError(f"Unexpected solver output: {header!r}")
        ms = float(header[5:])
        header = self.readline(deadline)
        if not header.startswith("BASIS "):
            raise RuntimeError(f"Expected a basis header, received {header!r}")
        size = int(header[6:])
        if not isfinite(ms) or ms < 0 or size < 0:
            raise ValueError(f"Invalid solver time {ms} or basis size {size}")
        return ms, tuple(self.readline(deadline) for _ in range(size))


def start_oscar(worker, timeout):
    worker.start()
    try:
        handshake = worker.readline(monotonic() + timeout).split()
    except TimeoutError:
        raise RuntimeError(f"OSCAR startup and precompilation exceeded {timeout} seconds") from None
    if len(handshake) != 3 or handshake[0] != "READY":
        raise RuntimeError(f"Invalid OSCAR startup handshake: {handshake}")
    return {"julia": handshake[1], "oscar": handshake[2]}


def trial_order(methods, trial):
    # Each consecutive pair runs in reverse orders. Rotate the middle slot too.
    offset = (trial // 2) % len(methods)
    order = methods[offset:] + methods[:offset]
    return order if trial % 2 == 0 else order[::-1]


def prepare(case, batch):
    try:
        if case["order"] not in ("lex", "grevlex") or type(case["n"]) is not int or case["n"] < 1:
            raise ValueError("invalid variable count or term order")
        for poly in case["polys"]:
            terms = table(poly)
            if len(terms) != len(poly) or any(len(e) != case["n"] or
                    any(type(power) is not int or power < 0 for power in e) for e in terms):
                raise ValueError("repeated monomial or invalid exponent vector")
        reference = canonical(case, map(table, case["expected"])) if "expected" in case else None
        return request_text([case], batch).encode(), reference
    except (TypeError, KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid benchmark input for {case.get('label', '<unnamed>')}: {exc}") from exc


def verify_classical(case, cached, unavailable, timeout):
    # With no recorded reference, adjudicate disagreements outside all benchmark timers.
    candidates = {}
    for method, outputs in cached.items():
        if outputs[0] is not None and unavailable[method].get(case["label"], {}).get("status") != "incorrect":
            candidates.setdefault(canonical_output(case, outputs[0]), []).append(method)
    if not candidates:
        return "no completed solve"
    if len(candidates) == 1 and len(next(iter(candidates.values()))) > 1:
        return "cross-checked by independent solvers"
    check = "std"
    if any("oscar-default" in methods for methods in candidates.values()):
        check = "stdfglm" if case["order"] == "lex" else "slimgb"
    statuses = {}
    for candidate in candidates:
        basis = [[[str(c.numerator), str(c.denominator), list(e)] for e, c in p] for p in candidate]
        try:
            singular_round([case], check, [1], [basis], timeout=timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            if len(candidates) == 1:
                return "unverified: independent Singular check unavailable"
            statuses[candidate] = "disagreement"
        except RuntimeError as exc:
            if "returned an incorrect reduced monic basis" not in str(exc):
                raise RuntimeError(f"Independent verification failed for {case['label']}: {exc}") from exc
            statuses[candidate] = "incorrect"
        else:
            statuses = {other: "incorrect" for other in candidates if other != candidate}
            break
    for candidate, status in statuses.items():
        reason = (f"Returned basis differs from independently computed Singular {check}" if status == "incorrect"
                  else "Different bases returned on this ideal; independent Singular check unavailable")
        for method in candidates[candidate]:
            unavailable[method][case["label"]] = {"status": status, "reason": reason}
            print(f"  {case['label']}, {method}: {reason}", flush=True)
    if any(x.get(case["label"], {}).get("status") == "disagreement" for x in unavailable.values()):
        return "unresolved disagreement"
    return f"checked by Singular {check}"


def measure(cases, args, workers, report):
    corpus = "expected" in cases[0]
    batch = args.batch if corpus else 1
    methods = [m for m in report["methods"] if m != "oscar-fglm" or
               (not corpus and cases[0]["order"] == "lex")]
    requests, expected = zip(*(prepare(case, batch) for case in cases))
    samples = {m: [] for m in methods}
    cached = {m: [None] * len(cases) for m in methods}
    unavailable = {m: {} for m in methods}
    if "oscar-f4" in methods:
        unavailable["oscar-f4"] = {case["label"]: {"status": "skipped", "reason": "msolve F4 supports grevlex only"}
                                   for case in cases if case["order"] != "grevlex"}
    round_orders = []
    exhausted = {}
    for trial in range(args.repeats):
        order = trial_order(methods, trial)
        round_orders.append(order)
        for method in order:
            times = [None] * len(cases)
            samples[method].append(times)
            if method in exhausted:
                continue
            deadline = monotonic() + args.corpus_time_limit if corpus else None
            tag = method.removeprefix("oscar-")
            if tag == "f4" and corpus and args.msolve_corpus_threads == "one":
                tag = "f4-serial"
            worker = workers["atlas" if method == "atlas" else "oscar"]
            for i, (case, request) in enumerate(zip(cases, requests)):
                label = case["label"]
                if label in unavailable[method]:
                    continue
                in_flight = False
                if deadline is not None and monotonic() >= deadline:
                    break
                if worker.process is None:
                    startup = monotonic()
                    if method == "atlas":
                        worker.start()
                    elif start_oscar(worker, args.startup_limit) != report["versions"]:
                        raise RuntimeError("The OSCAR environment changed during the benchmark")
                    if deadline is not None:
                        deadline += monotonic() - startup
                if method != "atlas":
                    request = tag.encode() + b"\n" + request
                remaining = deadline - monotonic() if deadline is not None else args.time_limit
                if remaining <= 0:
                    break
                try:
                    ms, rows = worker.solve(request, min(remaining, args.time_limit))
                except SolverNotice as exc:
                    unavailable[method][label] = {"status": exc.status, "trial": trial + 1, "reason": str(exc)}
                    print(f"  {label}, {method}: {exc.status}: {exc}", flush=True)
                    continue
                except TimeoutError:
                    worker.close()
                    if deadline is not None and remaining <= args.time_limit:
                        in_flight = True
                        break
                    unavailable[method][label] = {"status": "timeout", "trial": trial + 1,
                                                  "request_limit_seconds": args.time_limit}
                    print(f"  {method} timed out on {label}", flush=True)
                    continue
                except (RuntimeError, ValueError, OSError) as exc:
                    raise RuntimeError(f"Broken solver protocol for {label}, {method}, trial {trial+1}: {exc}") from exc
                if rows != cached[method][i]:
                    reason = None
                    try:
                        current = canonical_output(case, rows)
                    except (ValueError, ZeroDivisionError) as exc:
                        if not str(exc).startswith("Nonmonic basis"):
                            raise ValueError(f"Malformed basis for {label}, {method}, trial {trial+1}: {exc}") from exc
                        reason = str(exc)
                    else:
                        if corpus and current != expected[i]:
                            reason = "Returned basis differs from the recorded reduced monic basis"
                        elif cached[method][i] is not None and current != canonical_output(case, cached[method][i]):
                            reason = "Solver returned different bases in repeated trials"
                    if reason:
                        unavailable[method][label] = {"status": "incorrect", "trial": trial + 1, "reason": reason}
                        print(f"  {label}, {method}: {reason}", flush=True)
                        continue
                    cached[method][i] = rows
                times[i] = ms
            else:
                continue
            # A break indicates a corpus round exceeded its wall budget. A worker
            # that completed its previous request can immediately serve another method.
            done = sum(t is not None for t in times)
            exhausted[method] = {"trial": trial + 1, "limit_seconds": args.corpus_time_limit,
                                 "completed_case_count": done, "total_case_count": len(cases),
                                 "first_unfinished_index": i, "first_unfinished_label": label,
                                 "request_interrupted": in_flight}
            print(f"  {method} exceeded its {args.corpus_time_limit}s corpus round budget "
                  f"with {done}/{len(cases)} timings; remaining {method} rounds omitted", flush=True)
        print(f"{cases[0]['label']} ({len(cases)} ideals), round {trial+1}/{args.repeats}", flush=True)

    verification = "recorded bases" if corpus else verify_classical(cases[0], cached, unavailable, args.time_limit)
    for method, failures in unavailable.items():
        for i, case in enumerate(cases):
            if failures.get(case["label"], {}).get("status") in ("incorrect", "disagreement"):
                for times in samples[method]:
                    times[i] = None
    comparisons = {}
    for method in methods[1:]:
        if method in exhausted or "atlas" in exhausted:
            continue
        indices = [i for i in range(len(cases)) if all(
            values[i] is not None for solver in ("atlas", method) for values in samples[solver])]
        if indices:
            totals = {solver: median(sum(values[i] for i in indices) for values in samples[solver])
                      for solver in ("atlas", method)}
            comparisons[method] = {"indices": indices, "median_totals_ms": totals}
            print(f"  {method} {totals[method]:.3f} ms; atlas {totals['atlas']:.3f} ms "
                  f"on the same {len(indices)}/{len(cases)} ideals", flush=True)
    return {"labels": [c["label"] for c in cases], "orders": [c["order"] for c in cases],
            "input_sha256": hashlib.sha256(b"".join(requests)).hexdigest(),
            "batch": batch, "trial_orders": round_orders, "samples_ms": samples,
            "unavailable": unavailable, "corpus_budget_exhausted": exhausted,
            "verification": verification, "paired_comparisons": comparisons}


def report_path(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def fingerprints(project):
    files = [Path(__file__).resolve(), OSCAR_DRIVER, CORPUS, DRIVER,
             *(ROOT / "benchmarks" / name for name in ("bench_groebner_basis.cpp", "groebner.py",
               "groebner_native_comparison.py", "groebner_comparison.py", "groebner_hard.py", "groebner_singular.py"))]
    linked = subprocess.check_output(["ldd", str(DRIVER)], text=True)
    for row in linked.splitlines():
        fields = row.split()
        if len(fields) >= 3 and fields[1] == "=>" and fields[0].startswith(("libgroebner.", "libflint.")):
            files.append(Path(fields[2]).resolve())
    project_files = [project / "Project.toml", project / "Manifest.toml", *sorted(project.glob("Manifest-v*.toml"))]
    files.extend(p for p in project_files if p.is_file())
    return {report_path(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with (gzip.open if path.suffix == ".gz" else open)(tmp, "wt") as out:
        json.dump(report, out)
        out.write("\n")
    tmp.replace(path)


def run(args):
    available = sorted(os.sched_getaffinity(0))
    if args.cpu == "single":
        os.sched_setaffinity(0, {available[0]})
    cpus = sorted(os.sched_getaffinity(0))
    groups = case_groups()
    if args.suite == "corpus":
        groups = groups[:1]
    elif args.suite == "classical":
        groups = groups[1:]

    julia = shutil.which(args.julia)
    if julia is None:
        raise SystemExit(f"Could not find {args.julia!r}. Set --julia to the Julia executable.")
    if not DRIVER.is_file():
        raise SystemExit("Build the native driver with 'make -C native -f groebner.mk build/bench_groebner_basis' first.")
    project = args.project.expanduser().resolve()
    def oscar_command(executable, project_dir, driver):
        return [executable, "--startup-file=no", "--color=no", f"--project={project_dir}",
                f"--threads={len(cpus)}", str(driver)]

    command = oscar_command(julia, project, OSCAR_DRIVER)
    recorded_command = oscar_command(Path(julia).name, report_path(project), report_path(OSCAR_DRIVER))
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    workers = {"atlas": Worker([str(DRIVER)], env), "oscar": Worker(command, env)}
    hashes = fingerprints(project)
    with ExitStack() as cleanup:
        for worker in workers.values():
            cleanup.callback(worker.close)
        versions = start_oscar(workers["oscar"], args.startup_limit)
        print(f"Julia {versions['julia']}, OSCAR {versions['oscar']}; CPU set {cpus}", flush=True)
        report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "cpu": next(row.split(":", 1)[1].strip() for row in Path("/proc/cpuinfo").read_text().splitlines()
                              if row.startswith("model name")),
                  "initial_cpu_set": available, "measured_cpu_set": cpus,
                  "versions": versions, "julia_threads": len(cpus),
                  "msolve_corpus_thread_policy": args.msolve_corpus_threads,
                  "msolve_corpus_threads": 1 if args.msolve_corpus_threads == "one" else len(cpus),
                  "msolve_classical_threads": len(cpus),
                  "openblas_threads": 1,
                  "oscar_command": recorded_command, "sha256": hashes, "repeats": args.repeats,
                  "corpus_round_limit_seconds": args.corpus_time_limit,
                  "request_limit_seconds": args.time_limit, "startup_limit_seconds": args.startup_limit,
                  "units": "milliseconds per fully reduced monic basis",
                  "methods": ["atlas", *("oscar-" + m for m in args.methods)], "groups": []}
        for cases in groups:
            report["groups"].append(measure(cases, args, workers, report))
            save(args.output, report)
        if any(failure["status"] in ("incorrect", "disagreement")
               for group in report["groups"] for failures in group["unavailable"].values()
               for failure in failures.values()):
            raise SystemExit(f"Basis checks failed; see the per-case results in {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default=os.environ.get("JULIA", "julia"))
    parser.add_argument("--project", type=Path, default=Path(os.environ.get("JULIA_PROJECT", ROOT / "benchmarks/oscar")),
                        help="Julia project containing Oscar.jl")
    parser.add_argument("--cpu", choices=("single", "host"), default="single")
    parser.add_argument("--suite", choices=("all", "corpus", "classical"), default="all")
    parser.add_argument("--methods", nargs="+", choices=("default", "f4", "fglm"),
                        default=["default", "f4", "fglm"], help="FGLM is measured on classical lex ideals")
    parser.add_argument("--repeats", type=int, default=6, help="number of paired, alternating rounds")
    parser.add_argument("--batch", type=int, default=8, help="bases per timed verifier batch")
    parser.add_argument("--time-limit", type=int, default=30, help="warm request limit per method and ideal (seconds)")
    parser.add_argument("--corpus-time-limit", type=int, default=240, help="wall seconds per warm method and corpus round")
    parser.add_argument("--msolve-corpus-threads", choices=("one", "available"), default="one",
                        help="msolve corpus thread count (default one); classical F4 uses the available CPUs")
    parser.add_argument("--startup-limit", type=int, default=1800, help="separate OSCAR startup and JIT limit (seconds)")
    parser.add_argument("--output", required=True, type=Path, help="save raw results as JSON or .json.gz")
    args = parser.parse_args()
    if min(args.repeats, args.batch, args.time_limit, args.corpus_time_limit, args.startup_limit) < 1:
        parser.error("counts and limits must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be distinct")
    run(args)
