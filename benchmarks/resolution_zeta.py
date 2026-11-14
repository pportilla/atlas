#!/usr/bin/env python3
"""Same-host wall time for the local topological and monodromy zetas of surfaces."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
from queue import Empty, Queue
import re
import shutil
from statistics import median
import subprocess
import sys
from threading import Thread
from time import monotonic, perf_counter_ns


ROOT = Path(__file__).resolve().parents[1]
# Polynomial, local topological zeta, A'Campo zeta in the Atlas convention.
CASES = {
    "quadric_A1": ("x^2+y^2+z^2", "(s+3)/((s+1)*(2*s+3))", "1/(1-s^2)"),
    "cubic_cone": ("x^3+y^3+z^3", "1/(s+1)", "1/(1-s^3)^3"),
    "surface_A6": ("x^7+y^2-z^2", "(s+8)/((s+1)*(7*s+8))", "1/(1-s^7)"),
}
PREFIX = "ATLASBENCH "
ATLAS_WORKER = '''import sys
sys.path.insert(0, sys.argv[1])
from engine.resolve import resolve
print("ATLASBENCH ready", flush=True)
for line in sys.stdin:
    result = resolve({"polynomial": line.strip(), "variables": ["x", "y", "z"],
                      "mode": "embedded", "compute": "cpu", "search": "quick",
                      "monodromy": True})
    for field in ("topological_zeta", "monodromy_zeta"):
        if result[field]["status"] != "computed":
            raise RuntimeError(result[field])
    print("ATLASBENCH " + result["topological_zeta"]["expression"] + "|" +
          result["monodromy_zeta"]["expression"], flush=True)
'''
SINGULAR_SETUP = '''LIB "reszeta.lib";
proc measureZetas(poly f) {
    list re = resolve(ideal(f), "A");
    list zetas = zetaDL(re, 1, "local", "A");
    print("ATLASBENCH " + zetas[1] + "|" + zetas[2]);
}
ring R = 0,(x,y,z),dp;
print("ATLASBENCH ready");
'''


def read_output(stream, output):
    for line in stream:
        output.put(line.rstrip())
    output.put(None)


class Solver:
    def __init__(self, label, root, singular):
        self.label = label
        self.is_singular = root is None
        self.root = root
        self.command = ([singular, "-q", "--no-rc"] if self.is_singular else
                        [sys.executable, "-I", "-B", "-u", "-c", ATLAS_WORKER, str(root)])
        self.setup = SINGULAR_SETUP if self.is_singular else ""
        self.end = "quit;\n" if self.is_singular else ""

    def job(self, polynomial):
        return f"measureZetas({polynomial});\n" if self.is_singular else polynomial + "\n"

    def fresh(self, polynomial, env, timeout):
        payload = self.setup + self.job(polynomial) + self.end
        start = perf_counter_ns()
        result = subprocess.run(self.command, input=payload, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout, env=env)
        elapsed = (perf_counter_ns() - start) / 1e6
        lines = result.stdout.splitlines()
        answers = [line[len(PREFIX):] for line in lines if line.startswith(PREFIX)]
        if (result.returncode or len(answers) != 2 or answers[0] != "ready"
                or any(line.lstrip().startswith("?") for line in lines)):
            raise RuntimeError(f"{self.label} failed for {polynomial}:\n{result.stdout}")
        return elapsed, answers[1]


class Session:
    def __init__(self, solver, env, timeout):
        self.solver = solver
        self.timeout = timeout
        self.proc = subprocess.Popen(solver.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, env=env)
        self.output = Queue()
        self.reader = Thread(target=read_output, args=(self.proc.stdout, self.output), daemon=True)
        self.reader.start()

    def write(self, source):
        self.proc.stdin.write(source)
        self.proc.stdin.flush()

    def answer(self):
        deadline = monotonic() + self.timeout
        diagnostics = []
        while True:
            try:
                line = self.output.get(timeout=max(0, deadline - monotonic()))
            except Empty:
                raise TimeoutError(f"{self.solver.label} timed out: {diagnostics}") from None
            if line is None or line.lstrip().startswith("?"):
                raise RuntimeError(f"{self.solver.label} failed: {diagnostics + [line]}")
            if line.startswith(PREFIX):
                return line[len(PREFIX):]
            diagnostics.append(line)

    def __enter__(self):
        try:
            self.write(self.solver.setup)
            if self.answer() != "ready":
                raise RuntimeError(f"{self.solver.label} did not initialize")
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *_):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.reader.join()
        self.proc.stdin.close()
        self.proc.stdout.close()

    def warm(self, polynomial):
        start = perf_counter_ns()
        self.write(self.solver.job(polynomial))
        answer = self.answer()
        return (perf_counter_ns() - start) / 1e6, answer


def check_answer(solver, case, answer):
    import sympy as sp
    s = sp.Symbol("s")

    def expression(raw, compact=False):
        # Singular prints 2s3 for 2*s^3 in a rational-function coefficient.
        raw = raw.replace("T", "s").replace(" ", "")
        if not re.fullmatch(r"[0-9s()+*/^\-]+", raw):
            raise ValueError(f"Unexpected univariate expression {raw!r}")
        if compact:
            raw = re.sub(r"s([0-9]+)", r"s^\1", raw)
            raw = re.sub(r"([0-9])s", r"\1*s", raw)
        return sp.sympify(raw.replace("^", "**"), {"s": s})

    top, mon = (expression(raw, solver.is_singular) for raw in answer.split("|"))
    if solver.is_singular:
        # chi(s) = det(s I-h | H²); the Milnor fibre has only H⁰ and H².
        degree = sp.Poly(mon, s).degree()
        mon = 1 / ((1 - s) * s**degree * mon.subs(s, 1 / s))
    for actual, expected in zip((top, mon), CASES[case][1:]):
        if sp.cancel(actual - expression(expected)) != 0:
            raise ValueError(f"{solver.label}/{case}: got {answer!r}, expected {CASES[case][1:]}")


def atlas_fingerprint(root):
    digest = hashlib.sha256()
    paths = sorted((root / "engine").glob("*.py"))
    binary_names = {"plane", "newton_fan", "toricfan", "face_gb"}
    binary_names.update(f"lib{name}.{suffix}" for name in ("groebner", "chain", "resolution")
                        for suffix in ("so", "dylib"))
    paths += sorted(path for path in (root / "native/build").glob("*") if path.name in binary_names)
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, help="Additional built Atlas checkout")
    parser.add_argument("--singular", default="Singular")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120, help="Seconds per request")
    parser.add_argument("--output", type=Path, help="Write samples and measurements as JSON")
    args = parser.parse_args()
    singular = shutil.which(args.singular)
    if singular is None or args.repeats < 1 or args.timeout <= 0:
        parser.error("Install Singular and supply positive repetitions and timeout.")
    solvers = [Solver("atlas", args.atlas_root.resolve(), None)]
    if args.baseline:
        solvers.append(Solver("baseline", args.baseline.resolve(), None))
    solvers.append(Solver("singular", None, singular))
    env = {k: v for k, v in os.environ.items() if not k.startswith("ATLAS_")}
    env["ATLAS_GROEBNER"] = "native"
    cpuinfo = Path("/proc/cpuinfo")
    models = (line.split(":", 1)[1].strip() for line in
              (cpuinfo.read_text().splitlines() if cpuinfo.is_file() else []) if line.startswith("model name"))
    version = subprocess.check_output([singular, "--version"], text=True).splitlines()[0]
    data = {"date_utc": datetime.now(timezone.utc).isoformat(), "cpu": next(models, platform.processor()),
            "platform": platform.platform(), "python": sys.version, "singular": version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "atlas": {solver.label: {"root": str(solver.root), "sha256": atlas_fingerprint(solver.root)}
                      for solver in solvers if solver.root}, "cases": CASES, "samples": []}
    jobs = list(itertools.product(CASES, solvers))
    with ExitStack() as stack:
        sessions = {solver: stack.enter_context(Session(solver, env, args.timeout)) for solver in solvers}
        for case, solver in jobs:
            _, answer = sessions[solver].warm(CASES[case][0])
            check_answer(solver, case, answer)
        for rep in range(args.repeats):
            for case, solver in (jobs if rep % 2 == 0 else reversed(jobs)):
                for scope in (("fresh", "warm") if rep % 2 == 0 else ("warm", "fresh")):
                    polynomial = CASES[case][0]
                    if scope == "fresh":
                        elapsed, answer = solver.fresh(polynomial, env, args.timeout)
                    else:
                        elapsed, answer = sessions[solver].warm(polynomial)
                    check_answer(solver, case, answer)
                    data["samples"].append({"case": case, "solver": solver.label, "scope": scope,
                                            "repeat": rep + 1, "wall_ms": elapsed, "result": answer})
    data["medians_ms"] = {}
    for case, solver in jobs:
        values = {scope: median(row["wall_ms"] for row in data["samples"]
                                if (row["case"], row["solver"], row["scope"]) == (case, solver.label, scope))
                  for scope in ("fresh", "warm")}
        data["medians_ms"].setdefault(case, {})[solver.label] = values
        print(f"{case:12} {solver.label:10} fresh {values['fresh']:8.2f} ms  warm {values['warm']:8.2f} ms")
    if args.output:
        args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
