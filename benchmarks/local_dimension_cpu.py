"""CPU timings for local-dimension and local-smoothness over QQ.

Build the typed CPU runner with: make -C native -f groebner.mk build/bench_local_dimension
Run: python3 benchmarks/local_dimension_cpu.py --output /tmp/local-dimension.json
Each case starts from the same input polynomials, variable order and rational point.
"""
import argparse
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import native_polynomial
from engine.polynomial import InputError, parse_expression
from engine.rational_point import normalize_point

OPERATIONS = ("local-dimension", "local-smoothness")
SINGULAR = ["Singular", "-q", "--no-rc", "--threads=1", "--flint-threads=1", "--random=47", "--ticks-per-sec=1000000"]


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple
    ideal: tuple
    point: tuple
    dimension: int
    embedding: int
    order: str = "grevlex"


def corpus():
    u, v, w = "(x-1/2)", "(y+2/3)", "(z-1/3)"
    xy, xyz = ("x", "y"), ("x", "y", "z")
    p = ("1/2", "-2/3", "1/3")
    matrix = ("a", "b", "c", "d", "e", "f")
    # The 2x2 minors of a generic 2x3 matrix define a fourfold,
    # although their three quadratic generators have no linear parts.
    def minors(a, b, c, d, e, f):
        return (f"{a}*{e}-{b}*{d}", f"{a}*{f}-{c}*{d}", f"{b}*{f}-{c}*{e}")

    transformed = ("((a-1/2)+(b+2/3)^2)", "(b+2/3)", "(c-1/3)", "(d+(b+2/3)*(c-1/3))", "e", "f")
    return [
        Case("regular-plane", xy, (), (), 2, 2),
        Case("shifted-cusp", xy, (f"{v}^2-{u}^3",), p[:2], 1, 2),
        Case("smooth-on-cusp", xy, (f"{v}^2-{u}^3",), ("3/2", "1/3"), 1, 1),
        Case("fat-point", xy, ("x^2", "y^3"), (), 0, 2),
        Case("plane-and-line", xyz, ("x*y", "x*z"), (), 2, 3),
        Case("on-line-only", xyz, ("x*y", "x*z"), ("1", "0", "0"), 1, 1),
        Case("embedded-point", xy, ("x*y", "y^2"), (), 1, 2),
        Case("shifted-complete-intersection", xyz, (f"{u}*{v}+{w}^2", f"{u}*{v}+{w}^2+{u}^2"), p, 1, 3, "lex"),
        Case("shifted-oblique-line", xyz, (f"2*{u}+3*{v}+{w}+{u}^2", f"{u}-2*{v}+3*{w}",
                                          f"2*({u}-2*{v}+3*{w})"), p, 1, 1),
        Case("outside-support", xy, ("x*y", "y^2"), ("1/2", "1"), -1, -1),
        Case("rank-one-matrix-vertex", matrix, minors(*matrix), (), 4, 6),
        Case("shifted-nonlinear-matrix-vertex", matrix, minors(*transformed),
             ("1/2", "-2/3", "1/3", "0", "0", "0"), 4, 6),
    ]


def expected(case, operation):
    if case.dimension < 0:
        return (-1,)
    if operation == "local-dimension":
        return (case.dimension,)
    return (case.dimension, case.embedding, len(case.variables) - case.embedding, int(case.dimension == case.embedding))


def values(result, operation):
    if operation == "local-dimension":
        if result["type"] != "integer" or result["value"] != result["local_dimension"]:
            raise ValueError(f"Invalid dimension record: {result}")
        return (result["value"],)
    if result["type"] != "boolean" or type(result["value"]) is not bool:
        raise ValueError(f"Invalid smoothness record: {result}")
    return (result["local_dimension"], result["embedding_dimension"], result["jacobian_rank"], int(result["value"]))


def prepare_atlas(case):
    names = list(case.variables)
    variables = tuple(map(native_polynomial.Symbol, names))
    operands = {"ideal": [native_polynomial.from_terms(variables, parse_expression(g, names)) for g in case.ideal]}
    if case.point:
        operands["point"] = normalize_point(list(case.point), len(names))
    return {"field": "QQ", "variables": names, "order": case.order}, operands


def time_atlas(case, operation, batch):
    from engine.local_dimension import calculate_cpu
    ring, operands = prepare_atlas(case)

    def call():
        try:
            return calculate_cpu(operation, ring, operands, canonical=True)
        except InputError as error:
            if case.dimension < 0 and "point" in str(error).lower():
                return None
            raise

    def validate(record):
        if record is None:
            result = (-1,)
        else:
            output, execution = record
            if execution["requested"] != "cpu" or execution["backend"] != "cpu":
                raise ValueError(f"Unexpected device for {case.name}: {execution}")
            result = values(output, operation)
        if result != expected(case, operation):
            raise ValueError(f"{case.name}: Atlas gave {result}, expected {expected(case, operation)}")

    validate(call())
    start = perf_counter_ns()
    for _ in range(batch):
        result = call()
    elapsed = (perf_counter_ns() - start) / 1000 / batch
    validate(result)
    return elapsed


def time_typed(case, operation, batch):
    source = "\n".join([",".join(case.variables), ",".join(case.point), *case.ideal]) + "\n"
    runner = ROOT / "native/build/bench_local_dimension"
    run = subprocess.run([str(runner), str(batch), operation, case.order], input=source, text=True,
                         check=True, capture_output=True, timeout=120)
    result = json.loads(run.stdout)
    if case.dimension < 0:
        if result.get("off_scheme") is not True:
            raise ValueError(f"{case.name}: missing off-scheme rejection in typed CPU runner: {result}")
    else:
        typed = dict(result, type="boolean" if operation == "local-smoothness" else "integer")
        if values(typed, operation) != expected(case, operation):
            raise ValueError(f"{case.name}: unexpected typed CPU result {result}")
    return result["elapsed_ns"] / 1000 / batch


def singular_source(case, operation, batch=None, *, generic=False):
    order = {"lex": "Ds", "grevlex": "ds"}[case.order]
    names = ",".join(case.variables)
    # Declaration and map construction are outside the warm timer in both engines.
    # Execute the map on the original ideal inside the timer whenever a shift is needed.
    shifted = any(q != "0" for q in case.point)
    if shifted:
        local = ",".join(f"u{i}" for i in range(len(case.variables)))
        shifts = ",".join(f"u{i}+({q})" for i, q in enumerate(case.point))
        rings = (f"ring Source=0,({names}),dp; ideal basis=ideal({','.join(case.ideal) or '0'});\n"
                 f"ring Local=0,({local}),{order}; map move=Source,{shifts};")
        prepare = "ideal J=move(basis);"
    else:
        rings = f"ring Local=0,({names}),{order}; ideal J=ideal({','.join(case.ideal) or '0'});"
        prepare = ""
    # Principal ideal theorem and the zero ideal avoid an unnecessary standard
    # basis. A full-rank linear part proves smoothness of the complete intersection.
    body = [prepare, "if(size(jet(J,0))){return(intvec(-1));}", "int d;"]
    if operation == "local-smoothness":
        body += ["int r; if(size(J)==1){r=(size(jet(J,1))!=0);}else{r=rank(jacob(jet(J,1)));}",
                 "int e=nvars(basering)-r;"]
    dimension = ("if(size(J)==0){d=nvars(basering);}"
                 "else{if(size(J)==1){d=nvars(basering)-1;}")
    if operation == "local-smoothness":
        dimension += "else{if(r==size(J) || e==0){d=e;}else{d=dim(std(J));}}}"
    else:
        dimension += "else{d=dim(std(J));}}"
    body += [dimension, "return(intvec(d,e,r,d==e));" if operation == "local-smoothness" else "return(intvec(d));"]
    if generic:
        body = [prepare, "int d=dim(std(J));"]
        if operation == "local-smoothness":
            body += ["if(d<0){return(intvec(-1));}", "int r=rank(jacob(jet(J,1)));",
                     "int e=nvars(basering)-r;", "return(intvec(d,e,r,d==e));"]
        else:
            body += ["return(intvec(d));"]
    source = [rings, "proc solve(){" + "\n".join(body) + "}", "intvec got=solve();"]
    if batch is not None:
        source += ["int start=rtimer; int i;", f"for(i=0;i<{batch};i++){{got=solve();}}",
                   'print("TIME "+string(rtimer-start));']
    source += ['print("RESULT "+string(got));', "quit;"]
    return "\n".join(source) + "\n"


def time_singular(case, operation, batch, *, generic=False):
    source = singular_source(case, operation, batch, generic=generic)
    start = perf_counter_ns()
    p = subprocess.run(SINGULAR, input=source, text=True, capture_output=True, check=True, timeout=120)
    cold_us = (perf_counter_ns() - start) / 1000
    records = {}
    for line in p.stdout.splitlines():
        if line.startswith(("TIME ", "RESULT ")):
            key, value = line.split(" ", 1)
            records[key] = value
        elif line.strip():
            raise RuntimeError(f"Unexpected Singular output: {p.stdout}")
    if p.stderr or tuple(map(int, records["RESULT"].split(","))) != expected(case, operation):
        raise ValueError(f"{case.name}: unexpected Singular result {p.stdout} {p.stderr}")
    return int(records["TIME"]) / batch if batch is not None else cold_us


def cold_atlas(case, operation):
    args = [str(ROOT / "atlas"), "algebra", operation, "--vars", ",".join(case.variables),
            "--order", case.order, "--compute", "cpu", "--json"]
    if case.point:
        args += ["--point", ",".join(case.point)]
    for g in case.ideal:
        args += ["--gen", g]
    start = perf_counter_ns()
    p = subprocess.run(args, text=True, capture_output=True, timeout=120)
    elapsed = (perf_counter_ns() - start) / 1000
    record = json.loads(p.stdout)
    if case.dimension < 0:
        if p.returncode != 2 or record.get("error", {}).get("code") != "invalid_request":
            raise ValueError(f"{case.name}: invalid CLI error {record}")
    elif p.returncode or record["execution"]["backend"] != "cpu" or values(record["result"], operation) != expected(case, operation):
        raise ValueError(f"{case.name}: unexpected CPU CLI result {record}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[c.name for c in corpus()])
    parser.add_argument("--batch", type=int, default=64, help="kernel calls per warm sample")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cold-repeats", type=int, default=3, help="process samples per engine on all cases; zero disables")
    parser.add_argument("--cpu", type=int, help="pin both engines to the same available logical CPU (Linux)")
    parser.add_argument("--output", type=Path, help="write full JSON in addition to the printed table")
    args = parser.parse_args()
    if args.batch < 1 or args.repeats < 1 or args.cold_repeats < 0:
        parser.error("positive batch and repeats, nonnegative cold repeats required")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    from engine.native_library import _LIBRARY
    if not (ROOT / "native/build/bench_local_dimension").is_file():
        parser.error("build the benchmark runner with make -C native -f groebner.mk build/bench_local_dimension")
    cpu_file = Path("/proc/cpuinfo")
    cpu_name = next((s.split(":", 1)[1].strip() for s in cpu_file.read_text().splitlines()
                     if s.startswith("model name")), platform.machine()) if cpu_file.exists() else platform.machine()
    artifacts = [Path(__file__), _LIBRARY, ROOT / "native/build/bench_local_dimension", ROOT / "native/build/local_algebra",
                 ROOT / "native/local_operations.cpp", ROOT / "native/point_geometry.cpp", ROOT / "native/local_cli.cpp",
                 ROOT / "native/local_basis.cpp", ROOT / "native/local_basis.hpp",
                 ROOT / "benchmarks/local_dimension_kernel.cpp",
                 ROOT / "atlas", *(ROOT / "engine" / name for name in
                                   ("local_dimension.py", "local_dimension_input.py", "algebra_execution.py",
                                    "native_library.py", "native_polynomial.py"))]
    report = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cpu": cpu_name,
              "atlas_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in artifacts if p.exists()},
              "python": sys.version.split()[0], "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "command": ["python3", "benchmarks/local_dimension_cpu.py", *sys.argv[1:]],
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "unit": "microseconds per call", "batch": args.batch, "repeats": args.repeats,
              "cold_repeats": args.cold_repeats, "scope": (
                  "Warm Atlas public CPU handler including policy/stages and native request/response; "
                  "separate warmed Atlas C++ CPU kernel on a parsed ideal and point; "
                  "two warm Singular procedures in a parsed rational local ring. Generic uses dim(std(J)) and, "
                  "for smoothness, the rank of the first jet. Specialized uses dimension bounds and shortcuts "
                  "for principal ideals and maximal rank. Report both medians and their minimum. Nonzero-point translation "
                  "or evaluation and incidence are included; parsing, process startup, replay generation and printing excluded. "
                  "All timed operations compute the same scalar(s); both sides warm up then recompute. "
                  "Cold measures the installed ./atlas CLI + JSON and fresh Singular -q + source + printed invariant tuple. "
                  "Singular uses one thread and its native wall clock. Warm engine order rotates per sample."),
              "rows": []}
    selected = [c for c in corpus() if not args.case or c.name in args.case]
    for case in selected:
        for operation in OPERATIONS:
            engines = ("atlas_handler", "atlas_typed", "singular_generic", "singular_specialized")
            samples = {engine: [] for engine in engines}
            methods = {"atlas_handler": time_atlas, "atlas_typed": time_typed,
                       "singular_specialized": time_singular,
                       "singular_generic": lambda c, op, n: time_singular(c, op, n, generic=True)}
            cold = {"atlas": [], "singular_generic": [], "singular_specialized": []}
            for trial in range(args.repeats):
                offset = trial % len(engines)
                for engine in engines[offset:] + engines[:offset]:
                    samples[engine].append(methods[engine](case, operation, args.batch))
            for trial in range(args.cold_repeats):
                cold_engines = tuple(cold)
                offset = trial % len(cold_engines)
                for engine in cold_engines[offset:] + cold_engines[:offset]:
                    cold[engine].append(cold_atlas(case, operation) if engine == "atlas" else
                                        methods[engine](case, operation, None))
            medians = {k: statistics.median(v) for k, v in samples.items()}
            medians["singular_best"] = min(medians["singular_generic"], medians["singular_specialized"])
            cold_medians = {k: statistics.median(v) for k, v in cold.items() if v}
            if cold_medians:
                cold_medians["singular_best"] = min(cold_medians["singular_generic"], cold_medians["singular_specialized"])
            report["rows"].append({"case": asdict(case), "operation": operation, "expected": expected(case, operation),
                                   "warm_us": samples, "warm_median_us": medians, "cold_us": cold,
                                   "cold_median_us": cold_medians})
            print(f"{case.name:29} {operation:17} Atlas handler {medians['atlas_handler']:8.2f} µs   "
                  f"Atlas C++ {medians['atlas_typed']:8.2f} µs   Singular best {medians['singular_best']:8.2f} µs", flush=True)
    changed = [name for name, digest in report["sha256"].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"Benchmark inputs changed during measurements; rerun on one build: {', '.join(changed)}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
