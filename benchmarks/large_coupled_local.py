"""Exact local dimension/smoothness over QQ on shifted sparse coupled systems.

Run from a native build: python3 benchmarks/large_coupled_local.py --output results.json
Linux process limits apply identically to the Atlas CLI and the Singular commands.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("local-dimension", "local-smoothness")
SINGULAR = ("Singular", "-q", "--no-rc", "--threads=1", "--flint-threads=1", "--random=47")


@dataclass(frozen=True)
class Case:
    name: str
    order: str
    variables: tuple
    ideal: tuple
    point: tuple
    chart: tuple
    chart_ideal: tuple
    chart_point: tuple
    jacobian_determinant: str
    dimension: int
    embedding: int


def make_case(name, coordinates, equations, point, image, determinant, dimension, embedding, order="grevlex"):
    names = tuple("xyzw"[:len(point)])
    rows = tuple(re.sub(r"\b[a-d]\b", lambda m: "(" + coordinates[ord(m[0])-97] + ")", f)
                 for f in equations)
    return Case(name, order, names, rows, tuple(map(str, point)), coordinates, equations,
                tuple(map(str, image)), str(determinant), dimension, embedding)


def corpus():
    axes = ("a*b", "a*c", "b*c")
    squares = ("(a+b)^2", "(a+c)^2", "(b+c)^2")
    p = (1, 1, 0)
    origin = (0, 0, 0)
    cases = []
    for n in (7, 1001):
        cases.append(make_case(f"axis-union-{n}", (f"x^{n}-1", f"y^{n}-1", "z"), axes,
                               p, origin, n*n, 1, 3))
    for n in (17, 1000001, 2**64+7):
        cases.append(make_case(f"parameter-squares-{n}", (f"x^{n}-1", "y", "z"), squares,
                               (1, 0, 0), origin, n, 0, 3, "lex"))
    # In the quotient x=1 the high-power term in the third generator vanishes.
    # Initial generators alone have dimension 1. The exact local ring is Artinian.
    n = 1000001
    cases.append(Case("graph-cancellation", "grevlex", ("x","y","z"),
                      ("x-1","y^2+z^3",f"y*z+(x^{n}-1)*z^2"), ("1","0","0"),
                      ("x-1","y","z"), ("a","b^2+c^3","b*c"), ("0","0","0"), "1", 0, 2))
    n = 1001
    cases.append(Case("graph-cancellation-curve", "grevlex", ("x","y","z","w"),
                      ("x-1", "y^2+z^3", f"y*z+z^2*((x^{n}-1)*w^{n}+(x-1)*(w+1))"),
                      ("1","0","0","1"), ("x-1","y","z","w-1"), ("a","b^2+c^3","b*c"),
                      ("0","0","0","0"), "1", 1, 3))
    primitive = (f"x^{n}*y^{n}+x+y-3", "x-y", "z")
    cases.append(make_case("independent-higher-jets", primitive, ("a^2", "a^2+b^3", "a^2+b^3+c^4"),
                           p, origin, -2*(n+1), 0, 3))
    cases.append(make_case("nonreduced-line", primitive, ("a^2", "a*b", "b^2"),
                           p, origin, -2*(n+1), 1, 3))
    for n in (7, 1001):
        oblique = (f"x^{n}*y^{n}+x+y-3+z", "x-y+z", "z+x-1")
        cases.append(make_case(f"oblique-axis-union-{n}", oblique, axes, p, origin, -n, 1, 3))
    mixed = ("x^1001+x-2", "y^1001+y-2", "z", "w")
    rows = ("a*b", "a*c", "a*d")
    for key, point, image, dim, emb in (
        ("crossing", (1, 1, 0, 0), (0, 0, 0, 0), 3, 4),
        ("line", (-1, 1, 0, 0), (-4, 0, 0, 0), 1, 1),
        ("hyperplane", (1, -1, "1/2", 0), (0, -4, "1/2", 0), 3, 3),
        ("outside", (-1, -1, 0, 0), (-4, -4, 0, 0), -1, -1),
    ):
        cases.append(make_case("mixed-"+key, mixed, rows, point, image, 1002**2, dim, emb, "lex"))
    return cases


def expected(case, op):
    if case.dimension < 0 or op == "local-dimension":
        return [case.dimension]
    return [case.dimension, case.embedding, len(case.variables)-case.embedding, int(case.embedding == case.dimension)]


def singular_ring(label, names, order, wide):
    # Singular allows setting maxExp on the construction list, before the ring exists.
    if not wide:
        return f"ring {label}=0,({','.join(names)}),{order};"
    return (f"ring base{label}=0,({','.join(names)}),{order}; list L{label}=ringlist(base{label});"
            f'attrib(L{label},"maxExp",8000000); def {label}=ring(L{label}); setring {label};')


def singular_input(case, op, method):
    chart = method == "chart"
    variables = tuple("abcd"[:len(case.variables)]) if chart else case.variables
    equations = case.chart_ideal if chart else case.ideal
    point = case.chart_point if chart else case.point
    smooth = op == "local-smoothness"
    wide = not chart and any(int(n) > 500000 for row in equations for n in re.findall(r"\^(\d+)",row))
    out = []
    out += [singular_ring("Source", variables, "dp", wide), "ideal I="+",".join(equations)+";",
            "map ev=Source,"+",".join(point)+";",
            'if(size(ev(I))){print("RESULT -1");quit;}']
    if smooth and not chart:
        # At a closed point n - #generators <= d <= n - rank.
        out += ["int n=nvars(Source); matrix deriv=jacob(I); matrix atpoint=ev(deriv);",
                "int r=rank(atpoint); int e=n-r;",
                'if(r==size(I) || r==n){print("RESULT "+string(intvec(e,e,r,1)));quit;}']
    if method == "components" and not smooth:
        out += ["matrix deriv=jacob(I); matrix atpoint=ev(deriv); int r=rank(atpoint);",
                'if(r==size(I) || r==nvars(Source)){print("RESULT "+string(nvars(Source)-r));quit;}']
    if method == "components":
        # Every closed point of an equidimensional affine variety has its dimension.
        # Embedded components cannot increase the maximum among components through p.
        # If the whole affine scheme attains the height lower bound then
        # every germ on its support has this same dimension. This avoids
        # costly decomposition for many complete intersections.
        out += ["ideal G=std(I); int d=dim(G);",
                'if(d==nvars(Source)-size(I) || d==0){print("RESULT "+string('
                + ("intvec(d,e,r,d==e)" if smooth else "intvec(d)") + '));quit;}',
                'LIB "primdec.lib";', "list parts=equidim(G); ideal C; d=-1; int i;",
                "for(i=size(parts);i>=1;i--){ C=parts[i]; if(size(ev(C))==0){d=dim(std(C));break;}}"]
    else:
        out.append(singular_ring("Local", tuple(f"t{i}" for i in range(len(variables))),
                                 "Ds" if case.order == "lex" and not wide else "ds", wide))
        shift = ",".join(f"t{i}+({q})" for i, q in enumerate(point))
        out += [f"map shift=Source,{shift}; ideal J=shift(I);", "int d=dim(std(J));"]
        if smooth and chart:
            # Independent first-infinitesimal-neighborhood colength in the supplied étale chart.
            out += ["int e=vdim(std(J+maxideal(2)))-1;", "int r=nvars(Local)-e;"]
    result = "intvec(d,e,r,d==e)" if smooth else "intvec(d)"
    out += [f'print("RESULT "+string({result}));', "quit;"]
    return "\n".join(out)+"\n"


def atlas_args(root, case, op):
    args = [str(root/"atlas"), "algebra", op, "--vars", ",".join(case.variables), "--order", case.order,
            "--compute", "cpu", "--json", "--point", ",".join(case.point)]
    for row in case.ideal:
        args += ["--gen", row]
    return args


def run(case, operation, method, args):
    atlas = method.startswith("atlas")
    if atlas:
        root = args.baseline_root if method == "atlas-baseline" else args.atlas_root
        command, source = atlas_args(root, case, operation), ""
    else:
        command, source = SINGULAR, singular_input(case, operation, method)
    def limit():
        ceiling = args.memory_mib*1024**2
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    start = perf_counter()
    env = {**os.environ, "ATLAS_GROEBNER": "native"}
    for setting in ("ATLAS_GROEBNER_GPU", "PYTHONINTMAXSTRDIGITS"):
        env.pop(setting, None)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True, preexec_fn=limit, env=env)
    try:
        stdout, stderr = process.communicate(source, timeout=max(0, args.seconds-(perf_counter()-start)))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return {"status": "timeout", "lower_bound_s": args.seconds}
    elapsed = (perf_counter()-start)*1000
    diagnostics = (stdout+" "+stderr).strip()
    if re.search(r"no more memory|out of memory|bad_alloc|cannot allocate|could not allocate|allocating.*failed",
                 diagnostics, re.IGNORECASE):
        return {"status": "allocation_failure", "diagnostic": diagnostics[-450:]}
    if atlas:
        record = json.loads(stdout)
        if record.get("status") == "error":
            message = record["error"]["message"]
            if process.returncode != 2:
                raise ValueError(f"Unexpected Atlas error exit {process.returncode}: {message}")
            if message == "Large coordinate translation." and record["error"]["code"] == "computation_failed":
                return {"status": "translation_cap", "diagnostic": message}
            if case.dimension != -1 or "point must lie on" not in message or record["error"]["code"] != "invalid_request":
                raise ValueError(f"{case.name} {operation} {method}: {diagnostics[:1000]}")
            got = [-1]
        else:
            result, execution = record["result"], record["execution"]
            if execution["requested"] != "cpu" or execution["backend"] != "cpu" or process.returncode:
                raise ValueError(record)
            got = [result["local_dimension"]]
            if operation == "local-dimension":
                assert result["type"] == "integer" and result["value"] == got[0]
            else:
                assert result["type"] == "boolean" and type(result["value"]) is bool
                got += [result["embedding_dimension"], result["jacobian_rank"], int(result["value"])]
    else:
        if "poly` ^ `bigint` failed" in diagnostics or "OVERFLOW in power" in diagnostics:
            error = next(line for line in diagnostics.splitlines() if "failed" in line or "OVERFLOW" in line)
            return {"status": "unsupported_exponent", "diagnostic": error.strip()}
        if "? OVERFLOW" in diagnostics:
            return {"status": "arithmetic_overflow", "diagnostic": diagnostics[:350]}
        if process.returncode or re.search(r"^\s*\?", stdout, re.M) or stderr:
            raise ValueError(f"{case.name} {operation} {method}: {process.returncode} {diagnostics[:1400]}")
        matches = re.findall(r"^RESULT ([-0-9,]+)\s*$", stdout, re.M)
        if len(matches) != 1:
            raise ValueError(diagnostics)
        got = [int(x) for x in matches[0].split(",")]
    if got != expected(case, operation):
        raise ValueError(f"{case.name} {operation} {method}: got {got}, expected {expected(case, operation)}")
    return {"status": "ok", "milliseconds": elapsed, "result": got}


def fingerprint(root):
    inputs = ("atlas", "native/build/libgroebner.so", "native/build/local_algebra", "native/point_geometry.cpp",
              "native/local_dimension_graph.cpp", "native/local_io.cpp", "native/local_power_coordinates.cpp",
              "engine/local_dimension.py", "native/local_cli.cpp")
    return {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in inputs if (root/f).is_file()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas-root", type=Path, default=ROOT)
    ap.add_argument("--baseline-root", type=Path, help="previous native build, optional")
    ap.add_argument("--cpu", type=int, help="pin every child to this logical CPU")
    ap.add_argument("--seconds", type=float, default=3)
    ap.add_argument("--memory-mib", type=int, default=768)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--case", action="append", choices=[c.name for c in corpus()])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.seconds <= 0 or args.memory_mib < 64 or args.repeats < 1:
        ap.error("time and repeats must be positive, memory at least 64 MiB")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    for path in (args.atlas_root, args.baseline_root):
        if path and not (path/"native/build/local_algebra").is_file():
            ap.error(f"missing native local_algebra under {path}")
    cases = [c for c in corpus() if not args.case or c.name in args.case]
    methods = (["atlas-baseline"] if args.baseline_root else []) + ["atlas", "translated", "components", "chart"]
    report = {"date_utc": datetime.now(timezone.utc).isoformat(), "seconds_per_process": args.seconds,
              "address_space_mib": args.memory_mib, "repeats": args.repeats, "affinity": sorted(os.sched_getaffinity(0)),
              "host": platform.uname().machine, "cpu": next(s.split(":",1)[1].strip() for s in
                     Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),
              "singular": subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "atlas_sha256": fingerprint(args.atlas_root),
              "baseline_sha256": fingerprint(args.baseline_root) if args.baseline_root else None,
              "command": ["python3","benchmarks/large_coupled_local.py",*sys.argv[1:]],
              "scope": "Fresh process wall time including startup, input construction and solution. Chart is a separately supplied mathematical oracle, excluded from original-input comparisons. Only first failure sampled; no successful times inferred from failures.",
              "cases": [asdict(c) for c in cases], "results": []}
    for case in cases:
        for operation in OPERATIONS:
            records = {m: [] for m in methods}
            for repeat in range(args.repeats):
                for m in methods[repeat % len(methods):]+methods[:repeat % len(methods)]:
                    if records[m] and records[m][-1]["status"] != "ok":
                        continue
                    records[m].append(run(case, operation, m, args))
            rows = {}
            for m, samples in records.items():
                rows[m] = {"samples": samples}
                if all(s["status"] == "ok" for s in samples):
                    rows[m]["median_ms"] = statistics.median(s["milliseconds"] for s in samples)
            report["results"].append({"case": case.name, "operation": operation, "methods": rows})
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(case.name, operation, *(f"{m}={data.get('median_ms',data['samples'][-1]['status'])}" for m,data in rows.items()), flush=True)


if __name__ == "__main__":
    main()
