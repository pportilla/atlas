"""Compare native QQ primary decomposition with Singular's GTZ and SY algorithms.

Run from the repository with python3 benchmarks/primary_decomposition.py.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_library import affine_request
from engine import native_polynomial as np
from engine.polynomial import format_poly, parse_expression, sparse_row


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple
    generators: tuple
    order: str = "grevlex"


def corpus():
    from benchmarks.groebner_hard import katsura
    names, equations = katsura(3)
    a, b, c = "(a+2*b)", "(b+3*a)", "(c+2*b+3)"
    h = f"({a}^2+{b}^2)"
    d, e, f = "(a+2*b+1)", "(b+a+1)", "(c+3*a+2)"
    return (
        Case("principal", ("x", "y"), ("(x-y)^4*(x^2+y^2)^2*(2*x+1)",)),
        Case("monomial_embedded", ("x", "y", "z"), ("x^4*y", "x^2*y^3", "y^2*z^3")),
        Case("monomial_primary", ("x", "y", "z"), ("x^3", "y^3", "z^2", "x*y^2", "x^2*z")),
        Case("singular_manual", ("x", "y", "z"), ("(z^2+1)*(z^3+2)^2", "y-z^2"), "lex"),
        Case("finite_quadratic", ("x", "y"), ("((x+y)^2+1)^2*((x+y)-1)",
                                              "(y-2*x)^2"), "lex"),
        Case("mixed_line_point", ("x", "y"), ("x^2*(x-1)", "x^2*(y-1)^2")),
        Case("embedded_curve", ("x", "y", "z"), ("x*y", "x*z+y^2")),
        Case("exceptional_fiber", ("t", "x", "y"),
             tuple(f"({p})*({q})" for p in ("t", "y", "x^2+1")
                   for q in ("t*x-1", "(y-t^2)^2"))),
        Case("katsura_3", tuple(names), tuple(format_poly(p, names) for p in equations)),
        # Added after the original nine-case baseline. A fixed-seed change of
        # coordinates exposed the expense of computing a lex basis of a product.
        Case("common_factor_holdout", ("a", "b", "c"),
             (f"{h}*({a}^2+2)", f"{h}*({c}-{b}^2)^2"), "lex"),
        Case("comaximal_product_holdout", ("a", "b", "c"),
             tuple(f"({p})*({q})" for p in (f"{d}^3", f"{e}^2")
                   for q in (f"{d}^2+3", f"({f}-{e}^2)^2"))),
        # Additional controls from review of the generic fiber implementation.
        Case("two_parameter_tensor", ("x", "y", "u", "v"),
             ("(u^2*x^2+v^2)^2", "(u^2*y^2+v^2)^2")),
        Case("complex_thick_fiber", ("a", "b", "c"),
             ("(a^2+1)^2", "(a*b+c)^2", "(b^2+a^2)^2")),
    )


def stress_cases():
    # Deterministic affine substitutions in products of two small ideals. This
    # separate corpus exposed items 4 and 11, now named in the main comparison.
    rng = random.Random(12)
    variables = ("a", "b", "c")
    for index in range(32):
        forms = [f"({v}+{rng.randint(1, 3)}*{rng.choice([w for w in variables if w != v])}+{rng.randrange(4)})"
                 for v in variables]
        a, b, c = forms
        left = (f"{a}^{rng.randint(1, 3)}", f"{b}^2") if index % 2 else (f"{a}^2+{b}^2",)
        right = ((f"{a}^2+{rng.randint(1, 3)}", f"({c}-{b}^2)^2") if index % 3 else (f"{c}^2",))
        yield Case(f"stress_{index}", variables, tuple(f"({p})*({q})" for p in left for q in right),
                   "grevlex" if index % 2 else "lex")


def singular(source, *, seed, timeout):
    cmd = ["Singular", "-q", "--no-rc", f"--random={seed}", "--ticks-per-sec=1000000"]
    run = subprocess.run(cmd, input=source + "\nquit;\n", cwd=ROOT, text=True,
                         capture_output=True, timeout=timeout, check=True)
    for line in (run.stdout + run.stderr).splitlines():
        if line.lstrip().startswith("?"):
            raise RuntimeError(f"Singular error: {run.stdout}\n{run.stderr}")
    return run.stdout


def declare(case):
    order = {"lex": "lp", "grevlex": "dp"}[case.order]
    return (f"ring R=0,({','.join(case.variables)}),{order};\n" +
            f"ideal inputIdeal=ideal({','.join(case.generators) or '0'});\n")


def time_singular(case, method, seed, timeout):
    # Each trial runs in a new interpreter with a discarded warm-up, so
    # loading primdec.lib and the short first-call initialization do not count.
    source = ('LIB "primdec.lib";\n' + declare(case) +
              f'list D={method}(inputIdeal); kill D;\n'
              f'int start=rtimer; list D={method}(inputIdeal); int elapsed=rtimer-start;\n'
              'print("TIME "+string(elapsed));')
    output = singular(source, seed=seed, timeout=timeout)
    micros, = [int(s[5:]) for s in output.splitlines() if s.startswith("TIME ")]
    return micros / 1000


def payload(case):
    names = list(case.variables)
    coords = tuple(map(np.Symbol, names))
    rows = [np.row(coords, np.from_terms(coords, parse_expression(f, names))) for f in case.generators]
    return ["primary_decomposition", " ".join(names), case.order, *rows]


def partition(lines, names):
    lines = iter(lines)
    count = int(next(lines))
    pairs = []
    for _ in range(count):
        nq, np_ = map(int, next(lines).split())
        pairs.append({kind: [format_poly(sparse_row(next(lines)), names) for _ in range(n)]
                      for kind, n in (("primary", nq), ("prime", np_))})
    if next(lines, None) is not None:
        raise ValueError("Extra decomposition data")
    return pairs


def time_native(data):
    affine_request(data)
    start = perf_counter_ns()
    result = affine_request(data)
    return (perf_counter_ns() - start) / 1e6, result


def compare_in_singular(case, components, timeout=60, method="primdecGTZ"):
    # Embedded primary ideals are nonunique. Check primaryness and the actual
    # intersection independently, in addition to matching the associated primes.
    if method not in ("primdecGTZ", "primdecSY"):
        raise ValueError("Unknown Singular decomposition procedure")
    source = ['LIB "primdec.lib";', declare(case),
              'proc sameIdeal(ideal A, ideal B){return(size(reduce(A,std(B)))==0 && size(reduce(B,std(A)))==0);}',
              f"proc decompose(ideal I) {{return({method}(I));}}",
              "list qs; list ps;"]
    for i, component in enumerate(components, 1):
        for key, var in (("primary", "qs"), ("prime", "ps")):
            source.append(f"{var}[{i}]=ideal({','.join(component[key]) or '0'});")
    source.append('''
    ideal allQ=1;
    int i,j,found;
    list ref=decompose(inputIdeal);
    if (sameIdeal(inputIdeal,ideal(1))) {ref=list();}
    int ok=size(ref)==size(qs);
    list checkQ; ideal rest;
    for(i=1; i<=size(qs); i++) {
      checkQ=decompose(qs[i]);
      if(size(checkQ)!=1) {ok=0;}
      else {
        if (!sameIdeal(checkQ[1][2],ps[i])) {ok=0;}
      }
      found=0;
      for(j=1;j<=size(ref);j++) {
        if(sameIdeal(ref[j][2],ps[i])) {found++;}
      }
      if(found!=1) {ok=0;}
      rest=1;
      for(j=1;j<=size(qs);j++) {
        if(j!=i) {
          rest=intersect(rest,qs[j]);
          if(sameIdeal(ps[i],ps[j])) {ok=0;}
        }
      }
      if(sameIdeal(rest,inputIdeal)) {ok=0;}
      allQ=intersect(allQ,qs[i]);
    }
    if(!sameIdeal(allQ,inputIdeal)) {ok=0;}
    print("VERIFIED "+string(ok));''')
    answer = singular("\n".join(source), seed=51, timeout=timeout)
    if "VERIFIED 1" not in answer.splitlines():
        raise ValueError(f"Singular rejected {case.name}: {components}\n{answer}")


def summary(samples):
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def comparison(samples):
    result = {method: summary(times) for method, times in samples.items()}
    if "atlas" in samples:
        reference = min(result[m]["median_ms"] for m in ("primdecGTZ", "primdecSY"))
        result["speedup_vs_faster_reference"] = reference / result["atlas"]["median_ms"]
        paired = [min(gtz, sy) / atlas for atlas, gtz, sy in
                  zip(samples["atlas"], samples["primdecGTZ"], samples["primdecSY"])]
        result["wins_vs_both"] = sum(ratio > 1 for ratio in paired)
        result["worst_paired_speedup"] = min(paired)
    return result


def cpu_name():
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def native_binaries():
    from engine.native_library import _LIBRARY
    files = {"atlas_library_sha256": _LIBRARY, "atlas_launcher_sha256": ROOT / "atlas"}
    cli = ROOT / "native/build/primary_decomposition"
    if cli.exists():
        files["atlas_native_cli_sha256"] = cli
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}


def timed_native_subprocess(case, timeout):
    # A separate process lets the OS enforce a cutoff even inside a native
    # computation. This timing includes Python startup, parsing, and transport.
    cmd = [sys.executable, "-c", (
        "import json, sys; from benchmarks.primary_decomposition import Case, affine_request, payload; "
        "case = Case(**json.load(sys.stdin)); print(affine_request(payload(case))[0])"
    )]
    start = perf_counter_ns()
    try:
        run = subprocess.run(cmd, input=json.dumps(asdict(case)), cwd=ROOT, capture_output=True,
                             text=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_ms": (perf_counter_ns() - start) / 1e6}
    return {"status": "completed", "elapsed_ms": (perf_counter_ns() - start) / 1e6,
            "components": int(run.stdout)}


def cold_sample(case, method, seed, timeout, python_cli=False):
    if method == "atlas":
        cmd = [str(ROOT / "atlas"), "algebra", "primary-decomposition", "--vars", ",".join(case.variables),
               "--order", case.order, "--compute", "cpu", "--json"]
        if python_cli:
            cmd.insert(0, sys.executable)
        for generator in case.generators:
            cmd += ["--gen", generator]
        source = None
    else:
        cmd = ["Singular", "-q", "--no-rc", f"--random={seed}"]
        source = ('LIB "primdec.lib";\n' + declare(case) +
                  f'list D={method}(inputIdeal); print(D);\nquit;\n')
    start = perf_counter_ns()
    run = subprocess.run(cmd, input=source, cwd=ROOT, text=True, capture_output=True,
                         timeout=timeout, check=True)
    elapsed = (perf_counter_ns() - start) / 1e6
    if method == "atlas":
        if json.loads(run.stdout)["status"] != "ok":
            raise ValueError(run.stdout)
    elif not run.stdout.strip() or any(line.lstrip().startswith("?") for line in run.stdout.splitlines()):
        raise ValueError(run.stdout)
    return elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--cold-repeats", type=int, default=0, help="separate fresh CLI and interpreter timings with full output")
    ap.add_argument("--python-cli", action="store_true", help="run the cold Atlas trials through python3 atlas")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--cpu", type=int, help="pin both engines to this CPU on Linux")
    ap.add_argument("--singular-only", action="store_true", help="measure the reference implementations alone")
    ap.add_argument("--native-probe", action="store_true",
                    help="one native subprocess per case with a hard timeout, including startup")
    ap.add_argument("--stress-probe", action="store_true",
                    help="fresh Atlas, Singular GTZ and SY commands on 32 seeded affine examples")
    ap.add_argument("--timeout", type=float, default=90, help="seconds per subprocess")
    ap.add_argument("--case", action="append", help="select specific named cases for profiling")
    args = ap.parse_args()
    if args.repeats < 1 or args.cold_repeats < 0 or args.timeout <= 0:
        ap.error("positive repeats and timeout required")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    cases = tuple(stress_cases()) if args.stress_probe else corpus()
    if args.case:
        missing = set(args.case) - {case.name for case in cases}
        if missing:
            ap.error("unknown cases: " + ", ".join(sorted(missing)))
        cases = tuple(case for case in cases if case.name in args.case)
    if args.native_probe:
        report = {"cpu": cpu_name(), "cpu_affinity": args.cpu, "timeout_seconds": args.timeout,
                  **native_binaries(), "cases": []}
        for case in cases:
            result = timed_native_subprocess(case, args.timeout)
            report["cases"].append(dict(asdict(case), **result))
            print(case.name, result, flush=True)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    version = subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0]
    if args.stress_probe:
        report = {"cpu": cpu_name(), "cpu_affinity": args.cpu, "singular": version, "repeats": args.repeats,
                  **native_binaries(), "seed": 12, "timeout_seconds": args.timeout,
                  "scope": "Fresh Atlas CLI, Singular primdecGTZ and primdecSY including parsing and full output. "
                           "Engine order rotates each trial. Atlas answers independently verified by SY.",
                  "cases": []}
        for case in cases:
            samples = {"atlas": [], "primdecGTZ": [], "primdecSY": []}
            for trial in range(args.repeats):
                methods = list(samples)
                for method in methods[trial % 3:] + methods[:trial % 3]:
                    try:
                        elapsed = cold_sample(case, method, 17 + trial, args.timeout)
                    except subprocess.TimeoutExpired:
                        elapsed = None
                    samples[method].append(elapsed)
            if any(t is None for t in samples["atlas"]):
                raise RuntimeError(f"Atlas timed out on {case.name}")
            components = partition(affine_request(payload(case)), case.variables)
            compare_in_singular(case, components, args.timeout, "primdecSY")
            record = asdict(case)
            for method, times in samples.items():
                completed = [t for t in times if t is not None]
                record[method] = {"samples_ms": times, "timeouts": len(times) - len(completed),
                                  "median_ms": statistics.median(times) if len(completed) == len(times) else None}
            record["wins_vs_both"] = sum(all(a < (s if s is not None else 1000 * args.timeout) for s in (g, y))
                                         for a, g, y in zip(samples["atlas"], samples["primdecGTZ"], samples["primdecSY"]))
            record["verified_components"] = len(components)
            report["cases"].append(record)
            print(case.name, record["atlas"]["median_ms"], record["primdecSY"]["median_ms"], flush=True)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    report = {"cpu": cpu_name(), "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "singular": version, "repeats": args.repeats, "units": "wall-clock milliseconds",
              "scope": "Warm Atlas affine_request including input row decoding and result serialization, versus warmed Singular primdecGTZ and primdecSY in a loaded ring, excluding source parsing, output printing and process startup. One discarded warm-up before each sample. No cached answers. Engine order rotates on each trial. Singular seeds 17 + trial number.",
              "cases": []}
    if args.cold_repeats:
        report["cold_scope"] = ("Fresh atlas CLI/worker returning full JSON, versus fresh Singular interpreter "
                                "including primdec.lib loading, calculation and printing its complete list. "
                                "File caches warm; full output captured in both cases. Output formats differ.")
        report["cold_repeats"] = args.cold_repeats
        report["atlas_command"] = "python3 atlas" if args.python_cli else "./atlas"
    if not args.singular_only:
        report.update(native_binaries())
    methods = ["primdecGTZ", "primdecSY"] if args.singular_only else ["atlas", "primdecGTZ", "primdecSY"]
    for case in cases:
        samples = {method: [] for method in methods}
        result = None
        data = payload(case) if not args.singular_only else None
        for trial in range(args.repeats):
            for method in methods[trial%len(methods):] + methods[:trial%len(methods)]:
                if method == "atlas":
                    elapsed, result = time_native(data)
                else:
                    elapsed = time_singular(case, method, 17+trial, args.timeout)
                samples[method].append(elapsed)
        record = dict(asdict(case), **comparison(samples))
        if result is not None:
            components = partition(result, case.variables)
            compare_in_singular(case, components, args.timeout, "primdecSY")
            record["verified_components"] = len(components)
            record["result_sha256"] = hashlib.sha256("\n".join(result).encode()).hexdigest()
        if args.cold_repeats:
            cold = {method: [] for method in methods}
            for trial in range(args.cold_repeats):
                for method in methods[trial%len(methods):] + methods[:trial%len(methods)]:
                    cold[method].append(cold_sample(case, method, 17+trial, args.timeout, args.python_cli))
            record["fresh_command"] = comparison(cold)
        report["cases"].append(record)
        print(case.name + "  " + "  ".join(f"{m} {statistics.median(t):.3f} ms" for m,t in samples.items()),flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
