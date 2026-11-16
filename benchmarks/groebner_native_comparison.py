"""Time the native Atlas and libSingular solvers on identical parsed inputs."""
import argparse
from fractions import Fraction
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess

from groebner import line, table
from groebner_comparison import CORPUS, DRIVER, PROBLEMS, ROOT, primitive
from groebner_singular import singular_round

DRIVERS = {"atlas": DRIVER, "libSingular-std-pure": DRIVER.with_name("bench_singular_basis")}


def case_groups():
    records = json.loads(CORPUS.read_text())
    for i, case in enumerate(records):
        case["label"] = f"verifier-{i}"
    groups = [records]
    for label, (names, polys), order in PROBLEMS:
        groups.append([{"label": f"{label} {order}", "order": order, "n": len(names),
                        "polys": [[[str(c), "1", list(e)] for e, c in p.items() if c] for p in polys]}])
    for cases in groups:
        for case in cases:
            case["polys"] = [primitive(p, case["order"]) for p in case["polys"]]
    return groups


def request_text(cases, batch):
    rows = []
    for case in cases:
        rows.extend([f"{len(case['polys'])} {batch}", f"{case['n']} {int(case['order'] == 'lex')} 0"])
        rows.extend(line(p) for p in case["polys"])
    return "\n".join(rows) + "\n"


def native_round(driver, request, cases, timeout):
    command = [str(driver)]
    if driver == DRIVERS["libSingular-std-pure"]:
        singular = shutil.which("Singular")
        if singular is None:
            raise RuntimeError("Singular must be on PATH for libSingular resource initialization")
        command.append(singular)
    result = subprocess.run(command, input=request, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{driver.name} failed: {result.stderr}")
    rows = iter(result.stdout.splitlines())
    times, bases = [], []
    for case in cases:
        timer, ms = next(rows).split()
        basis, count = next(rows).split()
        ms, count = float(ms), int(count)
        if timer != "TIME" or basis != "BASIS" or not isfinite(ms) or ms < 0 or count < 0:
            raise ValueError(f"Invalid native timing or basis header for {case['label']}")
        times.append(ms)
        bases.append(tuple(next(rows) for _ in range(count)))
    if next(rows, None) is not None:
        raise ValueError(f"Trailing output from {driver.name}")
    return times, bases


def canonical(case, polynomials):
    """Validate monicity in the requested order, then compare independently of storage order."""
    basis = []
    for terms in polynomials:
        if not terms or any(len(e) != case["n"] or min(e) < 0 or not c for e, c in terms.items()):
            raise ValueError(f"Malformed polynomial in the basis for {case['label']}")
        if case["order"] == "lex":
            leading = max(terms)
        else:
            leading = max(terms, key=lambda e: (sum(e), tuple(-x for x in reversed(e))))
        if terms[leading] != 1:
            raise ValueError(f"Nonmonic basis for {case['label']}")
        basis.append(tuple(sorted(terms.items())))
    return tuple(sorted(basis))


def canonical_output(case, rows):
    polys = []
    for row in rows:
        terms = {}
        for text in row.split(";"):
            coefficient, *exponents = text.split()
            e = tuple(map(int, exponents))
            if e in terms:
                raise ValueError(f"Repeated monomial for {case['label']}")
            terms[e] = Fraction(coefficient)
        polys.append(terms)
    return canonical(case, polys)


def measure(cases, args):
    corpus = len(cases) > 1
    batch, timeout = (args.batch, 240) if corpus else (1, args.time_limit)
    request = request_text(cases, batch)
    expected = [canonical(c, map(table, c["expected"])) if "expected" in c else None for c in cases]
    cached = {name: [None] * len(cases) for name in DRIVERS}
    samples = {name: [] for name in DRIVERS}
    timeouts = {}
    methods = list(DRIVERS)
    for trial in range(args.repeats):
        for method in methods if trial % 2 == 0 else reversed(methods):
            if method in timeouts:
                continue
            try:
                times, outputs = native_round(DRIVERS[method], request, cases, timeout)
            except subprocess.TimeoutExpired:
                if corpus:
                    raise
                timeouts[method] = {"process_limit_seconds": timeout, "trial": trial + 1}
                print(f"{method} timed out on {cases[0]['label']}", flush=True)
                continue
            for i, (case, rows) in enumerate(zip(cases, outputs)):
                if rows != cached[method][i]:
                    current = canonical_output(case, rows)
                    if expected[i] is None:
                        expected[i] = current
                    if expected[i] != current:
                        raise ValueError(f"{method} disagrees on {case['label']}")
                    cached[method][i] = rows
            samples[method].append(times)
        print(f"{cases[0]['label']} ({len(cases)} ideals), round {trial+1}/{args.repeats}", flush=True)

    # A per-method timeout must not prevent measuring the other solver. Where
    # only one solver finished a classical ideal, verify its returned basis
    # against an independent Singular method, outside both benchmark timers.
    if not corpus and expected[0] is not None and any(not trials for trials in samples.values()):
        rows = [[[str(c.numerator), str(c.denominator), list(e)] for e, c in p] for p in expected[0]]
        singular_round(cases, "std", [1], [rows], timeout=timeout)
    for method, trials in samples.items():
        if method not in timeouts:
            print(f"  {method}: {median(map(sum, trials)):.3f} ms", flush=True)
    return {"labels": [c["label"] for c in cases], "orders": [c["order"] for c in cases],
            "input_sha256": hashlib.sha256(request.encode()).hexdigest(),
            "batch": batch, "samples_ms": samples, "timeouts": timeouts}


def fingerprints():
    paths = {CORPUS, Path(__file__).resolve(), ROOT / "benchmarks/groebner_comparison.py",
             ROOT / "benchmarks/groebner_hard.py", *DRIVERS.values()}
    for path in DRIVERS.values():
        paths.add(ROOT / "benchmarks" / (path.name + ".cpp"))
        # Record the exact native kernels that the timed binaries resolve.
        linked = subprocess.check_output(["ldd", str(path)], text=True)
        for row in linked.splitlines():
            fields = row.split()
            if len(fields) >= 3 and fields[1] == "=>" and fields[0].startswith(("libgroebner.", "libsingular-")):
                paths.add(Path(fields[2]).resolve())
    return {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def run(args):
    available = sorted(os.sched_getaffinity(0))
    if args.cpu == "single":
        os.sched_setaffinity(0, {available[0]})
    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "cpu": next(s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                          if s.startswith("model name")),
              "initial_cpu_set": available, "measured_cpu_set": sorted(os.sched_getaffinity(0)),
              "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "sha256": fingerprints(), "repeats": args.repeats, "methods": list(DRIVERS),
              "units": "milliseconds per fully reduced monic basis", "groups": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    groups = case_groups()
    if args.suite == "classical":
        groups = groups[1:]
    elif args.suite == "corpus":
        groups = groups[:1]
    for cases in groups:
        report["groups"].append(measure(cases, args))
        args.output.write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", choices=("single", "host"), default="single")
    parser.add_argument("--suite", choices=("all", "corpus", "classical"), default="all")
    parser.add_argument("--repeats", type=int, default=6, help="alternating paired trials (even for balanced order)")
    parser.add_argument("--batch", type=int, default=8, help="bases per timed verifier batch")
    parser.add_argument("--time-limit", type=int, default=30, help="whole classical process timeout (seconds)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 5 or min(args.batch, args.time_limit) < 1:
        parser.error("use at least five paired trials, a positive batch and a positive timeout")
    run(args)
