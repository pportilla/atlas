"""Compare native Atlas and Singular from parsed QQ generators to a reduced monic basis."""
import argparse
import hashlib
import json
from math import gcd, lcm
import os
from pathlib import Path
from statistics import median
import subprocess

from groebner import line, parse, table
from groebner_hard import PROBLEMS
from groebner_singular import singular_round


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "native/build/bench_groebner_basis"
CORPUS = ROOT / "benchmarks/groebner-corpus.json"
METHODS = ("atlas", "groebner", "std", "slimgb", "std-pure")


def primitive(poly, order):
    """Give both engines the same representative that Atlas parses for an ideal generator."""
    terms = table(poly)
    if not terms:
        return []
    def monomial(e):
        return e if order == "lex" else (sum(e), tuple(-x for x in reversed(e)))
    powers = sorted(terms, key=monomial, reverse=True)
    den = lcm(*(c.denominator for c in terms.values()))
    coefficients = [int(terms[e] * den) for e in powers]
    content = gcd(*coefficients)
    if not content:
        return []
    if coefficients[0] < 0:
        content = -content
    return [[str(c // content), "1", list(e)] for e, c in zip(powers, coefficients) if c]


def atlas_round(cases, batches, timeout=240):
    request = []
    for case, batch in zip(cases, batches):
        request.extend([f"{len(case['polys'])} {batch}", f"{case['n']} {int(case['order'] == 'lex')} 0"])
        request.extend(line(p) for p in case["polys"])
    output = subprocess.run([str(DRIVER)], input="\n".join(request) + "\n", text=True,
                            capture_output=True, check=True, timeout=timeout).stdout
    rows = iter(output.splitlines())
    times, bases = [], []
    for _ in cases:
        timer, ms = next(rows).split()
        basis, size = next(rows).split()
        if (timer, basis) != ("TIME", "BASIS"):
            raise ValueError("Unexpected native driver output")
        times.append(float(ms))
        bases.append("".join(next(rows) + "\n" for _ in range(int(size))))
    if next(rows, None) is not None:
        raise ValueError("Unexpected trailing native driver output")
    return times, bases


def references(cases):
    _, texts = atlas_round(cases, [1] * len(cases))
    expected = []
    for case, text in zip(cases, texts):
        basis = parse(text)
        if "expected" in case:
            if basis != [table(p) for p in case["expected"]]:
                raise ValueError(f"Atlas disagrees with the recorded basis of {case['label']}")
            expected.append(case["expected"])
        else:
            # Each Singular method must match this candidate term for term.
            expected.append([[[str(c.numerator), str(c.denominator), list(e)] for e, c in p.items()] for p in basis])
    return texts, expected


def measure(cases, methods, batch, args):
    batches = [batch] * len(cases)
    texts, expected = references(cases)
    times = {method: [] for method in methods}
    timeouts = {}
    timeout = args.time_limit if len(cases) == 1 else 240
    for trial in range(args.repeats):
        offset = trial % len(methods)
        for method in methods[offset:] + methods[:offset]:
            if method in timeouts:
                continue
            try:
                if method == "atlas":
                    values, bases = atlas_round(cases, batches, timeout=timeout)
                    if bases != texts:
                        raise ValueError("Atlas produced inconsistent basis coefficients")
                else:
                    values = singular_round(cases, method, batches, expected, timeout=timeout)
            except subprocess.TimeoutExpired:
                if len(cases) != 1:
                    raise
                timeouts[method] = {"process_limit_seconds": timeout, "trial": trial + 1}
                print(f"  {method} timed out on {cases[0]['label']}", flush=True)
                continue
            times[method].append(values)
        print(f"  {cases[0]['label']} ({len(cases)} ideals), round {trial+1}/{args.repeats}", flush=True)
    return {"labels": [c["label"] for c in cases], "orders": [c["order"] for c in cases],
            "batch": batch, "samples_ms": times, "timeouts": timeouts}


def run(args):
    available = sorted(os.sched_getaffinity(0))
    if args.cpu == "single":
        os.sched_setaffinity(0, {available[0]})
    cpus = sorted(os.sched_getaffinity(0))
    records = json.loads(CORPUS.read_text())
    for i, case in enumerate(records):
        case["label"] = f"verifier-{i}"
    hard = [{"label": f"{label} {order}", "order": order, "n": len(names),
             "polys": [[[str(c), "1", list(e)] for e, c in p.items() if c] for p in polys]}
            for label, (names, polys), order in PROBLEMS]
    for case in records + hard:
        case["polys"] = [primitive(p, case["order"]) for p in case["polys"]]

    def fingerprint(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    lib = next(DRIVER.parent.glob("libgroebner.*"))
    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "cpu": next(s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                          if s.startswith("model name")),
              "initial_cpu_set": available, "measured_cpu_set": cpus,
              "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in
                         (DRIVER, lib, CORPUS, Path(__file__).resolve(), ROOT/"benchmarks/groebner_singular.py",
                          ROOT/"benchmarks/bench_groebner_basis.cpp", ROOT/"benchmarks/groebner_hard.py")},
              "repeats": args.repeats, "units": "milliseconds per fully reduced monic basis",
              "methods": list(METHODS), "groups": []}
    def save(group):
        report["groups"].append(group)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report) + "\n")
        for method, samples in group["samples_ms"].items():
            if method not in group["timeouts"]:
                print(f"  {method}: {median(map(sum, samples)):.3f} ms", flush=True)
    if args.suite in ("all", "corpus"):
        save(measure(records, METHODS, args.batch, args))
    if args.suite in ("all", "classical"):
        for case in hard:
            # FGLM needs a zero-dimensional ideal and changes from dp to lp.
            # Compare verified modular computation on the four grevlex problems.
            extra = "stdfglm" if case["order"] == "lex" else "modStd"
            save(measure([case], (*METHODS, extra), 1, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", choices=("single", "host"), default="single")
    parser.add_argument("--suite", choices=("all", "corpus", "classical"), default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8, help="solves per timed batch for verifier ideals")
    parser.add_argument("--time-limit", type=int, default=30, help="seconds per classical method process")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.repeats, args.batch, args.time_limit) < 1:
        parser.error("repeats, batch and time-limit must be positive")
    run(args)
