"""Time Liao–Fateman multivariate GCDs over QQ and check their known monic answers.

From the repository root, with the Atlas build environment active:
    make -C native -f groebner.mk build/bench_polynomial_gcd
    python3 benchmarks/polynomial_gcd.py --output gcd.json
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.primary_decomposition import cpu_name
from engine import fork_server
from engine.native_library import _LIBRARY
from engine.session_algebra import gcd, polynomial
from engine.session_polynomial import coordinates
from tests.test_polynomial_gcd import replay_workload

KERNEL = ROOT / "native/build/bench_polynomial_gcd"
METHODS = ("atlas_session", "flint_kernel", "singular")
EXTRA_METHODS = ("atlas_session_first_use", "singular_first_use", "atlas_session_batch")
SOURCE = "https://people.eecs.berkeley.edu/~fateman/282/readings/liao.pdf#page=5"


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple
    left: str
    right: str
    expected: str


def liao_fateman(width):
    # The paper uses x,y1,...,yv, so width = v+1.
    names = ("x", *(f"y{i}" for i in range(1, width)))
    tail = "+".join(names[1:])
    total = "+".join(names)
    squares = "+".join(f"{v}^2" for v in names)
    yield Case("LF1 coprime", names, f"({total}+1)*({total}+2)",
               f"({squares}+1)*(-3*y1*x^2+y1^2-1)", "1")
    h = f"({total}+1)^2"
    yield Case("LF2 dense quartic", names, f"{h}*(x-({tail})-2)^2", f"{h}*({total}+2)^2", h)
    powers = "+".join(f"{v}^{width}" for v in names)
    h = f"({powers}+1)"
    yield Case("LF3 sparse", names, f"{h}*({powers}-2)", f"{h}*({powers}+2)", h)
    plus1, minus2, plus2 = ("*".join(f"({v}{shift})" for v in names) for shift in ("+1", "-2", "+2"))
    h = f"({plus1}-3)"
    yield Case("LF5 fully dense", names, f"{h}*({minus2}+3)", f"{h}*({plus2}-3)", h)


def cases():
    for width in (3, 6):
        yield from liao_fateman(width)
    yield next(case for case in liao_fateman(8) if case.name == "LF5 fully dense")
    yield Case("Atlas 401-bit coefficient", ("x", "y", "z"), *replay_workload())


def kernel_sample(case, count):
    data = "\n".join((" ".join(case.variables), case.left, case.right, case.expected, ""))
    out = subprocess.check_output([KERNEL, str(count)], input=data, text=True, timeout=120)
    return json.loads(out)


def singular_sample(case, count, seed, *, first_use=False):
    calculation = "d=gcd(f,g); d=d/leadcoef(d);"
    setup = "poly d;"
    if first_use:
        if count != 1:
            raise ValueError("First-use timing requires exactly one call.")
        timed = calculation
    else:
        setup += calculation
        timed = f"for(i=1;i<={count};i++) {{ {calculation} }}"
    src = f"""
ring R=0,({','.join(case.variables)}),dp;
poly f={case.left};
poly g={case.right};
poly expected={case.expected};
{setup}
int i;
int start=rtimer;
{timed}
int elapsed=rtimer-start;
print(elapsed);
print(d==expected);
print(size(f)); print(size(g)); print(size(d));
quit;
"""
    command = ["Singular", "-q", "--no-rc", "--threads=1", "--ticks-per-sec=1000000", f"--random={seed}"]
    run = subprocess.run(command, input=src, text=True, capture_output=True, check=True, timeout=120)
    rows = run.stdout.splitlines()
    if run.stderr or len(rows) != 5 or rows[1] != "1":
        raise RuntimeError(f"Singular failed for {case.name}: {run.stdout} {run.stderr}")
    return 1000 * int(rows[0]) / count, list(map(int, rows[2:]))


def session_operands(case):
    ring = coordinates(*case.variables)[0].parent
    return tuple(polynomial(source, ring=ring) for source in (case.left, case.right, case.expected))


def check_session_result(result, f, expected):
    if result.parent is not f.parent or result != expected:
        raise AssertionError("Atlas returned an unexpected polynomial or parent")


def session_sample(operands):
    f, g, expected = operands
    start = perf_counter_ns()
    result = gcd(f, g, order="grevlex", compute="cpu")
    elapsed = perf_counter_ns() - start
    check_session_result(result, f, expected)
    return elapsed


def session_batch_sample(operands, count):
    f, g, expected = operands
    start = perf_counter_ns()
    for _ in range(count):
        result = gcd(f, g, order="grevlex", compute="cpu")
    elapsed = perf_counter_ns() - start
    check_session_result(result, f, expected)
    return elapsed / count


def batch_size(pilot_ns):
    # Target roughly 25 ms per timed batch.
    return min(4096, max(1, math.ceil(25_000_000 / max(1, pilot_ns))))


def measure(case, trials):
    operands = session_operands(case)
    session_sample(operands)
    pilot = kernel_sample(case, 1)
    singular_ns, terms = singular_sample(case, 1, 1)
    if terms != pilot["terms"]:
        raise AssertionError(f"Different term counts for {case.name}")
    batches = dict(atlas_session=1, flint_kernel=batch_size(pilot["ns_per_gcd"]), singular=batch_size(singular_ns))
    samples = {method: [] for method in METHODS + EXTRA_METHODS}
    for trial in range(trials):
        offset = trial % len(METHODS)
        for method in METHODS[offset:] + METHODS[:offset]:
            if method == "atlas_session":
                ns = session_sample(operands)
            elif method == "flint_kernel":
                ns = kernel_sample(case, batches[method])["ns_per_gcd"]
            else:
                ns, _ = singular_sample(case, batches[method], trial + 2)
            samples[method].append(ns / 1_000_000)
    # Keep the original three-way trial order and single-call measurement above.
    batches.update(atlas_session_first_use=1, singular_first_use=1,
                   atlas_session_batch=batch_size(session_batch_sample(operands, 3)))
    for trial in range(trials):
        for method in EXTRA_METHODS if trial % 2 == 0 else reversed(EXTRA_METHODS):
            if method == "atlas_session_first_use":
                ns = session_sample(session_operands(case))
            elif method == "singular_first_use":
                ns, _ = singular_sample(case, 1, trial + 2, first_use=True)
            else:
                ns = session_batch_sample(operands, batches[method])
            samples[method].append(ns / 1_000_000)
    return {**asdict(case), "terms_left_right_gcd": terms, "calls_per_trial": batches,
            "samples_ms_per_gcd": samples, "median_ms_per_gcd": {k: median(v) for k, v in samples.items()},
            "exact_agreement": True}, pilot["flint_version"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("Use at least two measured trials.")
    if not KERNEL.is_file():
        parser.error("Build the helper with make -C native -f groebner.mk build/bench_polynomial_gcd")
    report = {
        "source": SOURCE, "started_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_revision": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip(),
        "native_library_sha256": hashlib.sha256(_LIBRARY.read_bytes()).hexdigest(),
        "cpu": cpu_name(), "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "system": platform.platform(), "python": platform.python_version(),
        "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[:3],
        "trials": args.trials, "field": "QQ", "order": "grevlex (dp)", "conditions": {
            "atlas_session": "Public gcd on expanded session polynomials, CPU, with the prompt's fork server "
                             "enabled. One call per timed trial after one warmup on the same operands. "
                             "Timing ends when the exact session polynomial is returned.",
            "atlas_session_first_use": "First public CPU gcd on independently constructed, expanded operands in "
                                       "a fresh parent for each trial. Includes any first-use GCD preparation.",
            "atlas_session_batch": "Repeated public CPU gcd calls on the warmed operands. Each iteration computes "
                                   "a new result. Three pilot calls calibrate a batch targeting 25 ms.",
            "flint_kernel": "Generic FLINT gcd and monic normalization on prebuilt inputs (historical baseline). "
                            "Fresh helper per batch, one untimed warmup, steady-clock wall time.",
            "singular": "Prepared poly objects, gcd and division by leadcoef inside Singular. One thread, no rc, "
                        "fresh interpreter per batch, one untimed warmup, wall time via rtimer at 1000000 ticks/s.",
            "singular_first_use": "One gcd and monic normalization on prebuilt poly objects in a fresh Singular "
                                  "interpreter, before any other gcd. Wall time at microsecond resolution.",
            "sampling": "Sequential, rotating the original three methods, then alternating the order of the extra "
                        "measurements. Median of trial means. Input construction, warmup, and exact checks "
                        "excluded. Singular seeds 2 through trials+1, seed 1 for pilot.",
            "agreement": "Each method checks exact equality with the specified monic polynomial outside timing, "
                         "on every trial. The pilot also compares expanded term counts."
        }, "cases": []}
    with fork_server.serving(prepare_now=True):
        for case in cases():
            print(f"Running {case.name}, {len(case.variables)} variables", file=sys.stderr, flush=True)
            row, report["flint_version"] = measure(case, args.trials)
            report["cases"].append(row)
    data = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
