"""Time ideal operators on prepared ideals over QQ and check both containments in Singular."""
import argparse
import hashlib
import json
import operator
import os
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.normalization import ideal as singular_ideal
from benchmarks.primary_decomposition import cpu_name
from engine import native_polynomial
from engine.native_library import _LIBRARY, affine_request
from engine.session_algebra import ideal, polynomial
from engine.session_polynomial import coordinates


# Fixed before timing. The rational case exercises expansion and coefficients beyond 64 bits.
CASES = {
    "plane_monomials": (("x^2", "y"), ("x", "y^2")),
    "rational_forms": (
        ("(x+2*y)^8", "(18446744073709551629/7)*x^6*y"),
        ("(2*x-y)^7", "(1/13)*x*y^5"),
    ),
}
OPERATIONS = {
    "sum": (operator.add, "I+J"),
    "product": (operator.mul, "I*J"),
    "square": (operator.pow, "std(I^2)"),
}


def inputs(left, right):
    x, y = coordinates("x", "y")

    def build(generators):
        return ideal(tuple(polynomial(text, ring=x.parent) for text in generators), ring=x.parent)

    return build(left), build(right)


def wire(operation, left, right):
    variables = left.parent.generators
    if operation == "square":
        header = ["ideal_power", "x y", left.order, "2"]
        generators = left.generators
    else:
        header = ["ideal_" + operation, "x y", left.order, f"{len(left.generators)} {len(right.generators)}"]
        generators = left.generators + right.generators
    return [*header, *(native_polynomial.row(variables, f._value) for f in generators)]


def atlas_sample(calculate, batch):
    result = calculate()
    start = perf_counter_ns()
    for _ in range(batch):
        result = calculate()
    return (perf_counter_ns() - start) / (1000 * batch), result


def run_singular(source):
    run = subprocess.run(["Singular", "-q", "--no-rc", "--threads=1", "--ticks-per-sec=1000000"],
                         input=source + "\nquit;\n", text=True, capture_output=True, check=True, timeout=60)
    if run.stderr.strip() or any(line.strip().startswith("?") for line in run.stdout.splitlines()):
        raise RuntimeError(run.stdout + run.stderr)
    return run.stdout.strip()


def singular_sample(declarations, calculation, batch):
    program = declarations + f"""
ideal answer={calculation};
int i; int start=rtimer;
for(i=1;i<={batch};i++) {{ answer={calculation}; }}
print(rtimer-start);
"""
    return int(run_singular(program)) / batch


def check(declarations, calculation, results):
    # Input and output lists may differ by order, scalar multiples or redundant generators.
    source = declarations + f"ideal expected=std({calculation});\n"
    for i, result in enumerate(results):
        source += f"ideal received{i}=std({singular_ideal(result)});\n"
        source += f"print(size(reduce(received{i},expected)) + size(reduce(expected,received{i})));\n"
    if run_singular(source).splitlines() != ["0"] * len(results):
        raise AssertionError(f"Singular disagrees with Atlas for {calculation}.")


def measure(left_source, right_source, operation, batch, trials):
    left, right = inputs(left_source, right_source)
    apply, singular_expression = OPERATIONS[operation]
    request = wire(operation, left, right)
    declarations = ("ring R=0,(x,y),dp; option(redSB); short=0;\n"
                    f"ideal I={singular_ideal(left_source)}; ideal J={singular_ideal(right_source)};\n")
    methods = ("atlas_session", "atlas_native_wire", "singular")
    samples = {name: [] for name in methods}
    for trial in range(trials):
        generators = []
        start = trial % len(methods)
        for method in methods[start:] + methods[:start]:
            if method == "atlas_session":
                operand = 2 if operation == "square" else right
                elapsed, result = atlas_sample(lambda: apply(left, operand), batch)
                generators.append([str(f) for f in result.generators])
            elif method == "atlas_native_wire":
                elapsed, result = atlas_sample(lambda: affine_request(request), batch)
                generators.append(json.loads(result[0])["generators"])
            else:
                elapsed = singular_sample(declarations, singular_expression, batch)
            samples[method].append(round(elapsed, 3))
        check(declarations, singular_expression, generators)
    return {"atlas_operation": {"sum": "I + J", "product": "I * J", "square": "I^2"}[operation],
            "singular_operation": singular_expression, "mutual_containment_verified": True,
            "median_us": {name: statistics.median(rows) for name, rows in samples.items()},
            "samples_us": samples}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--trials", type=int, default=5)
    cli.add_argument("--batch", type=int, default=16)
    cli.add_argument("--cpu", type=int)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.trials < 2 or args.batch < 1:
        cli.error("Use at least two trials and one operation per batch.")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    report = {
        "cpu": cpu_name(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "atlas_native_sha256": hashlib.sha256(_LIBRARY.read_bytes()).hexdigest(),
        "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
        "ring": "QQ[x,y], grevlex", "trials": args.trials, "operations_per_trial": args.batch,
        "conditions": (
            "Each trial repeats an operation on the same prepared pair. One preceding invocation per method is "
            "discarded. Trial order rotates. Warm session timings use the same typed operators as the Atlas prompt, "
            "including construction of their complete results. Native wire timings include sparse input decoding, "
            "computation and output serialization; rows are prepared before timing. Singular runs the indicated "
            "expression on parsed ideals on one thread. Times exclude top-level program startup, ring creation, "
            "initial input construction and verification. Each trial compares the generated ideals from both Atlas "
            "paths by mutual reduction in Singular. Ordinary power returns a reduced basis in Atlas, so Singular "
            "uses std(I^2) with option(redSB)."
        ), "cases": {},
    }
    for name, (left, right) in CASES.items():
        report["cases"][name] = {"left": left, "right": right, "operations": {
            op: measure(left, right, op, args.batch, args.trials) for op in OPERATIONS}}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
