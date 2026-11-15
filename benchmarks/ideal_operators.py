"""Time ideal operators on prepared ideals over QQ and check both containments in Singular."""
import argparse
import hashlib
import io
import json
import operator
import os
import platform
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
from engine.session import new_session
from engine.session_algebra import ideal, polynomial
from engine.session_polynomial import Polynomial, coordinates
from engine.session_transcript import SessionTranscript


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


def singular_sample(declarations, calculation, batch, setup=""):
    program = declarations + f"""
ideal answer={calculation};
int i; int start=rtimer;
for(i=1;i<={batch};i++) {{ {setup} answer={calculation}; }}
print(rtimer-start);
"""
    return int(run_singular(program)) / batch


def prepared_sample(request, batch):
    result = subprocess.run([ROOT / "native/build/bench_ideal_operators", str(batch)],
                            input="\n".join(request) + "\n", text=True, capture_output=True, check=True)
    ns, *generators = result.stdout.splitlines()
    return int(ns) / (1000 * batch), generators


def cli_sample(session, line, batch):
    from unittest.mock import patch
    from engine.app import _session_loop, make_parser, shell_command_names
    from engine.session_panel import SessionPanel

    parser = make_parser()
    panel = SessionPanel(io.StringIO())
    reads = 0
    started = ended = 0

    def read(_prompt):
        nonlocal reads, started, ended
        reads += 1
        if reads == 2:
            started = perf_counter_ns()
        if reads == batch + 2:
            ended = perf_counter_ns()
            return "quit"
        return line

    with patch("builtins.input", read):
        _session_loop(parser, panel, session, shell_command_names(parser))
    return (ended - started) / (1000 * batch), session.variables["answer"]


def fresh_sample(left, right, apply, square, batch, *, include_declaration=False):
    def declared(operand):
        return ideal(operand.generators, ring=operand.parent, order=operand.order)

    if include_declaration:
        start = perf_counter_ns()
        for _ in range(batch):
            result = apply(declared(left), 2 if square else declared(right))
        return (perf_counter_ns() - start) / (1000 * batch), result

    pairs = [(declared(left), 2 if square else declared(right)) for _ in range(batch)]
    start = perf_counter_ns()
    for a, b in pairs:
        result = apply(a, b)
    return (perf_counter_ns() - start) / (1000 * batch), result


def fresh_polynomial_pairs(left, right, batch):
    """The first ideal declaration in each ring starts with already constructed polynomials."""
    terms = [[native_polynomial.terms(g.parent.generators, g._value) for g in operand.generators]
             for operand in (left, right)]
    pairs = []
    for _ in range(batch):
        parent = coordinates("x", "y")[0].parent
        groups = [tuple(Polynomial(parent, native_polynomial.from_terms(parent.generators, row)) for row in group)
                  for group in terms]
        pairs.append((parent, *groups))
    return pairs


def fresh_declaration_sample(left, right, apply, square, batch):
    pairs = fresh_polynomial_pairs(left, right, batch)
    start = perf_counter_ns()
    for parent, a, b in pairs:
        first = ideal(a, ring=parent, order=left.order)
        result = apply(first, 2 if square else ideal(b, ring=parent, order=right.order))
    return (perf_counter_ns() - start) / (1000 * batch), result


def check(declarations, calculation, results):
    # Input and output lists may differ by order, scalar multiples or redundant generators.
    source = declarations + f"ideal expected=std({calculation});\n"
    for i, result in enumerate(results):
        source += f"ideal received{i}=std({singular_ideal(result)});\n"
        source += f"print(size(reduce(received{i},expected)) + size(reduce(expected,received{i})));\n"
    if run_singular(source).splitlines() != ["0"] * len(results):
        raise AssertionError(f"Singular disagrees with Atlas for {calculation}.")


def measure(left_source, right_source, operation, batch, trials, kernel):
    left, right = inputs(left_source, right_source)
    apply, singular_expression = OPERATIONS[operation]
    request = wire(operation, left, right)
    declarations = ("ring R=0,(x,y),dp; option(redSB); short=0;\n"
                    f"ideal I={singular_ideal(left_source)}; ideal J={singular_ideal(right_source)};\n")
    operand = 2 if operation == "square" else right
    session = new_session(None)
    session.variables.update(I=left, J=right)
    transcript = SessionTranscript()
    name = {"sum": "ideal_sum(I, J)", "product": "ideal_product(I, J)", "square": "ideal_power(I, 2)"}[operation]
    line = f"answer = {name}"
    infix = {"sum": "answer = I + J", "product": "answer = I * J", "square": "answer = I^2"}[operation]

    def prompt():
        for _ in session.run(line, transcript=transcript):
            pass
        return session.variables["answer"]

    # For timings that include declaration, both systems start from the same
    # already expanded polynomials. Only the input ideals are rebuilt.
    sources = [("I", left_source)] if operation == "square" else [("I", left_source), ("J", right_source)]
    poly_declarations = ""
    assignments = []
    for label, polys in sources:
        ids = [f"p{label}{i}" for i in range(len(polys))]
        poly_declarations += "".join(f"poly {var}={poly};\n" for var, poly in zip(ids, polys))
        assignments.append(f"{label}={','.join(ids)};")
    script = "I = ideal(a, b)\n"
    if operation != "square":
        script += "J = ideal(c, d)\n"
    script += line
    session.variables.update(zip(("a", "b", "c", "d"), left.generators + right.generators))

    def declare_in_script():
        for _ in session.run(script):
            pass
        return session.variables["answer"]

    methods = ("atlas_session", "atlas_prompt", "atlas_cli", "atlas_cli_infix", "atlas_first_use", "atlas_declare_and_first_use",
               "atlas_export_record", "atlas_native_wire",
               *(("atlas_native",) if kernel else ()), "singular", "singular_declare_and_first_use",
               "atlas_first_declare_fresh_polynomials", "atlas_script_declare_and_first_use")
    samples = {name: [] for name in methods}
    for trial in range(trials):
        generators = []
        start = trial % len(methods)
        for method in methods[start:] + methods[:start]:
            if method == "atlas_session":
                elapsed, result = atlas_sample(lambda: apply(left, operand), batch)
                generators.append([str(f) for f in result.generators])
            elif method == "atlas_prompt":
                elapsed, result = atlas_sample(prompt, batch)
                generators.append([str(f) for f in result.generators])
            elif method in ("atlas_cli", "atlas_cli_infix"):
                elapsed, result = cli_sample(session, line if method == "atlas_cli" else infix, batch)
                generators.append([str(f) for f in result.generators])
            elif method in ("atlas_first_use", "atlas_declare_and_first_use"):
                elapsed, result = fresh_sample(left, right, apply, operation == "square", batch,
                                               include_declaration=method == "atlas_declare_and_first_use")
                generators.append([str(f) for f in result.generators])
            elif method == "atlas_first_declare_fresh_polynomials":
                elapsed, result = fresh_declaration_sample(left, right, apply, operation == "square", batch)
                generators.append([str(f) for f in result.generators])
            elif method == "atlas_script_declare_and_first_use":
                elapsed, result = atlas_sample(declare_in_script, batch)
                generators.append([str(f) for f in result.generators])
            elif method == "atlas_export_record":
                pending = [apply(left, operand) for _ in range(batch)]
                started = perf_counter_ns()
                for value in pending:
                    record = value.record
                elapsed = (perf_counter_ns() - started) / (1000 * batch)
                generators.append(record["result"]["generators"])
            elif method == "atlas_native_wire":
                elapsed, result = atlas_sample(lambda: affine_request(request), batch)
                generators.append(json.loads(result[0])["generators"])
            elif method == "atlas_native":
                elapsed, rows = prepared_sample(request, batch)
                generators.append(rows)
            elif method == "singular_declare_and_first_use":
                elapsed = singular_sample(declarations + poly_declarations, singular_expression,
                                          batch, setup=" ".join(assignments))
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
    cli.add_argument("--skip-kernel", action="store_true", help="Use for older builds without the prepared C++ driver.")
    args = cli.parse_args()
    if args.trials < 2 or args.batch < 1:
        cli.error("Use at least two trials and one operation per batch.")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    try:
        from engine.session_ideal_native import _compiled
        binding = "CPython stable ABI" if _compiled() is not None else "ctypes"
    except ImportError:  # Historical build before prepared handles.
        binding = "sparse wire"
    report = {
        "cpu": cpu_name(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "atlas_native_sha256": hashlib.sha256(_LIBRARY.read_bytes()).hexdigest(),
        "python": platform.python_version(), "atlas_binding": binding,
        "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
        "ring": "QQ[x,y], grevlex", "trials": args.trials, "operations_per_trial": args.batch,
        "conditions": (
            "Each trial repeats an operation on the same prepared pair. One preceding invocation per method is "
            "discarded for warm operations. Trial order rotates. Session measures one operator on typed ideals. "
            "Prompt measures a named assignment using the prompt evaluator and transcript, excluding terminal I/O. "
            "CLI and CLI infix time named and operator assignments through the interactive command loop with input in memory and the terminal "
            "panel disabled. It includes routing, multiline detection, evaluation, and transcript updates. "
            "First use takes newly declared ideals of the same initialized polynomials without a warm-up per ideal. "
            "Declare and first use includes construction of the input ideals from those polynomials. "
            "First declaration on fresh polynomials excludes creation of the separate rings and polynomial values, "
            "and includes their very first ideal construction. Script with declarations evaluates named functions "
            "and binds I, J as needed and answer through the shared script evaluator. "
            "Record measures first export of a complete record from an already computed result. Native wire includes "
            "sparse input decoding and JSON output. Prepared native measures C++ arithmetic on parsed inputs and "
            "the completed result stays in memory. Singular measures an assignment on initialized ideals in its "
            "interpreter, with one thread. Timings exclude top-level program startup, input declarations, and "
            "agreement checks. Each trial checks mutual containment in Singular. Ordinary power returns a reduced "
            "basis in Atlas, so Singular uses std(I^2) with option(redSB)."
        ), "cases": {},
    }
    if binding == "CPython stable ABI":
        module = _LIBRARY.parent / "_atlas_ideal.abi3.so"
        report["atlas_binding_sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    for name, (left, right) in CASES.items():
        report["cases"][name] = {"left": left, "right": right, "operations": {
            op: measure(left, right, op, args.batch, args.trials, not args.skip_kernel) for op in OPERATIONS}}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
