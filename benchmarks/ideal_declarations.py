"""Time ideal declarations from existing QQ polynomials, including transport to the kernel."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from platform import python_version
from statistics import median
import subprocess
import sys
from time import perf_counter_ns
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.ideal_operators import CASES, inputs, run_singular
from benchmarks.primary_decomposition import cpu_name
from engine import native_polynomial as polynomial, session_ideal_native as binding
from engine.native_library import _LIBRARY
from engine.session_algebra import Ideal, _polynomial_inputs, _unpack, ideal
from engine.session_polynomial import Polynomial, coordinates


STAGES = ("validate_and_wrap", "extract_and_check_sizes", "encode_rows_and_payload",
          "native_prepare_and_own", "bind_and_release", "instrumented_total")


def group_factory(ideals, batch, fresh):
    generators = tuple(i.generators for i in ideals)
    if not fresh:
        # Make cache hits explicit before timing. The polynomials and their live parent are shared.
        for group in generators:
            for g in group:
                polynomial.row(g.parent.generators, g._value)
        return lambda: [generators] * batch
    sparse = [[polynomial.terms(g.parent.generators, g._value) for g in group] for group in generators]

    def groups():
        # Fresh parent identity ensures every encoding misses the Python sparse-row cache.
        pairs = []
        for _ in range(batch):
            parent = coordinates("x", "y")[0].parent
            pairs.append([[Polynomial(parent, polynomial.from_terms(parent.generators, terms))
                          for terms in group] for group in sparse])
        return pairs

    return groups


def declarations(groups, thin=False):
    start = perf_counter_ns()
    for pair in groups:
        for gens in pair:
            parent = gens[0].parent
            if thin:
                # The representation used before Atlas retained native operands.
                value = Ideal(*_polynomial_inputs(_unpack((gens,)), "grevlex", None, ring=parent))
            else:
                value = ideal(gens, ring=parent, order="grevlex")
            del value
    return (perf_counter_ns() - start) / (1000 * len(groups))


def stages(groups, compiled, library):
    """Instrument the same eligible path in ideal() and NativeIdeal.prepare()."""
    totals = [0] * len(STAGES)
    for pair in groups:
        for gens in pair:
            t0 = perf_counter_ns()
            parent, polys, order = _polynomial_inputs(_unpack((gens,)), "grevlex", None, ring=gens[0].parent)
            declared = Ideal(parent, polys, order)
            t1 = perf_counter_ns()
            axes = parent.generators
            values = tuple(f._value for f in polys)
            if (len(axes) > 32 or len(polys) > 128 or sum(len(v._terms) for v in values) > 512
                    or any(c.numerator.bit_length() > 16384 or c.denominator.bit_length() > 16384
                           or any(k.bit_length() > 16384 for k in powers)
                           for v in values for powers, c in v._terms)):
                raise ValueError("Example exceeds the prepared-ideal limits.")
            t2 = perf_counter_ns()
            rows = [polynomial.row(axes, v) for v in values]
            if sum(map(len, rows)) > 4096:
                raise ValueError("Example exceeds the sparse-row budget.")
            payload = "".join(row + "\n" for row in rows).encode("ascii")
            t3 = perf_counter_ns()
            if compiled is not None:
                owner = compiled.prepare(len(axes), payload, len(rows))
            else:
                ptr = library.atlas_ideal_prepare(len(axes), payload, len(rows))
                if not ptr:
                    raise ArithmeticError(library.atlas_ideal_error().decode())
                owner = binding.NativeIdeal(library, ptr)
            t4 = perf_counter_ns()
            owner._values = values
            declared.__dict__["_native_ideal"] = owner
            del owner, declared
            t5 = perf_counter_ns()
            increments = (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)
            for i, elapsed in enumerate(increments):
                totals[i] += elapsed
    return [n / (1000 * len(groups)) for n in totals]


def singular_declarations(left, right, batch, trials):
    source = "ring R=0,(x,y),dp; option(redSB); short=0;\n"
    for letter, polys in (("I", left), ("J", right)):
        for i, poly in enumerate(polys):
            source += f"poly p{letter}{i}={poly};\n"
        source += f"ideal {letter}=p{letter}0,p{letter}1;\n"
    rows = {}
    for key, body in (("empty_loop", ""), ("one_ideal", "I=pI0,pI1;"),
                      ("both_ideals", "I=pI0,pI1; J=pJ0,pJ1;")):
        program = source + body + f"""
int i; int start=rtimer;
for(i=1;i<={batch};i++) {{ {body} }}
print(rtimer-start);
print((I[1]==pI0) and (I[2]==pI1) and (J[1]==pJ0) and (J[2]==pJ1));
"""
        times = []
        for _ in range(trials):
            ticks, match = run_singular(program).splitlines()
            if match != "1":
                raise AssertionError("Singular's declarations changed the ordered generators.")
            times.append(int(ticks) / batch)
        rows[key] = samples(times)
    return rows


def samples(times):
    return {"median_us": round(median(times), 3), "samples_us": [round(t, 3) for t in times]}


def measure(left, right, trials, batch):
    ideals = inputs(left, right)
    compiled, library = binding._compiled(), binding._api()
    if library is None:
        raise RuntimeError("Build the current native kernel before measuring declarations.")
    controls = group_factory(ideals, batch, False)
    report = {"left": left, "right": right, "atlas_wrappers_only": samples(
        [declarations(controls(), thin=True) for _ in range(trials)])}
    backends = [("ctypes", None)]
    if compiled is not None:
        backends.insert(0, ("abi3", compiled))
    for backend, owner_class in backends:
        with nullcontext() if owner_class else patch.object(binding, "_compiled", new=lambda: None):
            # Confirm expression order, parent identity and the native retained input.
            for existing in ideals:
                declared = ideal(existing.generators, ring=existing.parent, order=existing.order)
                assert declared.parent is existing.parent and declared.generators == existing.generators
                assert declared._native_ideal.size() == len(declared.generators)
            for fresh in (False, True):
                groups = group_factory(ideals, batch, fresh)
                totals, measured = [], []
                for _ in range(trials):
                    totals.append(declarations(groups()))
                    measured.append(stages(groups(), owner_class, library))
                values = dict(zip(STAGES, (samples(col) for col in zip(*measured))))
                key = "first_row_encoding" if fresh else "cached_row_encoding"
                report.setdefault(backend, {})[key] = {"public_declarations": samples(totals), "stages": values}
    report["singular"] = singular_declarations(left, right, batch, trials)
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cpu", type=int)
    cli.add_argument("--trials", type=int, default=7)
    cli.add_argument("--batch", type=int, default=256)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.trials < 2 or args.batch < 1:
        cli.error("Use at least two trials and one declaration pair per batch.")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    extension = _LIBRARY.with_name("_atlas_ideal.abi3.so")
    report = {
        "cpu": cpu_name(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": python_version(), "singular": subprocess.check_output(
            ["Singular", "--version"], text=True).splitlines()[0],
        "kernel_sha256": hashlib.sha256(_LIBRARY.read_bytes()).hexdigest(),
        "binding_sha256": hashlib.sha256(extension.read_bytes()).hexdigest() if binding._compiled() else None,
        "trials": args.trials, "pairs_per_trial": args.batch,
        "conditions": "Microseconds per pair of ideal declarations on previously expanded polynomials, including "
        "release. Python stage timings replay the eligible public path with a clock at each boundary. Actual public "
        "declarations are timed separately without that instrumentation. Each first-row trial uses independent live "
        "parents built outside the timer. Cached rows use the same live polynomials. The wrappers-only control "
        "performs validation and creates the Python ideal values without a native allocation. Singular uses a fresh "
        "single-threaded process per sample, starts its clock after creating four native polynomials and both ideals, "
        "and checks the ordered generators after the loop. Native sparse-row decoding and FLINT packing are included "
        "together in the native_prepare_and_own stage. No ideal arithmetic is timed.",
        "cases": {name: measure(*inputs_, args.trials, args.batch) for name, inputs_ in CASES.items()},
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
