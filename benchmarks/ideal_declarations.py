"""Time ideal declarations with existing polynomials, then include creating the polynomials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from platform import python_version
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.ideal_operators import CASES, fresh_polynomial_pairs, inputs, run_singular
from benchmarks.primary_decomposition import cpu_name
from engine import native_polynomial
from engine.native_library import _LIBRARY
from engine.session import new_session
from engine.session_algebra import ideal
from engine.session_ideal_native import _compiled
from engine.session_polynomial import coordinates


def samples(times):
    return {"median_us": round(median(times), 3), "samples_us": [round(t, 3) for t in times]}


def declarations(pairs, explicit):
    start = perf_counter_ns()
    for parent, first, second in pairs:
        if explicit:
            a = ideal(first, ring=parent, order="grevlex")
            b = ideal(second, ring=parent, order="grevlex")
        else:
            a = ideal(*first)
            b = ideal(*second)
        del a, b
    return (perf_counter_ns() - start) / (1000 * len(pairs))


def create_polynomials(source, batch):
    # Each input ring exists before timing and has never seen these expressions.
    sessions = []
    for _ in range(batch):
        session = new_session(None)
        x, y = coordinates("x", "y")
        session.variables.update(x=x, y=y)
        sessions.append(session)
    start = perf_counter_ns()
    for session in sessions:
        for _ in session.run(source):
            pass
    elapsed = (perf_counter_ns() - start) / (1000 * batch)
    return elapsed, session.variables


def singular_samples(source, program, check, batch, trials):
    loop = source + f"""
int i;
for(i=1;i<={batch};i++) {{ {program} }}
int start=rtimer;
for(i=1;i<={batch};i++) {{ {program} }}
print(rtimer-start);
print({check});
"""
    elapsed = []
    for _ in range(trials):
        ticks, match = run_singular(loop).splitlines()
        if match != "1":
            raise AssertionError("Singular changed a generator.")
        elapsed.append(int(ticks) / batch)
    return samples(elapsed)


def measure(left, right, trials, batch, creation_batch):
    a, b = inputs(left, right)
    parent = a.parent
    names = ("pI0", "pI1", "pJ0", "pJ1")
    sources = (*left, *right)
    script = "\n".join(f"{v} = {expr}" for v, expr in zip(names, sources))
    construction = "I = ideal(pI0, pI1)\nJ = ideal(pJ0, pJ1)"
    reference = [str(p) for p in a.generators + b.generators]
    equal_source = " and ".join(f"(({observed})==({expected}))" for observed, expected in zip(reference, sources))
    if run_singular(f"ring R=0,(x,y),dp; print({equal_source});") != "1":
        raise AssertionError("Atlas and Singular disagree on the ordered generators.")
    result = {"left": left, "right": right, "ordered_generators_verified_in_singular": True}
    for fresh in (False, True):
        for explicit in (False, True):
            times = []
            for trial in range(trials + 1):
                pairs = (fresh_polynomial_pairs(a, b, batch) if fresh
                         else [(parent, a.generators, b.generators)] * batch)
                elapsed = declarations(pairs, explicit)
                if trial:
                    times.append(elapsed)
                ring, first, second = pairs[-1]
                for gens in (first, second):
                    check = ideal(gens, ring=ring, order="grevlex") if explicit else ideal(*gens)
                    if check.parent is not ring or check.generators != gens:
                        raise AssertionError("Atlas changed the parent or the ordered polynomial generators.")
                    rows = check._native_ideal.rows()
                    native_values = [native_polynomial.from_row(ring.generators, row) for row in rows]
                    if native_values != [g._value for g in gens]:
                        raise AssertionError("The native ideal changed the ordered polynomial generators.")
            key = "first_declarations" if fresh else "repeated_declarations"
            result.setdefault(key, {})["explicit" if explicit else "positional"] = samples(times)
    for key, program in (("create_polynomials", script), ("create_polynomials_and_ideals", script + "\n" + construction)):
        times = []
        for trial in range(trials + 1):
            elapsed, bound = create_polynomials(program, creation_batch)
            if trial:
                times.append(elapsed)
            if [str(bound[name]) for name in names] != reference:
                raise AssertionError("Atlas polynomial construction disagrees with the reference input.")
            if key == "create_polynomials_and_ideals":
                packed = bound["I"].generators + bound["J"].generators
                if [str(f) for f in packed] != reference or any(f.parent is not bound["x"].parent for f in packed):
                    raise AssertionError("Atlas ideal construction changed the generators or their parent.")
        result[key] = samples(times)

    # Both interpreters evaluate the same four expressions in existing QQ[x,y] rings.
    literal = "".join(f"poly {v}={expr};" for v, expr in zip(names, sources))
    assign = "".join(f"{v}={expr};" for v, expr in zip(names, sources))
    init = "ring R=0,(x,y),dp; option(redSB); short=0;\n" + literal + "ideal I=pI0,pI1; ideal J=pJ0,pJ1;\n"
    declare = "I=pI0,pI1; J=pJ0,pJ1;"
    equal = " and ".join(f"({v}==({expr}))" for v, expr in zip(names, sources))
    equal += " and (I[1]==pI0) and (I[2]==pI1) and (J[1]==pJ0) and (J[2]==pJ1)"
    result["singular"] = {
        key: singular_samples(init, program, equal, count, trials) for key, program, count in (
            ("empty_loop", "", batch), ("one_ideal", "I=pI0,pI1;", batch),
            ("both_ideals", declare, batch), ("create_polynomials", assign, creation_batch),
            ("create_polynomials_and_ideals", assign + declare, creation_batch))}
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cpu", type=int)
    cli.add_argument("--trials", type=int, default=7)
    cli.add_argument("--batch", type=int, default=256)
    cli.add_argument("--creation-batch", type=int, default=32)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.trials < 2 or min(args.batch, args.creation_batch) < 1:
        cli.error("Use at least two trials and one declaration pair per batch.")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    extension = _LIBRARY.with_name("_atlas_ideal.abi3.so")
    report = {
        "cpu": cpu_name(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": python_version(), "singular": subprocess.check_output(
            ["Singular", "--version"], text=True).splitlines()[0],
        "kernel_sha256": hashlib.sha256(_LIBRARY.read_bytes()).hexdigest(),
        "binding_sha256": hashlib.sha256(extension.read_bytes()).hexdigest() if _compiled() else None,
        "trials": args.trials, "pairs_per_trial": args.batch, "creation_pairs_per_trial": args.creation_batch,
        "conditions": "Microseconds per pair of ideals. The positional call is ideal(f,g). The explicit call also "
        "supplies ring and order. Repeated declarations use the same live polynomials. For first declarations, each "
        "pair of ideals is the first ever constructed from its polynomials, in a fresh parent. Both conditions "
        "exclude ring creation and polynomial construction. Creation samples evaluate and assign the four "
        "polynomial expressions from x and y in Atlas scripts or the Singular interpreter, optionally including both "
        "ideal declarations. Each Atlas script runs in a separately created QQ[x,y] ring with a fresh session. "
        "Singular reassigns the same four names in an existing ring, one thread. Both systems verify all four ordered "
        "generators outside the timer. Each condition discards a warm-up batch of the same size. No sum, product "
        "or ideal power is timed.",
        "cases": {name: measure(*inputs_, args.trials, args.batch, args.creation_batch) for name, inputs_ in CASES.items()},
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
