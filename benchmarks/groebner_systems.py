"""Compare the F4 trace of the modular Groebner route with msolve and Groebner.jl.

Each system is given to every program as the same list of integer
polynomials, in the graded reverse lexicographic order with the first
variable largest. Katsura systems come from groebner_hard.py; the others
were written by groebner-systems/julia/export_systems.jl from the examples of
Groebner.jl.

- modulo the kernel's trace prime: the kernel's F4 trace and one replay,
  msolve with exact and probabilistic linear algebra, Groebner.jl with
  deterministic and randomized linear algebra and its learn and apply;
- over Q, for the systems of --rational: the kernel's certified basis with
  its phases, and Groebner.jl with and without its certificate.

Every basis is compared with the kernel's, over Q modulo a 61-bit prime.
Over Q, and modulo the prime when the homogenized generators are a complete
intersection, the bases are equal term by term. Otherwise the kernel's trace
ends with the basis of the homogenized ideal, whose images are a Groebner
basis of the ideal that is not reduced (the certified route reduces it after
lifting, which is not timed here). It is then compared in two steps: its
minimal leading monomials are those of the reduced basis, and msolve reduces
it to the reduced basis.

A timing is the best of up to --repeats runs; a measurement stops repeating
once it has used --budget seconds. Groebner.jl first solves a small system
and does not count a first run shorter than five seconds, so that
compilation is not timed. The report is saved after every system.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from engine.native_library import _kernel  # noqa: E402
from groebner_f4 import CHECK_PRIME, canonical, certified, kernel_terms, msolve_terms, summary, timed  # noqa: E402
from groebner_hard import katsura  # noqa: E402

JULIA = HERE / "groebner-systems" / "julia"
SYSTEMS = HERE / "groebner-systems" / "systems"


def load(name):
    """The number of variables and the polynomials, each a list of (exponents, integer)."""
    family, _, size = name.rpartition("-")
    if family == "katsura":
        names, polynomials = katsura(int(size))
        return len(names), [[(e, c) for e, c in p.items() if c] for p in polynomials]
    lines = (SYSTEMS / f"{name}.txt").read_text().splitlines()
    polynomials = []
    for line in lines[1:]:
        if line.strip():
            polynomials.append([(tuple(map(int, term.split()[1:])), int(term.split()[0])) for term in line.split(";")])
    return int(lines[0]), polynomials


def text(polynomials):
    return "".join(";".join(f"{c} {' '.join(map(str, e))}" for e, c in p) + "\n" for p in polynomials)


def expression(names, polynomial):
    out = ""
    for exponents, coefficient in polynomial:
        factors = [f"{v}^{k}" if k > 1 else v for v, k in zip(names, exponents) if k]
        size = abs(coefficient)
        body = "*".join(([str(size)] if size != 1 or not factors else []) + factors)
        out += ("-" if coefficient < 0 else "+" if out else "") + body
    return out


def leading(terms):
    return max(terms, key=lambda t: (sum(t[0]), tuple(-x for x in reversed(t[0]))))[0]


def minimal_leading(elements):
    """The leading monomials that no other leading monomial divides."""
    heads = sorted({leading(terms) for terms in elements}, key=sum)
    kept = []
    for head in heads:
        if not any(all(a <= b for a, b in zip(other, head)) for other in kept):
            kept.append(head)
    return set(kept)


def file_terms(path):
    """A basis written by run_groebner.jl as lists of (exponents, residue)."""
    return [[(tuple(map(int, term.split()[1:])), int(term.split()[0])) for term in row.split(";")]
            for row in Path(path).read_text().splitlines() if row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="katsura-9,katsura-10,katsura-11,cyclic-7,cyclic-8,noon-7,noon-8,eco-11,eco-12,reimer-6")
    parser.add_argument("--rational", default="katsura-9,cyclic-7,noon-7,eco-11", help="systems also solved over Q")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=float, default=120, help="seconds after which a measurement is not repeated")
    parser.add_argument("--limit", type=float, default=1800, help="seconds allowed to each external run")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--skip", default="", help="comma separated: msolve, groebnerjl")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    rational = set(filter(None, args.rational.split(",")))

    library = _kernel()
    if library is None or not hasattr(library, "atlas_groebner_f4_profile"):
        raise SystemExit("The kernel with the F4 trace is not built. Run make.")
    library.atlas_groebner_f4_profile.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
                                                 ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_long),
                                                 ctypes.POINTER(ctypes.c_void_p)]
    library.atlas_groebner_modular.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_long),
                                              ctypes.POINTER(ctypes.c_double)]
    library.atlas_groebner_modular.restype = ctypes.c_void_p
    library.atlas_groebner_text.restype = ctypes.c_char_p
    library.atlas_groebner_use_f4.argtypes, library.atlas_groebner_use_f4.restype = [ctypes.c_int], None
    library.atlas_groebner_modular_route.restype = ctypes.c_int

    work = Path(tempfile.mkdtemp(prefix="atlas-systems-"))
    julia = shutil.which("julia") if "groebnerjl" not in skip and (JULIA / "Project.toml").is_file() else None
    msolve = shutil.which("msolve") if "msolve" not in skip else None
    report = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "threads": args.threads,
              "machine": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count()},
              "programs": {"msolve": msolve, "julia": julia}, "systems": {}}

    def save():
        if args.out:
            args.out.write_text(json.dumps(report, indent=1) + "\n")

    def repeat(run):
        """Up to --repeats samples of run(), stopping once the budget is used or a run fails."""
        samples, start = [], time.perf_counter()
        while len(samples) < args.repeats:
            sample = run()
            if sample is None:
                break
            samples.append(sample)
            if time.perf_counter() - start > args.budget:
                break
        return samples

    def groebnerjl(system, characteristic, mode, threads, basis):
        """The seconds of each run by kind, from run_groebner.jl; None when it fails or exceeds the limit."""
        command = [julia, "-t", str(threads), f"--project={JULIA}", str(JULIA / "run_groebner.jl"), str(system),
                   str(characteristic), mode, str(args.repeats), str(basis), str(CHECK_PRIME), str(args.budget)]
        elapsed, output = timed(command, args.limit)
        if elapsed is None:
            return None
        for line in reversed(output.splitlines()):
            if line.startswith("{"):
                return json.loads(line)["seconds"]
        return None

    for name in args.systems.split(","):
        variables, polynomials = load(name)
        names = [f"x{i + 1}" for i in range(variables)]
        data = (f"{variables} 0 0\n" + text(polynomials)).encode()
        system = work / f"{name}.txt"
        system.write_text(f"{variables}\n" + text(polynomials))
        expressions = ",\n".join(expression(names, p) for p in polynomials)
        entry = {"variables": variables, "generators": len(polynomials), "load_1min": os.getloadavg()[0]}
        report["systems"][name] = entry
        rows, agreement = {}, {}

        def profile(threads, replays, want_basis=False):
            seconds, counts, pointer = (ctypes.c_double * 2)(), (ctypes.c_long * 9)(), ctypes.c_void_p()
            if library.atlas_groebner_f4_profile(data, len(data), replays, threads, seconds, counts,
                                                 ctypes.byref(pointer) if want_basis else None):
                raise SystemExit(f"{name}: {library.atlas_groebner_error().decode()}")
            basis = None
            if want_basis:
                basis = ctypes.string_at(pointer).decode()
                library.atlas_groebner_release(pointer)
            return list(seconds), list(counts), basis

        _, counts, basis = profile(args.threads, 0, True)
        prime = counts[8]
        ours = kernel_terms(basis, prime)
        reference = canonical(ours, prime)
        del basis

        def same(theirs):
            """Whether a reduced basis is the kernel's, or the reduced form of the kernel's."""
            if theirs == reference:
                return True
            if minimal_leading(ours) != {leading(list(terms)) for terms in theirs}:
                return False
            if not msolve:
                return "leading monomials only"
            # msolve reduces the kernel's basis: the ideal it generates must be the same.
            source, result = work / "ours.ms", work / "ours-basis.ms"
            source.write_text(",".join(names) + f"\n{prime}\n" + ",\n".join(expression(names, p) for p in ours) + "\n")
            if timed(["msolve", "-g", "2", "-t", str(args.threads), "-f", str(source), "-o", str(result)], args.limit)[0] is None:
                return "leading monomials only"
            return "after reduction" if canonical(msolve_terms(result, names), prime) == theirs else False
        entry["prime"] = {"prime": prime, "basis_elements": counts[0], "basis_terms": counts[1], "matrices": counts[2],
                          "combinations_reduced": counts[3], "dependent_combinations": counts[4],
                          "hilbert_bound_met": bool(counts[6]), "seconds": rows, "same_basis_as_atlas": agreement}
        print(f"{name}: {variables} variables, {counts[0]} elements, {counts[1]} terms modulo {prime}; "
              f"{counts[3]} combinations in {counts[2]} matrices, Hilbert bound {'met' if counts[6] else 'not met'}", flush=True)

        def show(label, samples):
            rows[label] = summary(samples)
            row = rows[label]
            print(f"  {label:58s} " + (f"best {row['best_s']:9.3f}  median {row['median_s']:9.3f}  ({row['runs']})" if row
                                      else "failed or over the limit"), flush=True)
            save()

        for threads in sorted({1, args.threads}):
            replays = []

            def trace():
                seconds, counts, _ = profile(threads, 1)
                if counts[7] != 1:
                    raise SystemExit(f"{name}: a replay disagreed with the trace.")
                replays.append(seconds[1])
                return seconds[0]
            show(f"Atlas F4 trace, {threads} thread(s)", repeat(trace))
            if threads == 1:
                show("Atlas F4 replay of one prime, 1 thread", replays)
        if msolve:
            source = work / "system.ms"
            source.write_text(",".join(names) + f"\n{prime}\n{expressions}\n")
            for label, option in (("exact", "2"), ("probabilistic", "42")):
                for threads in sorted({1, args.threads}):
                    result = work / "basis.ms"
                    result.unlink(missing_ok=True)
                    samples = repeat(lambda: timed(["msolve", "-l", option, "-t", str(threads), "-g", "2", "-f", str(source),
                                                    "-o", str(result)], args.limit)[0])
                    show(f"msolve F4, {label}, {threads} thread(s)", samples)
                    if samples and result.is_file():
                        try:
                            theirs = msolve_terms(result, names)
                            # The reduced basis of the ideal itself, next to the homogenized one of the trace.
                            entry["prime"].update(reduced_elements=len(theirs), reduced_terms=sum(map(len, theirs)))
                            agreement[f"msolve {label}, {threads} thread(s)"] = same(canonical(theirs, prime))
                        except (ValueError, KeyError):
                            agreement[f"msolve {label}, {threads} thread(s)"] = "no basis printed"
        if julia:
            for mode in ("deterministic", "randomized"):
                for threads in sorted({1, args.threads}):
                    result = work / "basis.gjl"
                    result.unlink(missing_ok=True)
                    seconds = groebnerjl(system, prime, mode, threads, result)
                    show(f"Groebner.jl, {mode}, {threads} thread(s)", (seconds or {}).get(mode, []))
                    if seconds and result.is_file():
                        agreement[f"Groebner.jl {mode}, {threads} thread(s)"] = same(canonical(file_terms(result), prime))
            seconds = groebnerjl(system, prime, "learn_apply", 1, "-")
            show("Groebner.jl learn, 1 thread", (seconds or {}).get("learn", []))
            show("Groebner.jl apply to one prime, 1 thread", (seconds or {}).get("apply", []))
        print("  same basis as the kernel:", agreement, flush=True)
        del reference, ours
        save()

        if name in rational:
            rows, agreement = {}, {}
            entry["rational"] = {"seconds": rows, "same_basis_as_atlas": agreement}
            row, basis = certified(library, data, True)
            row["trace"] = {1: "Buchberger", 2: "F4, homogenized", 3: "F4, direct"}.get(library.atlas_groebner_modular_route(), "?")
            rows["Atlas certified"] = row
            print("  over Q, Atlas certified:", row, flush=True)
            save()
            if basis is not None and julia:
                reference = canonical(kernel_terms(basis, CHECK_PRIME), CHECK_PRIME)
                entry["rational"].update(basis_elements=basis.count("\n"), basis_terms=basis.count(";") + basis.count("\n"))
                del basis
                for mode, label in (("default", "Groebner.jl over Q, certify=false"), ("certify", "Groebner.jl over Q, certify=true")):
                    result = work / "basis-q.gjl"
                    result.unlink(missing_ok=True)
                    seconds = groebnerjl(system, 0, mode, args.threads, result)
                    rows[f"{label}, {args.threads} threads"] = summary((seconds or {}).get(mode, []))
                    if seconds and result.is_file():
                        agreement[label] = canonical(file_terms(result), CHECK_PRIME) == reference
                    print(f"  {label}:", rows[f"{label}, {args.threads} threads"] or "failed or over the limit", flush=True)
                    save()
                print("  same basis as the kernel:", agreement, flush=True)
        entry["load_1min_end"] = os.getloadavg()[0]
        save()
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
