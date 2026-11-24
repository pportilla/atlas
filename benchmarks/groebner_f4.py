"""Time the F4 trace of the modular Groebner route on a Katsura ideal.

Two comparisons on the same input, with msolve and giac as references:

- one prime: the reduced graded reverse lexicographic basis modulo the
  kernel's trace prime, by the kernel's F4 trace, msolve (exact and
  probabilistic linear algebra) and giac;
- over Q: the kernel's certified basis with its phases, giac (probabilistic
  and certified), msolve's solve, and the kernel's previous Buchberger trace.

Every basis is compared with the kernel's: msolve's modular basis term by
term, giac's through the values of each element at two points modulo a
prime (its number of elements alone above --giac-terms terms, where that
evaluation takes giac longer than the basis). The certified basis over Q is
also reduced modulo the trace prime and compared with msolve's basis there.
Each timing is the best of --repeats runs, the median alongside. The report
is saved after every measurement.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
from statistics import median
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.native_library import _kernel  # noqa: E402
from groebner_hard import katsura, payload  # noqa: E402

POINTS = ((3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59),
          (61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137))
CHECK_PRIME = 2305843009213693951  # 2^61 - 1, for the comparison over Q


def expression(names, polynomial):
    """A polynomial as text that msolve and giac both read."""
    text = ""
    for exponents, coefficient in polynomial.items():
        if not coefficient:
            continue
        factors = [f"{v}^{k}" if k > 1 else v for v, k in zip(names, exponents) if k]
        size = abs(coefficient)
        body = "*".join(([str(size)] if size != 1 or not factors else []) + factors)
        text += ("-" if coefficient < 0 else "+" if text else "") + body
    return text


def timed(command, limit, **options):
    start = time.perf_counter()
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=limit, **options)
    except subprocess.TimeoutExpired:
        return None, ""
    return time.perf_counter() - start, done.stdout + done.stderr


def summary(samples):
    samples = [s for s in samples if s is not None]
    return {"best_s": round(min(samples), 3), "median_s": round(median(samples), 3), "runs": len(samples)} if samples else None


def ratios(elements, prime):
    """For each element, its value at one point over its value at another, modulo the prime."""
    out = []
    for terms in elements:
        values = []
        for point in POINTS:
            total = 0
            for exponents, coefficient in terms:
                monomial = coefficient
                for x, e in zip(point, exponents):
                    if e:
                        monomial = monomial * pow(x, e, prime) % prime
                total += monomial
            values.append(total % prime)
        out.append(values[0] * pow(values[1], prime - 2, prime) % prime)
    return sorted(out)


def kernel_terms(text, prime):
    """The kernel's basis text as lists of (exponents, residue)."""
    def residue(coefficient):
        numerator, _, denominator = coefficient.partition("/")
        return int(numerator) * pow(int(denominator or 1), -1, prime) % prime
    return [[(tuple(map(int, term.split()[1:])), residue(term.split()[0])) for term in row.split(";")]
            for row in text.splitlines()]


def msolve_terms(path, names):
    index = {name: k for k, name in enumerate(names)}
    text = Path(path).read_text()
    body = text[text.index("[") + 1:text.rindex("]")]
    elements = []
    for polynomial in body.split(","):
        terms = []
        for term in polynomial.strip().replace("-", "+-").split("+"):
            if not term.strip():
                continue
            factors = term.strip().split("*")
            exponents = [0] * len(names)
            for factor in factors[1:]:
                name, _, power = factor.partition("^")
                exponents[index[name]] += int(power or 1)
            terms.append((tuple(exponents), int(factors[0])))
        elements.append(terms)
    return elements


def canonical(elements, prime):
    """Monic elements as a set, whatever the order of elements and terms."""
    out = set()
    for terms in elements:
        terms = [(e, c % prime) for e, c in terms if c % prime]
        lead = max(terms, key=lambda t: (sum(t[0]), tuple(-x for x in reversed(t[0]))))[1]
        inverse = pow(lead, -1, prime)
        out.add(frozenset((e, c * inverse % prime) for e, c in terms))
    return out


def giac_session(names, polynomials, prime, threads, certified, with_ratios=True):
    """A giac script: the basis (modulo `prime`, or over Q), then its size and ratios."""
    system = ",".join(expression(names, p) for p in polynomials)
    modulus = prime or CHECK_PRIME
    # giac certifies its result when proba_epsilon is zero; its default allows
    # a probabilistic one.
    lines = [f"threads:={threads};"] + (["proba_epsilon:=0;"] if certified else []) + [
        f"F:=[{system}]:;", f"V:=[{','.join(names)}]:;"]
    # time() gives the processor and the real seconds of the statement.
    lines.append(f"T:=time(G:=gbasis(F % {prime},V,revlex)):;" if prime else "T:=time(G:=gbasis(F,V,revlex)):;")
    lines += ["print(\"SECONDS\",T);", "print(\"SIZE\",size(G));"]
    if with_ratios:
        a, b = (",".join(map(str, point[:len(names)])) for point in POINTS)
        value = (f"irem(irem(subst(G[k],V,[{a}]),{modulus})"
                 f"*powmod(irem(subst(G[k],V,[{b}]),{modulus}),{modulus - 2},{modulus}),{modulus})")
        lines += [f"R:=[seq({value},k,0,size(G)-1)]:;", "print(\"RATIOS\",R);"]
    return "\n".join(lines) + "\n"


def giac_result(output, modulus):
    """The real seconds of the basis computation, the number of elements and their ratios."""
    seconds, size, values = None, None, None
    for line in output.splitlines():
        if ">>" in line:
            continue  # the echo of the statement
        if "SECONDS" in line:
            seconds = float(re.findall(r"[-+0-9.e]+", line.split("SECONDS", 1)[1])[-1])
        if "SIZE" in line:
            size = int(re.findall(r"\d+", line)[-1])
        if "RATIOS" in line:
            values = sorted(int(v) % modulus for v in re.findall(r"-?\d+", line))
    return seconds, size, values


def certified(library, data, f4):
    """The certified basis through the modular route with its phases; no text when it gives up."""
    library.atlas_groebner_use_f4(int(f4))
    counts, seconds = (ctypes.c_long * 7)(), (ctypes.c_double * 4)()
    start = time.perf_counter()
    handle = library.atlas_groebner_modular(data, len(data), counts, seconds)
    elapsed = time.perf_counter() - start
    library.atlas_groebner_use_f4(1)
    if not handle:
        return {"gave_up": library.atlas_groebner_error().decode(), "total_s": round(elapsed, 3)}, None
    text = library.atlas_groebner_text(handle).decode()
    library.atlas_groebner_free(handle)
    row = {"total_s": round(elapsed, 3), "primes": counts[0], "rejected": counts[1], "traces": counts[2]}
    row.update({k: round(v, 3) for k, v in zip(("trace_s", "replay_s", "lift_s", "certify_s"), seconds)})
    row["candidate_s"] = round(sum(seconds[:3]), 3)
    return row, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=11, help="Katsura parameter: n+1 variables")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--replays", type=int, default=4, help="primes replayed at once when timing replays")
    parser.add_argument("--parts", default="prime,rational", help="comma separated: prime, rational")
    parser.add_argument("--giac-repeats", type=int, default=1, help="runs of each giac computation modulo the prime")
    parser.add_argument("--giac-terms", type=int, default=150000,
                        help="largest basis, in terms, whose elements giac evaluates for the comparison")
    parser.add_argument("--limit", type=float, default=3600, help="seconds allowed to each external run")
    parser.add_argument("--skip", default="", help="comma separated: msolve, giac, giac-certified, buchberger, msolve-solve")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    parts, skip = set(args.parts.split(",")), set(filter(None, args.skip.split(",")))

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

    names, polynomials = katsura(args.size)
    data = payload(polynomials, len(names), "grevlex")
    if parts == {"buchberger"}:
        # The previous trace alone, for the parent run to time under its limit.
        row, text = certified(library, data, False)
        row["basis_sha256"] = hashlib.sha256(text.encode()).hexdigest() if text is not None else None
        print(json.dumps(row))
        return
    system = ",\n".join(expression(names, p) for p in polynomials)
    work = Path(tempfile.mkdtemp(prefix="atlas-f4-"))
    report = {"input": f"Katsura-{args.size} ({len(names)} variables), grevlex", "threads": args.threads,
              "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "machine": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                          "load_1min_start": os.getloadavg()[0]},
              "systems": {"msolve": shutil.which("msolve"), "giac": shutil.which("giac")}}

    def save():
        report["machine"]["load_1min_end"] = os.getloadavg()[0]
        if args.out:
            args.out.write_text(json.dumps(report, indent=1) + "\n")

    def profile(threads, replays, want_basis=False):
        seconds, counts, text = (ctypes.c_double * 2)(), (ctypes.c_long * 9)(), ctypes.c_void_p()
        if library.atlas_groebner_f4_profile(data, len(data), replays, threads, seconds, counts,
                                             ctypes.byref(text) if want_basis else None):
            raise SystemExit(library.atlas_groebner_error().decode())
        basis = None
        if want_basis:
            basis = ctypes.string_at(text).decode()
            library.atlas_groebner_release(text)
        return list(seconds), list(counts), basis

    def msolve_basis(prime, option, threads):
        source = work / "system.ms"
        source.write_text(",".join(names) + f"\n{prime}\n{system}\n")
        return timed(["msolve", "-l", option, "-t", str(threads), "-g", "2", "-f", str(source),
                      "-o", str(work / "basis.ms")], args.limit)[0]

    _, counts, basis = profile(args.threads, 0, True)
    prime = counts[8]
    if "prime" in parts:
        rows, agreement = {}, {}
        ours = kernel_terms(basis, prime)
        report["prime"] = {"prime": prime, "basis_elements": counts[0], "basis_terms": counts[1], "matrices": counts[2],
                           "combinations_reduced": counts[3], "dependent_combinations": counts[4],
                           "recorded_words": counts[5], "hilbert_bound_met": bool(counts[6]),
                           "seconds": rows, "same_basis_as_atlas": agreement}
        reference = canonical(ours, prime)
        evaluate = counts[1] <= args.giac_terms
        our_ratios = ratios(ours, prime) if evaluate else None
        print(f"Katsura-{args.size} modulo {prime}: {counts[0]} elements, {counts[1]} terms, "
              f"{counts[3]} combinations reduced in {counts[2]} matrices", flush=True)

        def show(name):
            row = rows[name]
            print(f"  {name:62s} " + (f"best {row['best_s']:9.3f}  median {row['median_s']:9.3f}" if row
                                      else f"over the limit of {args.limit:.0f} s"), flush=True)
            save()

        for threads in sorted({1, args.threads}):
            concurrent = min(args.replays, threads)
            trace, replay = [], []
            for _ in range(args.repeats):
                seconds, counts, _ = profile(threads, concurrent)
                if counts[7] != concurrent:
                    raise SystemExit("A replay disagreed with the trace.")
                trace.append(seconds[0])
                replay.append(seconds[1])
            rows[f"Atlas F4 trace, {threads} thread(s)"] = summary(trace)
            show(f"Atlas F4 trace, {threads} thread(s)")
            rows[f"Atlas F4 replay, {concurrent} prime(s) at once on {threads} thread(s)"] = summary(replay)
            show(f"Atlas F4 replay, {concurrent} prime(s) at once on {threads} thread(s)")
        if "msolve" not in skip and shutil.which("msolve"):
            for label, option in (("exact", "2"), ("probabilistic", "42")):
                for threads in sorted({1, args.threads}):
                    samples = [msolve_basis(prime, option, threads) for _ in range(args.repeats)]
                    name = f"msolve F4, {label} linear algebra, {threads} thread(s)"
                    rows[name] = summary(samples)
                    if samples[-1] is not None:
                        agreement[f"msolve {label}, {threads} thread(s)"] = (
                            canonical(msolve_terms(work / "basis.ms", names), prime) == reference)
                    show(name)
        if "giac" not in skip and shutil.which("giac"):
            for threads in sorted({1, args.threads}):
                samples = []
                for _ in range(args.giac_repeats):
                    elapsed, output = timed(["giac"], args.limit,
                                            input=giac_session(names, polynomials, prime, threads, False, evaluate))
                    seconds, size, values = giac_result(output, prime) if elapsed is not None else (None, None, None)
                    samples.append(seconds)
                name = f"giac gbasis, default settings, {threads} thread(s)"
                rows[name] = summary(samples)
                if samples[-1] is not None:
                    agreement[f"giac, {threads} thread(s)" + ("" if evaluate else " (number of elements only)")] = (
                        size == len(ours) and (not evaluate or values == our_ratios))
                show(name)
        print("  agreement with the kernel's basis:", agreement, flush=True)
        del ours, reference
        save()
    del basis

    if "rational" in parts:
        rows, agreement = {}, {}
        report["rational"] = {"seconds": rows, "same_basis_as_atlas": agreement}
        rows["Atlas certified, F4 trace"], text = certified(library, data, True)
        print("  Atlas certified, F4 trace:", rows["Atlas certified, F4 trace"], flush=True)
        if text is None:
            raise SystemExit("The modular route gave up.")
        digest = hashlib.sha256(text.encode()).hexdigest()
        elements, terms = text.count("\n"), text.count(";") + text.count("\n")
        report["rational"].update(basis_elements=elements, basis_terms=terms, basis_sha256=digest)
        save()
        evaluate = terms <= args.giac_terms
        our_ratios = ratios(kernel_terms(text, CHECK_PRIME), CHECK_PRIME) if evaluate else None
        if "msolve" not in skip and shutil.which("msolve") and msolve_basis(prime, "2", args.threads) is not None:
            # The certified basis modulo the trace prime against msolve's basis there.
            agreement[f"msolve exact modulo {prime}"] = (canonical(kernel_terms(text, prime), prime)
                                                         == canonical(msolve_terms(work / "basis.ms", names), prime))
            print("  agreement so far:", agreement, flush=True)
            save()
        del text
        if shutil.which("giac"):
            for label, proved in (("giac", False), ("giac-certified", True)):
                if label in skip:
                    continue
                elapsed, output = timed(["giac"], args.limit,
                                        input=giac_session(names, polynomials, 0, args.threads, proved, evaluate))
                name = f"giac gbasis over Q, {'certified' if proved else 'probabilistic'}, {args.threads} thread(s)"
                seconds, size, values = giac_result(output, CHECK_PRIME) if elapsed is not None else (None, None, None)
                rows[name] = None if seconds is None else {"total_s": round(seconds, 3)}
                if seconds is not None:
                    agreement[name + ("" if evaluate else " (number of elements only)")] = (
                        size == elements and (not evaluate or values == our_ratios))
                print(f"  {name}:", rows[name] or f"over the limit of {args.limit:.0f} s", flush=True)
                save()
        if "msolve-solve" not in skip and shutil.which("msolve"):
            source = work / "system-q.ms"
            source.write_text(",".join(names) + f"\n0\n{system}\n")
            elapsed, _ = timed(["msolve", "-t", str(args.threads), "-f", str(source), "-o", str(work / "solve.ms")], args.limit)
            name = f"msolve solve over Q (parametrization and real roots, no basis over Q), {args.threads} thread(s)"
            rows[name] = None if elapsed is None else {"total_s": round(elapsed, 3)}
            print(f"  {name}:", rows[name] or f"over the limit of {args.limit:.0f} s", flush=True)
            save()
        if "buchberger" not in skip:
            # In a process of its own: the previous trace may run for long before it gives up.
            elapsed, output = timed([sys.executable, __file__, "--size", str(args.size), "--parts", "buchberger"], args.limit)
            name = "Atlas certified, Buchberger trace (before this change)"
            if elapsed is None:
                rows[name] = None
            else:
                rows[name] = json.loads(output.strip().splitlines()[-1])
                previous = rows[name].pop("basis_sha256", None)
                if previous:
                    agreement["Buchberger trace"] = previous == digest
            print(f"  {name}:", rows[name] or f"over the limit of {args.limit:.0f} s", flush=True)
        print("  agreement with the kernel's basis:", agreement, flush=True)
    save()
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
