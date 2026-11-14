"""Time the native Groebner kernel on bases recorded from the ideal-blowup verifier.

Every problem in groebner-corpus.json is a basis the verifier asked for, with
SymPy's reduced basis and SymPy's time. The kernel's answer is compared term
by term. With --singular the same problems are timed in Singular's std; for
both engines only the computation is timed, not parsing or printing.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.groebner import _kernel  # noqa: E402


def line(poly):
    return ";".join((num if den == "1" else f"{num}/{den}") + "".join(f" {e}" for e in exps)
                    for num, den, exps in poly)


def table(poly):
    return {tuple(exps): Fraction(int(num), int(den)) for num, den, exps in poly}


def parse(text):
    basis = []
    for row in text.split("\n")[:-1]:
        terms = {}
        for term in row.split(";"):
            coefficient, *powers = term.split(" ")
            terms[tuple(map(int, powers))] = Fraction(coefficient)
        basis.append(terms)
    return basis


def kernel_times(corpus, repeats):
    library = _kernel()
    if library is None:
        raise SystemExit("The kernel is not built. Run make.")
    seconds, wrong = [], 0
    for record in corpus:
        header = f"{record['n']} {1 if record['order'] == 'lex' else 0} 0\n"
        payload = (header + "".join(line(p) + "\n" for p in record["polys"])).encode()
        best = None
        for _ in range(repeats):
            handle = library.atlas_groebner(payload, len(payload))
            if not handle:
                raise SystemExit("The kernel refused a corpus problem.")
            elapsed = library.atlas_groebner_seconds(handle)
            best = elapsed if best is None else min(best, elapsed)
            text = library.atlas_groebner_text(handle).decode()
            library.atlas_groebner_free(handle)
        seconds.append(best)
        wrong += parse(text) != [table(p) for p in record["expected"]]
    return seconds, wrong


def singular_times(corpus, repeats):
    def text(poly, names):
        return "+".join("*".join([f"({num if den == '1' else num + '/' + den})"] +
                                 [f"{names[i]}^{e}" for i, e in enumerate(exps) if e])
                        for num, den, exps in poly) or "0"
    script = ['system("--ticks-per-sec",1000000);', "int t;"]
    for index, record in enumerate(corpus):
        names = [f"x{i}" for i in range(record["n"])]
        order = {"grevlex": "dp", "lex": "lp"}[record["order"]]
        script += [f"ring r{index}=0,({','.join(names)}),{order};",
                   f"ideal i={','.join(text(p, names) for p in record['polys']) or '0'};",
                   "t=rtimer; ideal g=std(i); t=rtimer-t;", f'print("R {index} "+string(t));', "kill g; kill i;"]
    script = "\n".join(script) + "\nquit;\n"
    best = [None] * len(corpus)
    for _ in range(repeats):
        run = subprocess.run(["Singular", "-q"], input=script, capture_output=True, text=True, timeout=3600)
        for row in run.stdout.splitlines():
            if row.startswith("R "):
                _, index, micros = row.split()
                value = int(micros) / 1e6
                best[int(index)] = value if best[int(index)] is None else min(best[int(index)], value)
    if any(value is None for value in best):
        raise SystemExit("Singular did not finish every problem.")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(Path(__file__).with_name("groebner-corpus.json")))
    parser.add_argument("--repeats", type=int, default=5, help="runs per problem; the best one counts")
    parser.add_argument("--singular", action="store_true", help="also time Singular's std (needs Singular on PATH)")
    parser.add_argument("--output", help="write the per-problem times as JSON")
    args = parser.parse_args()
    corpus = json.load(open(args.corpus))
    kernel, wrong = kernel_times(corpus, args.repeats)
    singular = None
    if args.singular:
        if not shutil.which("Singular"):
            raise SystemExit("Singular is not on PATH.")
        singular = singular_times(corpus, min(args.repeats, 3))
    groups = [("all", lambda r: True), ("grevlex", lambda r: r["order"] == "grevlex"),
              ("lex", lambda r: r["order"] == "lex"), ("50 generators or more", lambda r: len(r["polys"]) >= 50)]
    print(f"{len(corpus)} problems, {wrong} wrong")
    for label, pick in groups:
        chosen = [i for i, r in enumerate(corpus) if pick(r)]
        row = (f"{label:22s} {len(chosen):4d} problems   kernel {1000*sum(kernel[i] for i in chosen):8.2f} ms"
               f"   SymPy {sum(corpus[i]['sympy_seconds'] for i in chosen):7.2f} s")
        if singular:
            total = sum(singular[i] for i in chosen)
            row += f"   Singular std {1000*total:8.2f} ms   Singular/kernel {total/sum(kernel[i] for i in chosen):5.2f}"
        print(row)
    if singular:
        print("kernel slower than Singular on", sum(k > s for k, s in zip(kernel, singular)), "problems")
    if args.output:
        json.dump({"kernel_seconds": kernel, "singular_std_seconds": singular, "wrong": wrong}, open(args.output, "w"))
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
