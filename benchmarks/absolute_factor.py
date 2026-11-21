"""Compare public absolute factorization over QQ with Singular's absfact.lib."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
import statistics
import sys
import subprocess
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.resolution_zeta import Session
from tests.absolute_factor_oracle import check

D = "(y^3-x^5)"
SHIFT = "(y^2-x^3)"
SHIFT2 = "(y^2-x^3+x^4+y^3)"


@dataclass(frozen=True)
class Case:
    expression: str
    # Rational factors, their number of distinct absolute factors, and multiplicity.
    orbits: tuple


CASES = {
    "two-lines": Case("x^2+y^2", (("x^2+y^2", 2, 1),)),
    "irreducible-quadric": Case("x^2+y*z", (("x^2+y*z", 1, 1),)),
    "non-normal-cubic": Case("((x+y*z)^3-2*(y+z)^3)", (("(x+y*z)^3-2*(y+z)^3", 3, 1),)),
    "two-quadratic-fields": Case(
        "(-2/3)*(x^2+y^2)^2*((x+y*z)^2-2*(y+z)^2)*(x^2+y*z)",
        (("x^2+y^2", 2, 2), ("(x+y*z)^2-2*(y+z)^2", 2, 1), ("x^2+y*z", 1, 1)),
    ),
    "three-quadratic-translates": Case(
        f"(z^2-{D})*((z-{SHIFT})^2-{D})*((z-{SHIFT2})^2-{D})",
        ((f"z^2-{D}", 1, 1), (f"(z-{SHIFT})^2-{D}", 1, 1), (f"(z-{SHIFT2})^2-{D}", 1, 1)),
    ),
}


DECLARATION = 'x,y,z=coordinates("x","y","z",coeffs=QQ)'
ATLAS_RESULT = 'F=absolute_factor(p,order="lex")\nprint_json(result=F.result)'
SINGULAR_HEADER = r'''LIB "absfact.lib";
ring R=0,(x,y,z),lp;
proc writeAbsolute(poly p) {
    def S=absFactorize(p,"a");
    setring S;
    ideal H=absolute_factors[1];
    ideal M=absolute_factors[3];
    intvec powers=absolute_factors[2];
    ring P=0,(x,y,z,a),lp;
    short=0;
    ideal H=imap(S,H);
    ideal M=imap(S,M);
    string out="@unit "+string(H[1]);
    for (int i=2;i<=size(H);i++) {
        out=out+";@orbit|"+string(powers[i])+"|"+string(M[i])+"|"+string(H[i]);
    }
    print("ATLASBENCH "+out);
}
'''


def script(case, solver):
    if solver == "Atlas":
        return f"{DECLARATION}\np={case.expression}\n{ATLAS_RESULT}\n"
    return SINGULAR_HEADER + f"writeAbsolute({case.expression});\nquit;\n"


def checked(case, solver, output):
    if solver == "Singular":
        output = output.replace(";", "\n")
    check(case, solver, output)


def run(case, solver, trial, timeout):
    command = ([ROOT / "atlas", "run", "-"] if solver == "Atlas" else
               ["Singular", "-q", "--no-rc", f"--random={trial+1}"])
    source = script(case, solver)
    start = perf_counter()
    result = subprocess.run(command, input=source, capture_output=True, text=True, cwd=ROOT, timeout=timeout)
    elapsed = (perf_counter()-start)*1000
    if result.returncode or any(line.lstrip().startswith("?") for line in result.stdout.splitlines()):
        raise RuntimeError(result.stderr or result.stdout)
    output = result.stdout
    if solver == "Singular":
        output = next(line.removeprefix("ATLASBENCH ") for line in output.splitlines()
                      if line.startswith("ATLASBENCH "))
    checked(case, solver, output)
    return elapsed


class WarmSolver:
    """Use the same Atlas session functions and Singular procedures in a live interpreter."""

    def __init__(self, label):
        self.label = label
        if label == "Singular":
            self.command = ["Singular", "-q", "--no-rc", "--random=1"]
            self.setup = SINGULAR_HEADER + 'print("ATLASBENCH ready");\n'
        else:
            program = f"""import sys
sys.path.insert(0, {str(ROOT)!r})
from engine.session import new_session
session = new_session(None)
list(session.run({DECLARATION!r}))
print("ATLASBENCH ready", flush=True)
for line in sys.stdin:
    print("ATLASBENCH ", end="")
    list(session.run("p="+line+{ATLAS_RESULT!r}))
"""
            self.command = [sys.executable, "-I", "-u", "-c", program]
            self.setup = ""

    def job(self, polynomial):
        return f"writeAbsolute({polynomial});\n" if self.label == "Singular" else polynomial + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    with ExitStack() as stack:
        warm = {label: stack.enter_context(Session(WarmSolver(label), os.environ, args.timeout))
                for label in ("Atlas", "Singular")}
        for name in args.case or CASES:
            case = CASES[name]
            for label, session in warm.items():
                _, answer = session.warm(case.expression)
                checked(case, label, answer)
            for mode in ("fresh", "warm"):
                samples = {"Atlas": [], "Singular": []}
                for n in range(args.trials):
                    for label in (samples if n % 2 == 0 else reversed(samples)):
                        if mode == "fresh":
                            elapsed = run(case, label, n, args.timeout)
                        else:
                            elapsed, answer = warm[label].warm(case.expression)
                            checked(case, label, answer)
                        samples[label].append(elapsed)
                for label, times in samples.items():
                    values = ", ".join(f"{t:.3f}" for t in times)
                    print(f"{name} | {mode} | {label} | {values} | median {statistics.median(times):.3f} ms",
                          flush=True)


if __name__ == "__main__":
    main()
