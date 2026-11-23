#!/usr/bin/env python3
"""Polynomial graph and quadratic-translate zetas, measured from process startup on the same rational input."""
import argparse
import json
from pathlib import Path
import platform
import re
from statistics import median
import subprocess
from time import perf_counter

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
POLYNOMIAL = "z*(z-y^2+x^3)*(z-y^2+x^3-x^4-y^3)"
SINGULAR_RESOLUTION = f'''LIB "reszeta.lib";
ring R=0,(x,y,z),dp;
list re=resolve(ideal({POLYNOMIAL}),"A");
list result=zetaDL(re,1,"local","A",1);
ring Qs=(0,s),Z,dp;
number mon=1;
int i;
for(i=1;i<=size(result[2]);i++) {{
   mon=mon*(1-s^(result[6][result[2][i][1][1]]))^(-result[2][i][2]);
}}
print("ATLASBENCH "+result[1]+"|"+string(mon));
'''
TOP = "(-356*s^4+4912*s^3+13699*s^2+10903*s+2717)/((s+1)^2*(18*s+11)*(22*s+13)*(32*s+19))"


FOUR_TOP = "(-2619*s^4-2996*s^3+8580*s^2+10126*s+2717)/((s+1)^2*(21*s+11)*(25*s+13)*(36*s+19))"
FOUR_MON = "(1-s^7)*(1-s^9)/((1-s^21)*(1-s^25)*(1-s^36))"


def verify(top, mon, expected):
    for value, reference in zip((top, mon), expected):
        if sp.cancel(sp.sympify(value) - reference) != 0:
            raise ValueError(f"Wrong zeta: {value}")


def trial(method, command, source, timeout, expected):
    start = perf_counter()
    try:
        result = subprocess.run(command, input=source, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if method != "Singular from polynomial":
            raise
        return None
    elapsed = (perf_counter() - start) * 1000
    result.check_returncode()
    if method == "Atlas from polynomial":
        record = json.loads(result.stdout)
        verify(record["topological_zeta"]["expression"], record["monodromy_zeta"]["expression"].replace("T", "s"), expected)
    else:
        if re.search(r"^\s*\?", result.stdout, re.M):
            raise ValueError(result.stdout)
        answer, = [line.removeprefix("ATLASBENCH ") for line in result.stdout.splitlines()
                   if line.startswith("ATLASBENCH ")]
        answer = re.sub(r"(\d)s", r"\1*s", re.sub(r"s(\d+)", r"s^\1", answer))
        verify(*answer.split("|"), expected)
    return round(elapsed, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    case = parser.add_mutually_exclusive_group()
    case.add_argument("--quadratic", choices=("cuspidal", "conics"),
                      help="three translates with a common quadratic radicand")
    case.add_argument("--four", action="store_true", help="include the fourth graph z=y")
    args = parser.parse_args()
    if args.repeats < 2 or args.timeout <= 0:
        parser.error("Use at least two trials and a positive timeout.")
    if args.quadratic == "cuspidal":
        saved = ROOT / "tests" / "fixtures" / "quadratic-translates.json"
        reference = json.loads(saved.read_text())
        polynomial = reference["polynomial"]
        expected = (reference["topological_zeta"], reference["monodromy_zeta"]["expression"].replace("T", "s"))
        specialized = None
    elif args.quadratic == "conics":
        polynomial = "(z^2-x*y)*((z-x-2*y)^2-x*y)*((z-3*x-5*y)^2-x*y)"
        expected = ("(3*s^2+1)/((2*s+1)*(s+1)^2)", "1/(1-s^6)^9")
        specialized = ("Singular conic resolution from polynomial", "quadratic_contact_conics.sing")
    else:
        polynomial = POLYNOMIAL + ("*(z-y)" if args.four else "")
        s = sp.Symbol("s")
        expected = ((FOUR_TOP, FOUR_MON) if args.four else
                    (TOP, 1/sp.prod(sp.cyclotomic_poly(k, s) for k in (2, 16, 18, 22, 32))))
        charts = "four_graph_contact.sing" if args.four else "three_graph_contact.sing"
        specialized = ("Singular with prescribed plane charts", charts)
    expected = tuple(map(sp.sympify, expected))
    jobs = [
        ("Atlas from polynomial", [ROOT / "atlas", "resolve", polynomial, "--vars", "x,y,z",
                                   "--compute", "cpu", "--zetas-only", "--json", "--quick"], None),
        ("Singular from polynomial", ["Singular", "-q", "--no-rc"], SINGULAR_RESOLUTION.replace(POLYNOMIAL, polynomial)),
    ]
    if specialized:
        label, filename = specialized
        jobs.append((label, ["Singular", "-q", "--no-rc", ROOT / "benchmarks" / filename], None))
    samples = {name: [] for name, _, _ in jobs}
    for repeat in range(args.repeats):
        for name, command, source in (jobs if repeat % 2 == 0 else reversed(jobs)):
            samples[name].append(trial(name, command, source, args.timeout, expected))
    print(json.dumps({
        "conditions": "Fresh processes, serial alternating order, startup included. Null denotes timeout.",
        "polynomial": polynomial, "platform": platform.platform(),
        "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
        "timeout_seconds": args.timeout, "wall_ms": samples,
        "median_ms": {name: median(rows) if None not in rows else None for name, rows in samples.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
