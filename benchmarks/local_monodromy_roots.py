#!/usr/bin/env python3
"""Compare local monodromy eigenvalue decisions with Singular Weyl elimination and full b-functions."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.kernel_json import kernel_json
from engine.point_geometry import encode_terms
from engine.polynomial import parse


# Expected first local b-root in each residue class; None certifies absence.
# Positive weights let Singular's global bfct also compute the origin's
# entire local b-function.
CASES = [
    ("tangent-conics", "xyz", "(x*z-y^2)*(x*z+y*z-y^2)", {"-2/3": "-2/3"}, (1, 1, 1)),
    ("e7-cylinder", "xyzw", "y*x^2+y^2*(y^2+z^3)", {"-3/4": "-3/4"}, (9, 6, 4, 1)),
    ("cubic-parameters", "xyzw", "y*x^2+y^2*(y^2+z^3+w^3)", {"-3/4": "-3/4"}, (9, 6, 4, 4)),
    ("mixed-parameters", "xyzw", "y*x^2+y^2*(y^2+z^3+z*w)", {"-3/4": "-3/4"}, (9, 6, 4, 8)),
    ("cusp", "xy", "x^2+y^3", {"1/6": "-5/6", "-1/6": "-7/6", "-1/2": None}, (3, 2)),
    ("five-square-suspension", "xyzvw", "x^2+y^2+z^2+v^2+w^2", {"-1/2": "-5/2"}, (1,)*5),
    ("distant-double-line", "xy", "(x^2+y^3)*(x-1)^2", {"-1/2": None}, None),
    ("transverse-line", "xyz", "z^2-x*y^4", {"-3/4": "-3/4"}, (2, 1, 3)),
    ("transverse-tilted-line", "xyz", "z^2-x*(x+y)^4", {"-3/4": "-3/4"}, (2, 2, 5)),
]


def singular_run(executable, program, timeout):
    cmd = [executable, "-q", "--no-rc", "--threads=1", "--random=47", "--ticks-per-sec=1000000"]
    run = subprocess.run(cmd, input=program + "\nquit;\n", text=True,
                         capture_output=True, check=True, timeout=timeout)
    if any(line.lstrip().startswith("?") for line in run.stdout.splitlines()):
        raise RuntimeError(run.stdout + run.stderr)
    rows = [line.split()[1:] for line in run.stdout.splitlines() if line.startswith("RESULT ")]
    elapsed, = [int(line[8:])/1000 for line in run.stdout.splitlines() if line.startswith("ELAPSED ")]
    return elapsed, rows


def local_program(variables, equation, expected):
    width = len(variables)
    checks = []
    for pole in expected:
        residue = Fraction(pole) % 1
        checks.append('found=0;')
        for index in range(width):
            a = residue - 1 - index
            checks.append(f'if (!found && present({a},LD,F,derivs,{width})) {{ '
                          f'print("RESULT {pole} {a}"); found=1; }}')
        checks.append(f'if (!found) {{ print("RESULT {pole} absent"); }}')
    return f'''
LIB "dmod.lib";
proc present(number alpha, ideal annihilator, poly f, poly derivs, int n) {{
    ideal specialized=subst(annihilator,var(nvars(basering)),alpha),f;
    ideal support=eliminate(specialized,derivs);
    for (int i=1; i<=n; i++) {{ support=subst(support,var(i),0); }}
    return(size(support)==0);
}}
ring R=0,({','.join(variables)}),dp;
poly f={equation};
int started=rtimer;
def D=Sannfs(f);
setring D;
poly F=imap(R,f);
poly derivs={'*'.join('D'+v for v in variables)};
int found;
{chr(10).join(checks)}
print("ELAPSED "+string(rtimer-started));
'''


def full_program(variables, equation):
    return f'''
LIB "bfun.lib";
ring R=0,({','.join(variables)}),dp;
poly f={equation};
int started=rtimer;
list bdata=bfct(f);
print("ELAPSED "+string(rtimer-started));
for (int i=1; i<=size(bdata[1]); i++) {{
    print("RESULT "+string(bdata[1][i])+" "+string(bdata[2][i]));
}}
'''


def measure(case, trials, executable, timeout):
    name, variables, equation, expected, weights = case
    terms = parse(equation, list(variables))
    if weights and len({sum(a*w for a, w in zip(powers, weights)) for powers in terms}) != 1:
        raise ValueError("Invalid positive weights for the full b-function comparison")
    payload = dict(width=len(variables), terms=encode_terms(terms), values=list(expected))
    local = local_program(variables, equation, expected)
    full = full_program(variables, equation)
    times = {"Atlas_local_ms": [], "Singular_local_ms": []}
    roots = None
    if weights:
        times["Singular_full_b_ms"] = []
    for trial in range(trials+1):
        for key in (list(times) if trial%2 else reversed(times)):
            if key == "Atlas_local_ms":
                start = perf_counter_ns()
                row = kernel_json("atlas_monodromy", "local_eigenvalues", **payload)
                elapsed = (perf_counter_ns()-start)/1e6
                actual = {pole: ref["root"] for pole, ref in row["by_pole"].items()}
            elif key == "Singular_local_ms":
                elapsed, rows = singular_run(executable, local, timeout)
                actual = {pole: None if a == "absent" else a for pole, a in rows}
            else:
                elapsed, rows = singular_run(executable, full, timeout)
                calculated = {str(Fraction(a)): int(multiplicity) for a, multiplicity in rows}
                if roots is not None and roots != calculated:
                    raise ValueError(f"Full b-function changed across trials on {name}")
                roots = calculated
                actual = {}
                for pole in expected:
                    matches = [Fraction(a) for a in roots if (Fraction(a)-Fraction(pole)).denominator == 1]
                    actual[pole] = str(max(matches)) if matches else None
            if actual != expected:
                raise ValueError(f"{name} {key} returned {actual}, expected {expected}")
            if trial:
                times[key].append(round(elapsed, 3))
    return dict(case=name, variables=variables, polynomial=equation, expected=expected,
                weights=weights, full_b_roots=roots, trials=times,
                median_ms={key: median(samples) for key, samples in times.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per Singular trial")
    parser.add_argument("--singular", default="Singular")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Pass at least one measured trial")
    report = {
        "Singular": subprocess.check_output([args.singular, "--version"], text=True).splitlines()[0],
        "conditions": "One discarded warmup and alternating measurements. QQ inputs pre-parsed. "
                      "Atlas uses JSON/FFI and a fresh native computation. Singular uses one thread, "
                      "a fresh interpreter per sample, and rtimer around the operation, excluding "
                      "library load, program startup and polynomial declaration. Singular local uses "
                      "Sannfs, exact specialization, derivative elimination and origin evaluation. "
                      "Full bfct is timed only for positively weighted homogeneous inputs.",
        "cases": [],
    }
    for case in CASES:
        row = measure(case, args.trials, args.singular, args.timeout)
        report["cases"].append(row)
        print(row["case"], row["median_ms"], flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
