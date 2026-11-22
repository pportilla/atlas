"""Time rational chart factors over a marked field and compare every factor and multiplicity with Singular."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.native_library import affine_request
from engine.polynomial import format_poly, parse_expression, sparse_row

# The shape eliminant encountered in the intrinsic x²+y²+(z²-t³)² charts.
SHAPE = "x^6*(y^2+1)*((2*x+4*y+z)^2+y^2+1)^2+256*x^3*(y^2+1)*(2*x+4*y+z)+4096"
MIXED = "((x*y+z+1)^2+(x^2+y*z)^2)^2*(1298074214633706907132624082305024*x+y*z+1)^3"
CASES = (("cusp-shape", SHAPE, -1), ("split-repeated", MIXED, -1), ("nonsplit-repeated", MIXED, 2),
         ("primitive-product", "(x*(z^2-y))^2+1", -1), ("primitive-sum", "x^2+(z^2-y)^2", 2))
NAMES = ("x", "y", "z", "a")


def compare(equation, d, trials):
    relation = f"a^2-({d})"
    request = ["field_factorization", "4", parse_expression(relation, NAMES, as_row=True),
               parse_expression(equation, NAMES, as_row=True)]
    answer = affine_request(request)
    branches = [format_poly(sparse_row(row), NAMES) for row in answer[1::2]]
    powers = answer[::2]
    program = (f"ring r=(0,a),(x,y,z),lp;minpoly={relation};poly f={equation};\n"
               f"ideal ref={','.join(branches)};intvec powers={','.join(powers)};\n"
               "int i,j,hit,seen;int ok=1;list fac;poly g;int start=rtimer;fac=factorize(f);\n"
               'print("TIME "+string(rtimer-start));\n'
               "for(i=1;i<=size(fac[1]);i++){g=fac[1][i];if(deg(g)>0){seen++;hit=0;"
               "for(j=1;j<=size(ref);j++){if(g*leadcoef(ref[j])==ref[j]*leadcoef(g)"
               " && fac[2][i]==powers[j]){hit++;}}if(hit!=1){ok=0;}}}\n"
               'print("AGREE "+string(seen==size(ref) && ok==1));quit;\n')
    times = {"atlas_s": [], "singular_s": []}
    for trial in range(trials):
        for system in (times if trial % 2 == 0 else reversed(times)):
            if system == "atlas_s":
                start = perf_counter()
                result = affine_request(request)
                times[system].append(perf_counter()-start)
                if result != answer:
                    raise ValueError("Native marked factors changed between trials.")
            else:
                p = subprocess.run(["Singular", "-q", "--no-rc", "--ticks-per-sec=1000000"],
                                   input=program, text=True, capture_output=True, check=True, timeout=60)
                lines = (p.stdout+p.stderr).splitlines()
                if "AGREE 1" not in lines or any(line.lstrip().startswith("?") for line in lines):
                    raise ValueError(lines)
                timing, = [line.split()[1] for line in lines if line.startswith("TIME ")]
                times[system].append(int(timing)/1_000_000)
    return dict(equation=equation, relation=relation, factors=len(branches), multiplicities=powers, **times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("Use at least two trials.")
    record = dict(conditions="Exact factorization in QQ[a]/(minpoly)[x,y,z]. Each trial times the "
                  "uncached native affine call including row I/O or Singular factorize on the same polynomial. "
                  "Timers exclude interpreter and coefficient field setup. Each Singular result is checked "
                  "factor-by-factor up to a field unit, including multiplicity.",
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  singular=subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
                  cases={name: compare(equation, d, args.trials) for name, equation, d in CASES})
    args.output.write_text(json.dumps(record, indent=2)+"\n")
