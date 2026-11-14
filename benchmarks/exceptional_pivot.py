"""Compare graph strict transforms and residual dimension tests with Singular."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import kernel, native_polynomial as np


def samples(repeat):
    x, y, z = symbols = tuple(map(np.Symbol, "xyz"))
    p = x**2+1
    chart = (z-p*y,)
    large = 2**140+17
    equation = large*z**3+p**4
    numerator = large*y**3+p
    a, b, w, d, e, t = variables = tuple(map(np.Symbol, "abwdet"))
    # The stage-five chart of the nodal double quadric in four variables.
    # The residual is recorded in normal form modulo the exceptional locus.
    base = (a**5-e*t+1, b**5-d*e*t+1, e)
    section = -5*w*a**4*b**4+2*w**2*a*b-3*w**2*a-5*w**2*b+d*t+w**2+1
    cases = [
        ("principal", lambda: kernel.graph_strict_part(symbols, chart, p, equation),
         lambda r: r is not None and r == (3, (numerator,))),
        ("residual_dimension", lambda: kernel.residual_dimension_below(variables, base, (section,), 3),
         lambda r: r is True),
    ]
    output = {}
    for name, run, correct in cases:
        times = []
        for _ in range(repeat):
            start = perf_counter_ns()
            answer = run()
            times.append((perf_counter_ns()-start)/1000)
            if not correct(answer):
                raise RuntimeError(f"Atlas disagreed on {name}: {answer}")
        output[name] = {"Atlas_us": times}
    singular = shutil.which("Singular")
    if not singular:
        raise SystemExit("Singular is required for the comparison.")
    program = f"""
system("--ticks-per-sec",1000000);
ring r=0,(x,y,z),dp;
poly p=x^2+1;
ideal a=z-p*y;
poly f={large}*z^3+p^4;
poly expected={large}*y^3+p;
int k, orders, start, elapsed; ideal I, cut;
for(k=0;k<{repeat};k++) {{
  start=rtimer; cut=std(a+ideal(p)); I=std(a+ideal(f)); orders=0;
  while(size(reduce(I,cut))==0) {{I=std(quotient(I,ideal(p))); orders++;}}
  elapsed=rtimer-start;
  if(orders==3 && size(reduce(ideal(expected),I))==0 && size(reduce(I,std(a+ideal(expected))))==0)
    {{print("principal "+string(elapsed));}}
}}
ring r2=0,(w,d,t,a,b,e),(dp(3),dp(3));
ideal base=a^5-e*t+1,b^5-d*e*t+1,e;
poly section={str(section).replace("**","^")};
int k, start, elapsed, n;
for(k=0;k<{repeat};k++) {{
  start=rtimer; n=dim(std(base+ideal(section))); elapsed=rtimer-start;
  if(n==2) {{print("residual_dimension "+string(elapsed));}}
}}
quit;
"""
    result = subprocess.run([singular, "-q", "--no-rc"], input=program, text=True,
                            capture_output=True, check=True, timeout=60)
    for name, record in output.items():
        times = [int(line.split()[1]) for line in result.stdout.splitlines() if line.startswith(name+" ")]
        if len(times) != repeat:
            raise RuntimeError(f"Singular failed the {name} agreement check: {result.stdout}")
        record["Singular_us"] = times
        record["median_us"] = {solver: median(record[solver+"_us"]) for solver in ("Atlas", "Singular")}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("Use at least two trials.")
    print(json.dumps({"machine": platform.machine(), "field": "QQ",
                      "timing": "microseconds, warm processes, exact inputs already built, no correctness-check time",
                      "Singular_residual_order": "dp(w,d,t) then dp(a,b,e)",
                      "samples": samples(args.trials)}, indent=2))
