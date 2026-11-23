"""Time the boundary defect of smooth curve centres in Atlas and Singular."""
import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import native_polynomial as np
from engine.blowup_ideal import ideal_basis
from engine.centre_weights import centre_rows
from engine.kernel import exprs
from engine.kernel_json import kernel_json


def measure(trials, batch):
    x, y, z = variables = tuple(map(np.Symbol, "xyz"))
    results = {}
    for exponent in (1, 2, 4):
        # On z=0, y=0 contains the whole curve. On z=1 it cuts out x^e.
        centre = (z*(z-1), y-z*x**exponent)
        arguments = dict(width=3, dimension="1", ideal=centre_rows(variables, centre),
                         divisors=[centre_rows(variables, (y,))])
        expected = ideal_basis(variables, (x**(exponent-1), z-1, y))
        atlas = []
        for _ in range(trials):
            start = perf_counter_ns()
            for _ in range(batch):
                rows = kernel_json("atlas_graph_wide", "centre_boundary_defect", **arguments)
            atlas.append((perf_counter_ns()-start)/1000/batch)
            actual = (1,) if rows is None else exprs(variables, rows)
            if ideal_basis(variables, actual) != expected:
                raise AssertionError(f"Atlas defect differs at exponent {exponent}.")
        program = f'''
LIB "elim.lib"; ring R=0,(x,y,z),dp;
proc outside(ideal C,ideal E){{ def A=sat(C,E); if(typeof(A)=="list"){{return(A[1]);}} return(A); }}
ideal C=z*(z-1),y-z*x^{exponent}; ideal E=y; ideal W=std(ideal(x^{exponent-1},z-1,y));
ideal cut,defect; int i,j,start;
system("--ticks-per-sec",1000000);
for(i=0;i<{trials};i++){{
  start=rtimer;
  for(j=0;j<{batch};j++){{
    cut=outside(C,E)+E;
    defect=std(cut+minor(jacob(cut),3));
  }}
  print("TIME "+string(rtimer-start));
  if(size(reduce(defect,W)) || size(reduce(W,defect))){{ print("WRONG"); quit; }}
}}
quit;'''
        result = subprocess.run(["Singular", "-q", "--no-rc"], input=program, text=True,
                                capture_output=True, timeout=30, check=True)
        singular = [int(line[5:])/batch for line in result.stdout.splitlines() if line.startswith("TIME ")]
        if len(singular) != trials or "WRONG" in result.stdout or "?" in result.stdout:
            raise RuntimeError(result.stdout + result.stderr)
        results[exponent] = {"atlas_us": atlas, "singular_us": singular,
                             "median_us": {"Atlas": median(atlas), "Singular": median(singular)}}
    return {"conditions": "QQ[x,y,z], CPU, in-process batches after parsing. Atlas includes JSON transport. "
                          "Singular saturates and computes Jacobian minors at the expected codimension. "
                          "Both check equality of the defect ideals after timing.",
            "trials": trials, "batch": batch, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()
    if min(args.trials, args.batch) < 1:
        parser.error("--trials and --batch must be positive")
    print(json.dumps(measure(args.trials, args.batch), indent=2))
