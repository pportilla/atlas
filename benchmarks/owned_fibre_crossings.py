"""Compare exact Jacobian rank tests on a reducible owned fibre with Singular."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import native_polynomial as np
from engine.centres import GiveUp
from engine.local_blowups import Chart
from engine.repeated_tangent import certify_fibre


def measure(trials, batch):
    x, y, z = variables = tuple(map(np.Symbol, "xyz"))
    scale = 2**140 + 17
    rows = {}
    for name, exponent, expected in (("crossing", 1, True), ("tangency", 2, False)):
        f = (1+x**6)*y**exponent + scale*z
        c = Chart(variables, (), (("E", 1, 1, (z,)), ("strict", 1, 1, (f,))),
                  (x*(x-1),), (), (), ())
        atlas = []
        for _ in range(trials):
            start = perf_counter_ns()
            for _ in range(batch):
                try:
                    certify_fibre(c, 3)
                    answer = True
                except GiveUp as error:
                    if "fails normal crossings" not in str(error):
                        raise
                    answer = False
                if answer != expected:
                    raise AssertionError(f"Unexpected Atlas result for {name}.")
            atlas.append((perf_counter_ns()-start)/1000/batch)
        # E is a coordinate plane. The rank-one Jacobian of f and the
        # rank-two Jacobian of (z,f) on their respective cuts suffice.
        singular = f"""
system("--ticks-per-sec",1000000);
ring r=0,(x,y,z),dp;
poly f=(1+x^6)*y^{exponent}+{scale}*z;
poly fibre=x*(x-1);
int i,j,ok;
int start;
for(i=0;i<{trials};i++){{
  start=timer;
  for(j=0;j<{batch};j++){{
    ok=(reduce(1,std(ideal(fibre,f,diff(f,x),diff(f,y),diff(f,z))))==0)
       && (reduce(1,std(ideal(fibre,z,f,diff(f,x),diff(f,y))))==0);
    if(ok!={int(expected)}){{ print("WRONG"); quit; }}
  }}
  print("TIME "+string(timer-start));
}}
quit;
"""
        out = subprocess.run(["Singular", "-q", "--no-rc"], input=singular, text=True,
                             capture_output=True, check=True)
        times = [int(line[5:])/batch for line in out.stdout.splitlines() if line.startswith("TIME ")]
        if len(times) != trials or "WRONG" in out.stdout:
            raise RuntimeError(out.stdout + out.stderr)
        rows[name] = {"snc": expected, "atlas_us": atlas, "singular_us": times}
    cpuinfo = Path("/proc/cpuinfo")
    processor = platform.processor()
    if cpuinfo.is_file():
        processor = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), processor)
    return {"machine": platform.machine(), "processor": processor,
            "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
            "conditions": "Warm CPU processes, fibres and equations as in the script. "
                          "Atlas reads owned chart ideals. Singular uses the explicit Jacobians for z and f.",
            "batch": batch, "trials": trials, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()
    if min(args.trials, args.batch) < 1:
        parser.error("--trials and --batch must be positive.")
    print(json.dumps(measure(args.trials, args.batch), indent=2))
