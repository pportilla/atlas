"""Time the intrinsic reduced exceptional section on the two developable charts."""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.kernel_json import kernel_json
from engine.polynomial import encode_sparse_row, parse_expression


def chart(names, ambient, maps, strict, divisor):
    def rows(items):
        return [encode_sparse_row(parse_expression(f, list(names))) for f in items]
    return dict(width=len(names), ambient=rows(ambient), maps=rows(maps), holes=[], theta=-1,
                components=[["strict", rows([strict]), True], ["B1", rows([divisor]), False]])


FIRST = chart("xyr", [], ["x", "y", "(y-x^2)*r+x*y"], "4*x*r-r^2-4*y", "x^2-y")
SECOND = chart("xyzv", ["-x*y*v+z*v+x^2-y"], ["x", "y", "z"],
               "4*x*v-4*y*v^2-1", "x*y-z")
EXPECTED = dict(status="complete", convention="Exceptional divisor on the smooth strict transform in "
                "the germ over the origin over Qbar.", vertices=[dict(id="B1", dimension=1)], cells=[])


def atlas(batch):
    start = perf_counter_ns()
    for _ in range(batch):
        if kernel_json("atlas_chart_geometry", "single_curve", charts=[FIRST, SECOND]) != EXPECTED:
            raise ValueError("Atlas did not certify the single geometric vertex")
    return (perf_counter_ns() - start) / (1000 * batch)


def singular(batch):
    # The two inclusions independently prove that the reduced ideal is
    # the parametrized line with r free. Check its fibre is nonempty and
    # that the strict transform misses the entire fibre on the other chart.
    source = f"""
    LIB "primdec.lib";
    ring a=0,(x,y,r),dp;
    ideal D=x^2-y,4*x*r-r^2-4*y;
    ideal G=2*x-r,4*y-r^2;
    ideal M=x,y,(y-x^2)*r+x*y;
    ideal A;
    ring b=0,(x,y,z,v),dp;
    ideal B=-x*y*v+z*v+x^2-y,4*x*v-4*y*v^2-1,x,y,z;
    int i,bad;
    system("--ticks-per-sec",1000000);
    int start=rtimer;
    for (i=1;i<={batch};i++) {{
      if (reduce(1,std(B))!=0) {{ bad=1; }}
      setring a;
      A=std(radical(D));
      if (size(reduce(A,std(G))) || size(reduce(G,A)) || reduce(1,std(D+M))==0) {{ bad=1; }}
      setring b;
    }}
    print("SECTION "+string(bad)+" "+string(rtimer-start));
    quit;
    """
    result = subprocess.run(["Singular", "-q", "--no-rc"], input=source, text=True,
                            capture_output=True, timeout=120, check=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith("SECTION ")]
    if len(lines) != 1 or lines[0].split()[1] != "0" or "error occurred" in result.stdout:
        raise ValueError(result.stdout + result.stderr)
    return int(lines[0].split()[2]) / batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.trials < 1:
        parser.error("batch and trials must be positive")
    os.sched_setaffinity(0, {args.cpu})
    atlas(1)
    samples = {"atlas": [], "singular": []}
    for i in range(args.trials):
        for runner in ((atlas, singular) if i % 2 == 0 else (singular, atlas)):
            samples[runner.__name__].append(runner(args.batch))
    report = dict(cpu=args.cpu, batch=args.batch, trials=args.trials, samples_us=samples,
                  median_us={name: median(v) for name, v in samples.items()},
                  singular_version=subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
                  conditions="Intrinsic reduced section of s2-developable-123 over QQ. "
                             "Atlas JSON call in process; Singular radical, two ideal inclusions and fibre checks "
                             "in fresh sessions, timed internally after startup.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["median_us"])


if __name__ == "__main__":
    main()
