"""Compare all affine pivot charts of triangular and non-graph centres over QQ."""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.native_library import ideal_request
from engine.polynomial import format_poly, parse_expression, sparse_row


BATCH = 25


def singular_program(names, equations, answer):
    k = len(equations)
    ratios = [f"r{i+1}" for i in range(k-1)]
    names = names + ratios
    source = ['LIB "elim.lib";system("--ticks-per-sec",1000000);option(redSB);',
              f'ring R=0,({",".join(names)}),lp;', f'ideal C={",".join(equations)};',
              'int total;int start;int iteration;int ok=1;']
    tail = iter(answer)
    for i in range(k):
        count, strict = map(int, next(tail).split())
        if strict != -1:
            raise AssertionError("Ambient charts expected")
        expected = ",".join(format_poly(sparse_row(next(tail)), names) for _ in range(count)) or "0"
        equations = ",".join(f'C[{j+1}]-C[{i+1}]*{r}'
                             for j, r in zip((j for j in range(k) if i != j), ratios))
        source += [f'ideal G{i}={equations};ideal S{i};start=rtimer;for(iteration=0;iteration<{BATCH};iteration++){{'
                   f'S{i}=sat(G{i},ideal(C[{i+1}]));S{i}=std(S{i});}}total=total+rtimer-start;',
                   f'ideal B{i}=std(ideal({expected}));'
                   f'ok=ok && size(reduce(S{i},B{i}))==0 && size(reduce(B{i},S{i}))==0;']
    if list(tail):
        raise AssertionError("Extra chart rows")
    source.append('print("CHECK "+string(ok)+" "+string(total));quit;')
    return "\n".join(source)


def measure(names, equations, trials):
    rows = [parse_expression(f, names, bounded=False, as_row=True) for f in equations]
    header = f"charts {len(names)} 0 -1 0"
    samples = {k: [] for k in ("baseline", "atlas", "singular")}
    os.environ["ATLAS_LINEAR_ALGEBRA"] = "0"
    expected = ideal_request(header, rows)
    program = singular_program(names, equations, expected)
    for i in range(trials):
        for label in (samples if i % 2 == 0 else reversed(samples)):
            if label == "singular":
                result = subprocess.run(["Singular", "-q", "--no-rc"], input=program, text=True,
                                        capture_output=True, check=True, timeout=120)
                checks = [l.split() for l in result.stdout.splitlines() if l.startswith("CHECK ")]
                if (len(checks) != 1 or checks[0][1] != "1"
                        or any(l.lstrip().startswith("?") for l in result.stdout.splitlines())):
                    raise ValueError(result.stdout)
                samples[label].append(int(checks[0][2]) / BATCH)
            else:
                os.environ["ATLAS_LINEAR_ALGEBRA"] = "1" if label == "atlas" else "0"
                start = perf_counter_ns()
                for _ in range(BATCH):
                    result = ideal_request(header, rows)
                samples[label].append((perf_counter_ns()-start)/(1000 * BATCH))
                if result != expected:
                    raise ValueError("Chart presentation changed")
    return dict(equations=equations, variables=names, samples_us=samples,
                median_us={k: median(v) for k, v in samples.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    os.sched_setaffinity(0, {args.cpu})
    conditions = ("Warm Atlas ideal_request including QQ sparse text transport. Singular microsecond "
                  "rtimer for sat and reduced lex std on each of the same pivot graph ideals. "
                  "All chart bases checked for mutual containment over QQ outside the timer. "
                  "Alternating order, single logical CPU.")
    report = dict(cpu=args.cpu, trials=args.trials, batch=BATCH, conditions=conditions,
                  singular=subprocess.check_output(["Singular","--version"], text=True).splitlines()[0], cases={})
    for name, axes, centre in (
            ("triangular", "xyz", ["y-x^2", "z-x*y"]),
            ("mixed_linear", "xyz", ["y+z-x^2", "2*y-z+x^3"]),
            ("three_rows", "xyzt", ["y-x^3", "z-x*y", "t-y-z"]),
            ("rational_wide", "xyz", ["12384761283471987341239876213498761298347/11*y-x^8", "z-x*y"]),
            ("conic", "xyz", ["x^2+y^2-1", "z"]),
            ("common_factor", "xyz", ["x*y", "x*z"])):
        report["cases"][name] = measure(list(axes), centre, args.trials)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
