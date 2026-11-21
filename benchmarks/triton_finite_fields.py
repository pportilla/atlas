"""Compare complete extension-field batches on Atlas CPU, physical Triton and Singular."""
import argparse
import json
import platform
import random
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.algebra import dispatch
from engine.gpu_devices import triton_device


def cases(count):
    for p, poly, actions in [
        (5, [1, 1, 0, 1], ("reduce", "add", "subtract", "multiply", "divide", "inverse")),
        (2, [1, 1, 0, 1, 1, 0, 0, 0, 1], ("multiply", "inverse")),
        (536870879, [1, 0, 1], ("multiply", "inverse")),
        (2147483647, [1, 0, 1], ("multiply", "inverse")),
    ]:
        d = len(poly)-1
        rng = random.Random(9397 + p)
        left = [[rng.randrange(-p, p) for _ in range(d)] for _ in range(count)]
        rhs = [[rng.randrange(p) for _ in range(d)] for _ in range(count)]
        right = [row if any(row) else [1] for row in rhs]
        unreduced = [row+[rng.randrange(p) for _ in range(d-1)] for row in left]
        for action in actions:
            args = {"action": action, "modulus": p, "field_polynomial": poly,
                    "values": right if action == "inverse" else unreduced if action == "reduce" else left}
            if action not in ("inverse", "reduce"):
                args["other"] = right
            yield f"{p}^{d} {action}", args


def expression(coefficients, variable):
    terms = [f"({c})*{variable}^{j}" if j else f"({c})"
             for j, c in enumerate(coefficients) if int(c)]
    return "+".join(terms) or "0"


def singular(data, expected, trials):
    p, poly, action = (data[key] for key in ("modulus", "field_polynomial", "action"))
    count = len(expected)
    # Singular's algebraic-number inversion uses Factory, whose characteristic limit is 2**29.
    # At the largest prime, compute with polynomial representatives and a Bezout lift.
    representatives = p > 2**29
    field_variable = "x" if representatives else "a"
    typename = "poly" if representatives else "number"
    if representatives:
        code = [f"ring R={p},x,dp;", f"poly m={expression(poly, 'x')};",
                "ideal relation=std(ideal(m)); matrix coefficients;"]
    else:
        code = [f"ring R=({p},a),x,dp;", f"minpoly={expression(poly, 'a')};"]
    code.append("list lhs,rhs,actual,want;int k,trial,ok,start,elapsed;")
    for i, row in enumerate(data["values"], 1):
        if action == "reduce":
            code.append(f"lhs[{i}]=poly({expression(row, 'x')});")
        else:
            code.append(f"lhs[{i}]={typename}({expression(row, field_variable)});")
    for i, row in enumerate(data.get("other", []), 1):
        code.append(f"rhs[{i}]={typename}({expression(row, field_variable)});")
    for i, row in enumerate(expected, 1):
        code.append(f"want[{i}]={typename}({expression(row, field_variable)});")
    if representatives:
        statement = {
            "multiply": "actual[k]=reduce(lhs[k]*rhs[k],relation);",
            "inverse": "coefficients=lift(ideal(lhs[k],m),ideal(1));actual[k]=reduce(coefficients[1,1],relation);",
        }[action]
    else:
        operation = {
            "reduce": "leadcoef(subst(lhs[k],x,a))", "add": "lhs[k]+rhs[k]", "subtract": "lhs[k]-rhs[k]",
            "multiply": "lhs[k]*rhs[k]", "divide": "lhs[k]/rhs[k]", "inverse": "number(1)/lhs[k]",
        }[action]
        statement = f"actual[k]={operation};"
    code.append(f"""for(trial=0;trial<={trials};trial++){{
      start=rtimer;for(k=1;k<={count};k++){{{statement}}}elapsed=rtimer-start;
      ok=1;for(k=1;k<={count};k++){{if(actual[k]!=want[k]){{ok=0;}}}}
      print("CHECK "+string(ok)+" "+string(elapsed));
    }}
    quit;""")
    run = subprocess.run(["Singular", "-q", "--no-rc", "--threads=1", "--ticks-per-sec=1000000"],
                         input="\n".join(code), text=True, capture_output=True, check=True, timeout=240)
    rows = [line.split() for line in run.stdout.splitlines() if line.startswith("CHECK ")]
    if len(rows) != trials+1 or any(ok != "1" for _, ok, _ in rows) or "?" in run.stdout:
        raise ValueError(f"Singular disagreed on {p}, {action}: {run.stdout[:2000]}")
    return [int(tick)/1000 for _, _, tick in rows[1:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--trials", type=int, default=7)
    opts = parser.parse_args()
    if opts.count < 1 or opts.trials < 1:
        parser.error("Provide positive batch size and trial count")
    device = triton_device(0)
    if not device:
        parser.error("A physical Triton device is required")
    version = subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0]
    report = {"device": device, "platform": platform.platform(), "singular": version, "entries": opts.count,
              "warmups": 1, "trials": opts.trials, "cases": {}}
    for name, data in cases(opts.count):
        request = {"operation": "modular-arithmetic", "ring": {"variables": []}, "operands": data}
        expected = dispatch(dict(request, compute="cpu"))["result"]
        timings = {"cpu": [], "gpu": []}
        for trial in range(opts.trials+1):
            for mode in (("gpu", "cpu") if trial % 2 else ("cpu", "gpu")):
                start = perf_counter_ns()
                record = dispatch(dict(request, compute=mode))
                duration = (perf_counter_ns()-start)/1e6
                backend = "triton" if mode == "gpu" else "cpu"
                if record["execution"]["backend"] != backend or record["result"] != expected:
                    raise RuntimeError(f"{name} failed to dispatch or disagreed on {mode}")
                if trial:
                    timings[mode].append(duration)
        timings["singular"] = singular(data, expected["values"], opts.trials)
        report["cases"][name] = {key: {"ms": times, "median_ms": median(times)} for key, times in timings.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
