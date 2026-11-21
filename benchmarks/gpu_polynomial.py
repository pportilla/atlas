#!/usr/bin/env python3
"""Time the same sparse polynomial requests on FLINT CPU and a physical Triton device."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.algebra_execution import algebra_execution
from engine.gpu_devices import triton_device
from engine.gpu_polynomial import compute_row
from engine.polynomial import format_poly, sparse_row


def cases():
    dense = ";".join(f"{i%19-9} {i}" for i in range(512) if i%19 != 9)
    # All degrees are occupied, including coefficients on both sides of zero.
    consecutive = ";".join(f"{(i%23+1)*(-1)**i} {i}" for i in range(512))
    sparse = ";".join(f"{i-35}/41 {3*i} {2*i%11}" for i in range(70) if i!=35)
    line = ";".join(f"{(i%11)+1}/7 {i} {i%4}" for i in range(32))
    yield "add", ("add", 1, 1, [dense, consecutive])
    yield "subtract", ("subtract", 1, 1, [dense, consecutive])
    yield "sparse-product", ("multiply", 2, 2, [sparse, sparse])
    yield "dense-product", ("multiply", 1, 1, [consecutive, consecutive])
    yield "wide-product", ("multiply", 0, 0, [str(2**2048+1),str(3**1300-5)])
    yield "divide", ("divide", 2, 2, [sparse, "-19/23 0 0"])
    yield "power", ("power", 1, 1, ["1 0;1 1;1 2"], 35)
    yield "differentiate", ("differentiate", 2, 2, [sparse], 0)
    yield "evaluate", ("substitute", 2, 0, [line, "2", "-1/3"])
    yield "compose", ("substitute", 2, 2, ["1 3 1;2 4 2", "1 0 1;-1 1 0", "2 1 1;3 0 0"])


def singular(call, reference, trials):
    op, n, m, rows, *parameter = call
    variables = ["x", "y"][:max(1, n, m)]

    def expression(row, count):
        return format_poly(sparse_row(row), variables[:count])

    code = [f"ring R=0,({','.join(variables)}),lp;", f"poly f={expression(rows[0], n)};"]
    if op == "substitute":
        images = [expression(row, m) for row in rows[1:]]
        code.append(f"map image=R,{','.join(images)};")
        calculate = "image(f)"
    elif op == "power":
        calculate = "f^" + str(parameter[0])
    elif op == "differentiate":
        calculate = "diff(f," + variables[parameter[0]] + ")"
    else:
        code.append(f"poly g={expression(rows[1], m)};")
        calculate = "f" + {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op] + "g"
    code.extend([f"poly target={expression(reference, m)};", f"poly val={calculate};",
                 "int start,elapsed,i;", f"for(i=1;i<={trials};i++){{",
                 f'start=rtimer; val={calculate}; elapsed=rtimer-start;',
                 'print("RESULT "+string(elapsed)+" "+string(val==target));}', "quit;"])
    cmd = ["Singular", "-q", "--no-rc", "--threads=1", "--ticks-per-sec=1000000"]
    proc = subprocess.run(cmd, input="\n".join(code), text=True, capture_output=True, check=True)
    values = [line.split() for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if len(values) != trials or any(ok != "1" for _, _, ok in values) or "?" in proc.stdout:
        raise ValueError("Singular disagreed: " + proc.stdout)
    times = [int(elapsed) / 1000 for _, elapsed, _ in values]
    return {"ms": times, "median_ms": statistics.median(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--singular", action="store_true", help="Also time Singular and check every result")
    args = parser.parse_args()
    device = triton_device(args.index)
    if not device:
        parser.error("A physical Triton device is required for the comparison.")
    if args.trials < 1:
        parser.error("Choose a positive trial count.")
    result = {"device": device, "trials": args.trials, "cases": {}}
    for name, call in cases():
        reference = compute_row(*call)
        timings = {}
        for backend in ("cpu", "gpu"):
            with algebra_execution(backend, backend="triton", index=args.index) as run:
                if compute_row(*call) != reference:
                    raise ArithmeticError(f"{name} warmup disagreed")
                elapsed = []
                for _ in range(args.trials):
                    start = time.perf_counter_ns()
                    answer = compute_row(*call)
                    elapsed.append((time.perf_counter_ns() - start) / 1e6)
                    if answer != reference:
                        raise ArithmeticError(f"{name} disagreed on {backend}")
                receipt = run.report()
                if backend == "gpu" and (receipt["backend"] != "triton" or receipt["fallback_reasons"]):
                    raise RuntimeError(f"{name} did not complete on Triton: {receipt}")
            timings[backend] = {"ms": elapsed, "median_ms": statistics.median(elapsed)}
        if args.singular:
            timings["singular"] = singular(call, reference, args.trials)
        result["cases"][name] = timings
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
