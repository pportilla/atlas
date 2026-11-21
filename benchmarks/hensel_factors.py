"""Compare verified Atlas and Singular lifts of the same ordered coprime factors."""

import argparse
import json
import os
import platform
from pathlib import Path
from random import Random
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.algebra import dispatch
from engine.gpu_devices import triton_device


def expression(coeffs):
    terms = []
    for k, c in reversed(list(enumerate(coeffs))):
        if not c:
            continue
        if k == 0:
            terms.append(str(c))
        else:
            term = "x" if k == 1 else f"x^{k}"
            terms.append(term if c == 1 else f"{c}*{term}")
    return "+".join(terms) or "0"


def cases():
    yield "quadratic", 7, "x^2-2", "x+4", "x+3", "x+39", "x+10"
    for d in (32, 128):
        p = 257
        rng = Random(d)
        g = [rng.randrange(1, p) for _ in range(d)] + [1]
        h = [rng.randrange(1, p) for _ in range(d + 1)] + [1]
        a = [c + p * rng.randrange(p) for c in g[:-1]] + [1]
        b = [c + p * rng.randrange(p) for c in h[:-1]] + [1]
        f = [0] * (len(a) + len(b) - 1)
        for i, c in enumerate(a):
            for j, e in enumerate(b):
                f[i + j] = (f[i + j] + c * e) % (p * p)
        yield f"dense-{d}", p, *map(expression, (f, g, h, a, b))


def singular(f, g, h, p, a, b, trials):
    source = f"""
ring Q=0,(x),dp;
ring K={p},(x),dp;
ring S=(integer,{p},2),(x),dp;
setring Q;
proc liftPair(poly f,poly g,poly h) {{
  poly e=(f-g*h)/number({p});
  setring K; poly e=imap(Q,e); poly g=imap(Q,g); poly h=imap(Q,h);
  list bez=extgcd(h,g);
  poly u=reduce(e*bez[2]/leadcoef(bez[1]),std(ideal(g))); poly v=(e-h*u)/g;
  setring Q; poly u=imap(K,u); poly v=imap(K,v);
  return(list(g+{p}*u,h+{p}*v));
}}
int i,start; poly f,g,h; list answer; poly A,B;
for(i=0;i<={trials};i++) {{
 start=rtimer; f={f}; g={g}; h={h}; answer=liftPair(f,g,h); A=answer[1];B=answer[2];
 setring S; poly A=imap(Q,A);poly B=imap(Q,B);print(rtimer-start);
 print((A==({a})) && (B==({b})) && ((A*B)==imap(Q,f))); kill A,B; setring Q;
}}
quit;
"""
    proc = subprocess.run(
        ["Singular", "-q", "--no-rc", "--threads=1", "--ticks-per-sec=1000000"],
        input=source,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    lines = proc.stdout.splitlines()
    if len(lines) != 2 * (trials + 1) or lines[1::2] != ["1"] * (trials + 1):
        raise RuntimeError(proc.stdout + proc.stderr)
    return [int(t) / 1000 for t in lines[2::2]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Use a positive number of trials.")
    os.environ["RESOLUTION_GPU_BACKEND"] = "triton"
    device = triton_device(0)
    if device is None:
        parser.error("Select a working physical Triton device at index zero.")
    import triton

    singular_version = subprocess.run(
        ["Singular", "--version"], text=True, capture_output=True, check=True
    ).stdout.splitlines()[0]
    out = {
        "device": device,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "singular": singular_version,
        "trials": args.trials,
        "milliseconds": {},
    }
    for label, p, f, g, h, a, b in cases():
        request = {
            "operation": "hensel-factors",
            "ring": {"variables": ["x"]},
            "operands": {"polynomial": f, "factors": [g, h], "modulus": str(p)},
        }
        times = {"cpu": [], "gpu": []}
        for k in range(args.trials + 1):
            for mode in ("cpu", "gpu") if k % 2 == 0 else ("gpu", "cpu"):
                start = perf_counter_ns()
                report = dispatch(dict(request, compute=mode))
                elapsed = (perf_counter_ns() - start) / 1e6
                result = report["result"]
                if (result["left"], result["right"], result["modulus"]) != (a, b, str(p * p)):
                    raise ValueError(f"{label}: incorrect {mode} lift")
                if mode == "gpu" and not any(
                    s["backend"] == "triton" and s["operation"] == "polynomial_multiply"
                    for s in report["execution"]["stages"]
                ):
                    raise RuntimeError(report["execution"])
                if k:
                    times[mode].append(elapsed)
        times["Singular"] = singular(f, g, h, p, a, b, args.trials)
        out["milliseconds"][label] = {
            "prime": p,
            "samples": times,
            "medians": {mode: median(obs) for mode, obs in times.items()},
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
