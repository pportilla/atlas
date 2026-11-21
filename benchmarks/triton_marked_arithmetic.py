"""Compare exact operations in QQ(i)[x,y] via public Atlas dispatch and Singular."""
import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.algebra import dispatch
from engine.gpu_devices import triton_device


def workloads():
    f="(x+a*y+1)^5"
    g="(a*x+y-2)^3"
    wide=str(2**255+19)
    return [
        ("sum", "add", f, {"other_polynomial": g}, "f+g"),
        ("difference", "subtract", f, {"other_polynomial": g}, "f-g"),
        ("product", "multiply", f, {"other_polynomial": g}, "f*g"),
        ("wide-product", "multiply", wide+"*a*x+y", {"other_polynomial":g}, "f*g"),
        ("scalar-division", "divide", f, {"other_polynomial":"1+a"}, "f/(1+a)"),
        ("derivative", "differentiate", f, {"variable":"y"}, "diff(f,y)"),
        ("power", "power", "a*x+y+1", {"exponent":"8"}, "f^8"),
        ("substitution", "substitute", f, {"images":["a*y","(1+a)*x"]}, "phi(f)"),
    ]


def singular(source, args, expression, target, trials):
    # Parse the same defining expressions inside the timed region of each system.
    setup = f"f={source};"
    if "other_polynomial" in args:
        setup += 'g='+args["other_polynomial"]+';'
    if "images" in args:
        setup += 'map phi=R,'+','.join(args["images"])+';'
    cleanup = "kill phi;" if "images" in args else ""
    code=f"""
ring R=(0,a),(x,y),dp;
minpoly=a^2+1;
poly f,g,v; poly target={target};
int i,start,elapsed;
for(i=0;i<={trials};i++) {{
 start=rtimer; {setup} v={expression}; elapsed=rtimer-start;
 print("CASE "+string(v==target)+" "+string(elapsed)); {cleanup}
}}
quit;
"""
    proc=subprocess.run(["Singular","-q","--no-rc","--threads=1","--ticks-per-sec=1000000"],
                        input=code, text=True, capture_output=True, check=True)
    rows=[l.split() for l in proc.stdout.splitlines() if l.startswith("CASE ")]
    if len(rows)!=trials+1 or any(r[1]!="1" for r in rows) or "?" in proc.stdout:
        raise RuntimeError(proc.stdout+proc.stderr)
    return [int(r[2])/1000 for r in rows[1:]]


def run(trials):
    device=triton_device(0)
    if device is None: raise RuntimeError("A physical Triton device is required.")
    result={"device":device,"trials":trials,"cases":{}}
    for name, action, source, args, expression in workloads():
        req={"operation":"polynomial-arithmetic","ring":{"variables":["x","y"],"field":{"generator":"a","embedding":"I"}},
             "operands":dict(action=action, polynomial=source, **args)}
        reference=dispatch(dict(req, compute="cpu"))["result"]
        samples={"cpu":[],"gpu":[]}
        for trial in range(trials+1):
            for mode in (("cpu","gpu") if trial%2 else ("gpu","cpu")):
                start=perf_counter_ns()
                out=dispatch(dict(req,compute=mode))
                elapsed=(perf_counter_ns()-start)/1e6
                if out["result"]!=reference or out["execution"]["backend"]!=("cpu" if mode=="cpu" else "triton"):
                    raise ArithmeticError(f"Inconsistent result or backend in {name}, {mode}: {out}")
                if trial: samples[mode].append(elapsed)
        samples["singular"]=singular(source,args,expression,reference["polynomial"],trials)
        result["cases"][name]={"samples_ms":samples,"median_ms":{k:median(v) for k,v in samples.items()}}
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=7)
    args=parser.parse_args()
    if args.trials<1: parser.error("Use a positive trial count.")
    print(json.dumps(run(args.trials),indent=2))
