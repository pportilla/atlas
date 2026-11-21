"""Compare public modular batches with exact CPU output and the same arithmetic in Singular."""
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


def cases(count):
    p = 65521
    a = [2*i - count for i in range(count)]
    b = [i%127+1 for i in range(count)]
    for action in ("reduce","add","subtract","multiply","divide","inverse","reconstruct","hensel","crt"):
        values = b if action=="inverse" else a
        extras = {}
        if action in ("add","subtract","multiply","divide","crt"):
            extras["other"] = b
        if action=="reconstruct":
            values = [((i%41-20)*pow(23,-1,p))%p for i in range(count)]
        if action=="hensel":
            values = [-2 if i%2 else 2 for i in range(count)]
            extras["other"] = [-4,0,1]
        if action=="crt":
            extras["other_modulus"] = 65519
        yield {"action":action,"values":values,"modulus":p,**extras}


def singular(data, expected, repeats):
    # Time a warm Singular loop over prepared bigint/number entries, excluding result checks.
    action, p = data["action"],data["modulus"]
    n=len(data["values"])
    m=data.get("other_modulus",p)
    a=",".join(map(str,data["values"]))
    b=",".join(map(str,data["other"])) if "other" in data and action!="hensel" else "1"
    target=",".join(expected)
    operations = {
        "add":"value=(a[1,k]+b[1,k])%p;", "subtract":"value=(a[1,k]-b[1,k])%p;",
        "multiply":"value=(a[1,k]*b[1,k])%p;", "reduce":"value=a[1,k]%p;",
        "divide":"L=extgcd(b[1,k],p);value=(a[1,k]*bigint(L[2]))%p;",
        "inverse":"L=extgcd(a[1,k],p);value=bigint(L[2])%p;",
        "crt":"L=extgcd(p,m);value=(a[1,k]%p)+p*(((b[1,k]-(a[1,k]%p))*bigint(L[2]))%m);",
        "hensel":"u=a[1,k]%p;L=extgcd(2*u,p);value=u+p*((((4-u*u) div p)*bigint(L[2]))%p);",
        "reconstruct":"result[k]=farey(a[1,k],p);",
    }
    body=operations[action]+("" if action=="reconstruct" else "result[k]=value;")
    code=f'''
ring R=0,x,lp;
bigint p={p}; bigint m={m}; bigint u,value;
bigintmat a[1][{n}]={a}; bigintmat b[1][{len(data["other"]) if "other" in data and action!="hensel" else 1}]={b};
list target={target}; list result,L; int k,j,ok,start,elapsed;
'''
    code+=f'''for(j=0;j<={repeats};j++) {{
  start=rtimer;
  for(k=1;k<={n};k++) {{ {body} }}
  elapsed=rtimer-start;
  ok=1;for(k=1;k<={n};k++) {{ if (number(result[k])!=number(target[k])) {{ok=0;}} }}
  print("CHECK "+string(ok)+" "+string(elapsed));
}}
quit;
'''
    command=["Singular","-q","--no-rc","--threads=1","--ticks-per-sec=1000000"]
    run=subprocess.run(command,input=code,text=True,capture_output=True,check=True,timeout=90)
    lines=[line.split() for line in run.stdout.splitlines() if line.startswith("CHECK ")]
    if any(ok!="1" for _,ok,_ in lines) or len(lines)!=repeats+1 or "?" in run.stdout:
        raise RuntimeError(f"Singular failed {action}: {run.stdout[:1000]}")
    return [int(t)/1000 for _,_,t in lines[1:]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--singular", action="store_true")
    opts=parser.parse_args()
    if opts.count < 1 or opts.trials < 1:
        parser.error("Choose positive batch size and trial count")
    device=triton_device(0)
    if not device:
        parser.error("A physical Triton device is required")
    report={"device":device,"count":opts.count,"trials":opts.trials,"cases":{}}
    for data in cases(opts.count):
        timings={"cpu":[],"gpu":[]}
        request={"operation":"modular-arithmetic","ring":{"variables":[]},"operands":data}
        expected=dispatch({**request,"compute":"cpu"})["result"]
        for trial in range(opts.trials+1):
            for mode in (("gpu","cpu") if trial%2 else ("cpu","gpu")):
                start=perf_counter_ns()
                record=dispatch({**request,"compute":mode})
                elapsed=(perf_counter_ns()-start)/1e6
                if record["result"]!=expected or record["execution"]["backend"]!=("triton" if mode=="gpu" else "cpu"):
                    raise RuntimeError(f"{data['action']}: mismatched answer or missing device work")
                if trial:
                    timings[mode].append(elapsed)
        if opts.singular:
            timings["singular"]=singular(data,expected["values"],opts.trials)
        report["cases"][data["action"]]={k:{"ms":v,"median_ms":median(v)} for k,v in timings.items()}
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
