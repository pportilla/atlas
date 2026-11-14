"""Compare the exact torus singular schemes used to choose finite Newton repairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import native_polynomial as np
from engine.centre_weights import centre_rows
from engine.newton_repair import repair_operation

CASES = (
    ("isolated", "(x-2)^2+(y+3)^2", ("x-2", "y+3"), "points"),
    ("curve", "x^17*y^4*(y-2)^3", ("(y-2)^2",), "positive_dimension"),
    ("empty", "x^17*y^4*(x+y)", ("1",), "empty"),
    ("two_points", "(x^2+1)^2+(y-2)^2", ("x^2+1", "y-2"), "points"),
)


def singular_program(f, ideal, loops):
    return f'''LIB "elim.lib";ring R=0,(x,y),lp;
proc torusSaturation(ideal equations, ideal unit){{def result=sat(equations,unit);
if(typeof(result)=="list"){{return(result[1]);}}return(result);}}
poly f={f};ideal wanted={','.join(ideal)};
system("--ticks-per-sec",1000000);int i;int start=rtimer;ideal actual;
for(i=1;i<={loops};i++){{actual=std(torusSaturation(ideal(f,diff(f,x),diff(f,y)),ideal(x*y)));}}
int elapsed=rtimer-start;
int ok=size(reduce(actual,std(wanted)))==0 && size(reduce(wanted,actual))==0;
print("RESULT "+string(elapsed)+" "+string(ok));quit;
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=5)
    parser.add_argument("--loops",type=int,default=30)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.trials<1 or args.loops<1:
        parser.error("Each comparison needs at least one trial and one operation.")
    x,y=variables=np.symbols("x y")
    from engine.polynomial import parse_expression
    from engine.blowup_ideal import ideal_basis
    from engine.kernel import exprs

    def expression(s):
        return np.from_terms(variables,parse_expression(s,["x","y"]))

    data={}
    for name,f,ideal,kind in CASES:
        row=centre_rows(variables,(expression(f),))[0]
        expected=ideal_basis(variables,tuple(map(expression,ideal)))
        def atlas_trial():
            start=perf_counter_ns()
            for _ in range(args.loops):
                r=repair_operation("singular_orbit",width=2,equation=row)
            elapsed=(perf_counter_ns()-start)/1000/args.loops
            if r["kind"]!=kind or ideal_basis(variables,exprs(variables,r["ideal"]))!=expected:
                raise ValueError(f"Wrong Newton singular scheme for {name}")
            return elapsed
        def singular_trial():
            out=subprocess.run(["Singular","-q","--no-rc"],input=singular_program(f,ideal,args.loops),text=True,
                               capture_output=True,check=True,timeout=30).stdout.splitlines()
            if len(out)!=1 or out[0].split()[::2]!=["RESULT","1"]:
                raise ValueError(f"Singular failed the exact ideal comparison: {out}")
            return int(out[0].split()[1])/args.loops
        atlas,other=[],[]
        for trial in range(args.trials):
            if trial%2==0: atlas.append(atlas_trial())
            other.append(singular_trial())
            if trial%2: atlas.append(atlas_trial())
        data[name]={"equation":f,"singular_ideal":ideal,"kind":kind,"atlas_us":atlas,"singular_us":other,
                    "median_us":{"atlas":median(atlas),"singular":median(other)}}
    result={"conditions":"QQ[x,y], torus x*y != 0. Both systems compute the saturated Jacobian ideal; "
            "both compare it to the indicated ideal outside the timer. Atlas also returns its dimension class. "
            "Atlas times native calls including transport; Singular times in-process std(sat(J,xy)) "
            "after input parsing. Timers exclude process startup. Alternating batches per trial.",
            "trials":args.trials,"loops_per_trial":args.loops,"cpu_affinity":sorted(os.sched_getaffinity(0)),
            "library_sha256":hashlib.sha256((ROOT/'native/build/libgroebner.so').read_bytes()).hexdigest(),
            "singular_version":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],"cases":data}
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
