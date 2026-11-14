"""Compare the certificates of the same two toric parent maps and their transition."""
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
sys.path.insert(0,str(ROOT))
from engine import native_polynomial as np
from engine.centre_weights import centre_rows
from engine.repair_gluing import _repair_geometry
from engine.kernel import exprs


def singular(projection, expected, loops):
    # Matrices are the same local exponents used by the production torus map.
    # The denominator without t^2-2 reduction would fail the first map check.
    return f'''LIB "linalg.lib";ring R=0,(x,y,t),dp;
matrix A[2][2]=1,{projection},0,1;matrix B[2][2]=1,0,0,1;
matrix L[2][2]=1/9,0,0,-2;matrix M[2][2];ideal field=std(t^2-2);
ideal left=x/9+(t^2-2)*x,-2*x^{projection}*y;ideal right=x/9,-2*y;
proc checkMap(matrix S,ideal coord){{int j;int k;poly r;for(j=1;j<=2;j++){{
r=0;for(k=1;k<=2;k++){{r=r+L[j,k]*x^int(S[1,k])*y^int(S[2,k]);}}
if(reduce(coord[j]-r,field)!=0){{return(0);}}}}return(1);}}
int ok;int count;system("--ticks-per-sec",1000000);int start=rtimer;
for(count=1;count<={loops};count++){{ok=(det(A)^2==1 && det(B)^2==1 && det(L)!=0);
ok=ok && checkMap(A,left) && checkMap(B,right);if(ok){{M=A*inverse(B);}}}}
int elapsed=rtimer-start;ok=ok && M[1,1]==1 && M[1,2]=={expected} && M[2,1]==0 && M[2,2]==1;
print("RESULT "+string(elapsed)+" "+string(ok));quit;
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",default=5,type=int)
    parser.add_argument("--loops",default=30,type=int)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if min(args.loops,args.trials)<1:
        parser.error("Both loop counts must be positive")
    x,y,t=axes=np.symbols("x y t")
    relation=centre_rows(axes,(t*t-2,))[0]
    def parent(exponent):
        return dict(width=3,names=["X","Y"],modulus=relation,
                    rays=[["1",str(exponent)],["0","1"]],linear=[["1/9","0"],["0","-2"]],
                    images=centre_rows(axes,(x/9+(t*t-2)*x if exponent else x/9,-2*x**exponent*y)))
    cases=[]
    for exponent in (1,16,97):
        left,right=parent(exponent),parent(0)
        def atlas():
            before=perf_counter_ns()
            for _ in range(args.loops):
                source=_repair_geometry("parent",**left)
                target=_repair_geometry("parent",**right)
                transition=_repair_geometry("transition",source=source,target=target)
            elapsed=(perf_counter_ns()-before)/1000/args.loops
            if exprs((x,y),transition)!=(x,1,x**exponent*y,1):
                raise ArithmeticError("Invalid marked transition")
            return elapsed
        def other():
            code=singular(exponent,exponent,args.loops)
            out=subprocess.run(["Singular","-q","--no-rc"],text=True,input=code,capture_output=True,
                               timeout=30,check=True).stdout.splitlines()
            if len(out)!=1 or out[0].split()[::2]!=["RESULT","1"]:
                raise ArithmeticError(f"Singular reported {out}")
            return int(out[0].split()[1])/args.loops
        samples={"atlas":[],"singular":[]}
        for i in range(args.trials):
            for name,call in ((("atlas",atlas),("singular",other)) if i%2==0 else (("singular",other),("atlas",atlas))):
                samples[name].append(call())
        cases.append(dict(exponent=exponent,expected="x, x^n*y",microseconds=samples,
                          median_us={name:median(t) for name,t in samples.items()}))
    result=dict(conditions="Both systems verify the same two saved polynomial base maps on t²=2, "
                "the unimodular rays and nonsingular rational map, then change bases. "
                "Atlas times three native calls including marshaling and returns numerator/denominator rows. "
                "Singular times its in-process polynomial checks and matrix change. Parsing and startup excluded.",
                loops=args.loops,trials=args.trials,cpu_affinity=sorted(os.sched_getaffinity(0)),
                singular=subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
                library_sha256=hashlib.sha256((ROOT/"native/build/libgroebner.so").read_bytes()).hexdigest(),cases=cases)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
