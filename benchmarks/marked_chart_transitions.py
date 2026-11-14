"""Repeat exact marked-field cancellations and finite chart sections against Singular."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from engine import native_polynomial as np
from engine.atlas_finite import marked_primes
from engine.chart_gluing_kernel import rows, values
from engine.kernel_json import kernel_json
from engine.local_blowups import Chart
from engine.native_library import _LIBRARY


def run(trials, count):
    x,y,t = axes = np.symbols("x y theta")
    relation = t**2-2
    chart = Chart(axes,(relation,),(),(),(),(),(),coefficient_field=(t,relation,2,"root 1 (FLINT order)"))
    scale=2**180+13
    common=(x+t)*y**70
    encoded=rows(axes,(scale*(y+1)*common,(y-1)*common))
    field=rows(axes,(relation,))[0]

    def cofactor():
        result=kernel_json("atlas_marked_chart","cofactors",width=3,theta=2,relation=field,pair=encoded)
        if values(axes,result)!=(scale*(y+1),y-1):
            raise ValueError("Wrong marked polynomial cofactors.")

    def split():
        result=marked_primes(chart,(x*x-2,y-x*x),0)
        expected={(x-t,y-2),(x+t,y-2)}
        if set(result)!=expected:
            raise ValueError(f"Incorrect sheets: {result}")

    base='ring R=(0,T),(x,y),lp;minpoly=T^2-2;\n'
    cases={
        "cofactors": (cofactor, f'poly a={scale}*(y+1)*(x+T)*y^70;poly b=(y-1)*(x+T)*y^70;poly g,n,d;',
                      'g=gcd(a,b); g=g/leadcoef(g);n=a/g;d=b/g;',
                      f'n=={scale}*(y+1) && d==y-1'),
        "two_marked_points": (split,'ideal I=x^2-2,y-x^2;ideal A=x-T,y-2;ideal B=x+T,y-2;list pieces;',
                              'pieces=minAssGTZ(I);',
                              'size(pieces)==2 && ((eq(pieces[1],A) && eq(pieces[2],B)) || (eq(pieces[1],B) && eq(pieces[2],A)))'),
    }
    record={}
    for label,(call,setup,operation,check) in cases.items():
        times={"Atlas":[],"Singular":[]}
        body=('LIB "primdec.lib";\n'+base+'proc eq(ideal a,ideal b){return(size(reduce(a,std(b)))==0 && size(reduce(b,std(a)))==0);}\n'
              +setup+f'\nint i;{operation}\nint start=rtimer;for(i=1;i<={count};i++){{{operation}}}\n'
              +'int elapsed=rtimer-start;print("RESULT "+string(elapsed)+" "+string('+check+'));quit;\n')
        for j in range(trials):
            for name in (tuple(times) if j%2==0 else tuple(reversed(times))):
                if name=="Atlas":
                    call()
                    start=perf_counter_ns()
                    for _ in range(count):
                        call()
                    times[name].append((perf_counter_ns()-start)/count/1000)
                else:
                    p=subprocess.run(["Singular","-q","--no-rc","--random=47","--threads=1","--ticks-per-sec=1000000"],
                                     input=body,capture_output=True,text=True,check=True,timeout=60)
                    assertions=[row for row in p.stdout.splitlines() if row.startswith("RESULT ")]
                    if p.stderr or "?" in p.stdout or len(assertions)!=1 or assertions[0].split()[2]!="1":
                        raise ValueError(p.stdout+p.stderr)
                    times[name].append(int(assertions[0].split()[1])/count)
        record[label]=dict(median_us={k:statistics.median(v) for k,v in times.items()},samples_us=times)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=5)
    parser.add_argument("--batch",type=int,default=15)
    parser.add_argument("--cpu",type=int,default=None)
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    if options.cpu is not None:
        os.sched_setaffinity(0,{options.cpu})
    if options.trials<2 or options.batch<1:
        parser.error("Use at least two trials and one operation in each batch.")
    result={"cpu":options.cpu,"trials":options.trials,"batch":options.batch,
            "singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
            "library_sha256":sha256(_LIBRARY.read_bytes()).hexdigest(),
            "conditions":"Fixed embedding of Q(sqrt(2)). Prepared inputs, alternating warm in-process operation batches. Atlas includes row/FFI and exact result validation. Singular uses its native coefficient field; startup and final verification excluded.",
            "cases":run(options.trials,options.batch)}
    options.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
