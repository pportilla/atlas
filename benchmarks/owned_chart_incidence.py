"""Compare owned exceptional incidence on rational smooth graph components in Atlas and Singular."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import native_polynomial as np
from engine.affine_kernel import decode
from engine.local_blowups import Chart, cells

# Each listed component is a polynomial graph over QQ or a union of
# disjoint rational graph components. Thus every Singular minimal prime
# on these particular inputs remains prime after base change to Qbar.
CASES = {
    "three_coordinate_divisors": (["x","y","z"], ["x*y*z"], [], []),
    "punctured_graphs": (["y-x^2","z-x*y"], ["(y-x^2)*(z-x*y)"], [], [["x","z"]]),
    "disjoint_parallel_graphs": (["x*(x-1)","y-z^2"], ["x*(x-1)*(y-z^2)"], [], [["x-1","y"]]),
}
PROC = '''
proc cellCounts(list forms,ideal fibre,ideal region,list holes){
  int n=size(forms);int mask,j,k,m,over,owned,value;string out="";
  ideal intersection,visible,P;list primes;
  for(mask=1;mask<2^n;mask++){
    intersection=0;m=mask;
    for(j=1;j<=n;j++){if(m%2){intersection=intersection,forms[j];} m=m div 2;}
    visible=intersection+fibre+region;
    for(j=1;j<=size(holes);j++){visible=sat(visible,holes[j])[1];}
    if(dim(std(visible))<0){continue;}
    primes=minAssGTZ(intersection);value=0;over=1;
    for(k=1;k<=size(primes);k++){
      P=std(primes[k]);
      if(size(reduce(fibre,P))){over=0;break;}
      owned=(size(reduce(region,P))==0);
      for(j=1;j<=size(holes);j++){if(size(reduce(holes[j],P))==0){owned=0;}}
      value=value+owned;
    }
    out=out+string(mask)+"="+string(value*over-(1-over))+" ";
  }
  return(out);
}
'''

def singular(executable, spec, trials, batch):
    forms, fibre, region, holes = spec
    group = lambda entries: 'ideal('+','.join(entries or ['0'])+')'
    command = '\n'.join(['LIB "primdec.lib";','ring R=0,(x,y,z),dp;',PROC,
                         f'list F=list({",".join(forms)});',f'ideal B={group(fibre)};ideal W={group(region)};',
                         f'list H=list({",".join(group(h) for h in holes)});',
                         'string answer=cellCounts(F,B,W,H);print("ANSWER "+answer);',
                         f'int i,j,t;for(i=0;i<{trials};i++){{t=timer;for(j=0;j<{batch};j++)'
                         '{answer=cellCounts(F,B,W,H);}print("TIME "+string(timer-t));}', 'exit;'])
    output=subprocess.check_output([executable,'-q','--no-rc','--threads=1','--flint-threads=1',
                                    '--ticks-per-sec=1000000'],input=command,text=True,cwd=ROOT)
    lines=output.splitlines()
    if any(line.lstrip().startswith('?') for line in lines):
        raise RuntimeError(output)
    answer=next(line[7:] for line in lines if line.startswith('ANSWER '))
    return dict(tuple(map(int,v.split('='))) for v in answer.split()),[
        int(line[5:])/batch/1000 for line in lines if line.startswith('TIME ')]

def atlas(spec, trials, batch):
    axes=np.symbols('x y z')
    inputs,fibre,region,holes=spec
    read=lambda text:decode(axes,text)
    components=tuple((str(i),1,1,(read(f),)) for i,f in enumerate(inputs))
    chart=Chart(axes,(),components,tuple(map(read,fibre)),tuple(map(read,region)),
                tuple((str(x),x) for x in axes),(),tuple(tuple(map(read,h)) for h in holes))
    def run():
        return cells((chart,),3,rational_counts={})
    answer=run()
    samples=[]
    for _ in range(trials):
        t=perf_counter_ns()
        for _ in range(batch):
            assert run()==answer
        samples.append((perf_counter_ns()-t)/batch/1e6)
    return {sum(1<<int(i) for i in key): n if n is not None else -1 for key,n in answer.items()},samples

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--trials',type=int,default=7)
    parser.add_argument('--batch',type=int,default=20)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    exe=shutil.which('Singular')
    if not exe:parser.error('Singular is required.')
    report=dict(trials=args.trials,batch=args.batch,affinity=sorted(os.sched_getaffinity(0)),
                conditions='Warm in-process Atlas chart kernel versus Singular minimal primes, saturation and ideal containment. Every geometric minimal prime in these inputs is defined over QQ. Times in milliseconds per chart; excludes setup and process startup.',cases={})
    for name,spec in CASES.items():
        counts,at=atlas(spec,args.trials,args.batch)
        agree,si=singular(exe,spec,args.trials,args.batch)
        if counts!=agree or len(si)!=args.trials:raise ValueError((name,counts,agree,si))
        report['cases'][name]=dict(inputs=spec,counts=counts,atlas_ms=at,singular_ms=si,
                                   median_ms={'Atlas':statistics.median(at),'Singular':statistics.median(si)})
    data=json.dumps(report,indent=2)+'\n'
    if args.output:args.output.write_text(data)
    else:print(data)

if __name__=='__main__':
    main()
