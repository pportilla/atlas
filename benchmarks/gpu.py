#!/usr/bin/env python3
"""Real CPU vs OpenCL solve. Missing/fallback GPU is never counted as a GPU run."""
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.native_backend import gpu_devices
from engine.resolve import resolve


def example(degree):
    rng=random.Random(9721+degree)
    words=[]
    for i in range(degree+1):
        for j in range(degree-i+1):
            powers=(i,j,degree-i-j)
            n=rng.choice([-9,-7,-5,-3,-2,-1,1,2,3,5,7,9])
            factors=[str(abs(n))]+[v if e==1 else f'{v}^{e}' for v,e in zip('xyz',powers) if e]
            words+=['-' if n<0 else '+','*'.join(factors)]
    return {'polynomial':''.join(words).lstrip('+')+f'+x^{degree+1}', 'variables':list('xyz'),'mode':'intrinsic'}


def geometry(r):
    whole=[r[k] for k in ('polynomial','mode','charts','steps','dual_complex')]
    c=r['certificate']
    whole.append([c[k] for k in ('prime','degree','rank','rows')])
    return hashlib.sha256(json.dumps(whole,sort_keys=True).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--degrees',type=int,nargs='+',default=[4,8,11])
    ap.add_argument('--repeats',type=int,default=7);ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if not 1<=args.repeats<=100: ap.error('use 1–100 repeats')
    devices=gpu_devices()
    if not devices: ap.exit(2,'No OpenCL GPU visible. Build with make gpu and check your driver.\n')
    try: cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except (OSError,StopIteration): cpu=platform.processor()
    report={'cpu':cpu,'devices':devices,'platform':platform.platform(),'python':platform.python_version(),'repeats':args.repeats,'cases':[]}
    os.environ['RESOLUTION_GPU_MIN_COLS']='1'
    for degree in args.degrees:
        req=example(degree);case={'degree':degree,'input':req,'samples':{},'median_ms':{},'cli_median_ms':{},'cli_samples_ms':{}}
        expected=None
        def check(r,backend):
            nonlocal expected
            c=r.get('certificate',{})
            if c.get('rank_backend')!=backend: raise SystemExit(f'Expected {backend} but got {c}. No timing comparison.')
            digest=geometry(r)
            if expected and digest!=expected: raise SystemExit('Geometry or certificate mismatch.')
            expected=digest
            case.update(matrix=[c['rows'],c['rank']],prime=c['prime'],geometry_sha256=digest,charts=len(r['charts']))
        for name,policy in [('cpu','off'),('opencl','auto')]:
            os.environ['RESOLUTION_GPU']=policy
            check(resolve(req),name) # warm-up for both paths
            case['samples'][name]=[]
            case['cli_samples_ms'][name]=[]
        for repeat in range(args.repeats):
            order=[('cpu','off'),('opencl','auto')]
            if repeat%2: order.reverse()
            for name,policy in order:
                os.environ['RESOLUTION_GPU']=policy
                t=perf_counter();r=resolve(req);wall=(perf_counter()-t)*1000;check(r,name)
                case['samples'][name].append({'solve':wall,'rank':r['certificate']['rank_elapsed_ms']})
                # Measure a newly started CLI (including its isolated worker and JSON output).
                t=perf_counter();cmd=[str(ROOT/'atlas'),'resolve',req['polynomial'],'--vars','x,y,z','--mode','intrinsic','--json']
                cli=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60)
                case['cli_samples_ms'][name].append((perf_counter()-t)*1000);check(json.loads(cli.stdout),name)
        for name in ('cpu','opencl'):
            case['median_ms'][name]={k:statistics.median(sample[k] for sample in case['samples'][name]) for k in ('rank','solve')}
            case['cli_median_ms'][name]=statistics.median(case['cli_samples_ms'][name])
        print(degree,case['matrix'],case['median_ms'],case['cli_median_ms'],flush=True)
        report['cases'].append(case)
    if args.output: args.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
