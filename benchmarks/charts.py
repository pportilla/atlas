#!/usr/bin/env python3
"""Time a multi-chart resolution on CPU and on GPU, including the full CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.native_backend import gpu_devices
from engine.resolve import resolve

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--power',type=int,default=27)
    ap.add_argument('--repeats',type=int,default=7)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if not 1<=args.power<=36 or not 1<=args.repeats<=50: ap.error('power 1–36 and repeats 1–50 required')
    devices=gpu_devices()
    if not devices: ap.exit(2,'No OpenCL GPU visible. Build with make gpu and check the driver.\n')
    f=f'x^2+y^19+z^37+x^3*(x+y+z)^{args.power}'
    request=dict(polynomial=f,variables=['x','y','z'],mode='intrinsic')
    data=dict(input=request,devices=devices,platform=platform.platform(),repeats=args.repeats,samples={'cpu':[],'opencl':[]})
    reference=None
    def check(r,mode):
        nonlocal reference
        info=r.get('certificate',{}).get('chart_batch',{})
        if info.get('backend')!=mode: raise SystemExit(f'Expected {mode}, got {info}.')
        geometry=[r[k] for k in ('polynomial','charts','dual_complex','fan','root_families')]
        digest=hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()
        if reference and digest!=reference: raise SystemExit('CPU and GPU atlases differ.')
        reference=digest
        data.update(geometry_sha256=digest,charts=len(r['charts']),chart_term_pairs=info['chart_term_pairs'],
            vertices=len(r['dual_complex']['vertices']),cells=len(r['dual_complex']['cells']))
        return info
    for mode,policy in [('cpu','cpu'),('opencl','gpu')]:
        check(resolve(request|{'compute':policy}),mode)
    for repeat in range(args.repeats):
        order=[('cpu','cpu'),('opencl','gpu')]
        if repeat%2: order.reverse()
        for mode,policy in order:
            t=perf_counter();r=resolve(request|{'compute':policy});solve_ms=(perf_counter()-t)*1000
            info=check(r,mode)
            cmd=[str(ROOT/'atlas'),'resolve',f,'--vars','x,y,z','--mode','intrinsic','--compute',policy,'--json']
            t=perf_counter();res=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60)
            cli_ms=(perf_counter()-t)*1000;check(json.loads(res.stdout),mode)
            data['samples'][mode].append({'native_transform_ms':info['native_elapsed_ms'],'batch_ms':info['elapsed_ms'],
                'solve_ms':solve_ms,'cli_ms':cli_ms})
    data['median_ms']={m:{k:statistics.median(x[k] for x in samples) for k in samples[0]} for m,samples in data['samples'].items()}
    print(data['chart_term_pairs'],'chart-term pairs;',data['charts'],'charts;',data['median_ms'])
    if args.output: args.output.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__': main()
