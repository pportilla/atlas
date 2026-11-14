"""Compare fresh public Atlas and Singular QQ ideal operations on identical inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median

from plane_conjugate_resolution import measure


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "eliminate": (['--gen', 'x^2-y', '--gen', 'y^2-z', '--gen', 'x*z-1', '--eliminate', 'x'],
                  'eliminate(ideal(x^2-y,y^2-z,x*z-1),x)'),
    "intersection": (['--gen', 'x^2-y', '--gen', 'z', '--other-gen', 'x*y', '--other-gen', 'z^2'],
                     'intersect(ideal(x^2-y,z),ideal(x*y,z^2))'),
    "quotient": (['--gen', 'x^2*y', '--gen', 'x*z', '--other-gen', 'x', '--other-gen', 'y'],
                 'quotient(ideal(x^2*y,x*z),ideal(x,y))'),
    "saturation": (['--gen', 'x^2*y', '--gen', 'x*z', '--poly', 'x'],
                   'sat(ideal(x^2*y,x*z),ideal(x))'),
}

def program(order, operation):
    return ('LIB "elim.lib";ring R=0,(x,y,z),' + ('lp' if order == 'lex' else 'dp') +
            ';option(redSB);short=0;ideal I=std(' + operation + ');')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--atlas-root', type=Path, default=ROOT)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.trials<1:
        parser.error('A positive trial count is required')
    if args.cpu is not None:
        os.sched_setaffinity(0,{args.cpu})
    singular=[shutil.which('Singular'),'-q','--no-rc']
    roots={'before':args.baseline.resolve(), 'after':args.atlas_root.resolve()}
    libraries={k:hashlib.sha256((p/'native/build/libgroebner.so').read_bytes()).hexdigest() for k,p in roots.items()}
    report=dict(cpu=args.cpu,trials=args.trials,native_sha256=libraries,
                conditions='Alternating fresh processes with one discarded warmup; every result compared within each order. Singular returns the reduced basis. Atlas returns a complete JSON record.', cases={})
    for op,(operands,target) in CASES.items():
        for order in ('lex','grevlex'):
            runs=[];expected=None;seen=None
            for i in range(-1,args.trials):
                for key in (('before','after','singular') if i%2==0 else ('singular','after','before')):
                    if key=='singular':
                        row,raw=measure(singular,ROOT,30,program(order,target)+'print("BASIS "+string(I));quit;')
                        if any(line.lstrip().startswith('?') for line in raw.splitlines()):
                            raise ValueError(raw)
                        reply=next(line for line in raw.splitlines() if line.startswith('BASIS '))
                        if seen is not None and seen!=reply:raise ValueError('Singular basis changed between runs')
                        seen=reply
                    else:
                        row,raw=measure(['./atlas','algebra',op,'--vars','x,y,z','--compute','cpu','--json',
                                         '--order',order,*operands],roots[key],30)
                        result=json.loads(raw)['result']
                        if expected is not None and result!=expected:raise ValueError('Atlas results differ')
                        expected=result
                    if row['status']!='complete': raise ValueError(f'Unfinished {op}')
                    if i>=0:runs.append(dict(solver=key,trial=i+1,**row))
            polys=','.join(expected['generators']) or '0'
            check=program(order,target)+'ideal J=std(ideal('+polys+'));print("PARITY "+string(size(reduce(I,J))==0 && size(reduce(J,I))==0));quit;'
            _,raw=measure(singular,ROOT,30,check)
            if 'PARITY 1' not in raw.splitlines():raise ValueError(raw)
            medians={k:{field:median([r[field] for r in runs if r['solver']==k]) for field in ('wall_s','cpu_s')}
                     for k in ('before','after','singular')}
            report['cases'][op+'-'+order]=dict(operands=operands,runs=runs,median=medians,atlas=expected,singular=seen,mutual_ideal_check=check)
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
