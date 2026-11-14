"""Warm CPU resolution timing, including parsing and any native helper process."""
import statistics
import sys
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.resolve import resolve

for f,vs in [('y^2-x^3','xy'),('(y^2-2*x^4)^2-x^11','xy'),('x*y-z^21','xyz'),
             ('x^2+y^3+z^5','xyz'),('x^3+y^3+z^3+w^3','xyzw')]:
    data=dict(polynomial=f,variables=list(vs),mode='intrinsic',compute='cpu')
    times=[]
    for i in range(8):
        t=perf_counter();r=resolve(data);d=(perf_counter()-t)*1000
        if i: times.append(d)
    assert r['status']=='resolved'
    print(f"{f:27s} {len(r['charts']):3d} charts  {len(r['dual_complex']['vertices']):3d} vertices   {statistics.median(times):8.3f} ms")
