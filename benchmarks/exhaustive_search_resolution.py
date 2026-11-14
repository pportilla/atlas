"""Compare the fresh exhaustive CLI before and after a search implementation change.

The regular benchmark verifies the full 36-chart Atlas record, both zetas,
their saved-record replays, and Singular's two-chart blowup. Timings of search,
certification, the exact same Atlas witness in Singular, and Singular's own
resolution are recorded separately from the end-to-end CLI trials.
"""
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
from unittest.mock import patch

import sheared_centre_resolution
from inclined_centre_resolution import CASES, compact_report
from plane_conjugate_resolution import measure
from whitney_related import ROOT, arguments, chart_expression, require_singular, singular_script


# Child imports resolve from the measured checkout through its working directory.
ATLAS_PHASES = """
import json,sys,time
from unittest.mock import patch
from engine import atlas,exhaustive_resolution,resolve
from engine.native_library import replay_scope
record,mode,case=json.load(sys.stdin)
if mode=='supplied':
    from engine.general_resolution import verify_candidate
    from engine.native_polynomial import Symbol
    from engine.resolution_search import _candidate
    x,y,z=map(Symbol,'xyz');variables=(x,y,z);f=(x+y+z)**2-z*(x-y)**2
    candidate=_candidate(variables,(2*x+z,2*y+z),'supplied singular line')
    with replay_scope('cpu'):
        start=time.perf_counter();cpu=time.process_time()
        result=verify_candidate(variables,f,candidate,'embedded')
        duration=time.perf_counter()-start;work=time.process_time()-cpu
    assert len(result['charts'])==2 and result['dual_complex']['status']=='complete'
    print(json.dumps({'supplied_centre_ms':1000*duration,'supplied_centre_cpu_ms':1000*work,'record':result}))
elif mode=='saved':
    with replay_scope('cpu'):
        charts=atlas.from_json(record['atlas'],native=True)
        start=time.perf_counter();cpu=time.process_time()
        answer=atlas.outputs(charts,3,complex=False,integrals=False)
        duration=time.perf_counter()-start;work=time.process_time()-cpu
    assert answer=={'cells':None,'reason':None} and len(charts)==36
    print(json.dumps({'owned_chart_checks_ms':1000*duration,'owned_chart_checks_cpu_ms':1000*work}))
else:
    phases={'search_ms':0.0,'outputs_ms':0.0,'search_calls':0,'output_calls':0}
    def watch(function,field,count):
        def wrapped(*args,**kw):
            start=time.perf_counter()
            try:return function(*args,**kw)
            finally:
                phases[field]+=(time.perf_counter()-start)*1000
                phases[count]+=1
        return wrapped
    with patch.object(atlas,'outputs',watch(atlas.outputs,'outputs_ms','output_calls')),patch.object(exhaustive_resolution,'search',watch(exhaustive_resolution.search,'search_ms','search_calls')):
        start=time.perf_counter()
        result=resolve.resolve(dict(polynomial=case,variables=list('xyz'),compute='cpu',mode='embedded',search='exhaustive'))
        phases['resolve_ms']=(time.perf_counter()-start)*1000
    result.pop('elapsed_ms')
    assert result==record, 'instrumented resolver changed the complete timed record'
    print(json.dumps(phases))
"""


def supplied_centre_check(record, singular, timeout):
    # Compare the Atlas graph and its strict/exceptional ideals directly with
    # Singular's affine opens for the same given ideal, in both directions.
    program = ['LIB "resolve.lib";ring graph=0,(x,y,z,r),dp;',
               'ideal centre=2*x+z,2*y+z;poly f=(x+y+z)^2-z*(x-y)^2;',
               'ring source=0,(x,y,z),dp;poly original=(x+y+z)^2-z*(x-y)^2;',
               'list charts=blowUp(ideal(original),ideal(2*x+z,2*y+z));int i,j,ok,which;intvec counts=0,0;',
               'for(i=1;i<=size(charts);i++){def C=charts[i];setring C;',
               'map into=graph,bM[1],bM[2],bM[3],var(3);ideal generators=into(centre);',
               'which=0;if(generators[1]*var(3)==generators[2]){which=1;}',
               'if(generators[2]*var(3)==generators[1]){which=2;}',
               'ok=(size(aS)==0 && size(sT)==1 && size(eD)==1 && which!=0);']
    for index, chart in enumerate(record["charts"], 1):
        x, y, z, ratio = chart["coordinates"]
        if ([x, y, z] != list("xyz") or chart["map"] != dict(zip("xyz","xyz"))
                or chart["projective_coordinates"] != (["1",ratio] if index==1 else [ratio,"1"])):
            raise ValueError("Unexpected Atlas graph map for the given centre")
        def group(ideal):
            return ",".join(re.sub(rf"\b{ratio}\b", "r", chart_expression(g, chart["coordinates"])) for g in ideal)
        relation = "r*(2*x+z)-(2*y+z)" if index==1 else "r*(2*y+z)-(2*x+z)"
        program.extend([f'if(which=={index}){{counts[which]=counts[which]+1;setring graph;',
                        f'ideal ambient=std(ideal({group(chart["ambient_ideal"])}));ideal rel=std(ideal({relation}));',
                        f'ideal h={group(chart["strict_transform"])};ideal e={group(chart["exceptional"][0]["ideal"])};',
                        'ok=ok && size(reduce(ambient,rel))==0 && size(reduce(rel,ambient))==0;',
                        'setring C;ideal strict=std(into(h));ideal exceptional=std(into(e));',
                        'ok=ok && size(reduce(strict,std(sT)))==0 && size(reduce(sT,strict))==0;',
                        'ok=ok && size(reduce(exceptional,std(eD)))==0 && size(reduce(eD,exceptional))==0;',
                        'setring graph;kill ambient,rel,h,e;setring C;}'])
    program.extend(['print("SAME_OPEN "+string(which)+" "+string(ok));setring source;kill C;}',
                    'print("PIVOTS "+string(counts));quit;'])
    _, answer = measure(singular, ROOT, timeout, "\n".join(program))
    checks = sorted(line for line in answer.splitlines() if line.startswith(("SAME_OPEN ", "PIVOTS ")))
    if any(line.lstrip().startswith("?") for line in answer.splitlines()) or checks != ["PIVOTS 1,1","SAME_OPEN 1 1","SAME_OPEN 2 1"]:
        raise ValueError("Atlas and Singular differ on the given-centre charts.\n"+answer[-2000:])
    return checks


def slices(args, main):
    record = main["agreed_records"]["geometry"]
    polynomial = CASES["inclined"].polynomial
    singular = [shutil.which("Singular"), "-q", "--no-rc", "--ticks-per-sec=1000"]
    chart_inputs = []

    def capture(command, cwd, timeout, program):
        chart_inputs.append(program)
        return measure(command, cwd, timeout, program)

    with patch.object(sheared_centre_resolution, "measure", capture):
        checks = sheared_centre_resolution.atlas_geometry(record, CASES["inclined"], singular, args.timeout)
    (chart_program,) = chart_inputs
    if len(checks) != 36:
        raise ValueError("Expected the complete 36-chart witness")
    resolution_program = singular_script(CASES["inclined"], checked=False).replace(
        'LIB "resolve.lib";', 'int loadStart=timer;LIB "resolve.lib";int loadTime=timer-loadStart;').replace(
        "list resolved=resolve(", "int workStart=timer;list resolved=resolve(").replace(
        'print("COUNT ', 'print("TIMING "+string(loadTime)+" "+string(timer-workStart));print("COUNT ')
    chart_checks = "\n".join(checks)
    roots = {"before":args.baseline.resolve(), "after":args.atlas_root.resolve()}
    supplied_program = ('LIB "resolve.lib";ring src=0,(x,y,z),dp;poly f=(x+y+z)^2-z*(x-y)^2;'
                        'ideal centre=2*x+z,2*y+z;int start=timer;list charts=blowUp(ideal(f),centre);'
                        'print("SUPPLIED "+string(size(charts))+" "+string(timer-start));quit;')
    supplied_record = None
    labels = ["before_resolution", "after_resolution", "before_saved", "after_saved",
              "singular_36_chart_check", "singular_default_with_internal_timer", "after_supplied", "singular_supplied"]
    rows = {label: [] for label in labels}
    for trial in range(-1, args.trials):
        for label in (labels if trial % 2 == 0 else reversed(labels)):
            if label == "singular_supplied":
                sample, out = measure(singular, ROOT, args.timeout, supplied_program)
                recognized = re.fullmatch(r"SUPPLIED 2 (\d+)\s*", out)
                if not recognized:
                    raise ValueError("Singular did not return the two supplied-centre charts")
                sample["supplied_centre_cpu_ms"] = int(recognized[1])
            elif label.startswith("singular"):
                witness = label == "singular_36_chart_check"
                sample, out = measure(singular, ROOT, args.timeout, chart_program if witness else resolution_program)
                if witness:
                    results = [line for line in out.splitlines() if line.startswith("ATLAS ")]
                    if results != checks or any(line.lstrip().startswith("?") for line in out.splitlines()):
                        raise ValueError("The independent 36-chart witness verification failed")
                else:
                    require_singular(out, False)
                    (timing,) = [line for line in out.splitlines() if line.startswith("TIMING ")]
                    sample["load_cpu_ms"], sample["resolve_cpu_ms"] = map(int, timing.split()[1:])
            else:
                stage, mode = label.split("_")
                source = json.dumps([record, mode, polynomial])
                sample, out = measure([sys.executable,"-c",ATLAS_PHASES], roots[stage], args.timeout, source)
                sample.update(json.loads(out))
                if mode == "supplied":
                    result = sample.pop("record")
                    if supplied_record is None:
                        supplied_record = result
                    elif result != supplied_record:
                        raise ValueError("Atlas returned a different record for the given centre")
            if sample["status"] != "complete":
                raise TimeoutError(label)
            if trial >= 0:
                rows[label].append(sample)
    medians = {label:{field:statistics.median(r[field] for r in samples)
                      for field in samples[0] if field!="status"} for label,samples in rows.items()}
    return dict(runs=rows, medians=medians, chart_count=len(checks),
                chart_program_sha256=hashlib.sha256(chart_program.encode()).hexdigest(),
                chart_output_sha256=hashlib.sha256(chart_checks.encode()).hexdigest(),
                program=chart_program, singular_resolution_program=resolution_program,
                supplied_centre=dict(singular_program=supplied_program, atlas_record=supplied_record,
                                     singular_checks=supplied_centre_check(supplied_record, singular, args.timeout)))


def first_centre(case, timeout):
    program = singular_script(case).removesuffix("quit;\n") + 'print("TREE "+string(size(resolved[2])));quit;\n'
    command = [shutil.which("Singular"), "-q", "--no-rc"]
    _, output = measure(command, ROOT, timeout, program)
    require_singular(output, True)
    block = output.split("Upcoming Center", 1)[1].split("\n-", 1)[0]
    generators = re.findall(r"^_\[\d+\]=(.+)$", block, flags=re.MULTILINE)
    if len(generators) != 2 or "TREE 3" not in output.splitlines():
        raise ValueError("Expected one blowup with two standard opens")
    # Short Singular display elides products in 2y and 15x. The displayed
    # first centres of these examples are all linear over QQ.
    expanded = [re.sub(r"(\d)([xyz])", r"\1*\2", text) for text in generators]
    program = [f'ring source=0,({",".join(case.variables)}),dp;',
               f'ideal reported=std(ideal({",".join(expanded)}));',
               f'ideal frame=std(ideal({",".join(case.pivots)}));',
               'print("SAME "+string(size(reduce(reported,frame))==0 && size(reduce(frame,reported))==0));quit;']
    _, answer = measure(command, ROOT, timeout, "\n".join(program))
    if answer.splitlines() != ["SAME 1"]:
        raise ValueError("The centre displayed by Singular differs from the checked blowup frame")
    return {"reported_generators":generators, "adapted_generators":case.pivots,
            "tree_nodes":3, "final_charts":2, "ideals_equal":True}


if __name__ == "__main__":
    args = arguments(CASES, __doc__)
    output = args.output
    compact_report(args)
    compressed = output.suffix == ".gz"
    data = output.read_bytes()
    report = json.loads(gzip.decompress(data) if compressed else data)
    report["singular_first_centres"] = {name:first_centre(CASES[name], args.timeout) for name in report["cases"]}
    if "inclined" in report["cases"]:
        report["main_resolution_and_chart_slices"] = slices(args, report["cases"]["inclined"])
    for phase, tree in report["roots"].items():
        manifest = Path(tree) / "source-hashes.json"
        if manifest.exists():
            values = json.loads(manifest.read_bytes())
            if not all(hashlib.sha256((Path(tree)/file).read_bytes()).hexdigest()==expected
                       for file, expected in values.items()):
                raise ValueError(f"Frozen files changed in {tree}")
            report.setdefault("full_frozen_manifests", {})[phase] = values
    report["benchmark_dependencies_sha256"]["inclined_centre_resolution.py"] = report["benchmark_sha256"]
    report["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content = json.dumps(report, indent=2).encode() + b"\n"
    output.write_bytes(gzip.compress(content, mtime=0) if compressed else content)
