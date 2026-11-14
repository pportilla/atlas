"""Compare public embedded resolutions before and after trying the reduced singular locus as a centre.

Every timed Atlas run returns the complete ordered geometry, with both zeta functions requested in a separate series.
The competing Singular runs use the same polynomial with default and checked resolve.lib. Outside the timers, verify
the saved Atlas charts and both zetas, and compare the two graph presentations on every intersection of their opens.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

from exhaustive_search_resolution import first_centre
from inclined_centre_resolution import CASES as LINEAR_CASES, InclinedCentre, singular_fibre
from plane_conjugate_resolution import measure
from sheared_centre_resolution import atlas_geometry
from whitney_related import ROOT, arguments, chart_expression, replay, require_singular, run, singular_script


CASES = {**LINEAR_CASES, "curved": InclinedCentre("xyz", "y-x^2", "z", "x", ("x^2-y","z"))}


def curved_fibre(case, singular, timeout):
    # Here one Singular pivot is quadratic. The two verified projection
    # formulas reconstruct both parametrizations of the same smooth graph
    # curve, and the reciprocal ratio glues its two standard blowup opens.
    script = singular_script(case).replace("poly original=", "short=0;poly original=").removesuffix("quit;\n") + '''
print("TREE "+string(size(resolved[2])));int i,ok,pivot;
for(i=1;i<=size(resolved[1]);i++){
    def C=resolved[1][i];setring C;ideal image=BO[5];map phi=source,image;
    poly exc=BO[4][1][1];poly h=BO[2][1];poly ratio=var(3);
    poly axial=image[1];poly p=axial^2-image[2];poly q=image[3];
    pivot=0;
    if(q==p*ratio && p==exc && axial==var(1) && image[2]==var(2)){pivot=1;}
    if(p==q*ratio && q==exc && axial==var(1) && image[3]==var(2)){pivot=2;}
    poly full=phi(original);poly expected=exc^2*h;poly jac=det(jacob(image));
    ok=(nvars(basering)==3 && size(BO[1])==0 && size(BO[4])==1 && size(BO[2])==1);
    ok=ok && pivot!=0 && full!=0 && expected!=0 && leadcoef(full)*expected==leadcoef(expected)*full;
    ok=ok && jac!=0 && leadcoef(jac)*exc==leadcoef(exc)*jac;
    ideal fibre=std(image);ideal line=std(ideal(exc,axial));
    ok=ok && size(reduce(fibre,line))==0 && size(reduce(line,fibre))==0;
    poly restriction=reduce(h,fibre);
    if(pivot==1){ok=ok && restriction!=0 && deg(restriction)==0;}
    if(pivot==2){ok=ok && restriction!=0 && restriction==leadcoef(restriction)*ratio^2;}
    ideal cut=exc,h;ideal bad=std(fibre+cut+minor(jacob(cut),2));
    print("CURVED "+string(pivot)+" "+string(ok && bad[1]==1));setring source;kill C;
}
quit;
'''
    _, answer = measure(singular, ROOT, timeout, script)
    require_singular(answer, True)
    checked = sorted(line for line in answer.splitlines() if line.startswith(("TREE ", "CURVED ")))
    if checked!=["CURVED 1 1","CURVED 2 1","TREE 3"]:
        raise ValueError(f"Singular curved-centre projections or fibre failed: {checked}")
    return {"centre_generators":case.pivots, "centre_checks":checked, "origin_fibre":"P^1",
            "exceptional_N_nu":[2,2],"strict_N_nu":[1,1],"euler_exceptional_open":1,"euler_exceptional_strict":1,
            "monodromy":"1/(1-T^2)","milnor_fibre_euler":2,"topological":"(s + 2)/(2*(s + 1)^2)"}


def fibre_oracle(case, singular, timeout):
    return curved_fibre(case,singular,timeout) if case==CASES["curved"] else singular_fibre(case,singular,timeout)


def graph_checks(record, case, singular, timeout):
    public, charts = record["charts"], record["atlas"]
    if len(charts) != 2 or len(public) != 2:
        raise ValueError("Expected the two standard graph opens")
    names = list(case.variables)
    pivots = [chart_expression(c["pivot_generator"], names) for c in public]
    tangent = chart_expression(case.tangent, names)
    partner = chart_expression(case.partner, names)
    program = [singular_script(case).removesuffix("quit;\n"), 'int i,ok;',
               'proc sameIdeal(ideal a,ideal b){return(size(reduce(a,std(b)))==0 && size(reduce(b,std(a)))==0);}']
    checks = []
    for ix,(chart, visible) in enumerate(zip(charts, public, strict=True)):
        coords = chart["coordinates"]
        ideal_map = dict(zip(names, names))
        if (coords[:-1] != names or visible["coordinates"] != coords or chart["map"] != ideal_map
                or visible["map"] != ideal_map or chart["removed"]):
            raise ValueError("Expected an identity projection from the full graph chart")
        r = coords[-1]
        if visible["projective_coordinates"] != ([r,"1"] if ix else ["1",r]):
            raise ValueError("Wrong homogeneous graph coordinates")
        grouped = {c["id"]:c for c in chart["components"]}
        if set(grouped) != {"E1","strict"}:
            raise ValueError("Expected exactly one exceptional and one strict divisor")
        if [(grouped[c]["multiplicity"],grouped[c]["nu"]) for c in ("E1","strict")] != [(2,2),(1,1)]:
            raise ValueError("Incorrect multiplicity or Jacobian order")
        if [e["id"] for e in visible["exceptional"]] != ["E1"]:
            raise ValueError("Unexpected public exceptional component")
        def rows(values):
            return ",".join(chart_expression(g, coords) for g in values) or "0"
        # These generators need only span the same two-dimensional space as the
        # Singular generators. On the jth Singular open compare after adjoining
        # the inverse of p_i/exceptional. There the Atlas ratio is (p_k/exceptional)/(p_i/exceptional).
        pi,pk = pivots[ix],pivots[1-ix]
        ring = f"g{ix}"
        axis = ",".join(names)
        program.extend([
            f"ring {ring}=0,({','.join(coords)}),dp;poly pivot={pi};poly other={pk};",
            f"ideal relation={r}*pivot-other;ideal fibre={axis};",
            f"ideal ambient={rows(chart['ambient_ideal'])};ideal visibleAmbient={rows(visible['ambient_ideal'])};",
            f"ideal submittedFibre={rows(chart['fibre'])};ideal owner={rows(chart['region'])};",
            f"ideal strict={rows(grouped['strict']['ideal'])};ideal publicStrict={rows(visible['strict_transform'])};",
            f"ideal exceptional={rows(grouped['E1']['ideal'])};ideal publicEx={rows(visible['exceptional'][0]['ideal'])};",
            f"ok=sameIdeal(ideal(pivot,other),ideal({tangent},{partner})) && sameIdeal(relation,ambient)"
            " && sameIdeal(relation,visibleAmbient) && sameIdeal(submittedFibre+ambient,fibre+ambient);",
            f"ok=ok && sameIdeal(owner+ambient,ideal({r if ix else '0'})+ambient)"
            " && sameIdeal(strict+ambient,publicStrict+ambient) && sameIdeal(exceptional+ambient,publicEx+ambient);",
            f'print("GRAPH {ix} "+string(ok));setring source;',
            'for(i=1;i<=size(resolved[1]);i++){def C=resolved[1][i];setring C;',
            f"ideal image=BO[5];map baseMap=source,image;poly exc=BO[4][1][1];poly h=BO[2][1];",
            f"setring source;poly p{ix}={pi};poly q{ix}={pk};setring C;",
            f"poly mappedOther=baseMap(q{ix});poly mappedPivot=baseMap(p{ix});",
            "poly unitFactor=mappedPivot/exc;poly otherFactor=mappedOther/exc;",
            "ok=unitFactor*exc==mappedPivot && otherFactor*exc==mappedOther && unitFactor!=0"
            " && sameIdeal(ideal(unitFactor,otherFactor),ideal(1));",
            "ring lifted=0,(w1,w2,w3,invUnit),dp;map extend=C,w1,w2,w3;ideal liftedImage=extend(image);",
            "ideal localize=invUnit*extend(unitFactor)-1;",
            f"map phi={ring},liftedImage[1],liftedImage[2],liftedImage[3],extend(otherFactor)*invUnit;",
            "ok=ok && sameIdeal(phi(ambient)+localize,localize)"
            " && sameIdeal(phi(strict)+localize,ideal(extend(h))+localize)"
            " && sameIdeal(phi(exceptional)+localize,ideal(extend(exc))+localize)"
            " && sameIdeal(phi(submittedFibre)+localize,extend(image)+localize);",
            f'print("OPEN {ix} "+string(i)+" "+string(ok));setring source;kill lifted,C;}}',
        ])
        checks += [f"GRAPH {ix} 1", f"OPEN {ix} 1 1", f"OPEN {ix} 2 1"]
    _, answer = measure(singular, ROOT, timeout, "\n".join([*program,"quit;"]))
    require_singular(answer, True)
    observed = [line for line in answer.splitlines() if line.startswith(("GRAPH ", "OPEN "))]
    if observed != checks:
        raise ValueError(f"Independent checks of graph presentations failed: {observed}")
    return checks


def verify(record, case, singular, timeout):
    if record["engine"] != "verified-ideal-blowup":
        return atlas_geometry(record, case, singular, timeout)
    result = graph_checks(record, case, singular, timeout)
    dual = record["dual_complex"]
    if dual["status"]!="complete" or dual["vertices"]!=[{"id":"E1","dimension":2}] or dual["cells"]:
        raise ValueError("Wrong dual complex for blowup of a smooth line")
    topo = record["topological_zeta"]
    if (topo["numerical_data"] != [{"component":"E1","multiplicity":2,"nu":2},{"component":"strict","multiplicity":1,"nu":1}]
            or topo["strata"] != [{"components":["E1"],"euler_characteristic":1},
                                  {"components":["E1","strict"],"euler_characteristic":1}]):
        raise ValueError("Numerical or Euler data disagree with the independently checked projective line")
    return result


def radical_oracle(case, singular, timeout):
    program = f'''
LIB "primdec.lib";ring source=0,({",".join(case.variables)}),dp;
poly f={case.polynomial};ideal jac=f,jacob(f);ideal centre=std(radical(jac));
ideal pencil=std(ideal({case.tangent},{case.partner}));int match=(size(reduce(centre,pencil))==0 && size(reduce(pencil,centre))==0);
print("REDUCED "+string(match)+" "+string(dim(centre)));quit;
'''
    _, raw = measure(singular, ROOT, timeout, program)
    if any(line.lstrip().startswith("?") for line in raw.splitlines()) or "REDUCED 1 1" not in raw.splitlines():
        raise ValueError("Singular's radical Jacobian ideal differs from the proposed line")
    return {"agrees":True, "dimension":1, "program":program}


def graph_tampering(record, singular, timeout):
    results = {}
    for name in ("map", "strict", "ambient", "multiplicity", "jacobian_order", "fibre", "owner", "pivot"):
        bad = deepcopy(record)
        chart = bad["atlas"][1]
        exceptional = next(c for c in chart["components"] if c["id"]=="E1")
        if name == "map":
            chart["map"]["x"] = "0"
        elif name == "strict":
            bad["charts"][1]["strict_transform"] = ["1"]
        elif name == "ambient":
            chart["ambient_ideal"] = ["1"]
        elif name == "multiplicity":
            exceptional["multiplicity"] += 1
        elif name == "jacobian_order":
            exceptional["nu"] += 1
        elif name == "fibre":
            chart["fibre"] = ["1"]
        elif name == "owner":
            chart["region"] = []
        else:
            bad["charts"][1]["pivot_generator"] = "1"
        try:
            verify(bad, CASES["inclined"], singular, timeout)
        except ValueError:
            results[name] = "rejected"
        else:
            raise ValueError(f"Incorrect {name} survived independent verification")
    return results


def replay_dual(tree, record, timeout):
    keys = ("status", "mode", "engine", "variables", "polynomial", "atlas")
    stripped = {name:record[name] for name in keys}
    _, output = measure(["./atlas","dual-complex","-"], tree, timeout, json.dumps(stripped))
    found = json.loads(output)
    # Binomial geometry adds toric ray metadata in its fast producer. Its
    # general saved-atlas replay returns the same labelled cells and vertices.
    expected = record["dual_complex"]
    if record["engine"]=="binomial-coordinate-blowups":
        expected = dict(expected, vertices=[{k:v[k] for k in ("id","dimension")} for v in expected["vertices"]])
    if found != expected:
        raise ValueError("Saved replay did not reconstruct the recorded dual complex")
    return {"chart_count":len(record["atlas"]), "vertices":len(found["vertices"]), "cells":len(found["cells"])}


PHASES = """
import json,sys,time
from engine import kernel,native_polynomial as np
from engine.general_resolution import expression,_verify
from engine.native_library import ideal_request,replay_scope
from engine.polynomial import parse
from engine.resolution_search import _candidate
data=json.load(sys.stdin);names=data['variables'];variables=tuple(map(np.Symbol,names))
f=expression(parse(data['polynomial'],names,bounded=False),variables);row=np.row(variables,f)
with replay_scope('cpu'):
    start=time.perf_counter();cpu=time.process_time()
    rows=ideal_request('smooth_reduced_centre '+str(len(names)),[row])
    discovered=time.perf_counter();dcpu=time.process_time()
    assert rows
    supplied=_candidate(variables,kernel.exprs(variables,rows),'smooth reduced singular locus')
    state,record=_verify(variables,f,supplied,'embedded',1)
    verified=time.perf_counter();vcpu=time.process_time()
assert state=='certified' and record['steps']==data['steps'] and record['charts']==data['charts']
# Also measure the combined operation actually used by the production helper.
# The supplied-centre verification above is an independent diagnostic split.
with replay_scope('cpu'):
    start2=time.perf_counter();cpu2=time.process_time()
    proof=ideal_request('smooth_reduced_graph '+str(len(names)),[row])
    combined=time.perf_counter()-start2;ccpu=time.process_time()-cpu2
assert proof[0].split()[0] in ('graph','centre') and proof[1:1+len(rows)]==rows and proof[1+len(rows)].startswith('pass ')
print(json.dumps(dict(centre_rows=rows,discovery_ms=1000*(discovered-start),discovery_cpu_ms=1000*(dcpu-cpu),
                      graph_ms=1000*(verified-discovered),graph_cpu_ms=1000*(vcpu-dcpu),combined_ms=1000*combined,
                      combined_cpu_ms=1000*ccpu,graph_request='smooth_reduced_graph '+str(len(names))+'\\n'+row+'\\n',
                      native_proof=proof)))
"""


def phases(tree, record, singular, trials, timeout):
    # This repeat isolates native work on the same centre from CLI startup and
    # payload dispatch. Each sample has a fresh process and one prepared input.
    source = json.dumps({k:record[k] for k in ("variables","polynomial","steps","charts")})
    singular_program = ('LIB "resolve.lib";ring q=0,(x,y,z),dp;poly f=(x+y+z)^2-z*(x-y)^2;'
                        'int start=timer;ideal jac=f,jacob(f);ideal centre=std(radical(jac));int discovered=timer;'
                        'list opens=blowUp(ideal(f),centre);int verified=timer;'
                        'if(size(opens)==2){print("TIMES "+string(discovered-start)+" "+string(verified-discovered));}quit;')
    rows = {"atlas":[], "singular":[]}
    centre = proof = None
    helper_samples = []
    for trial in range(-1,trials):
        for name in (rows if trial%2==0 else reversed(rows)):
            if name=="atlas":
                sample,raw = measure([sys.executable,"-c",PHASES],tree,timeout,source)
                sampled = json.loads(raw)
                candidate = sampled.pop("centre_rows")
                if centre is None:
                    centre = candidate
                if centre!=candidate:
                    raise ValueError("Native centre changed between timed samples")
                request = sampled.pop("graph_request")
                answer = sampled.pop("native_proof")
                if proof is None:
                    proof = answer
                if answer!=proof:
                    raise ValueError("The combined native proof changed")
                helper, emitted = measure([str(tree/"native"/"build"/"graph_verify")],tree,timeout,request)
                if emitted.splitlines()!=answer:
                    raise ValueError("Installed graph helper differs from the native operation")
                if trial>=0:
                    helper_samples.append(helper)
                sample.update(sampled)
            else:
                sample,raw = measure([*singular,"--ticks-per-sec=1000"],ROOT,timeout,singular_program)
                lines = raw.splitlines()
                if len(lines)!=1 or not lines[0].startswith("TIMES "):
                    raise ValueError("Singular did not discover and blow up the complete line")
                sample["discovery_cpu_ms"],sample["graph_cpu_ms"] = map(int,lines[0].split()[1:])
            if trial>=0:
                rows[name].append(sample)
    rows["native_helper"] = helper_samples
    return {"samples":rows, "native_proof":proof, "median":{name:{k:statistics.median(s[k] for s in samples) for k in samples[0] if k!="status"}
                                       for name,samples in rows.items()}, "centre_rows":centre,
            "singular_program":singular_program}


def main():
    args = arguments(CASES, __doc__)
    target = args.output
    with tempfile.TemporaryDirectory() as directory:
        args.output = Path(directory)/"run.json"
        run(args, cases=CASES, fibre_oracle=fibre_oracle, chart_oracle=verify,
            driver=__file__, chart_changes=True)
        data = json.loads(args.output.read_bytes())
    roots = {k:Path(v) for k,v in data["roots"].items()}
    singular = [shutil.which("Singular"), "-q", "--no-rc"]
    for name, result in data["cases"].items():
        case = CASES[name]
        records = result["agreed_records"]
        before, after = records["before_zetas"], records["after_zetas"]
        zeta_keys = {"monodromy_zeta": ("expression","factors","milnor_fiber_euler_characteristic"),
                     "topological_zeta": ("expression","numerator","denominator")}
        for zeta, fields in zeta_keys.items():
            if any(before[zeta][k]!=after[zeta][k] for k in fields):
                raise ValueError(f"Presentation-independent {zeta} differs between builds")
        if name=="unsheared":
            if not all(result["identical_full_record_between_roots"].values()):
                raise ValueError("The two-chart binomial control changed")
        elif not (before["engine"]=="canonical-blowups" and len(before["atlas"])==(30 if name=="curved" else 36)
                  and after["engine"]=="verified-ideal-blowup" and len(after["atlas"])==2):
            raise ValueError("The public CLI did not produce the expected single blowup after centre discovery")
        if name!="unsheared" and after["certificate"].get("center_source")!="smooth reduced singular locus":
            raise ValueError("The new reduced singular locus candidate was not certified")
        if name!="unsheared" and (len(after["steps"])!=1 or not after["steps"][0].get("generators")):
            raise ValueError("No single projective graph blowup was recorded")
        result["changed_fields"] = sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
        result["saved_dual_replay"] = {k:replay_dual(roots[k],r,args.timeout) for k,r in (("before",before),("after",after))}
        replay(roots["after"], before, args.timeout)
        result["legacy_record_replayed_by_after"] = {
            "dual_complex": replay_dual(roots["after"], before, args.timeout),
            "full_monodromy_and_topological_zeta": True,
        }
        if name=="curved":
            script = singular_script(case).replace("poly original=", "short=0;poly original=")
            _, answer = measure(singular, ROOT, args.timeout, script)
            report = answer.split("Upcoming Center",1)[1].split("\n-",1)[0].splitlines()
            if report[1:3]!=["_[1]=z","_[2]=x^2-y"]:
                raise ValueError("Unexpected first centre in the curved Singular example")
            result["singular_first_centre"] = {"reported_generators":["z","x^2-y"], "tree_nodes":3, "final_charts":2}
        else:
            result["singular_first_centre"] = first_centre(case,args.timeout)
        result["singular_jacobian_radical"] = radical_oracle(case,singular,args.timeout)
    if "inclined" in data["cases"]:
        record = data["cases"]["inclined"]["agreed_records"]["after_zetas"]
        data["graph_tampering"] = graph_tampering(record,singular,args.timeout)
        data["centre_and_graph_phases"] = phases(roots["after"],record,singular,args.trials,args.timeout)
    data["dependency_sha256"] = {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                 for n in ("whitney_related.py","inclined_centre_resolution.py","exhaustive_search_resolution.py",
                                           "sheared_centre_resolution.py","plane_conjugate_resolution.py")}
    for stage, root in roots.items():
        snapshot = root/"source-hashes.json"
        if snapshot.is_file():
            manifest = json.loads(snapshot.read_bytes())
            if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest for name,digest in manifest.items()):
                raise ValueError(f"Frozen source changed in {root}")
            data.setdefault("frozen_manifests", {})[stage] = manifest
    raw = (json.dumps(data,indent=2)+"\n").encode()
    target.write_bytes(gzip.compress(raw,mtime=0) if target.suffix==".gz" else raw)


if __name__ == "__main__":
    main()
