"""Compare CPU local quotient bases, Jacobian algebras and hypersurface numbers with Singular.

python3 benchmarks/point_quotient_invariants.py --cpu 5 --output /tmp/invariants.json
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import comb
import os
import re
import resource
import signal
from pathlib import Path
import select
import subprocess
import statistics
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.large_coupled_local import SINGULAR, singular_ring
from benchmarks.shifted_local_standard import launch, powers
from engine.polynomial import parse_expression, format_poly

QUOTIENT="local-quotient-basis"
JACOBIAN="local-jacobian-algebra"
MILNOR="milnor-number"
TJURINA="tjurina-number"
BASIS=(QUOTIENT,JACOBIAN)


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple
    point: tuple
    ideal: tuple
    polynomial: str
    dimension: int
    tjurina: int
    order: str = "grevlex"
    # Original-coordinate to a local chart with an invertible derivative.
    # The chart's ideal is used only for independent validation, never timed.
    model_ideal: tuple = ()
    model_polynomial: str = ""
    wide: bool = False
    scalar_only: bool = False

    def operations(self):
        if self.scalar_only:return (MILNOR,TJURINA)
        return (JACOBIAN,MILNOR,TJURINA) if self.polynomial else (QUOTIENT,)


def corpus():
    xy=("x","y"); xyz=("x","y","z")
    p=("1/2","-2/3")
    u,v="(x-1/2)","(y+2/3)"
    curve="y^2-x^3*(x-1)^2"
    cases=[Case("cusp-point",xy,("0","0"),(),curve,2,2), Case("node-point",xy,("1","0"),(),curve,1,1),
           Case("smooth-point",xy,("4","24"),(),curve,0,0),
           Case("outside-hypersurface",xy,("1","1"),(),curve,-2,-2),
           Case("unequal-shifted",xy,p,(),f"{u}^5+{v}^5+{u}^3*{v}^3",16,15),
           Case("nonisolated-shifted",xy,p,(),f"{u}^2*{v}",-1,-1),
           Case("zero-polynomial",xy,p,(),"0",-1,-1),
           Case("quotient-unit",xy,p,("x-1/4","y"),"",0,0),
           Case("quotient-zero-ideal",xy,p,(),"",-1,-1)]
    w="(z-2)"
    rows=tuple(f"(1-{u})*({g})" for g in (f"{u}^2+{v}^3",f"{v}^3",f"{w}^2",f"{u}*{w}-{v}^2+{w}^3"))
    for order in ("lex","grevlex"):
        cases.append(Case("quotient-tie-"+order,xyz,p+("2",),rows,"",7,7,order))
    for n in (37,1000001,2**64+7):
        sign=-1 if n>2**64 else 1
        a=f"(x^{n}-({sign}))"
        cases.append(Case(f"power-cusp-{n}",xy,(str(sign),"0"),(),f"{a}^3+y^2",2,2,
                          model_polynomial="h0^3+h1^2",wide=n>500000))
    n=1001
    a=f"(x^{n}-1)"
    cases.append(Case("power-unequal",xy,("1","0"),(),a+"^5+y^5+"+a+"^3*y^3",16,15,
                      model_polynomial="h0^5+h1^5+h0^3*h1^3"))
    d=2**65
    cases.append(Case("bigint-length",xy,("0","1"),(),f"x^{d+1}+(y-1)^2",d,d,
                      wide=True,scalar_only=True))
    a,b=f"(x^{n}-1)",f"(y^{n}-1)"
    cases.append(Case("coupled-nonisolated-quotient",xyz,("1","1","0"),(a+"*"+b,a+"*z",b+"*z"),
                      "",-1,-1,model_ideal=("h0*h1","h0*h2","h1*h2")))
    cases.append(Case("million-coordinate-slice",xyz,("1","0","0"),
                      ("x-1","y^2+z^3","y*z+(x^1000001-1)*z^2"),"",5,5,
                      model_ideal=("h0","h1^2+h2^3","h1*h2"),wide=True))
    cases.append(Case("wide-oblique-nonisolated",xy,("1","1"),(),"(x^1001*y^1001+x+y-3)^2",-1,-1,
                      model_polynomial="h0^2"))
    return cases


def deriv(row,axis):
    result={}
    for e,c in row.items():
        if e[axis]:
            out=list(e);out[axis]-=1
            result[tuple(out)]=c*e[axis]
    return result


def parsed(s,names):
    return parse_expression(s,list(names),_reference=True)


def singular_basis_output(case, local_ring, standard_basis, dimension):
    back = ",".join(f"{x}-({p})" for x, p in zip(case.variables, case.point))
    return (f"if({dimension}>0){{ideal B=kbase({standard_basis});setring S;"
            f"map back={local_ring},{back};ideal A=back(B);int vindex;"
            'for(vindex=1;vindex<=size(A);vindex++){print("B "+string(A[vindex]));}}')


def singular_certified_shortcuts(case, op):
    # An axis contained in the germ, or a repeated hypersurface factor through
    # the point, proves infinite colength without a full coordinate shift.
    src = []
    for i, x in enumerate(case.variables):
        line = ",".join(x if k == i else p for k,p in enumerate(case.point))
        src.append(f'map axis{i}=S,{line};if(size(axis{i}(J))==0){{print("L -1");quit;}}')
    if op != QUOTIENT and len(case.variables) > 1:
        src.extend(["list factors=factorize(F);ideal primes=factors[1];intvec multiplicities=factors[2];int fi;poly factor;",
                    'for(fi=1;fi<=size(primes);fi++){factor=primes[fi];if(multiplicities[fi]>1 && ev(factor)==0){print("L -1");quit;}}'])
    return src


def singular_jets(case, op):
    # Equal lengths at m^4 and m^5 imply m^4=0 in the local quotient by
    # Nakayama. Otherwise continue to the unrestricted local std procedure.
    h = [f"h{i}" for i in range(len(case.variables))]
    src = [
        """proc taylorAt(poly P,map at,int index,int budget,poly mon,number scale)
{
  if(index>nvars(basering)){return(mon*at(P)*scale);}
  int k; poly term=1; number fac=1; poly total=0; poly axis=var(index);
  for(k=0;k<=budget;k++){
    if(P==0){break;}
    total=total+taylorAt(P,at,index+1,budget-k,mon*term,scale/fac);
    P=diff(P,axis);term=term*axis;fac=fac*(k+1);
  }
  return(total);
}""","ideal truncated;int idx;for(idx=1;idx<=ncols(J);idx++){truncated[idx]=taylorAt(J[idx],ev,1,4,1,1);}",
        "ring Jet=0,("+",".join(h)+"),"+{"lex":"Ds","grevlex":"ds"}[case.order]+";",
        "short=0;map toJet=S,"+",".join(h)+";ideal jets=toJet(truncated);",
        "ideal J4=std(jets+maxideal(4));ideal J5=std(jets+maxideal(5));int d5=vdim(J5);",
        'if(d5==vdim(J4)){print("L "+string(d5));'
    ]
    if op in BASIS:
        src.append(singular_basis_output(case,"Jet","J5","d5"))
    return src+["quit;}setring S;"]


def singular_source(case,op,method):
    h=[f"h{i}" for i in range(len(case.variables))]
    src=[singular_ring("S",case.variables,"dp",case.wide),"short=0;","map ev=S,"+",".join(case.point)+";"]
    if op==QUOTIENT:
        src+=["ideal J="+(",".join(case.ideal) or "0")+";"]
    else:
        src+=[f"poly F={case.polynomial};", 'if(ev(F)!=0){print("OUTSIDE");quit;}',
              "ideal Jac=jacob(F);ideal J=Jac;"]
        if op==JACOBIAN:
            src+=['int ix;for(ix=1;ix<=nvars(S);ix++){print("D "+string(Jac[ix]));}']
        if op==TJURINA:src+=["J=J+ideal(F);"]
    src+=['if(size(ev(J))!=0){print("L 0");quit;} if(size(J)==0){print("L -1");quit;}']
    if method=="singular-components":
        src+=singular_certified_shortcuts(case,op)
    if method=="singular-jets":
        src+=singular_jets(case,op)
    if method=="singular-global-first":
        src+=["J=std(J);"]
    if method=="singular-unit-factors":
        src+=["ideal small;list fact;ideal terms;intvec exps;int i,j;poly v;poly g;",
              "for(i=1;i<=ncols(J);i++){if(J[i]==0){continue;}fact=factorize(J[i]);terms=fact[1];exps=fact[2];"
              "g=1;for(j=1;j<=size(terms);j++){v=terms[j];if(ev(v)==0){g=g*v^exps[j];}}small[size(small)+1]=g;}J=small;"]
    src+=[singular_ring("L",h,{"lex":"Ds","grevlex":"ds"}[case.order],case.wide),"short=0;",
          "map move=S,"+",".join(f"{v}+({p})" for v,p in zip(h,case.point))+";",
          'ideal I=move(J);ideal G=std(I);int d=vdim(G);print("L "+string(d));']
    if op in BASIS:
        src.append(singular_basis_output(case,"L","G","d"))
    return "\n".join(src+["quit;",""])


def expected(case,op):
    return case.tjurina if op==TJURINA else case.dimension


def timed_atlas(args,seconds,memory):
    # The Python CLI creates the math worker in a second session. Interrupt
    # both the child and any detached descendants to enforce a wall-time bound.
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(memory*1024**2,)*2)
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    env={**os.environ,"ATLAS_GROEBNER":"native"}
    env.pop("ATLAS_GROEBNER_GPU",None);env.pop("PYTHONINTMAXSTRDIGITS",None)
    st=perf_counter()
    proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                          env=env,start_new_session=True,preexec_fn=limits)
    try:
        out,err=proc.communicate(timeout=max(0,seconds-(perf_counter()-st)))
    except subprocess.TimeoutExpired:
        pending=[proc.pid]
        descendants=[]
        while pending:
            pid=pending.pop()
            descendants.append(pid)
            for child_file in Path(f"/proc/{pid}/task").glob("*/children"):
                try: pending.extend(int(c) for c in child_file.read_text().split() if int(c) not in descendants)
                except FileNotFoundError: pass
        for pid in reversed(descendants):
            try: os.kill(pid,signal.SIGKILL)
            except ProcessLookupError: pass
        proc.communicate(timeout=2)
        return {"status":"timeout","lower_bound_seconds":seconds},None
    ms=(perf_counter()-st)*1000
    diagnostic=(out+" "+err).strip()
    if re.search("bad_alloc|out of memory|no more memory|cannot (?:re)?allocate",diagnostic,re.I):
        return {"status":"allocation_failure","message":diagnostic[-260:]},None
    if "polynomial pullback exceeds FLINT's limits" in diagnostic:
        return {"status":"pullback_overflow","message":"The polynomial pullback exceeds FLINT's limits."},None
    try:record=json.loads(out)
    except ValueError as e: raise ValueError(f"Atlas exit {proc.returncode}: {diagnostic[-350:]}") from e
    if bool(record.get("error"))!=(proc.returncode==2):raise ValueError(record)
    error = record.get("error", {})
    if error.get("code") == "worker_unavailable":
        return {"status": "worker_unavailable", "message": error["message"]}, None
    return {"status":"ok","milliseconds":ms},record


def atlas_output(root,case,op,seconds,memory):
    args=[str(root/"atlas"),"algebra",op,"--compute","cpu","--vars",",".join(case.variables),
          "--order",case.order,"--point",",".join(case.point),"--json"]
    if op==QUOTIENT:
        for g in case.ideal:args+=["--gen",g]
    else:args+=["--poly",case.polynomial]
    info,data=timed_atlas(args,seconds,memory)
    if info["status"]!="ok":return info,None
    if expected(case,op)==-2:
        err=data.get("error",{})
        if err.get("code")!="invalid_request" or "pass through" not in err.get("message",""):raise ValueError(data)
        return info,{"dimension":-2}
    value=data["result"];execution=data["execution"]
    if (execution["backend"]!="cpu" or execution["requested"]!="cpu" or value["scope"]!="point"
            or value["point"]!=list(case.point) or value["ring"]!={"field":"QQ","variables":list(case.variables),"order":case.order}):
        raise ValueError(data)
    dim=value["dimension" if op in BASIS else "value"]
    if value["finite"] is not (dim is not None):raise ValueError(value)
    if value["type"]!=({QUOTIENT:"quotient-basis",JACOBIAN:JACOBIAN}.get(op,"integer")):raise ValueError(value)
    rec={"dimension":dim if dim is not None else -1}
    if op in BASIS:rec["basis"]=value["basis"]
    if op==JACOBIAN:rec["derivatives"]=value["generators"]
    return info,rec


def singular_output(case,op,method,seconds,memory):
    info,out=launch(SINGULAR,singular_source(case,op,method),seconds,memory)
    if info["status"]!="ok":return info,None
    vals={};basis=[];jac=[]
    for line in out.splitlines():
        if line=="OUTSIDE":return info,{"dimension":-2}
        key,_,val=line.partition(" ")
        if key=="L": vals["dimension"]=int(val)
        elif key=="B":basis.append(val)
        elif key=="D":jac.append(val)
        else:raise ValueError(out)
    if op in BASIS:vals["basis"]=basis if vals["dimension"]>=0 else None
    if op==JACOBIAN:vals["derivatives"]=jac
    return info,vals


def translated(poly,point,cutoff=None):
    """Exact Taylor coefficients in fixed variables; avoid dense shifts for finite jets."""
    if not poly:return {}
    degree=max(map(sum,poly)) if cutoff is None else cutoff-1
    result={}
    for d in range(degree+1):
        for k in powers(len(point),d):
            total=0
            for e,c in poly.items():
                term=c
                for a,b,p in zip(e,k,point):
                    if a<b or (not p and a!=b):
                        term=0
                        break
                    term*=comb(a,b)*p**(a-b)
                total+=term
            if total:result[k]=total
    return result


def generators(case,op):
    if op==QUOTIENT:return [parsed(g,case.variables) for g in case.ideal]
    f=parsed(case.polynomial,case.variables)
    rows=[deriv(f,i) for i in range(len(case.point))]
    return rows+[f] if op==TJURINA else rows


def verify(case,op,v,seconds,memory):
    d=v["dimension"]; n=len(case.point)
    if d!=expected(case,op):raise ValueError(f"{case.name}/{op}: got {d}, expected {expected(case,op)}")
    center=tuple(map(Fraction,case.point));h=tuple(f"h{i}" for i in range(n))
    if d==-2:
        at=translated(parsed(case.polynomial,case.variables),center,1)
        if not at:raise ValueError("off-hypersurface point reported on the hypersurface")
        return "nonzero exact value at point"
    original=generators(case,op)
    if op==JACOBIAN:
        got=[parsed(g,case.variables) for g in v["derivatives"]]
        if got!=original:raise ValueError(f"incorrect source-coordinate derivatives: {v}")
    if case.scalar_only:
        # x^(2^65)=0 admits a monic quadratic tower in x, x², x⁴, ...
        # Each extension doubles the vector-space rank, including the initial
        # QQ[t]/t² of rank 2. Singular verifies a two-step prototype and the
        # arbitrary-precision integer product; no basis of length 2^65 is built.
        if case.name!="bigint-length":raise ValueError(case.name)
        coef=2**65+1
        assert original[:2]==[{(coef-1,0):Fraction(coef)},{(0,1):Fraction(2),(0,0):Fraction(-2)}]
        if op==TJURINA:
            assert original[-1]=={(coef,0):1,(0,2):1,(0,1):-2,(0,0):1}
        code=('ring P=0,(w,t),dp; ideal G=std(ideal(t^2,w^2-t));int k;bigint total=bigint(1);'
              'for(k=1;k<=65;k++){total=total*2;}print("T "+string(vdim(G))+" "+string(total));quit;')
        info,ans=launch(SINGULAR,code,seconds,memory)
        if info["status"]!="ok" or ans.strip()!=f"T 4 {d}":raise ValueError(ans)
        return "exact derivative ideal and rank of 65 monic quadratic extensions"
    if op in BASIS:
        if (d<0 and v["basis"] is not None) or (d>=0 and len(v["basis"])!=d):raise ValueError(v)
    if d==0:
        if not any(translated(g,center,1) for g in original):raise ValueError("unit-ideal test failed")
        return "local unit"
    if case.model_polynomial or case.model_ideal:
        if case.model_polynomial:
            f=parsed(case.model_polynomial,h); eq=[deriv(f,i) for i in range(n)]
            if op==TJURINA:eq.append(f)
        else:eq=[parsed(g,h) for g in case.model_ideal]
    else:eq=[translated(g,center) for g in original]
    model=",".join(format_poly(g,h) for g in eq) or "0"
    # Local membership/length is independent of the tie-break order.
    src=[f"ring Ref=0,({','.join(h)}),ds;",f"ideal M={model};ideal MG=std(M);",
         f'print("CHECK "+string(vdim(MG)=={d}));']
    if op in BASIS and d>0:
        # In an Artinian local ring of length d the d-th maximal ideal power
        # vanishes. Hence jets in the *actual source directions* determine
        # the coordinates of the returned basis, even for an étale chart.
        actual=[format_poly(translated(g,center,d),h) for g in original]
        basis=[format_poly(translated(parsed(g,case.variables),center,d),h) for g in v["basis"]]
        src += [f"ring Jet=0,({','.join(h)}),"+{"lex":"Dp","grevlex":"dp"}[case.order]+";",
                "ideal A="+(",".join(actual) or "0")+f";ideal G=std(A+maxideal({d}));",
                "ideal C="+",".join(basis)+";ideal mon=kbase(G);",
                "matrix V=coeffs(reduce(C,G),mon);",
                f'print("CHECK "+string(vdim(G)=={d} && ncols(V)=={d} && nrows(V)=={d} && det(V)!=0));']
        if case.name.startswith("quotient-tie-"):
            kept,excluded=("h1^2","h0*h2") if case.order=="lex" else ("h0*h2","h1^2")
            src += ['int found,wrong,j;for(j=1;j<=size(C);j++){if(C[j]=='+kept+
                    '){found=1;}if(C[j]=='+excluded+'){wrong=1;}}',
                    'print("CHECK "+string(found==1 && wrong==0));']
    info,out=launch(SINGULAR,"\n".join(src+["quit;",""]),seconds,memory)
    if info["status"]!="ok" or any(s!="CHECK 1" for s in out.splitlines()) or not out.strip():
        raise ValueError(f"certificate failed for {case.name}/{op}: {info}, {out[:500]}")
    return "local length and full source-basis rank" if op in BASIS and d>0 else "local length"


def fingerprints(root):
    names=["atlas","engine/local_algebra.py","engine/point_germs.py","native/build/libgroebner.so","native/build/local_algebra",
           "native/local_cli.cpp","native/local_operations.cpp","native/local_point_basis.cpp","native/local_parameter_basis.cpp",
           "native/local_point_normal_form.cpp","native/local_point_quotient.cpp","native/local_quotient_initial.cpp",
           "native/monomial_quotient.cpp","native/point_geometry.cpp"]
    return {f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in names if (root/f).is_file()}


def singular_affinity():
    # This Singular package repins the interpreter during startup. Observe its
    # actual mask after parsing a command, while it waits for further input.
    with subprocess.Popen(SINGULAR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        try:
            process.stdin.write('print("ATLAS_AFFINITY_READY");\n')
            process.stdin.flush()
            if not select.select([process.stdout], [], [], 5)[0] or process.stdout.readline().strip() != "ATLAS_AFFINITY_READY":
                raise RuntimeError("Singular did not acknowledge the affinity probe")
            return sorted(os.sched_getaffinity(process.pid))
        finally:
            process.kill()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu",type=int)
    ap.add_argument("--atlas-root",type=Path,default=ROOT)
    ap.add_argument("--baseline-root",type=Path)
    ap.add_argument("--case",action="append",choices=[c.name for c in corpus()])
    ap.add_argument("--seconds",type=float,default=3)
    ap.add_argument("--memory-mib",type=int,default=768)
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--output",type=Path,required=True)
    opts=ap.parse_args()
    if opts.repeats<1 or opts.seconds<=0 or opts.memory_mib<64:ap.error("invalid measurement limits")
    if opts.cpu is not None:os.sched_setaffinity(0,{opts.cpu})
    singular_cpus = singular_affinity()
    if opts.cpu is not None and singular_cpus != [opts.cpu]:
        ap.error(f"Singular selected CPUs {singular_cpus} after startup. Set --cpu to its actual CPU for paired timings.")
    methods=(["before"] if opts.baseline_root else [])+["atlas","singular","singular-global-first","singular-unit-factors","singular-components","singular-jets"]
    report={"date_utc":datetime.now(timezone.utc).isoformat(),"command":sys.argv,"affinity":sorted(os.sched_getaffinity(0)),
            "cpu":next(v.split(":",1)[1].strip() for v in Path("/proc/cpuinfo").read_text().splitlines() if v.startswith("model name")),
            "seconds":opts.seconds,"address_space_mib":opts.memory_mib,"repeats":opts.repeats,"singular_affinity":singular_cpus,
            "atlas_sha256":fingerprints(opts.atlas_root),"baseline_sha256":fingerprints(opts.baseline_root) if opts.baseline_root else None,
            "singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
            "harness_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ("benchmarks/point_quotient_invariants.py","benchmarks/shifted_local_standard.py","benchmarks/large_coupled_local.py")},
            "cases":[],"results":[]}
    for case in corpus():
        if opts.case and case.name not in opts.case:continue
        report["cases"].append(asdict(case))
        for op in case.operations():
            samples={m:[] for m in methods}
            proofs={m:[] for m in methods}
            for i in range(opts.repeats):
                for m in methods[i%len(methods):]+methods[:i%len(methods)]:
                    if samples[m] and samples[m][-1]["status"]!="ok":continue
                    if m.startswith("singular"):trial,out=singular_output(case,op,m,opts.seconds,opts.memory_mib)
                    else:trial,out=atlas_output(opts.baseline_root if m=="before" else opts.atlas_root,case,op,opts.seconds,opts.memory_mib)
                    if out is not None and out not in [e["output"] for e in proofs[m]]:
                        proof=verify(case,op,out,max(10,opts.seconds),opts.memory_mib)
                        proofs[m].append({"output":out,"verified":proof})
                    samples[m].append(trial)
            medians={m:statistics.median(t["milliseconds"] for t in tests) for m,tests in samples.items()
                     if all(t["status"]=="ok" for t in tests)}
            report["results"].append({"case":case.name,"operation":op,"medians_ms":medians,"samples":samples,"certificates":proofs})
            opts.output.write_text(json.dumps(report,indent=2)+"\n")
            print(case.name,op,*(f"{m}={round(medians[m],2) if m in medians else samples[m][-1]['status']}" for m in methods),flush=True)


if __name__=="__main__":
    main()
