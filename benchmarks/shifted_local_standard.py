"""Compare shifted local standard bases and weak normal forms over QQ, using live Singular.

python3 benchmarks/shifted_local_standard.py --cpu 5 --output /tmp/shifted-standard.json
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import comb
import os
from pathlib import Path
import re
import resource
import signal
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.polynomial import parse_expression
from benchmarks.large_coupled_local import SINGULAR, singular_ring

OPS = ("local-standard-basis", "local-normal-form")


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple
    point: tuple
    ideal: tuple
    query: str
    member: bool
    order: str = "grevlex"
    # A stride >1 names the power coordinate used in the compressed oracle.
    strides: tuple = ()
    model: tuple = ()


def corpus():
    xy = ("x","y")
    xyz = ("x","y","z")
    a,b="(x-1/2)","(y+2)"
    cases = [
        Case("ordinary-weak",xy,("1/2","-2"),(f"{a}-{a}^2-{b}^2",),f"{a}+{a}^3-{b}^2",False),
        Case("ordinary-unit",xy,("1/2","-2"),(f"{a}*(1-{b})-{b}^2",),a,False,"lex"),
        Case("ordinary-finite",xy,("1/2","-2"),(f"{a}+{b}^2", f"{a}+{b}^3"),f"{b}^2",True),
        Case("unit-ideal",xy,("2","0"),("x^7-1","y^2"),"x+y",True),
    ]
    for order in ("lex","grevlex"):
        cases.append(Case("tie-"+order,xyz,("1/2","-2","0"),(f"{a}*z+{b}^2",),f"{a}*z",False,order))
    for n in (7,1001):
        u,v=f"(x^{n}-1)",f"(y^{n}-1)"
        cases.append(Case(f"axes-{n}",xyz,("1","1","0"),(f"{u}*{v}",f"{u}*z",f"{v}*z"),
                          "x*y+z",False,"grevlex",(n,n,1),("h0*h1","h0*h2","h1*h2")))
    for n in (17,1000001):
        u=f"(x^{n}-1)"
        cases.append(Case(f"squares-{n}",xyz,("1","0","0"),(f"({u}+y)^2",f"({u}+z)^2","(y+z)^2"),
                          "x-1+y*z",False,"lex",(n,1,1),
                          (f"({n}*h0+h1)^2",f"({n}*h0+h2)^2","(h1+h2)^2")))
    for n in (37,1000001):
        cases.append(Case(f"slice-{n}",xyz,("1","0","0"),("x-1","y^2+z^3",f"y*z+(x^{n}-1)*z^2"),
                          "x+y*z+z^5",False,"grevlex",(n,1,1),("h0","h1^2+h2^3","h1*h2")))
    n=2**64+7
    cases.append(Case("bigint-principal",xy,("-1","0"),(f"x^{n}+1+y",),"y",False,
                      "grevlex",(n,1),(f"{n}*h0+h1",)))
    cases.append(Case("bigint-unit",xy,("0","-1"),(f"x*(y^{n}+2)-(y+1)^2",),"x",False,
                      "grevlex",(1,n),("h0",)))
    for n in (7,1001):
        U,V,W=f"(x^{n}*y^{n}+x+y-3+z)","(x-y+z)","(z+x-1)"
        L,M,N=f"({n+1}*h0+{n+1}*h1+h2)","(h0-h1+h2)","(h2+h0)"
        cases.append(Case(f"primitive-oblique-{n}",xyz,("1","1","0"),(U+"*"+V,U+"*"+W,V+"*"+W),
                          "1+z",False,"grevlex",(n,n,1),(L+"*"+M,L+"*"+N,M+"*"+N)))
    return cases


def launch(command, source, seconds, memory):
    def limit():
        resource.setrlimit(resource.RLIMIT_AS,(memory*1024**2,)*2)
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    env = {**os.environ, "ATLAS_GROEBNER":"native"}
    env.pop("ATLAS_GROEBNER_GPU",None)
    env.pop("PYTHONINTMAXSTRDIGITS",None)
    start=perf_counter()
    p=subprocess.Popen(command, stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       start_new_session=True,text=True,env=env,preexec_fn=limit)
    try:
        out,err=p.communicate(source, timeout=max(0, seconds-(perf_counter()-start)))
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL)
        p.communicate()
        return {"status":"timeout","lower_bound_seconds":seconds}, ""
    ms=(perf_counter()-start)*1000
    msg=(out+" "+err).strip()
    if re.search("out of memory|no more memory|bad_alloc|cannot (?:re)?allocate",msg,re.I):
        return {"status":"allocation_failure","message":msg[-320:]}, ""
    if "poly` ^ `bigint` failed" in msg or "OVERFLOW in power" in msg:
        return {"status":"unsupported_exponent","message":next(line.strip() for line in msg.splitlines() if "failed" in line or "OVERFLOW" in line)}, ""
    if p.returncode or err or re.search(r"^\s*\?",out,re.M):
        if "The polynomial pullback exceeds FLINT's limits." in msg:
            return {"status":"pullback_overflow","message":"The polynomial pullback exceeds FLINT's limits."}, ""
        if "Large coordinate translation." in msg:
            return {"status":"translation_cap","message":"Large coordinate translation."}, ""
        raise RuntimeError(f"exit {p.returncode}: {msg[:1500]}")
    return {"status":"ok","milliseconds":ms},out


def source_ring(case):
    # Recreate the QQ ring with a larger exponent budget only when necessary.
    wide=any(n>500000 for n in case.strides)
    return singular_ring("Source",case.variables,"dp",wide)+"\nshort=0;"


def local_ring(case):
    names=[f"h{i}" for i in range(len(case.variables))]
    wide=any(n>500000 for n in case.strides)
    # Widened Ds in Singular 4.4.1 gives an incorrect basis for three parameter
    # squares, so keep the correct requested local tie order and ordinary bound.
    return singular_ring("Local",names,{"lex":"Ds","grevlex":"ds"}[case.order],wide and case.order!="lex")+"\nshort=0;"


def singular_solve(case,op):
    shifts=",".join(f"h{i}+({p})" for i,p in enumerate(case.point))
    back=",".join(f"{x}-({p})" for x,p in zip(case.variables,case.point))
    src=[source_ring(case),"ideal I="+",".join(case.ideal)+";",f"poly F={case.query};"]
    # Any nonzero principal equation is already a standard basis in a local
    # domain. Local units are recognized by evaluation before expanding shifts.
    src+=["map ev=Source,"+",".join(case.point)+";",
          'if(size(ev(I))){'+ ('print("B 1");' if op==OPS[0] else 'print("U 1");print("N 0");')+'quit;}']
    if op==OPS[0]:
        src+=['if(size(I)==1){print("B "+string(I[1]));quit;}']
    src+=[local_ring(case),f"map move=Source,{shifts};ideal J=move(I);ideal G=std(J);"]
    if op==OPS[0]:
        src+=["setring Source;",f"map back=Local,{back};ideal A=back(G);",
              'int i;for(i=1;i<=size(A);i++){print("B "+string(A[i]));}']
    else:
        src+=[f"ideal Q=move(F);list D=division(Q,G);poly U=D[3][1,1];poly N=D[2][1];",
              "number c=leadcoef(U);U=U/c;N=N/c;setring Source;",f"map back=Local,{back};",
              'print("U "+string(back(U)));print("N "+string(back(N)));']
    return "\n".join(src+["quit;",""])


def singular_prepared(case,op):
    """Simplify exact original-coordinate ideals before translating them."""
    direct=singular_solve(case,op)
    marker=local_ring(case)
    prelude=["ideal Original=I; ideal Gglobal=std(Original); I=Gglobal;"]
    if op==OPS[1]:
        # Global division uses the exact identity F = Fmod (mod I) with unit 1.
        prelude.append("F=reduce(F,Gglobal);")
    return direct.replace(marker,"\n".join(prelude)+"\n"+marker)


def singular_factored(case,op):
    """Discard source factors that are units at the supplied point."""
    prelude="""ideal reduced;list factors;ideal terms;intvec exps;int fi,fj;poly g;poly f;
for(fi=1;fi<=size(I);fi++){f=1;factors=factorize(I[fi]);terms=factors[1];exps=factors[2];
for(fj=1;fj<=size(terms);fj++){g=terms[fj];if(ev(g)==0){f=f*g^exps[fj];}}reduced[fi]=f;}
I=reduced;"""
    marker=local_ring(case)
    return singular_solve(case,op).replace(marker,prelude+"\n"+marker)


def polynomial(source,case):
    return parse_expression(source,list(case.variables),_reference=True)


def powers(width,degree):
    if width==1:
        yield (degree,)
        return
    for first in range(degree+1):
        for tail in powers(width-1,degree-first):
            yield (first,*tail)


def taylor_head(row,case):
    """Exact head via arbitrary integer binomial coefficients, independent of Atlas."""
    if not row:
        return None
    point=tuple(map(Fraction,case.point))
    for degree in range(max(map(sum,row))+1):
        nonzero=[]
        for e in powers(len(point),degree):
            c=0
            for a,coeff in row.items():
                term=coeff
                for k,j,p in zip(a,e,point):
                    if k<j or (p==0 and k!=j):
                        term=0
                        break
                    term*=comb(k,j)*(p**(k-j))
                c+=term
            if c: nonzero.append(e)
        if nonzero:
            return max(nonzero, key=(lambda e:e) if case.order=="lex" else (lambda e:tuple(-i for i in reversed(e))))
    raise ValueError("nonzero polynomial lost by translation")


def monomial(e):
    return "*".join(f"h{i}^{j}" for i,j in enumerate(e) if j) or "1"


def compressed(row,case,*,direct=False):
    """Lift each x^N to its order-four Taylor jet plus h^4 times a new local variable."""
    ans=[]
    for exps,c in row.items():
        factors=[f"({c})"]
        for i,(k,p,n) in enumerate(zip(exps,map(Fraction,case.point),(1,)*len(case.point) if direct else (case.strides or (1,)*len(case.point)))):
            # Split a power into its full stride multiples and a nonnegative remainder.
            q,r=divmod(k,n) if n>1 else (0,k)
            if r>4000 or q>4000: raise ValueError(f"compressed exponent {q},{r} is too wide")
            if r: factors.append(f"(h{i}+({p}))^{r}")
            if q:
                # After the degree-four jet, the divided difference by h^4
                # vanishes at the origin. Hence substituting it for s preserves units.
                terms=[f"({p**n})"]+[f"({comb(n,d)*p**(n-d)})*h{i}^{d}" for d in range(1,min(4,n)+1)]
                terms.append(f"h{i}^4*s{i}")
                factors.append("("+ "+".join(terms)+f")^{q}")
        ans.append("*".join(factors))
    return "+".join(ans) or "0"


def verify(case,op,record,seconds,memory):
    if op==OPS[0]:
        proposed=record["generators"]
    else:
        proposed=[record["unit"],record["remainder"]]
    I=[polynomial(g,case) for g in case.ideal]
    P=[polynomial(g,case) for g in proposed]
    Q=polynomial(case.query,case)
    h=tuple(f"h{i}" for i in range(len(case.variables)))
    direct=max(case.strides or (1,))<=64
    # When every expression already lies in QQ[x_i^N, ...], use the
    # diagonal étale chart itself for localized ideal membership. Otherwise
    # use the Taylor-lift certificate, which works on mixed powers too.
    relevant=I+P+([] if op==OPS[0] else [Q])
    aligned=bool(case.strides) and not direct and all(
        all(all(k%n==0 for k,n in zip(e,case.strides)) for e in row) for row in relevant)
    if aligned:
        def expr(row):
            out=[]
            for e,c in row.items():
                term=[f"({c})"]
                for i,(k,n,p) in enumerate(zip(e,case.strides,map(Fraction,case.point))):
                    if k: term.append(f"(h{i}+({p**n}))^{k//n}")
                out.append("*".join(term))
            return "+".join(out) or "0"
    else:
        def expr(row): return compressed(row,case,direct=direct)
    finite_jet=(not direct and not aligned and case.name.startswith("squares-"))
    if finite_jet:
        # The chart ideal consists of three squares of independent linear
        # forms. The fourth maximal-ideal power vanishes: degree-four
        # monomials in three coordinates all contain a square. Compute in
        # the exact four-jet quotient, with the chart bound verified below.
        def expr(row):
            terms=[]
            for d in range(4):
                for e in powers(len(h),d):
                    c=0
                    for a,coef in row.items():
                        term=coef
                        for k,j,v in zip(a,e,map(Fraction,case.point)):
                            if k<j or (v==0 and k!=j):
                                term=0
                                break
                            term*=comb(k,j)*v**(k-j)
                        c+=term
                    if c: terms.append(f"({c})*"+monomial(e))
            return "+".join(terms) or "0"
    pair_products=(case.name.startswith("primitive-oblique-") and op==OPS[0]
                   and all(abs(e[0]-e[1])<=3 for row in P for e in row))
    if pair_products:
        # In this chart, C is the image of x^N*y^N, not an independent
        # parameter for the singularity. Mutual *polynomial* membership in
        # this larger ring proves local membership after C=x^N*y^N. For
        # a degree-two change of generators this avoids a large local std.
        def chart_poly(row):
            terms=[]
            n=case.strides[0]
            for (a,b,c),v in row.items():
                q=min(a,b)//n
                factors=[f"({v})"]
                for var,d in zip(("w","x","y","z"),(q,a-q*n,b-q*n,c)):
                    if d: factors.append(f"{var}^{d}")
                terms.append("*".join(factors))
            return "+".join(terms) or "0"
    extra=tuple(f"s{i}" for i,k in enumerate(case.strides) if k>1 and not direct and not aligned and not finite_jet)
    ring=f"ring Check=0,({','.join(h+extra)}),ds;"
    full=lambda polys:",".join(expr(row) for row in polys) or "0"
    trivial_identity=(op==OPS[1] and P[0]=={(0,)*len(h):1} and P[1]==Q
                      and taylor_head(Q,case)==(0,)*len(h)
                      and all(taylor_head(row,case)!=(0,)*len(h) for row in I))
    source=[ring, "ideal J="+full(I)+";"+("J=J+maxideal(4);" if finite_jet else "")
            +("" if trivial_identity else "ideal G=std(J);")]
    if op==OPS[0]:
        if pair_products:
            source=["ring Explicit=0,(w,x,y,z),dp;",
                    "ideal J="+",".join(map(chart_poly,I))+";ideal G=std(J);",
                    "ideal A="+(",".join(map(chart_poly,P)) or "0")+";ideal AG=std(A);"]
        else:
            source+=["ideal A="+full(P)+";ideal AG=std(A);"]
        source+=[
                 'print("V "+string(size(reduce(A,G))==0));',
                 'print("V "+string(size(reduce(J,AG))==0));']
    else:
        relation="U*F-N==0" if trivial_identity else "reduce(U*F-N,G)==0"
        membership="jet(F,0)==0" if trivial_identity else "reduce(F,G)==0"
        source += [f"poly U={expr(P[0])};poly N={expr(P[1])};poly F={expr(Q)};",
                   'print("V "+string(jet(U,0)==1));',f'print("V "+string({relation}));',
                   f'print("V "+string((N==0)=={int(case.member)}));',
                   f'print("V "+string(({membership})=={int(case.member)}));']
    # Each model is in the original tangent directions. This includes the
    # full nondiagonal Jacobian of the primitive oblique coordinate chart.
    source += [f"ring Ref=0,({','.join(h)}),"+{"grevlex":"ds","lex":"Ds"}[case.order]+";",
               "ideal model="+(",".join(case.model) if case.model else full(I))+";"]
    # For homogeneous models the degree-compatible global order has identical
    # leading monomials. This is also an independent route around observed
    # local Ds interreduction errors in Singular 4.4.1 on parameter squares.
    if case.model and not case.name.startswith("slice-"):
        source[-2]=f"ring Ref=0,({','.join(h)}),"+{"grevlex":"dp","lex":"Dp"}[case.order]+";"
    source+=["ideal heads=std(lead(std(model)));"]
    if finite_jet:
        source+=['print("V "+string(size(reduce(maxideal(4),std(model)))==0));']
    if op==OPS[0]:
        hs=[monomial(taylor_head(row,case)) for row in P if row]
        source+=["ideal Aheads=std(ideal("+(",".join(hs) or "0")+"));",
                 'print("V "+string(size(reduce(Aheads,heads))==0));',
                 'print("V "+string(size(reduce(heads,Aheads))==0));']
    elif P[1]:
        source += [f"poly H={monomial(taylor_head(P[1],case))};",
                   'print("V "+string(reduce(H,heads)!=0));']
        if record.get("reduction")=="standard":
            # Certify every source-centered term, without using a finite-jet
            # truncation. Output tails in these cases need no auxiliary variable.
            tail=compressed(P[1],case,direct=direct)
            if re.search(r"\bs[0-9]+\b",tail):
                raise ValueError(f"full-tail proof requires expanding auxiliary coefficients: {case.name}")
            source+=["poly N="+tail+";",
                     'print("V "+string(reduce(N,heads)==N));']
    info,out=launch(SINGULAR,"\n".join(source+["quit;",""]),seconds,memory)
    if info["status"]!="ok": raise ValueError(f"checking {case.name} {op}: {info}")
    checks=[line for line in out.splitlines() if line]
    if not checks or any(line!="V 1" for line in checks):
        raise ValueError(f"certification failed for {case.name} {op}: {checks}, {proposed}")
    return len(checks)


def decode(method,out,case,op):
    if method.startswith("singular"):
        lines=out.splitlines()
        if op==OPS[0] and all(line.startswith("B ") for line in lines):
            return {"generators":[line[2:] for line in lines]}
        if len(lines)==2 and lines[0].startswith("U ") and lines[1].startswith("N "):
            return {"unit":lines[0][2:],"remainder":lines[1][2:]}
        raise ValueError(f"unexpected Singular output: {out[:300]}")
    obj=json.loads(out)
    result=obj["result"]
    if (obj["execution"]["backend"]!="cpu" or obj["execution"]["requested"]!="cpu" or result["type"]!=op
            or result["point"]!=list(case.point)):
        raise ValueError(obj)
    keys=("generators",) if op==OPS[0] else ("unit","remainder","reduction")
    return {key:result[key] for key in keys}


def run(case,op,method,root,args):
    if method.startswith("singular"):
        src={"singular":singular_solve,"singular-global-first":singular_prepared,
             "singular-unit-factors":singular_factored}[method](case,op)
        cmd=SINGULAR
    else:
        cmd=[str(root/"atlas"),"algebra",op,"--vars",",".join(case.variables),"--point",",".join(case.point),
             "--order",case.order,"--compute","cpu","--json"]
        for row in case.ideal:
            cmd+=["--gen",row]
        if op==OPS[1]: cmd+=["--poly",case.query]
        src=""
    info,out=launch(cmd,src,args.seconds,args.memory_mib)
    if info["status"]=="ok":
        info["output"]=decode(method,out,case,op)
    return info


def snapshot(root):
    names=("atlas","native/build/local_algebra","native/build/libgroebner.so","native/local_cli.cpp","native/local_operations.cpp",
           "native/local_basis.cpp","native/local_division.cpp","engine/local_standard.py","native/local_io.cpp",
           "native/local_point_basis.cpp","native/local_point_normal_form.cpp","native/local_power_coordinates.cpp","native/local_parameter_basis.cpp",
           "native/point_geometry.cpp")
    return {f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in names if (root/f).is_file()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas-root",type=Path,default=ROOT)
    ap.add_argument("--baseline-root",type=Path)
    ap.add_argument("--cpu",type=int)
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--seconds",type=float,default=3)
    ap.add_argument("--memory-mib",type=int,default=768)
    ap.add_argument("--case",action="append",choices=[c.name for c in corpus()])
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--resume",type=Path,help="continue an interrupted run on identical solver builds and settings")
    args=ap.parse_args()
    if args.cpu is not None: os.sched_setaffinity(0,{args.cpu})
    if args.repeats<1 or args.seconds<=0 or args.memory_mib<64: ap.error("invalid sampling limits")
    methods=(["before"] if args.baseline_root else [])+["atlas","singular","singular-global-first","singular-unit-factors"]
    report={"date_utc":datetime.now(timezone.utc).isoformat(),"command":sys.argv,"affinity":sorted(os.sched_getaffinity(0)),
            "cpu":next(v.split(":",1)[1].strip() for v in Path("/proc/cpuinfo").read_text().splitlines() if v.startswith("model name")),
            "singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
            "seconds":args.seconds,"address_space_mib":args.memory_mib,"repeats":args.repeats,
            "sha256":snapshot(args.atlas_root),"baseline_sha256":snapshot(args.baseline_root) if args.baseline_root else None,
            "harness_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"cases":[],"rows":[]}
    # Recheck saved certificates when resuming interrupted measurements. This
    # keeps measured solver times immutable across verifier-only corrections.
    prior={}
    if args.resume:
        saved=json.loads(args.resume.read_text())
        keys=("affinity","seconds","address_space_mib","repeats","sha256","baseline_sha256","singular")
        if any(report[k]!=saved[k] for k in keys):
            ap.error("resume requires identical native builds and measurement settings")
        report["resumed_harness_sha256"]=saved["harness_sha256"]
        prior={(row["case"],row["operation"]):row for row in saved["rows"]}
    for case in corpus():
        if args.case and case.name not in args.case: continue
        report["cases"].append(asdict(case))
        for op in OPS:
            if (case.name,op) in prior:
                row=prior[case.name,op]
                for entries in row["certificates"].values():
                    for entry in entries:
                        entry["independent_checks"]=verify(case,op,entry["output"],max(10,args.seconds),args.memory_mib)
                report["rows"].append(row)
                continue
            samples={m:[] for m in methods}
            proofs={m:[] for m in methods}
            for i in range(args.repeats):
                for m in methods[i%len(methods):]+methods[:i%len(methods)]:
                    bucket=samples[m]
                    if bucket and bucket[-1]["status"]!="ok": continue
                    sample=run(case,op,m,args.baseline_root if m=="before" else args.atlas_root,args)
                    if sample["status"]=="ok":
                        out=sample.pop("output")
                        if out not in [p["output"] for p in proofs[m]]:
                            checks=verify(case,op,out,max(10,args.seconds),args.memory_mib)
                            proofs[m].append({"output":out,"independent_checks":checks})
                    bucket.append(sample)
            medians={m:statistics.median(v["milliseconds"] for v in bucket) for m,bucket in samples.items()
                     if all(v["status"]=="ok" for v in bucket)}
            report["rows"].append({"case":case.name,"operation":op,"samples":samples,"certificates":proofs,"medians_ms":medians})
            args.output.write_text(json.dumps(report,indent=2)+"\n")
            print(case.name,op,*(f"{m}={round(medians[m],2) if m in medians else samples[m][-1]['status']}" for m in methods),flush=True)


if __name__=="__main__":
    main()
