"""Benchmark powers of ideals over QQ and verify reduced bases in Singular."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.normalization import comparison, ideal
from benchmarks.primary_decomposition import cpu_name, singular, summary
from engine import native_polynomial as np
from engine.native_library import affine_request
from engine.polynomial import parse_expression


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple
    generators: tuple
    exponent: int
    symbolic: bool
    order: str = "grevlex"

    @property
    def op(self):
        return "symbolic-power" if self.symbolic else "ideal-power"


def corpus():
    xy = ("x","y")
    xyz = ("x","y","z")
    xyzw = ("x","y","z","w")
    triangle = ("x*y","x*z","y*z")
    curve = ("y^2-x*z","y*z-x^3","z^2-x^2*y")
    twisted = ("x*z-y^2","x*w-y*z","y*w-z^2")
    a,b,c = "(x+y+2*z)", "(y-2*z)", "(z+1)"
    u,v,w = "(x+2*y)", "(y-z)", "(z+w)"
    h = "(x-z+1)"
    return (
        Case("zero_power_zero",xy,(),0,False),
        Case("unit_power",xy,("3",),17,False),
        Case("exponent_one",xy,("2*x+y","3*y^2"),1,False,"lex"),
        Case("monomial_principal",xyz,("3*x^2*z",),50,False),
        Case("rational_principal",xy,("2*x^2+3*y-1",),5,False),
        Case("nonradical_embedded",xyz,("x^3","x*y","y^2*z"),4,False),
        Case("coordinate_complete_intersection",xyz,("x","y","z"),5,False),
        Case("triangle_ordinary",xyz,triangle,4,False),
        Case("dense_plane",xy,("x^2+2*x*y+3*y^2+x+1","x*y+2*y^2+3*x-1"),3,False),
        Case("space_curve_ordinary",xyz,curve,3,False),
        Case("shifted_generators",xyz,(f"{a}*{b}",f"{a}*{c}",f"{b}*{c}"),2,False),
        Case("symbolic_zero",xy,(),5,True),
        Case("symbolic_unit",xy,("1",),2,True),
        Case("squarefree_principal",xy,("(y^2-x^3)*(2*x+1)",),3,True),
        Case("triangle_symbolic",xyz,triangle,4,True),
        Case("mixed_heights",xyz,("x*y","x*z"),3,True,"lex"),
        Case("five_cycle",("a","b","c","d","e"),("a*b","b*c","c*d","d*e","e*a"),3,True),
        Case("rational_and_conjugate_points",xy,("(x^2+1)*(x-2)","y-x"),3,True,"lex"),
        Case("nonlinear_complete_intersection",xyz,("y^2-x^3","z-x*y"),3,True),
        Case("twisted_cubic",xyzw,twisted,2,True),
        Case("singular_space_curve",xyz,curve,2,True),
        Case("transformed_triangle",xyz,(f"{a}*{b}",f"{a}*{c}",f"{b}*{c}"),2,True),
        Case("nonlinear_mixed_components",xyz,("x*(y^2-z^3)","x*(x-z^2)"),2,True),
        # Additional general-position examples retained after first measurements.
        Case("transformed_determinantal_cube",xyzw,(f"{u}*{w}-{v}^2",f"{u}*w-{v}*{w}",f"{v}*w-{w}^2"),3,True),
        Case("mixed_cusp_and_plane_fourth",xyz,tuple(f"{h}*({g})" for g in ("y^2-x^3","z-x*y")),4,True),
        Case("two_cubics_fourth",xy,("x^3+2*x*y+3*y^2+1","x*y^2-2*y^3+3*x^2-y"),4,False),
        Case("plane_and_space_curve_cube",xyz,tuple(f"{h}*({g})" for g in curve),3,True),
        Case("translated_grid_fourth",xyz,("x*y-1","(x-2)*(x-3)","z-y^2"),4,True,"lex"),
    )


def native_input(c):
    names = tuple(map(np.Symbol, c.variables))
    rows = [np.row(names,np.from_terms(names,parse_expression(g,c.variables))) for g in c.generators]
    return [c.op.replace("-","_"), " ".join(c.variables), c.order, str(c.exponent), *rows]


def call_wire(c):
    line, = affine_request(native_input(c))
    return json.loads(line)


def declarations(c):
    order = {"lex":"lp","grevlex":"dp"}[c.order]
    return f'ring R=0,({",".join(c.variables)}),{order}; option(redSB); short=0; ideal I={ideal(c.generators)};\n'


# For a radical complete intersection all powers are unmixed. The same
# conclusion holds for a zero-dimensional radical ideal (pairwise comaximal
# maximal ideals). For a general unmixed radical ideal in characteristic zero,
# saturating by the Jacobian singular locus removes embedded components:
# all embedded supports lie in the non-smooth locus, which avoids the minimal
# primes. Otherwise keep the primary components over the minimal primes of I.
def symbolic_procedure(method):
    decomp = "primdecGTZ" if method == "GTZ" else "primdecSY"
    saturation = ""
    if method == "Jacobian":
        saturation = """
        int equi=1; for(j=1;j<=size(D);j++){if(dim(std(D[j][2]))!=d){equi=0;}}
        if(equi){
            if(!defined(sat)){LIB "elim.lib";}
            ideal locus=G+minor(jacob(G),nvars(basering)-d);
            return(std(sat(Q,locus)));
        }"""
    return """
proc symbolic(ideal I,int n) {
    ideal G=std(I);
    if(size(G)==0){if(n==0){return(ideal(1));} return(G);}
    if(deg(G[1])==0){return(G);}
    int s=size(G); int i,j; int sf=1; int mon=1; intvec exp;
    for(i=1;i<=s;i++){
        exp=leadexp(G[i]); if(size(G[i])!=1){mon=0;}
        for(j=1;j<=nvars(basering);j++){if(exp[j]>1){sf=0;}}
    }
    if(mon && !sf){ERROR("nonradical monomial ideal");}
    if(s==1 && !sf){
        list F=factorize(G[1]); for(i=2;i<=size(F[1]);i++){
            if(F[2][i]>1){ERROR("nonradical principal ideal");}
        }
        return(std(G^n));
    }
    if(s>1 && !mon){
        poly h=G[1]; ideal K=G;
        for(i=2;i<=s && deg(h)>0;i++){h=gcd(h,G[i]);}
        if(deg(h)>0){
            list HF=factorize(h); for(i=2;i<=size(HF[1]);i++){
                if(HF[2][i]>1){ERROR("nonradical common factor");}
            }
            for(i=1;i<=s;i++){K[i]=G[i]/h;}
            K=std(K);
            if(size(reduce(quotient(K,ideal(h)),K))>0){ERROR("nonradical product");}
            // If K:h=K then (h) intersect K=h*K. In each symbolic power
            // h remains regular modulo K^(n), so the intersection is a product.
            return(std(ideal(h^n)*symbolic(K,n)));
        }
    }
    list D;
    if(!sf){
        if(!defined(DECOMP)){LIB "primdec.lib";}
        D=DECOMP(G); for(i=1;i<=size(D);i++){
            if(size(reduce(D[i][2],std(D[i][1])))>0){ERROR("nonradical ideal");}
        }
    }
    if(n==0){return(ideal(1));} if(n==1){return(G);}
    int d=dim(G);
    if(s==1 || d==0 || size(I)==nvars(basering)-d || s==nvars(basering)-d){return(std(I^n));}
    ideal J=1;
    if(mon){
        list U=indepSet(G,1); intvec v; ideal P;
        for(i=1;i<=size(U);i++){
            v=U[i]; P=0;
            for(j=1;j<=nvars(basering);j++){if(!v[j]){P=P,var(j);}}
            if(i==1){J=P^n;} else {J=intersect(J,P^n);}
        }
        return(std(J));
    }
    if(!defined(DECOMP)){LIB "primdec.lib";}
    if(size(D)==0){D=DECOMP(G);}
    ideal Q=I^n; SATURATION
    list E=DECOMP(Q); ideal B;
    for(i=1;i<=size(E);i++){
        for(j=1;j<=size(D);j++){
            if(size(reduce(E[i][2],std(D[j][2])))==0){
                J=intersect(J,E[i][1]); break;
            }
        }
    }
    return(std(J));
}
""".replace("SATURATION",saturation).replace("DECOMP",decomp)


def methods(c):
    return ["GTZ","Jacobian"] if c.symbolic else ["std"]


def singular_renderer(c):
    # Print the same JSON fields as the native transport. Singular stores its
    # reduced basis in the reverse of Atlas's descending head order.
    prefix = json.dumps({"type":"ideal", "ring":{"field":"QQ","variables":list(c.variables),
                                                "order":c.order}}, separators=(",",":"))[:-1] + ',"generators":['
    return rf"""
proc renderResult(ideal J) {{
    string out={json.dumps(prefix)}; int i;
    for(i=size(J);i>0;i--){{
        if(i<size(J)){{out=out+",";}}
        out=out+"\"" + string(J[i]/leadcoef(J[i]))+"\"";
    }}
    return(out+"]}}");
}}
"""


def run_singular(c, method, batch, seed, timeout, check=None, *, render=False):
    src = declarations(c)
    if c.symbolic:
        src += symbolic_procedure(method)
        call = f"symbolic(I,{c.exponent})"
    else:
        # Singular defines (0)^0 as (0). Both benchmarks use the ring-theoretic
        # convention I^0=(1), including I=(0).
        call = f"std(I^{c.exponent})" if c.exponent else "ideal(1)"
    if render:
        src += singular_renderer(c)
    if batch:
        convert = "answer=renderResult(J);" if render else ""
        src += (f'ideal J={call}; int i; string answer; {convert} int t=rtimer; '
                f'for(i=0;i<{batch};i++){{J={call};{convert}}} print("TIME "+string(rtimer-t));\n')
    elif render:
        src += f'ideal J={call}; print(renderResult(J));\n'

    else:
        src += f'ideal J={call}; print(J);\n'
    if check is not None:
        # Mutual inclusion and identical reduced monic generators. Singular
        # does not always return monic QQ generators, or the same list order.
        src += (f'ideal A={ideal(check["generators"])}; ideal B=std(A);'
                'int ok=(size(reduce(A,std(J)))==0 && size(reduce(J,B))==0 && size(A)==size(J));'
                'int k,l,found; for(k=1;k<=size(A);k++){found=0;'
                'for(l=1;l<=size(J);l++){if(A[k]*leadcoef(J[l])==J[l]){found=1;}}'
                'if(!found){ok=0;}} print("VERIFIED "+string(ok));\n')
    start=perf_counter_ns()
    output=singular(src,seed=seed,timeout=timeout)
    elapsed=(perf_counter_ns()-start)/1e6
    if render and check is not None:
        payload, = [json.loads(s) for s in output.splitlines() if s.startswith("{")]
        ring = {"variables":list(c.variables), "order":c.order, "field":"QQ"}
        if payload["type"] != "ideal" or payload["ring"] != ring:
            raise ValueError("Incorrect serialized Singular ring")
        expected = sorted(tuple(sorted(parse_expression(s,c.variables).items())) for s in check["generators"])
        actual = sorted(tuple(sorted(parse_expression(s,c.variables).items())) for s in payload["generators"])
        if actual != expected:
            raise ValueError("Incorrect serialized Singular generators on " + c.name)
    if check is not None and "VERIFIED 1" not in output.splitlines():
        raise ValueError(f"Singular disagrees on {c.name}/{method}: {check}\n{output}")
    if batch:
        dt, = [int(s[5:]) for s in output.splitlines() if s.startswith("TIME ")]
        return dt/1000/batch
    return elapsed


def run_typed(c, data, batch, timeout):
    command = [str(ROOT / "native/build/bench_ideal_powers"), c.op, c.order, str(c.exponent), str(batch)]
    rows = "\n".join([data[1], *data[4:]]) + "\n"
    run = subprocess.run(command, text=True, input=rows, capture_output=True, timeout=timeout, check=True)
    elapsed, *generators = run.stdout.splitlines()
    return int(elapsed) / 1e6 / batch, generators


def run_cli(c, timeout):
    cmd = [str(ROOT / "atlas"), "algebra", c.op, "--vars", ",".join(c.variables), "--order", c.order,
           "--exponent", str(c.exponent), "--compute", "cpu", "--json"]
    for g in c.generators:
        cmd.extend(["--gen", g])
    start = perf_counter_ns()
    run = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=True)
    elapsed = (perf_counter_ns() - start) / 1e6
    return elapsed, json.loads(run.stdout)["result"]


def time_case(c, args):
    refs = methods(c)
    report = asdict(c)
    names = refs if args.singular_only else ["atlas", *refs]
    data = None if args.singular_only else native_input(c)
    result = None if args.singular_only else call_wire(c)
    if result is not None:
        ring = {"field": "QQ", "variables": list(c.variables), "order": c.order}
        if result.get("type") != "ideal" or result["ring"] != ring:
            raise ValueError(f"Unexpected result on {c.name}: {result}")
        for method in refs:
            run_singular(c, method, 0, 31, args.timeout, check=result)
        report["verified"] = True

    wire = []
    serialized = {method: [] for method in refs}
    for warm, trials, key in ((True, args.repeats, "warm"), (False, args.cold_repeats, "fresh_command")):
        if not trials:
            continue
        samples = {name: [] for name in names}
        for trial in range(trials):
            offset = trial % len(names)
            for name in names[offset:] + names[:offset]:
                if name != "atlas":
                    elapsed = run_singular(c, name, args.batch if warm else 0, 17 + trial, args.timeout)
                    if warm and args.warm_json:
                        serialized[name].append(run_singular(c, name, args.batch, 17 + trial, args.timeout, render=True))
                elif warm:
                    elapsed, generators = run_typed(c, data, args.batch, args.timeout)
                    if generators != result["generators"]:
                        raise ValueError("Typed and wire bases differ on " + c.name)
                    affine_request(data)
                    start = perf_counter_ns()
                    for _ in range(args.batch):
                        affine_request(data)
                    wire.append((perf_counter_ns() - start) / 1e6 / args.batch)
                else:
                    elapsed, answer = run_cli(c, args.timeout)
                    if answer != result:
                        raise ValueError("CLI and wire differ on " + c.name)
                samples[name].append(elapsed)
        report[key] = comparison(samples)
        if not args.singular_only:
            report[key]["wins_vs_all"] = report[key].pop("wins_vs_both")
    if wire:
        report["warm_wire"] = summary(wire)
        if args.warm_json:
            for method in refs:
                run_singular(c, method, 0, 31, args.timeout, result, render=True)
            report["warm_json"] = comparison({"atlas": wire, **serialized})
            report["warm_json"]["wins_vs_all"] = report["warm_json"].pop("wins_vs_both")
    print(c.name, *(f"{name} {report['warm'][name]['median_ms']:.5f}" for name in names), flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--cold-repeats",type=int,default=7)
    parser.add_argument("--batch",type=int,default=5)
    parser.add_argument("--timeout",type=float,default=60)
    parser.add_argument("--cpu",type=int)
    parser.add_argument("--singular-only",action="store_true")
    parser.add_argument("--warm-json",action="store_true",help="Also time complete serialized ideal results in Singular")
    parser.add_argument("--case",action="append")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if min(args.repeats,args.batch)<1 or args.cold_repeats<0 or args.timeout<=0:parser.error("Invalid timing options")
    if args.cpu is not None:os.sched_setaffinity(0,{args.cpu})
    cases = corpus()
    if args.case:
        cases = [c for c in cases if c.name in args.case]
        if {c.name for c in cases} != set(args.case):
            parser.error("Unknown case")
    report={"cpu":cpu_name(),"cpu_affinity":args.cpu,"singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
            "benchmark_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "repeats":args.repeats,"cold_repeats":args.cold_repeats,"warm_batch_size":args.batch,"units":"milliseconds per power",
            "scope":"Complete reduced ideal on parsed objects with one discarded warm-up. Warm wire separately includes sparse parsing and result serialization. Fresh commands include loading, parsing and result printing. No result cache; run order rotates; Singular seed 17+trial. Ordinary uses built-in std(I^n). Symbolic verifies radicality, computes intersections of minimal coordinate prime powers for monomial ideals, splits squarefree coprime common factors, returns ordinary powers for complete intersections and finite reduced algebras, and compares GTZ primary filtering to Jacobian saturation on unmixed ideals (SY primary filtering for mixed dimensions).",
            "warm_json_scope": ("Native sparse input, ideal power and JSON output via the shared Atlas wire. "
                                "Singular starts from a parsed ideal, computes its power and serializes a JSON "
                                "object containing the verified ring metadata and complete monic basis.")
                                if args.warm_json else None,
            "cases":[]}
    if not args.singular_only:
        from engine.native_library import _LIBRARY
        for name,p in (("atlas_library",_LIBRARY),("atlas_typed",ROOT/"native/build/bench_ideal_powers"),
                       ("atlas_native_cli",ROOT/"native/build/local_algebra"),("atlas_launcher",ROOT/"atlas")):
            report[name+"_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    for c in cases:
        report["cases"].append(time_case(c,args))
        if args.output:args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
