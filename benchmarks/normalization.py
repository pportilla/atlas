"""Time normalization in Atlas and Singular, and compare the maps in the total fraction ring."""
import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.primary_decomposition import Case, cpu_name, declare, singular, summary
from engine import native_polynomial as np
from engine.native_library import affine_request
from engine.polynomial import parse_expression


def corpus():
    # Fixed before timing either system. Inputs are reduced, as required by
    # Singular's unqualified normal(). normal(I,"prim") also supports nilpotents.
    return (
        Case("smooth_graph", ("x","y"), ("y-x^2-x-1",)),
        Case("smooth_conic", ("x","y"), ("x^2+y^2-3",)),
        Case("normal_quadric_cone", ("x","y","z"), ("x*z-y^2",)),
        Case("cusp_2_3", ("x","y"), ("y^2-x^3",), "lex"),
        Case("shifted_cusp_3_5", ("x","y"), ("(y+x-2)^3-(x-1)^5",)),
        Case("rational_node", ("x","y"), ("y^2-x^2*(x+1)",)),
        Case("tacnode", ("x","y"), ("y^2-x^4",)),
        Case("space_curve_3_4_5", ("x","y","z"), ("y^2-x*z","y*z-x^3","z^2-x^2*y")),
        Case("whitney_umbrella", ("x","y","z"), ("z^2-x^2*y",)),
        Case("line_plus_cusp", ("x","y"), ("x*(y^2-x^3)",)),
        Case("mixed_plane_line", ("x","y","z"), ("x*(x-1)", "x*y")),
        Case("finite_conjugate_field", ("x","y"), ("x^2+1", "y-2*x"), "lex"),
        Case("conjugate_cusps", ("x","y"), ("y^2-(x^2+1)^3",)),
        # Post-baseline cases, retained after profiling the general algorithm.
        Case("ramified_cusp_3_7", ("x","y"), ("(y+2*x+1)^3-(x+1)^7",)),
        Case("translated_cusp_4_7", ("x","y"), ("(y-x-1)^4-(x-2)^7",)),
        Case("shifted_cyclic_cover", ("x","y"), ("(y-x)^3-x^7*(1+x)",)),
        Case("two_branch_divisors", ("x","y"), ("y^3-(x-1)^2*(x+1)^4",)),
        Case("cyclic_cover_surface", ("u","v","w"), ("w^3-(u+1)^2*(v^2+u)^4",)),
        Case("noncyclic_plane_quartic", ("x","y"), ("y^4-x^3*y^2+x^6*y-x^7",)),
        Case("noncyclic_plane_quintic", ("x","y"), ("y^3-2*x^2*y^2+x^4*y-x^5",)),
        Case("mixed_translated_curve", ("u","v","w"),
             ("((v+1)^2-(u+v)^3)*(u+w-1)","((w-u)^2-(v+1))*(u+w-1)")),
        Case("nonmonic_surface", ("x","z","y"), ("x*y^3+(x+z)^5*y+(x+z)^7",)),
        Case("non_complete_intersection_surface", ("x","y","z","w"),
             ("x^3-y^2","w^2-x*z^2","x*w-y*z","y*w-x^2*z")),
    )


def native_input(case):
    names = list(case.variables)
    coordinates = tuple(map(np.Symbol, names))
    rows = [np.row(coordinates, np.from_terms(coordinates, parse_expression(f, names)))
            for f in case.generators]
    return ["normalization", " ".join(names), case.order, *rows]


def normalize(case):
    line, = affine_request(native_input(case))
    return json.loads(line)


METHODS = ("atlas", "normal", "normal_prim")


def singular_call(method):
    return {"normal": "normal(inputIdeal)", "normal_prim": 'normal(inputIdeal,"prim")'}[method]


def run_singular(case, method, seed, timeout, *, warm):
    cmd = ["Singular", "-q", "--no-rc", f"--random={seed}", "--ticks-per-sec=1000000"]
    source = 'LIB "normal.lib";\n' + declare(case) + "printlevel=-1;\n"
    call = singular_call(method)
    if warm:
        source += (f'list first={call}; kill first;\n'
                   f'int start=rtimer; list N={call}; int elapsed=rtimer-start;\n'
                   'print("TIME "+string(elapsed));\n')
    else:
        # Singular packages each component presentation/map in a ring.
        # Printing only the top-level list would omit those output polynomials.
        source += (f'list N={call};\n'
                   'for(int i=1; i<=size(N[1]); i++){\n'
                   '  print(N[2][i]); def component=N[1][i]; setring component; '
                   '  print(norid); print(normap); setring R; kill component;\n}\n')
    start = perf_counter_ns()
    p = subprocess.run(cmd, input=source+"quit;\n", capture_output=True, text=True, check=True, timeout=timeout)
    elapsed = (perf_counter_ns()-start)/1e6
    if any(s.lstrip().startswith("?") for s in (p.stdout+p.stderr).splitlines()):
        raise RuntimeError(p.stdout+p.stderr)
    if warm:
        micros, = [int(s[5:]) for s in p.stdout.splitlines() if s.startswith("TIME ")]
        return micros/1000
    if not p.stdout.strip():
        raise RuntimeError("Singular returned no presentation.")
    return elapsed


def run_cli(case, timeout):
    cmd = [str(ROOT/"atlas"), "algebra", "normalization", "--vars", ",".join(case.variables),
           "--order", case.order, "--compute", "cpu", "--json"]
    for f in case.generators:
        cmd += ["--gen", f]
    start = perf_counter_ns()
    p = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    elapsed = (perf_counter_ns()-start)/1e6
    record = json.loads(p.stdout)
    if record["status"] != "ok":
        raise ValueError(record)
    return elapsed, record["result"]


def ideal(polynomials):
    return "ideal("+(",".join(polynomials) or "0")+")"


def verify(case, result, timeout=90):
    """Compare each embedded affine algebra with Singular's normalized fractional R-module."""
    expected_ring = {"field": "QQ", "variables": list(case.variables), "order": case.order}
    if result["type"] != "normalization" or result["ring"] != expected_ring:
        raise ValueError("Wrong operation or source coordinate ring")
    # Alpha-rename all coordinates. Source/target names can coincide with
    # Singular procedure names or with our verifier's temporary objects.
    def rename(expression, mapping):
        return re.sub(r"[A-Za-z_][A-Za-z_0-9]*",
                      lambda word: mapping[word.group()], expression)

    source_names = {name: f"atlasSrc{i}" for i, name in enumerate(case.variables)}
    case = Case(case.name, tuple(source_names.values()),
                tuple(rename(f, source_names) for f in case.generators), case.order)
    components = result["components"]
    code = ['LIB "normal.lib";', declare(case), 'printlevel=-1;',
            'proc eqIdeal(ideal A,ideal B){ return(size(reduce(A,std(B)))==0 && size(reduce(B,std(A)))==0); }',
            'list ref=normal(inputIdeal,"prim"); list refprime; int j; int ok=1;',
            'if(size(std(inputIdeal)) && std(inputIdeal)[1]==1){ref=list(list(),list());}',
            f'if(size(ref[1])!={len(components)}){{ok=0;}}',
            'for(j=1;j<=size(ref[1]);j++){ '
            ' def T=ref[1][j]; setring T; map mp=R,normap; setring R; '
            ' refprime[j]=std(preimage(T,mp,norid)); kill T; }',
            'ideal red=1; for(j=1;j<=size(refprime);j++){red=intersect(red,refprime[j]);}',
            f'if(eqIdeal(red,inputIdeal)!={int(result["input_reduced"])}){{ok=0;}}',
            'int matched; intvec used; ideal num,den,P,U; poly d;']
    for k, c in enumerate(components, 1):
        target = c["ring"]
        coords = {name: f"atlasDst{i}" for i, name in enumerate(target["variables"])}
        fractions = c["fractions"]
        if (target["field"]!="QQ" or len(fractions)!=len(coords)
                or len(c["images"])!=len(case.variables)):
            raise ValueError("Wrong number of map images or fraction coordinates")
        # Singular requires at least one variable. S/(dummy) realizes QQ.
        target_vars = ",".join(coords.values()) or "oracleDummy"
        relations = [rename(f, coords) for f in c["relations"]] if coords else ["oracleDummy", *c["relations"]]
        source_var = f"atlasTarget{k}"
        code += [f'P=std({ideal(rename(f, source_names) for f in c["prime"])}); matched=0;',
                 'for(j=1;j<=size(refprime);j++){if(eqIdeal(P,refprime[j])){matched=j;}}',
                 'if(!matched){ERROR("Atlas component has no Singular prime");}',
                 'if(matched<=size(used) && used[matched]){ok=0;} used[matched]=1;',
                 'U=ref[2][matched]; d=U[size(U)];',
                 f'num={ideal(rename(f["numerator"], source_names) for f in fractions)};',
                 f'den={ideal(rename(f["denominator"], source_names) for f in fractions)};']
        for i in range(1,len(coords)+1):
            # n/e ∈ U/d iff n*d ∈ e*U modulo P. d and e are nonzero on the branch.
            code += [f'if(reduce(den[{i}],P)==0 || reduce(num[{i}]*d,std(P+den[{i}]*U))!=0){{ok=0;}}']
        code += [f'ring {source_var}=0,({target_vars}),dp;',
                 f'ideal rel=std({ideal(relations)});',
                 f'map phi=R,{",".join(rename(f, coords) for f in c["images"]) or "0"};',
                 'ideal ns=phi(num); ideal ds=phi(den);',
                 'poly refden=phi(d); poly allDen=1; ideal refgens=phi(U);']
        for i in range(1,len(coords)+1):
            code += [f'if(reduce(var({i})*ds[{i}]-ns[{i}],rel)!=0){{ok=0;}} allDen=allDen*ds[{i}];']
        code += ['if(!eqIdeal(quotient(rel,ideal(allDen)),rel)){ok=0;}',
                 # g/d ∈ B iff phi(g) ∈ phi(d)*B. The two inclusions compare
                 # the actual subrings over R/P in its fraction field.
                 'if(size(reduce(refgens,std(rel+ideal(refden))))){ok=0;}',
                 'setring R;',
                 f'if(!eqIdeal(preimage({source_var},phi,rel),P)){{ok=0;}}']
    code.append('print("VERIFIED "+string(ok));')
    answer = singular("\n".join(code), seed=51, timeout=timeout)
    if "VERIFIED 1" not in answer.splitlines():
        raise ValueError(f"Singular rejected {case.name}: {result}\n{answer}")


def comparison(samples):
    out = {m: summary(ts) for m,ts in samples.items()}
    if "atlas" in out:
        references = [m for m in samples if m!="atlas"]
        paired = [min(ts)/a for a,ts in zip(samples["atlas"], zip(*(samples[m] for m in references)))]
        out["wins_vs_both"] = sum(x>1 for x in paired)
        out["worst_paired_speedup"] = min(paired)
        out["median_speedup_vs_faster_reference"] = min(out[m]["median_ms"] for m in references)/out["atlas"]["median_ms"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--cold-repeats",type=int,default=0)
    parser.add_argument("--cpu",type=int)
    parser.add_argument("--case",action="append")
    parser.add_argument("--timeout",type=float,default=90)
    parser.add_argument("--singular-only",action="store_true")
    parser.add_argument("--output",type=Path)
    a = parser.parse_args()
    if a.repeats<1 or a.cold_repeats<0:
        parser.error("Positive warm repeats and nonnegative cold repeats are required.")
    if a.cpu is not None:
        os.sched_setaffinity(0,{a.cpu})
    cases = corpus()
    if a.case:
        cases = [c for c in cases if c.name in a.case]
        if {c.name for c in cases}!=set(a.case):
            parser.error("Unknown case")
    methods = list(METHODS[1:] if a.singular_only else METHODS)
    report = {"cpu":cpu_name(),"cpu_affinity":a.cpu,"singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
              "units":"wall-clock milliseconds","repeats":a.repeats,"cold_repeats":a.cold_repeats,
              "scope":"Warm native affine request with sparse input decoding and complete result serialization, versus Singular normal(I) and normal(I, prim) with the same input ring already loaded. Discarded warmup for each trial. Fresh CLI measurements include startup, parsing and complete output of the presentation and maps. Order rotates and Singular seeds are 17 + trial.", "cases":[]}
    if not a.singular_only:
        from engine.native_library import _LIBRARY
        paths = {"atlas_library":_LIBRARY,"atlas_launcher":ROOT/"atlas"}
        if (ROOT/"native/build/normalization").exists():
            paths["atlas_native_cli"] = ROOT/"native/build/normalization"
        report.update({k+"_sha256":hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
    for case in cases:
        record = asdict(case)
        data = native_input(case) if not a.singular_only else None
        result = None
        for warm, count, key in ((True,a.repeats,"warm"),(False,a.cold_repeats,"fresh_command")):
            if not count: continue
            samples = {m:[] for m in methods}
            for i in range(count):
                for method in methods[i%len(methods):]+methods[:i%len(methods)]:
                    if method!="atlas":
                        elapsed = run_singular(case,method,17+i,a.timeout,warm=warm)
                    elif warm:
                        affine_request(data)
                        start = perf_counter_ns()
                        line, = affine_request(data)
                        elapsed = (perf_counter_ns()-start)/1e6
                        result = json.loads(line)
                    else:
                        elapsed, cli_result = run_cli(case,a.timeout)
                        if cli_result!=result:
                            raise ValueError(f"CLI and kernel disagree on {case.name}")
                    samples[method].append(elapsed)
            record[key] = comparison(samples)
        if result is not None:
            verify(case,result,a.timeout)
            record["verified_components"]=len(result["components"])
        report["cases"].append(record)
        print(case.name,*(f"{m} {record['warm'][m]['median_ms']:.3f}ms" for m in methods), flush=True)
        if a.output:
            a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
