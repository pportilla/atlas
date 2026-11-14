"""Compare two graph-chart calculations on the exact input of intrinsic blowup B8."""
import argparse
import gzip
import hashlib
from itertools import combinations_with_replacement
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import kernel, native_polynomial as np
from engine.blowup_ideal import blowup_charts
from engine.polynomial import format_poly


def inputs(capture):
    parent = capture["parent"]
    vars1 = tuple(map(np.Symbol, parent["variables"]))
    centre = tuple(np.from_row(vars1, r) for r in parent["chosen"])
    function = np.from_row(vars1, capture["function"])
    charts = blowup_charts(vars1, (), None, centre)
    parent = capture["second_parent"]
    vars2 = tuple(map(np.Symbol, parent["variables"]))
    fixed = tuple(np.from_row(vars2, r) for r in [*parent["ambient"], *parent["chosen"]])
    graph = (*vars2, *np.symbols("u v"))
    _, *rows = capture["second_parent_input"].splitlines()
    equations = tuple(np.from_row(graph, r) for r in rows)
    return vars1, centre, function, charts, vars2, fixed, graph, equations


def singular_expr(variables, value):
    return format_poly(np.terms(variables, value), ["a", "b", "c", "d", "e", "g"][:len(variables)])


def quotient_check_script(variables, centre, function, charts, quotients):
    # Singular recovers the order independently from ordinary centre ideal powers.
    # For each chart compare its representation with the actual Atlas polynomial,
    # and verify f=p^2*q modulo the original graph equations.
    pairs = list(combinations_with_replacement(range(len(centre)), 2))
    gens = ",".join(map(lambda g: singular_expr(variables, g), centre))
    products = ",".join(f"C[{i+1}]*C[{j+1}]" for i, j in pairs)
    script = [f"ring P=0,(a,b,c),dp; ideal C={gens};poly F={singular_expr(variables,function)};",
              "system(\"--ticks-per-sec\",1000000);int start=rtimer;",
              f"ideal J={products};matrix L=lift(J,ideal(F));",
              "ring S=0,(a,b,c,d,e),dp;"]
    # Name the coefficients before mapping between rings.
    for j in range(len(pairs)):
        script += ["setring P;", f"poly h{j}=L[{j+1},1];", "setring S;", f"poly k{j}=imap(P,h{j});"]
    for child in charts:
        ratios = ["1" if v==1 else singular_expr(child.variables,v) for v in child.ratios]
        i = child.pivot_index
        q = "+".join(f'k{j}*({ratios[k]})*({ratios[l]})' for j,(k,l) in enumerate(pairs))
        script.append(f"poly Q{i}={q};")
    script += ["setring P;",'print("TIME "+string(rtimer-start));',
               'print("CHECK "+string((matrix(J)*L)[1,1]==F));',"setring S;"]
    for chart, candidate in zip(charts, quotients):
        j = chart.pivot_index
        polys = ",".join(singular_expr(chart.variables,p) for p in chart.ambient)
        coef = singular_expr(chart.variables,candidate)
        piv = singular_expr(chart.variables,chart.pivot)
        script += [f"ideal A{j}={polys};ideal G{j}=std(A{j});poly p{j}={piv};poly result{j}={coef};",
                   f'print("CHECK "+string(reduce(Q{j}-result{j},G{j})==0));',
                   f'print("CHECK "+string(reduce(imap(P,F)-p{j}^2*result{j},G{j})==0));',
                   f'print("CHECK "+string(reduce(result{j},std(A{j}+ideal(p{j})))!=0));']
    return "\n".join([*script,"quit;"])


def witness_script(parent_vars, parent_equations, variables, equations):
    # Compute the centre basis independently. The two rational possibilities
    # are the roots of the captured centre's third-coordinate eliminant.
    ring = "ring R=0,(a,b,c,d,e,g),dp; "
    eqs = ",".join(singular_expr(variables,r) for r in equations)
    base = ",".join(singular_expr(parent_vars,r) for r in parent_equations)
    script = [ring+f"ideal I={eqs};ideal C={base};",
              "system(\"--ticks-per-sec\",1000000);int start=rtimer;",
              "ideal B=std(C);ideal D=c^2-225/676, a-26/15*c+1, b+1/4, d;",
              "int same=(size(reduce(B,std(D)))==0 && size(reduce(D,B))==0);",
              "ring S=0,(u),dp;map first=R,0,-1/4,15/26,0,0,0;",
              "map second=R,-2,-1/4,-15/26,0,0,0;",
              "int ok=(size(first(I))==0 && size(second(I))==0);setring R;",
              'print("TIME "+string(rtimer-start));print("CHECK "+string(same));setring S;',
              'print("CHECK "+string(ok));quit;']
    return "\n".join(script)


def discover_points_script(parent_vars, parent_equations, variables, equations):
    # Obtain the rational points without supplying either the separating
    # coordinate or its roots. A degree-one residue field gives every coordinate
    # as the constant normal form of its variable modulo the associated prime.
    closed = ",".join(singular_expr(variables,r) for r in equations)
    base = ",".join(singular_expr(parent_vars,r) for r in parent_equations)
    script = ['LIB "primdec.lib";ring P=0,(a,b,c,d),dp;',f'ideal C={base};',
              'ring R=0,(a,b,c,d,e,g),dp;',f'ideal I={closed};',
              'ring S=0,(u),dp;setring P;system("--ticks-per-sec",1000000);int start=rtimer;',
              'list centrePrimes=minAssGTZ(C);list points;int j;ideal B;',
              'for(j=1;j<=size(centrePrimes);j++){B=std(centrePrimes[j]);if(vdim(B)==1){',
              'points[size(points)+1]=list(reduce(a,B),reduce(b,B),reduce(c,B),reduce(d,B));}}',
              'list pt;poly q1,q2,q3,q4;int hits;',
              'for(j=1;j<=size(points);j++){pt=points[j];q1=pt[1];q2=pt[2];q3=pt[3];q4=pt[4];',
              'setring S;map ev=R,imap(P,q1),imap(P,q2),imap(P,q3),imap(P,q4),0,0;',
              'int hit=(size(ev(I))==0);setring P;hits=hits+hit;',
              'setring S;kill ev,hit;setring P;}',
              'print("TIME "+string(rtimer-start));print("CHECK "+string(size(points)==2 && hits==2));quit;']
    return "\n".join(script)


def slice_script(variables, ambient, strict, pivot, points):
    # Same family of coordinate hyperplanes through the first parent point.
    # A non-unit ideal after adding inv*p-1 certifies an actual off-centre point.
    names = "abcd"
    rows = ",".join(singular_expr(variables, f) for f in (*ambient, *strict))
    p = singular_expr(variables, pivot)
    lines = [f"ring R=0,(a,b,c,d,inv),dp;ideal I={rows};poly p={p};",
             'system("--ticks-per-sec",1000000);int start=rtimer;int yes;ideal J;']
    for axis, value in zip(names, points[0]):
        lines.append(f"if (!yes){{J=std(I+ideal({axis}-({value}),p*inv-1));yes=(J[1]!=1);}}")
    lines += ['print("TIME "+string(rtimer-start));print("CHECK "+string(yes));quit;']
    return "\n".join(lines)


def singular(script, checks):
    done = subprocess.run(["Singular","--no-rc","-q"],input=script,text=True,
                          capture_output=True,check=True,timeout=20)
    lines = done.stdout.splitlines()
    if not lines or not lines[0].startswith("TIME ") or lines[1:] != ["CHECK 1"]*checks:
        raise ValueError(done.stdout[:700])
    return int(lines[0].split()[1])/1000


def measure(run, trials, check):
    times = []
    for _ in range(trials):
        start = perf_counter_ns()
        answer = run()
        times.append((perf_counter_ns()-start)/1e6)
        check(answer)
    return times


def main():
    opt = argparse.ArgumentParser(description=__doc__)
    opt.add_argument("--trials",type=int,default=5)
    opt.add_argument("--output",type=Path,required=True)
    args=opt.parse_args()
    if args.trials < 1:
        opt.error("Provide at least one trial.")
    with gzip.open(ROOT/"tests/fixtures/b8-centre-and-incidence.json.gz", "rt") as f:
        capture=json.load(f)
    vars1, centre, function, charts, vars2, basis, graph, equations=inputs(capture)
    def divide():
        return kernel.centre_power_transforms(vars1, (), centre, function, charts)
    def points_and_lift():
        pts=kernel.graph_centre_points(vars2,basis)
        return pts, kernel.graph_centre_incidence(graph,equations,pts)
    def check_quotient(answer):
        if answer is None or answer[0] != 2 or len(answer[1]) != 3:
            raise ValueError("Missing expected parent centre-power certificate")
    def check_points(answer):
        expected={("0","-1/4","15/26","0"),("-2","-1/4","-15/26","0")}
        if set(answer[0]) != expected or answer[1] is not True:
            raise ValueError("Missing rational witness on the second parent's graph")
    answer=divide()
    check_quotient(answer)
    first_script=quotient_check_script(vars1,centre,function,charts,answer[1])
    second_script=witness_script(vars2,basis,graph,equations)
    discovery_script=discover_points_script(vars2,basis,graph,equations)
    parent=capture["second_parent"]
    ambient=tuple(np.from_row(vars2,r) for r in parent["ambient"])
    strict=tuple(np.from_row(vars2,r) for r in parent["strict"])
    pivot=np.from_row(vars2,parent["chosen"][-1])
    points=kernel.graph_centre_points(vars2,basis)
    third_script=slice_script(vars2,ambient,strict,pivot,points)
    def slice_of_parent():
        return kernel.graph_parent_slice(vars2,ambient,strict,pivot,points)
    def check_slice(answer):
        if answer is not True:
            raise ValueError("Expected an off-centre point on the original parent divisor.")
    data={}
    for name,run,check,script,nchecks in (
            ("all_three_quotients",divide,check_quotient,first_script,10),
            ("second_parent_given_witness",points_and_lift,check_points,second_script,2),
            ("second_parent_incidence",points_and_lift,check_points,discovery_script,1),
            ("third_pivot_parent_slice",slice_of_parent,check_slice,third_script,1)):
        atlas_ms=[];singular_ms=[]
        for trial in range(args.trials):
            if trial%2==0:
                atlas_ms.extend(measure(run,1,check))
            singular_ms.append(singular(script,nchecks))
            if trial%2:
                atlas_ms.extend(measure(run,1,check))
        data[name]={"atlas_ms":atlas_ms,"singular_ms":singular_ms,
                    "medians_ms":{"Atlas":statistics.median(atlas_ms),"Singular":statistics.median(singular_ms)}}
    result={"cpu_affinity":sorted(os.sched_getaffinity(0)), "library_sha256":hashlib.sha256(
        (ROOT/"native/build/libgroebner.so").read_bytes()).hexdigest(),
        "singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
        "conditions":"QQ, captured canonical B8. Input objects prepared before timing; Atlas includes "
        "native request and result transfer. Singular runs in fresh processes with timers after input parsing. "
        "Quotients: Atlas selects a centre power, Singular is given power 2, computes labelled ideal lift "
        "and transports the quotient to all three charts; independent normal-form checks are outside its timer. "
        "Incidence: Atlas discovers rational centre points; Singular verifies the supplied centre shape and "
        "the two stated rational solutions by ideal equality and substitution. This is a certificate check "
        "for Singular; the additional discovery comparison uses minAssGTZ and derives all QQ points without hints. "
        "Parent slice: both use the same supplied point, "
        "test hyperplanes in coordinate order and seek a nonempty complement of the pivot. "
        "Timings exclude chart construction.",
        "comparisons":data}
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
