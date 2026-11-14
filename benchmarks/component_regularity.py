"""Compare native regularity certificates to exact associated-prime tests in Singular."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import native_polynomial as np
from engine.native_library import ideal_request
from engine.polynomial import format_poly, sparse_row
from tests.test_component_regularity import b9_component_requests


def cases():
    x, y, z = variables = np.symbols("x y z")
    for name, eq, multiplier, regular in (
            ("embedded_prime_hit", (x*x, x*y), y, False),
            ("embedded_prime_avoided", (x*x, x*y), y+z, True),
            ("two_thick_components", (x*x, y*z), y+z, True),
            ("one_thick_component_hit", (x*x, y*z), y, False),
            ("components_without_rational_points", (x*x+1, y*z), x*(y+z), True)):
        rows = [np.row(variables, g) for g in (*eq, multiplier)]
        yield name, "graph_regular 3 2", rows, regular
    for case in b9_component_requests():
        yield f"B9_chart_{case['chart']}", case["header"], case["rows"], True


SINGULAR_PROCEDURE = """
proc componentRegular(ideal I,poly f)
{
    int i,j,k; int valid; int again=1; poly d,image;
    // Each scalar graph pivot gives an exact quotient-ring isomorphism.
    while(again)
    {
        again=0;
        for(i=1;i<=ncols(I) && !again;i++)
        {
            for(j=nvars(basering);j>=1 && !again;j--)
            {
                d=diff(I[i],var(j));
                if(d!=0 && deg(d)==0)
                {
                    image=var(j)-I[i]/leadcoef(d);
                    I[i]=0;
                    for(k=1;k<=ncols(I);k++){I[k]=subst(I[k],var(j),image);}
                    f=subst(f,var(j),image);
                    again=1;
                }
            }
        }
    }
    list C=primdecGTZ(I);
    ideal Q,P,W,combined;
    intvec freeAxes;
    for(i=1;i<=size(C);i++)
    {
        Q=C[i][1];
        if(i==1){combined=Q;}else{combined=intersect(combined,Q);}
    }
    if(size(reduce(I,std(combined))) || size(reduce(combined,std(I)))){return(-1);}
    for(i=1;i<=size(C);i++)
    {
        P=std(C[i][2]);
        freeAxes=indepSet(P);
        valid=0;
        // Nonzero on any exact fibre proves nonmembership in the prime.
        for(j=0;j<2 && !valid;j++)
        {
            W=P;
            for(k=1;k<=size(freeAxes);k++)
            {
                if(freeAxes[k]){W=W+ideal(var(k)-(1+j));}
            }
            if(reduce(f,std(W))!=0){valid=1;}
        }
        if(!valid && reduce(f,P)==0){return(0);}
    }
    return(1);
}
"""


def singular_program(name, width, rows, check_saturation):
    names = list("xyuzab"[:width])
    polynomials = [format_poly(sparse_row(row), names) for row in rows]
    if name == "B9_chart_23":
        declarations = [f"poly h{i}={poly};" for i, poly in enumerate(polynomials)]
        oracle = (Path(__file__).parent/"chart23_component_regularity.sing").read_text()
        return "\n".join(['LIB "primdec.lib";', f'ring R=0,({",".join(names)}),dp;option(redSB);',
                          'system("--ticks-per-sec",1000000);int beforeInput=rtimer;',
                          *declarations, 'int start=rtimer;', oracle])
    equations = ",".join(polynomials[:-1])
    lines = ['LIB "primdec.lib";', f'ring R=0,({",".join(names)}),dp;',
             SINGULAR_PROCEDURE, 'system("--ticks-per-sec",1000000);int beforeInput=rtimer;',
             f'ideal I={equations};poly f={polynomials[-1]};',
             'int start=rtimer;int answer=componentRegular(I,f);int finish=rtimer;',
             'print("US "+string(finish-start)+" "+string(finish-beforeInput));',
             'print("REGULAR "+string(answer));']
    if check_saturation:
        lines += ['int startSat=rtimer;ideal J=sat(I,ideal(f));',
                  'int same=(size(reduce(I,std(J)))==0 && size(reduce(J,std(I)))==0);',
                  'int finishSat=rtimer;',
                  'print("SAME "+string(same));print("SAT_US "+string(finishSat-startSat));']
    return "\n".join([*lines, 'quit;'])


def compare(case, trials):
    name, header, rows, expected = case
    _, width, _ = header.split()
    check_saturation = not name.startswith("B9_")
    script = singular_program(name, int(width), rows, check_saturation)
    timings = {"Atlas": [], "Singular": [], "Singular_with_input": []}
    if check_saturation:
        timings["Singular_full_saturation_and_equality"] = []

    def atlas():
        start = perf_counter_ns()
        result = ideal_request(header, rows)
        timings["Atlas"].append((perf_counter_ns() - start)/1e6)
        if result != ["yes" if expected else "unknown"]:
            raise ValueError((name, result))

    for iteration in range(trials):
        if iteration % 2 == 0:
            atlas()
        p = subprocess.run(["Singular", "--no-rc", "-q"], input=script, capture_output=True,
                           text=True, check=True, timeout=60)
        lines = p.stdout.splitlines()
        if check_saturation and lines and lines[-1].startswith("SAT_US "):
            timings["Singular_full_saturation_and_equality"].append(int(lines.pop().split()[1])/1000)
        results = [f"REGULAR {int(expected)}"]
        if check_saturation:
            results.append(f"SAME {int(expected)}")
        if not lines or not lines[0].startswith("US ") or lines[1:] != results:
            raise ValueError(f"{name}: {p.stdout[:800]} {p.stderr[:200]}")
        _, compute_us, input_us = lines[0].split()
        timings["Singular"].append(int(compute_us)/1000)
        timings["Singular_with_input"].append(int(input_us)/1000)
        if iteration % 2:
            atlas()
    return {"case": name, "regular": expected, "full_saturation_compared": check_saturation,
            "terms": [row.count(";")+1 for row in rows], "trials_ms": timings,
            "median_ms": {k: median(values) for k, values in timings.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Choose at least one trial.")
    data = {"cpu_affinity": sorted(os.sched_getaffinity(0)),
            "library_sha256": hashlib.sha256((ROOT/"native/build/libgroebner.so").read_bytes()).hexdigest(),
            "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
            "conditions": "Identical rational inputs. Atlas includes the native request and sparse "
                          "decoding of pre-encoded rows. Singular eliminates scalar graph pivots, computes a primary decomposition "
                          "and checks its intersection against the original ideal. For chart 23 Singular "
                          "verifies the complete intersection's height and decomposes its two "
                          "factor branches. It tests every "
                          "associated prime, including embedded ones, by exact fibre or normal form. "
                          "Both timings exclude process startup. Singular reports computation alone "
                          "and time including polynomial input. Small examples separately compare "
                          "ordinary saturation plus ideal equality under a separate timer. Alternating trial order, "
                          "one fixed CPU for the recorded run.",
            "comparisons": [compare(case, args.trials) for case in cases()]}
    args.output.write_text(json.dumps(data, indent=2)+"\n")


if __name__ == "__main__":
    main()
