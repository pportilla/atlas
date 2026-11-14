"""Compare graph-coordinate regularity certificates in Atlas and Singular over QQ."""
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
from engine.polynomial import format_poly
from tests.test_graph_regular import b9_frame


def cases():
    x, y, u, v = variables = np.symbols("x y u v")
    graph = (u-y*y-x*x, v-u*u-x)
    equation = (v-u*u)**3*(y-1)**2
    yield "triangular_coprime", variables, (*graph, equation), v-u*u+y, True
    yield "triangular_zero_divisor", variables, (*graph, equation), u-y*y, False
    yield "reducible_hypersurface", (x, y), (x*x*y,), x+y, True
    yield "B9_first_frame", *b9_frame(), True


SINGULAR_PROCEDURE = """
proc graphRegular(ideal E, poly F)
{
    int i,j,k; int again=1;
    poly d,image,h;
    int count;
    while(again)
    {
        again=0;
        for(i=1;i<=ncols(E) && !again;i++)
        {
            for(j=1;j<=nvars(basering) && !again;j++)
            {
                d=diff(E[i],var(j));
                if(d!=0 && deg(d)==0)
                {
                    image=var(j)-E[i]/leadcoef(d);
                    E[i]=0;
                    for(k=1;k<=ncols(E);k++){E[k]=subst(E[k],var(j),image);}
                    F=subst(F,var(j),image);
                    again=1;
                }
            }
        }
    }
    for(i=1;i<=ncols(E);i++){if(E[i]!=0){h=E[i];count++;}}
    if(count==0){return(F!=0);}
    if(count==1 && F!=0){return(deg(gcd(h,F))==0);}
    return(0);
}
"""


def singular_script(variables, equations, function, direct):
    names = [v.name for v in variables]
    def text(poly):
        return format_poly(np.terms(variables, poly), names)
    equations_text = ",".join(map(text, equations))
    lines = ['LIB "elim.lib";', f'ring R=0,({",".join(names)}),dp;', SINGULAR_PROCEDURE,
             'system("--ticks-per-sec",1000000);int beforeInput=rtimer;',
             f'ideal I={equations_text};poly f={text(function)};',
             'int start=rtimer;int answer=graphRegular(I,f);int finished=rtimer;',
             'print("TIME_US "+string(finished-start));print("WITH_INPUT_US "+string(finished-beforeInput));',
             'print("REGULAR "+string(answer));']
    if direct:
        # For these smaller inputs, verify the predicate by an unrelated full
        # saturation and reciprocal ideal containment, outside the timer.
        lines += ['ideal J=sat(I,ideal(f));int same=(size(reduce(J,std(I)))==0 && size(reduce(I,std(J)))==0);',
                  'print("SATURATION_UNCHANGED "+string(same));']
    return "\n".join([*lines, "quit;"])


def compare(case, trials):
    name, variables, eq, function, expected = case
    header = f"graph_regular {len(variables)} {len(eq)}"
    rows = [np.row(variables, f) for f in (*eq, function)]
    direct = name != "B9_first_frame"
    script = singular_script(variables, eq, function, direct)
    atlas, singular, singular_with_input = [], [], []
    def run_atlas():
        start = perf_counter_ns()
        decision = ideal_request(header, rows)
        atlas.append((perf_counter_ns()-start)/1e6)
        if decision != ["yes" if expected else "unknown"]:
            raise ValueError(f"{name}: unexpected Atlas decision {decision}")
    for trial in range(trials):
        if trial % 2 == 0:
            run_atlas()
        reply = subprocess.run(["Singular", "--no-rc", "-q"], input=script, capture_output=True,
                               text=True, timeout=45, check=True)
        lines = reply.stdout.splitlines()
        checks = [f"REGULAR {int(expected)}"]
        if direct:
            checks.append(f"SATURATION_UNCHANGED {int(expected)}")
        if (len(lines) < 2 or not lines[0].startswith("TIME_US ")
                or not lines[1].startswith("WITH_INPUT_US ") or lines[2:] != checks):
            raise ValueError(f"{name}: unexpected Singular output {reply.stdout[:1000]}")
        singular.append(int(lines[0].split()[1])/1000)
        singular_with_input.append(int(lines[1].split()[1])/1000)
        if trial % 2:
            run_atlas()
    return {"case": name, "regular": expected, "checked_by_full_saturation": direct,
            "terms": [len(np.terms(variables, f)) for f in (*eq, function)],
            "atlas_ms": atlas, "singular_ms": singular,
            "singular_with_input_ms": singular_with_input,
            "median_ms": {"Atlas": median(atlas), "Singular": median(singular),
                          "Singular_with_input": median(singular_with_input)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Provide at least one trial.")
    result = {"cpu_affinity": sorted(os.sched_getaffinity(0)),
              "library_sha256": hashlib.sha256((ROOT/"native/build/libgroebner.so").read_bytes()).hexdigest(),
              "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "conditions": "QQ on identical sparse input. Atlas includes native request, decoding and result transfer. "
                            "Singular uses a fresh process for each trial. It records both computation alone and elapsed time from the initial polynomial declarations. The procedures discover "
                            "scalar graph pivots independently and test the residual gcd. Small cases also verify the "
                            "result against ordinary full saturation outside the timer. Atlas and Singular run in "
                            "alternating order; centre selection and process startup are excluded.",
              "comparisons": [compare(case, args.trials) for case in cases()]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
