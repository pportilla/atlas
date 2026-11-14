"""Compare Atlas and Singular frame membership certificates over QQ."""
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
from engine.polynomial import format_poly, sparse_row
from tests.test_frame_membership import b9_restricted_frame, native_check


def cases():
    # Each tuple includes a known-nonempty flag and the expected shortcut result.
    eq = ["1 0 1;-1 2 0"]
    yield "small_defining_equation", eq, eq[0], ["1 1 0"], False, "yes", True
    yield "small_nonzero_scalar", eq, "-7/3 0 0", ["1 1 0"], True, "no", False
    yield "empty_open_scalar", ["1 1 0"], "1 0 0", ["1 1 0"], False, "scalar", True
    yield "small_inconclusive", eq, "1 0 2;-1 4 0", ["1 1 0"], False, "unknown", True
    packet = b9_restricted_frame()
    eq, units = packet["level_rows"], packet["units"]
    yield "B9_defining_equation", eq, eq[-1], units, False, "yes", True
    yield "B9_one_on_proven_nonempty_frame", eq, "1 0 0 0 0 0", units, True, "no", False


# The independent reference on the actual five-coordinate B9 chart. On the
# line x=z=0 the first two equations determine u and v. Common-factor
# removal proves the univariate fourth equation has a root off all opens.
B9_NONEMPTY = """
poly p=I[1];poly q=I[2];poly r=I[3];poly h=I[4];
poly a=opens[1];poly b=opens[2];poly c=opens[3];
ring line=0,(Y),dp;
poly U,V,H,W,C;
map phi=R,0,Y,0,0,0;
U=-phi(p);
phi=R,0,Y,0,U,0;
V=-phi(q);
phi=R,0,Y,0,U,V;
H=phi(h);
W=phi(a)*phi(b)*phi(c);
int valid=(phi(p)==0 && phi(q)==0 && phi(r)==0 && deg(H)==19 && deg(W)==24);
if (valid) {
    C=gcd(H,W);
    while(deg(C)>0){H=H/C;C=gcd(H,W);}
    valid=(deg(H)>0);
}
print("NONEMPTY "+string(valid));
"""


def singular_input(eq, query, opens, known, b9):
    width = len(query.split(";")[0].split())-1
    names = ["x","y","z","u","v"][:width]
    def poly(row):
        return format_poly(sparse_row(row), names)
    lines = [f'ring R=0,({",".join(names)}),dp;', 'system("--ticks-per-sec",1000000);',
             'proc membership(ideal I,poly f,int known) {',
             'if(f==0){return("yes");} int i; for(i=1;i<=ncols(I);i++){if(I[i]==f){return("yes");}}',
             'if(deg(f)==0){if(known){return("no");}return("scalar");}return("unknown");}',
             'int before=rtimer;', f'ideal I={",".join(map(poly,eq))};poly f={poly(query)};',
             f'int start=rtimer;string decision=membership(I,f,{int(known)});int finish=rtimer;',
             'print("RESULT "+decision);print("US "+string(finish-start)+" "+string(finish-before));',
             'ideal opens='+','.join(map(poly, opens))+';']
    if b9:
        # Reference generator equality in QQ, independent of sparse strings.
        lines += ['print("VANISHES "+string(f==I[4]));', B9_NONEMPTY]
    else:
        # Add t*unit-1 to compute membership in the localized ideal.
        lines += [f'ring S=0,(t,{",".join(names)}),dp; ideal J=imap(R,I);',
                  'ideal O=imap(R,opens);poly unit=1;int k;',
                  'for(k=1;k<=ncols(O);k++){unit=unit*O[k];}',
                  'ideal G=std(J+ideal(t*unit-1));',
                  'print("VANISHES "+string(reduce(imap(R,f),G)==0));',
                  'print("NONEMPTY "+string(reduce(1,G)!=0));']
    return "\n".join([*lines, "quit;"])


def run(case, trials):
    name, eq, query, units, known, result, vanishes = case
    width = len(query.split(";")[0].split())-1
    source = singular_input(eq, query, units, known, name.startswith("B9"))
    timings = {"Atlas": [], "Singular": [], "Singular_with_input": []}
    def atlas():
        start = perf_counter_ns()
        reply = native_check(width, eq, query, known)
        timings["Atlas"].append((perf_counter_ns()-start)/1e6)
        if reply != [result]:
            raise AssertionError((name,reply))
    for i in range(trials):
        if i % 2 == 0:
            atlas()
        output = subprocess.run(["Singular","--no-rc","-q"], text=True, input=source,
                                capture_output=True, check=True, timeout=30)
        lines = output.stdout.splitlines()
        if (len(lines)!=4 or lines[0]!=f"RESULT {result}" or not lines[1].startswith("US ")
                or lines[2]!=f"VANISHES {int(vanishes)}"
                or lines[3]!=f"NONEMPTY {int(name!='empty_open_scalar')}"):
            raise ValueError(f"{name}: {output.stdout[:1000]} {output.stderr[:200]}")
        _, compute, with_input = lines[1].split()
        timings["Singular"].append(int(compute)/1000)
        timings["Singular_with_input"].append(int(with_input)/1000)
        if i % 2:
            atlas()
    return {"case":name, "result":result, "vanishes":vanishes, "known_nonempty":known,
            "equation_terms":[r.count(";")+1 for r in eq], "trials_ms":timings,
            "median_ms":{key:median(seq) for key,seq in timings.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Use at least one trial.")
    result = {"singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
              "library_sha256":hashlib.sha256((ROOT/"native/build/libgroebner.so").read_bytes()).hexdigest(),
              "cpu_affinity":sorted(os.sched_getaffinity(0)),
              "conditions":"Same rational equations, candidate and nonemptiness knowledge in both systems. "
                           "Atlas times a native request with pre-encoded sparse rows, including payload transport. "
                           "Singular times the in-ring comparisons and the interval beginning before input declarations "
                           "in a fresh process. Both exclude process startup and all independent verification. "
                           "For small frames the reference uses a localized ideal. On B9 the reference checks "
                           "the generator identity and a univariate geometric nonemptiness witness. "
                           "Times concern the membership test after evidence is supplied.",
              "comparisons":[run(case,args.trials) for case in cases()]}
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
