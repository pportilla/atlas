"""Time finite QQ loci and compare their prime ideals to Singular's minimal associated primes."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "nonreduced_pair": dict(variables=["x", "y"], equations=["(x*y)^3", "(x-1)^2*(y-1)^2"], units=[]),
    "nonrational_open": dict(variables=["x", "y"], equations=["(x-1)*(x^2+1)", "x*y-1"], units=["x-1"]),
}


def polynomial(coefficients):
    return "(" + ("+".join(f"({q})*t^{k}" for k, q in enumerate(coefficients)) or "0") + ")"


def singular_source(case, orbits, repeats):
    variables = ",".join(case["variables"])
    lines = ['LIB "primdec.lib";', 'system("--ticks-per-sec",1000000);',
             f'ring T=0,(t,{variables}),lp;', 'list candidates;']
    for i, orbit in enumerate(orbits, 1):
        relations = [polynomial(orbit["mu"])]
        for var, c in zip(case["variables"], orbit["coordinates"], strict=True):
            relations.append(f"{var}-{polynomial(c)}")
        lines.append(f'candidates[{i}]=eliminate(ideal({",".join(relations)}),t);')
    lines += [f'ring R=0,({variables}),dp;', 'map proj=T,0,'+variables+';', 'list expected=proj(candidates);',
              'ideal I='+",".join(case["equations"])+';']
    if case["units"]:
        lines.append('ideal U='+('*'.join('('+u+')' for u in case["units"]))+';')
    solve = 'minAssGTZ(I)'
    if case["units"]:
        # Some Singular versions return (ideal, exponent), others just the ideal.
        lines.append('proc onopen(ideal I,ideal U){def S=sat(I,U);'
                     'if(typeof(S)=="list"){return(minAssGTZ(S[1]));} return(minAssGTZ(S));}')
        solve = 'onopen(I,U)'
    lines += [f'list actual={solve};', 'int k; int j; int n; int ok=size(actual)==size(expected);',
              'int start=rtimer;', f'for(k=1;k<={repeats};k++) {{actual={solve};}}', 'int elapsed=rtimer-start;',
              'for(k=1;k<=size(expected);k++){ n=0; for(j=1;j<=size(actual);j++){'
              'if(size(reduce(expected[k],std(actual[j])))==0 && size(reduce(actual[j],std(expected[k])))==0){n++;}}'
              'if(n!=1){ok=0;}}', 'print("LOCUS "+string(ok)+" "+string(elapsed));', 'quit;']
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    if min(args.trials, args.repeats) < 1:
        parser.error("trials and repeats must be positive")
    os.sched_setaffinity(0, {args.cpu})
    report = dict(cpu=args.cpu, trials=args.trials, repeats=args.repeats, cases={})
    for name, case in CASES.items():
        payload = json.dumps(dict(case, repeats=args.repeats))
        def atlas():
            return json.loads(subprocess.run([ROOT/"native/build/bench_finite_locus"], input=payload,
                                             text=True, capture_output=True, check=True, timeout=120).stdout)
        orbits = atlas()["orbits"]
        reference = singular_source(case, orbits, args.repeats)
        samples = {"atlas_us": [], "singular_us": []}
        for i in range(args.trials):
            for system in (("atlas", "singular") if i % 2 == 0 else ("singular", "atlas")):
                if system == "atlas":
                    record = atlas()
                    if record["orbits"] != orbits:
                        raise ValueError(f"Changing orbit coordinates in {name}")
                    samples["atlas_us"].append(record["us"])
                else:
                    result = subprocess.run(["Singular", "-q", "--no-rc"], input=reference, text=True,
                                            capture_output=True, timeout=120, check=True)
                    lines = result.stdout.splitlines()
                    answer = [line.split() for line in lines if line.startswith("LOCUS ")]
                    if any(s.lstrip().startswith("?") for s in lines) or len(answer) != 1 or answer[0][1] != "1":
                        raise ValueError(result.stdout + result.stderr)
                    samples["singular_us"].append(int(answer[0][2])/args.repeats)
        report["cases"][name] = dict(orbits=orbits, samples=samples, medians={k: median(v) for k,v in samples.items()})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
