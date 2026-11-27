"""Compare exact membership, open-fibre and crossing predicates on a shared set of QQ inputs."""
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.kernel_json import kernel_json
from engine.polynomial import parse_expression


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    equation: str
    generators: tuple
    expected: bool


CASES = (
    Case("radical_member", "radical_subset", "x", ("y-x^2", "y^2"), True),
    Case("principal_member", "ideal_subset", "x^3-y*x", ("x^2-y",), True),
    Case("nonmember", "ideal_subset", "x^2+1", ("x*y", "z"), False),
    Case("empty_principal_open", "frame_empty", "x^2", ("y-x^2", "x*y"), True),
    Case("unit_ideal", "empty_locus", "1", ("x*y-1", "x"), True),
    Case("zero_member", "ideal_subset", "0", ("x*y",), True),
    Case("transverse_strict_transform", "constant_crossing", "y-20*x*z+3*x^2*y^2*z", ("x*z",), True),
    Case("high_power_on_fibre", "constant_crossing", "y*(1+x*z)+z*(x*z)^12", ("x*z",), True),
    Case("tangency", "constant_crossing", "x+y^2", ("x",), False),
)


def atlas(case):
    names = ["x", "y", "z"]
    def label(atom):
        return parse_expression(atom, names, as_row=True)
    target, gens = label(case.equation), [label(e) for e in case.generators]
    if case.kind == "constant_crossing":
        fields = dict(names=names, identities=[0]*3, theta=-1, ambient=[], fibre=gens, region=[],
                      components=[["E", "1", "1", [label("x")]], ["X", "1", "1", [target]]],
                      coordinates=[], removed=[], check=False, crossings=True, own=False)
        return lambda: not kernel_json("atlas_canonical_blowup", "failures", dimension=3, packed_present=True,
                                       charts=[fields])[0]["loci"]
    if case.kind in ("ideal_subset", "radical_subset"):
        both = [target], gens
    elif case.kind == "frame_empty":
        both = gens, [target]
    else:
        both = gens, []
    return lambda: kernel_json("atlas_centre_maximum", "replay_inner", kind=case.kind, names=names,
                               identities=[0]*3, equations=both[0], extra=both[1], packed_present=True)


def singular(case, batch):
    upper = case.kind in ("frame_empty", "radical_subset")
    head = "0,(x,y,z,t),dp" if upper else "0,(x,y,z),dp"
    members = ",".join(case.generators)
    if upper:
        test = "reduce(1,std(I+ideal(t*f-1)))==0"
    elif case.kind == "constant_crossing":
        # The boundary divisor is x=0. The strict transform crosses each
        # boundary stratum iff every point avoids the frame cut below.
        test = "reduce(1,std(I+ideal(f,x*diff(f,x),diff(f,y),diff(f,z))))==0"
    else:
        test = "reduce(f,std(I))==0"
    full = f"""
system("--ticks-per-sec",1000000);
ring R={head};
ideal I={members};
poly f={case.equation};
int answer=0;
int j; int start=rtimer;
for (j=0;j<{batch};j++) {{ answer=answer+({test}); }}
print("exact "+string(answer)+" "+string(rtimer-start));
quit;
"""
    run = subprocess.run(["Singular", "-q", "--no-rc"], input=full, text=True, capture_output=True,
                         check=True, timeout=120)
    row = [s for s in run.stdout.splitlines() if s.startswith("exact ")]
    if len(row) != 1 or int(row[0].split()[1]) != batch * case.expected or "   ?" in run.stdout:
        raise ValueError(run.stdout + run.stderr)
    return int(row[0].split()[2]) / batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=int, default=5)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--case", choices=[c.name for c in CASES])
    parser.add_argument("--samples", type=Path, help="Save all sample timings and benchmark conditions.")
    args = parser.parse_args()
    if args.tests < 1 or args.batch < 1:
        parser.error("Use positive trial and batch counts.")
    report = dict(cpu=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  trials=args.tests, repeats=args.batch, linear_algebra=os.getenv("ATLAS_LINEAR_ALGEBRA", "1"),
                  cases={})
    print("test,Atlas_us,Singular_us")
    for case in CASES:
        if args.case is not None and args.case != case.name:
            continue
        solve = atlas(case)
        def timed_atlas():
            start = perf_counter_ns()
            for _ in range(args.batch):
                if solve() != case.expected:
                    raise ValueError(f"Atlas calculated a different result in {case.name}.")
            return (perf_counter_ns() - start) / (args.batch * 1000)
        result = [[], []]
        for k in range(args.tests + 1):
            for who in ((0, 1) if k % 2 == 0 else (1, 0)):
                elapsed = timed_atlas() if who == 0 else singular(case, args.batch)
                if k:
                    result[who].append(elapsed)
        report["cases"][case.name] = dict(atlas_us=result[0], singular_us=result[1])
        print(f"{case.name},{median(result[0]):.3f},{median(result[1]):.3f}", flush=True)

    if args.samples:
        args.samples.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
