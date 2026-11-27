"""Check each certified strict singularity against Singular's exact Jacobian ideal on the toric origin fibre."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.kernel_json import kernel_json
from engine.newton_fan import bounded_fan, build_fan, trial_supports
from engine.point_geometry import encode_terms
from engine.polynomial import parse


CASES = {
    "cone": "(x^2+y^2)^2-z^2*(x^2-y^2)",
    "quartic": "(x*y)^2+(y*z)^2+(z*x)^2+2*x^2*y*z",
    "sextic": "(x*y)^2+(y*z)^2+(z*x)^3+2*x*y*z*(x+y+z)",
    "smooth_tangency": "x^2+z^3",
}

REPAIR_CASES = {
    "repeated_linear": ("(x+y+z)^3+z^5", True),
    "repeated_conic": ("(x^2+y^2+z^2)^2+x*y*z^3", True),
    "isolated_nodes": ("x^3+y^3+z^3-3*x*y*z", False),
    "monomial_factor": ("z^2*(x^2+y^2+z^2)", False),
}


def measure_repair_orbits(args, report):
    report["conditions"] = ("Warm QQ, actual interior ray (1,1,1), 2D orbit; Atlas includes JSON and "
                            "face extraction. Singular includes face extraction and dimension of the "
                            "Laurent Jacobian ideal on the original 3-torus. Alternating order.")
    for name, (equation, expected) in REPAIR_CASES.items():
        terms = encode_terms(parse(equation, list("xyz")))
        code = f'''system("--ticks-per-sec",1000000);ring R=0,(x,y,z,t),dp;poly f={equation};
proc defect(poly g){{
 int e=0;poly q=jet(g,e);while(q==0){{e++;q=jet(g,e);}}
 ideal I=q,diff(q,x),diff(q,y),diff(q,z),t*x*y*z-1;
 return(dim(std(I))>=2);
}}
int j;int result;int start=rtimer;for(j=0;j<{args.batch};j++){{result=defect(f);}}
print("CHECK "+string(result)+" "+string(rtimer-start));quit;
'''
        samples = dict(atlas=[], singular=[])
        for i in range(args.trials + 1):
            for system in (samples if i % 2 == 0 else reversed(samples)):
                if system == "atlas":
                    start = perf_counter_ns()
                    for _ in range(args.batch):
                        result = kernel_json("atlas_canonical_obstruction", "nonisolated_repair_orbit",
                                             terms=terms, rays=[[1, 1, 1]])
                    elapsed = (perf_counter_ns()-start)/(1000*args.batch)
                else:
                    run = subprocess.run(["Singular", "-q", "--no-rc"], input=code, text=True, capture_output=True,
                                         check=True, timeout=120)
                    answer = [l.split() for l in run.stdout.splitlines() if l.startswith("CHECK ")]
                    if len(answer) != 1 or "   ?" in run.stdout:
                        raise ValueError(run.stdout + run.stderr)
                    result, elapsed = bool(int(answer[0][1])), int(answer[0][2])/args.batch
                if result != expected:
                    raise ValueError(f"{system}: incorrect repair orbit on {name}")
                if i:
                    samples[system].append(elapsed)
        report["cases"][name] = dict(equation=equation, expected=expected, samples_us=samples,
                                     median_us={k: median(v) for k, v in samples.items()})
        print(name, report["cases"][name]["median_us"], flush=True)


def charts(name, polynomial):
    if name == "smooth_tangency":
        return [[[1, 0, 0], [0, 1, 0], [1, 1, 1]]]
    fans = [build_fan(polynomial, exhaustive=True)]
    for support in kernel_json("atlas_canonical_obstruction", "relative_supports", supports=trial_supports(polynomial)):
        fans.append(bounded_fan(support))
    return [[fan["rays"][i] for i in cone] for fan in fans for cone in fan["cones"]]


def singular(equation, charts, batch):
    # Compute the monomial pullback and remove every common coordinate
    # factor in Singular itself. The three image coordinates generate
    # the scheme-theoretic fibre ideal, regardless of ray ordering.
    code = [f'system("--ticks-per-sec",1000000);ring R=0,(x,y,z),dp;poly f={equation};',
            'proc defect(poly q,ideal fibre){int j;for(j=1;j<=3;j++){'
            'while(reduce(q,std(ideal(var(j))))==0){q=q/var(j);}}'
            'return(reduce(1,std(fibre+ideal(q,diff(q,x),diff(q,y),diff(q,z))))!=0);}',
            'int j;int total;int result;int start;']
    for index, rays in enumerate(charts):
        images = ["*".join(f"{v}^{ray[col]}" for v, ray in zip("xyz", rays) if ray[col]) for col in range(3)]
        code += [f'map p{index}=R,{",".join(images)};ideal f{index}={",".join(images)};',
                 f'start=rtimer;for(j=0;j<{batch};j++){{result=defect(p{index}(f),f{index});}}'
                 f'total=total+rtimer-start;print("CHART "+string(result));']
    code += ['print("TIME "+string(total));quit;']
    run = subprocess.run(["Singular", "-q", "--no-rc"], input="\n".join(code), text=True,
                         capture_output=True, check=True, timeout=240)
    rows = run.stdout.splitlines()
    answers = [line.split()[1] == "1" for line in rows if line.startswith("CHART ")]
    times = [int(line.split()[1]) for line in rows if line.startswith("TIME ")]
    if len(answers) != len(charts) or len(times) != 1 or any(line.lstrip().startswith("?") for line in rows):
        raise ValueError(run.stdout + run.stderr)
    return answers, times[0] / (batch * len(charts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repair-orbits", action="store_true", help="Compare the positive-dimensional Newton repair orbit.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.trials, args.batch) < 1:
        parser.error("Supply positive trial and batch sizes.")
    os.sched_setaffinity(0, {args.cpu})
    report = dict(cpu=args.cpu, batch=args.batch, trials=args.trials,
                  conditions="Warm QQ; Atlas JSON including pullback vs Singular rtimer including pullback, "
                             "coordinate-factor removal and Jacobian plus origin fibre ideal. "
                             "Every certified chart cross-checked, same rays, alternated order, microseconds per chart.",
                  singular=subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0], cases={})
    if args.repair_orbits:
        measure_repair_orbits(args, report)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        return
    for name, equation in CASES.items():
        p = parse(equation, list("xyz"))
        terms, cones = encode_terms(p), charts(name, p)
        samples = dict(atlas=[], singular=[])
        def atlas():
            start = perf_counter_ns()
            for _ in range(args.batch):
                answers = [kernel_json("atlas_canonical_obstruction", "unresolved_chart", terms=terms,
                                       rays=rays, cones=[[0, 1, 2]], intrinsic=True) for rays in cones]
            return answers, (perf_counter_ns()-start)/(1000*args.batch*len(cones))
        for i in range(args.trials + 1):
            results = {}
            for system in (samples if i % 2 == 0 else reversed(samples)):
                answers, elapsed = atlas() if system == "atlas" else singular(equation, cones, args.batch)
                results[system] = answers
                if i:
                    samples[system].append(elapsed)
            if any(a and not b for a, b in zip(results["atlas"], results["singular"], strict=True)):
                raise ValueError(f"{name}: a certified obstruction occurred in a smooth fibre chart.")
            if name == "smooth_tangency" and any(results["singular"]):
                raise ValueError("The tangent strict chart should be smooth.")
        entry = dict(equation=equation, charts=len(cones), certified=sum(results["atlas"]),
                     singular_charts=sum(results["singular"]), samples_us=samples,
                     median_us={k: median(v) for k, v in samples.items()})
        print(name, entry["charts"], entry["certified"], entry["singular_charts"], entry["median_us"], flush=True)
        report["cases"][name] = entry
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
