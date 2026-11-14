"""Compare the generic chart-overlap decision with a Singular elimination oracle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import native_polynomial as np
from engine.kernel_json import kernel_json
from engine.native_library import _LIBRARY
from tests.gluing_examples import crossed_blowup_charts, implicit_cone_charts


# In these two graph presentations, eliminating all but one target
# coordinate gives its linear coordinate relations modulo the ambient
# equation. The cone has two denominators x and z for the same coordinate.
SINGULAR_ORACLE = """
proc coordinateDenominators(ideal G, int first, int last, int i) {
    poly keep=var(i);
    poly others=1;
    poly q;
    int k;
    for(k=first;k<=last;k++){ if(k!=i){others=others*var(k);} }
    ideal rel=eliminate(G,others);
    ideal dens=0;
    for(k=1;k<=size(rel);k++){
        q=diff(rel[k],keep);
        if(q!=0 && q*keep+subst(rel[k],keep,0)==rel[k]){dens=dens+ideal(q);}
    }
    return(dens);
}
proc regularDenominator(ideal G, ideal P, int first, int last) {
    ideal B=std(P);
    poly product=1;
    int i,k;
    poly keep,others,q,usable;
    ideal rel;
    for(i=first;i<=last;i++){
        keep=var(i); usable=0;
        if(size(D)){
            for(k=1;k<=size(D[i]);k++){
                q=D[i][k];
                if(reduce(q,B)!=0){usable=q;break;}
            }
        } else {
            others=1;
            for(k=first;k<=last;k++){ if(k!=i){others=others*var(k);} }
            rel=eliminate(G,others);
            for(k=1;k<=size(rel);k++){
                q=diff(rel[k],keep);
                if(q!=0 && q*keep+subst(rel[k],keep,0)==rel[k] && reduce(q,B)!=0){
                    usable=q;break;
                }
            }
        }
        if(usable==0){return(poly(0));}
        product=product*usable;
    }
    return(product);
}
proc matchPrimes(ideal G, ideal P, ideal Q, int leftWidth, int totalWidth) {
    poly u=regularDenominator(G,P,leftWidth+1,totalWidth);
    if(u==0){return(0);}
    poly v=regularDenominator(G,Q,1,leftWidth);
    if(v==0){return(0);}
    ideal H=u*v;
    ideal L=std(sat(G+P,H));
    ideal T=std(sat(G+Q,H));
    if(reduce(1,L)!=0 && size(reduce(L,T))==0 && size(reduce(T,L))==0){return(1);}
    return(-1);
}
"""


def singular_rows(polys):
    return ",".join(str(f).replace("**", "^") for f in polys)


def cases():
    for example in (crossed_blowup_charts, implicit_cone_charts):
        blocks, _, _, graph, pairs = example(np)
        for name, left, right, expected in pairs:
            for i, j in ((0, 1), (1, 0)):
                variables = (*blocks[i], *blocks[j])
                p, q = (left, right)[i], (left, right)[j]
                operands = {
                    "left_width": len(blocks[i]), "right_width": len(blocks[j]), "removed": [],
                    "graph": [np.row(variables, g) for g in graph],
                    "left_prime": [np.row(blocks[i], g) for g in p],
                    "right_prime": [np.row(blocks[j], g) for g in q],
                }
                setup = (f'ring R=0,({singular_rows(variables)}),dp;\n'
                         f'ideal G={singular_rows(graph)}; ideal P={singular_rows(p)}; ideal Q={singular_rows(q)};\n'
                         f'int leftWidth={len(blocks[i])}; int totalWidth={len(variables)};\n')
                yield f"{name}-{'forward' if i == 0 else 'reverse'}", operands, setup, expected


def time_singular(setup, batch, reuse):
    preparation = ('for(i=1;i<=leftWidth;i++){D[i]=coordinateDenominators(G,1,leftWidth,i);}\n'
                   'for(i=leftWidth+1;i<=totalWidth;i++){D[i]=coordinateDenominators(G,leftWidth+1,totalWidth,i);}\n'
                   if reuse else '')
    program = ('LIB "elim.lib";\n' + setup + SINGULAR_ORACLE
               + 'int i; list D;\n' + preparation
               + 'int result=matchPrimes(G,P,Q,leftWidth,totalWidth); int start=rtimer;\n'
               + f'for(i=0;i<{batch};i++){{result=matchPrimes(G,P,Q,leftWidth,totalWidth);}}\n'
               + 'int elapsed=rtimer-start; print("RESULT "+string(result)+" "+string(elapsed));\nquit;\n')
    command = ["Singular", "-q", "--no-rc", "--random=47", "--threads=1",
               "--flint-threads=1", "--ticks-per-sec=1000000"]
    proc = subprocess.run(command, input=program, capture_output=True, text=True, check=True, timeout=30)
    lines = proc.stdout.strip().splitlines()
    if proc.stderr or len(lines) != 1 or not lines[0].startswith("RESULT "):
        raise RuntimeError(proc.stderr + proc.stdout)
    _, value, elapsed = lines[0].split()
    return int(value), int(elapsed) / batch


def time_atlas(operands, batch, reuse):
    preparation = {key: operands[key] for key in ("left_width", "right_width", "graph")}
    decision = {key: value for key, value in operands.items() if key != "graph"}

    def prepare():
        return kernel_json("atlas_component_geometry", "prepare_overlap", **preparation)

    maps = prepare() if reuse else None

    def run():
        return kernel_json("atlas_component_geometry", "generic_overlap", **decision,
                           maps=maps if reuse else prepare())

    result = run()
    start = perf_counter_ns()
    for _ in range(batch):
        result = run()
    return result, (perf_counter_ns() - start) / (1000 * batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--cpu", type=int, help="pin both processes to this logical CPU on Linux")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 2 or args.batch < 1:
        parser.error("Choose at least two trials and a positive batch size.")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    cpu = platform.machine()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    result = {
        "cpu": cpu, "logical_cpu": args.cpu, "trials": args.trials, "batch": args.batch,
        "atlas_library_sha256": hashlib.sha256(_LIBRARY.read_bytes()).hexdigest(),
        "Singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
        "conditions": (
            "Both systems take the same closed diagonal over QQ and two chart-local prime ideals. "
            "Each decides if the ambient chart maps are regular in both directions at the generic "
            "points and whether the primes agree on that open. Singular derives coordinate fractions "
            "by elimination and compares graph ideals saturated by the regular denominators. "
            "Inputs are prebuilt, graph construction and process startup excluded. Atlas includes "
            "JSON/FFI transport and sparse-row decoding. Singular operates on parsed ideals. In "
            "fresh_map each system has only the graph; Singular can exit before deriving all the maps. "
            "In reused_map both first compute all coordinate denominators outside the timer. Atlas "
            "uses the same prepare_overlap and generic_overlap operations as resolution. "
            "Each sample discards one warmup, then times a batch of independent decisions, using "
            "perf_counter_ns for Atlas and microsecond rtimer for Singular. Execution order alternates."),
        "decisions": {"0": "no mutual regular open", "1": "same prime", "-1": "different on regular open"},
        "cases": {},
    }
    for name, operands, setup, expected in cases():
        result["cases"][name] = {"decision": expected}
        for reuse in (False, True):
            samples = {"Atlas": [], "Singular": []}
            for trial in range(args.trials):
                for engine in (samples if trial % 2 == 0 else reversed(samples)):
                    got, elapsed = (time_atlas(operands, args.batch, reuse) if engine == "Atlas"
                                    else time_singular(setup, args.batch, reuse))
                    if got != expected:
                        raise AssertionError((name, engine, got, expected))
                    samples[engine].append(round(elapsed, 3))
            result["cases"][name]["reused_map" if reuse else "fresh_map"] = {
                "median_us": {engine: statistics.median(times) for engine, times in samples.items()},
                "atlas_faster_pairs": sum(a < s for a, s in zip(samples["Atlas"], samples["Singular"])),
                "samples_us": samples,
            }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
