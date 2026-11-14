"""Compare the native open-locus decision with Singular on identical rational ideals."""
import argparse
import gzip
import hashlib
from itertools import product
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
from engine.native_library import ideal_request
from engine.polynomial import format_poly


def cases():
    x, y = variables = np.symbols("x y")
    roots = (x**2 + 1)**4 * (x-2)**3
    examples = [
        ("irrational-roots", [roots, y], [[x-2]], True),
        ("union-of-holes", [roots, y], [[x-2], [x**2+1]], False),
        ("multi-generator-hole", [roots, y], [[x-2, x**2+1]], True),
        ("whole-axis-minus-points", [y], [[x], [x-1]], True),
        ("off-axis", [x*y-1, x-17], [], True),
        ("degree-cap", [x**257-1, x*y-1], [], True),
        ("large-coefficient", [(2**131+7)*x-1, y], [[x]], True),
    ]
    for name, rows, removed, answer in examples:
        yield name, variables, rows, removed, answer
    with gzip.open(ROOT/"tests/fixtures/b8-strict-open-locus.json.gz", "rt") as file:
        captured = json.load(file)
    variables = tuple(map(np.Symbol, captured["variables"]))
    yield "B8", variables, [np.from_row(variables, r) for r in captured["rows"]], [], True


def singular_program(variables, generators, removed, trials, batch, use_b8_certificate):
    # For each choice of a generator from every removed ideal, adjoin
    # inv*product-1. Their union is exactly the requested open set.
    names = [f"a{i+1}" for i in range(len(variables))]
    def polynomial(f):
        return format_poly(np.terms(variables, f), names)
    equations = ",".join(map(polynomial, generators)) or "0"
    ring = f'ring R=0,({",".join(names)},inv),dp; ideal I={equations};'
    if use_b8_certificate:
        # The selected axis is known from the captured B8 fixture. A nonempty
        # intersection proves existence. This is explicitly a witness timing.
        setup = ring + ' ring L=0,(u),dp; map line=R,0,u,0,0,0,0;'
        work = 'B=std(line(I)); answer=(B[1]!=1);'
    else:
        setup = ring
        work = ''
        for choices in product(*removed):
            lifted = 'I'
            if choices:
                factor = '*'.join(f'({polynomial(f)})' for f in choices)
                lifted += f'+ideal(inv*({factor})-1)'
            work += f'if(!answer){{B=std({lifted});answer=(B[1]!=1);}}'
    return '\n'.join([setup, 'system("--ticks-per-sec",1000000);ideal B;int t,j,k,answer;',
                      f'for(t=0;t<{trials};t++){{k=rtimer;for(j=0;j<{batch};j++){{answer=0;{work}}}',
                      'print("RESULT "+string(answer)+" "+string(rtimer-k));}quit;'])


def singular(variables, rows, removed, trials, batch, certificate, timeout):
    script = singular_program(variables, rows, removed, trials, batch, certificate)
    done = subprocess.run(["Singular", "-q", "--no-rc"], input=script, capture_output=True,
                          text=True, check=True, timeout=timeout)
    output = done.stdout.splitlines()
    if len(output) != trials or any(not line.startswith("RESULT ") for line in output):
        raise ValueError(done.stdout[:1000] + done.stderr[:200])
    return [(bool(int(line.split()[1])), int(line.split()[2])/batch/1000) for line in output]


def atlas(variables, polynomials, removed, trials, batch):
    header = "open_locus " + " ".join(map(str, (len(variables), len(polynomials), len(removed),
                                                 *map(len, removed))))
    rows = [np.row(variables, f) for f in (*polynomials, *(f for h in removed for f in h))]
    samples = []
    for _ in range(trials):
        start = perf_counter_ns()
        for _ in range(batch):
            result = ideal_request(header, rows)
        samples.append((result == ["yes"], (perf_counter_ns()-start)/batch/1e6))
        if result not in (["yes"], ["no"]):
            raise ValueError(result)
    return samples


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--trials", type=int, default=7)
    args.add_argument("--batch", type=int, default=50)
    args.add_argument("--b8-timeout", type=float, default=5.0,
                      help="Seconds allowed for one unrestricted Singular B8 trial.")
    args.add_argument("--output", type=Path, required=True)
    config = args.parse_args()
    if min(config.trials, config.batch, config.b8_timeout) <= 0:
        args.error("Trials, batch size and cutoff must be positive.")
    library = ROOT/"native/build/libgroebner.so"
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")), platform.machine()) if cpuinfo.is_file() else platform.machine()
    report = {"cpu": cpu, "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "trials": config.trials, "batch": config.batch, "b8_timeout_s": config.b8_timeout,
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "conditions": "QQ, dp/grevlex, same polynomials. Atlas times the installed native whole "
                            "open_locus call including sparse-row transfer. Singular runs exact standard-basis "
                            "unit tests with polynomials already loaded, including Rabinowitsch open conditions. "
                            "B8 also has a separate Singular second-axis witness comparison. Each unrestricted "
                            "Singular B8 trial is a fresh process; its timeout includes parsing and startup. "
                            "Other timings exclude process and interpreter startup. All samples are milliseconds per call.",
              "cases": []}
    for name, variables, rows, removed, expected in cases():
        is_b8 = name == "B8"
        batch = min(5, config.batch) if is_b8 else config.batch
        a = atlas(variables, rows, removed, config.trials, batch)
        s = singular(variables, rows, removed, config.trials, batch, is_b8, 60)
        if any(answer != expected for answer, _ in [*a, *s]):
            raise ValueError(f"Nonemptiness results disagree on {name}.")
        record = {"name": name, "answer": expected, "batch": batch, "atlas_ms": [t for _, t in a],
                  "singular_ms": [t for _, t in s], "singular_method": "axis witness" if is_b8 else "global",
                  "medians_ms": {"Atlas": statistics.median(t for _, t in a),
                                 "Singular": statistics.median(t for _, t in s)}}
        if is_b8:
            global_trials = []
            for _ in range(min(3, config.trials)):
                try:
                    (answer, ms), = singular(variables, rows, removed, 1, 1, False, config.b8_timeout)
                    if answer != expected:
                        raise ValueError("Global Singular B8 decision disagrees with the witness.")
                    global_trials.append(ms)
                except subprocess.TimeoutExpired:
                    global_trials.append(None)
            record["unrestricted_singular_ms"] = global_trials
        report["cases"].append(record)
    config.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
