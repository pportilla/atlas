"""Compare interactive monomial searches over the installed Zoo with Singular.

Run python3 benchmarks/zoo_search.py --copies 100 --trials 5 --output results.json.
Inputs and the catalogue index are resident before timing. Each copy of a surface
equation gets a distinct row ID so the comparison can check every matching row.
"""
import argparse
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.zoo_search import _Catalogue

QUERIES = ("x*y^2", "1/2*x*y^3 - x^4", "x^2 + y^3 + z^5")


def singular_cases(entries, copies, trials):
    # All query terms have distinct powers. |f-q| = |f|-|q| exactly when
    # every term of q occurs in f with its coefficient. Test the opposite sign too.
    lines = ["ring R=0,(x,y,z),dp;", "ideal surfaces;", "int i,k,a,t,j,elapsed,matched;",
             "intvec hits; poly f,q;",
             "proc matches(poly f, ideal patterns) { int k,goal; poly q; "
             "for(k=1;k<=size(patterns);k++){ q=patterns[k]; goal=size(f)-size(q); "
             "if(size(f-q)==goal || size(f+q)==goal){return(1);} } return(0); }"]
    for i, e in enumerate(entries, 1):
        lines.append(f'surfaces[{i}]={e["equation"]};')
    count = len(entries) * copies
    # Load and parse outside the measured loops.
    lines.append(f"ideal rows; for(i=1;i<={count};i++) {{rows[i]=surfaces[1+(i-1)%{len(entries)}];}}")
    for i, query in enumerate(QUERIES):
        patterns = dict.fromkeys(query.translate(str.maketrans("xyz", "".join(p))) for p in permutations("xyz"))
        lines.append(f'ideal patterns{i}={",".join(patterns)};')
        lines.append(f"for(t=0;t<={trials};t++) {{ a=rtimer; "
                     f"for(i=1;i<={count};i++) {{ hits[i]=matches(rows[i],patterns{i}); }} "
                     'elapsed=rtimer-a; if(t>0){print("time "+string(elapsed));} }')
        lines.append('print("hits "+string(hits));')
    lines.append("quit;")
    command = ["Singular", "-q", "--no-rc", "--threads=1", "--ticks-per-sec=1000000"]
    proc = subprocess.run(command, input="\n".join(lines), capture_output=True, text=True, check=True)
    if proc.stderr:
        raise RuntimeError(proc.stderr)
    results = []
    timings = []
    for line in proc.stdout.splitlines():
        if line.startswith("time "):
            timings.append(int(line[5:]) / 1000)
        elif line.startswith("hits ") and len(timings) == trials:
            flags = list(map(int, line[5:].split(",")))
            if len(flags) != count:
                raise RuntimeError("Singular returned the wrong number of rows")
            results.append((timings, [str(i) for i, v in enumerate(flags) if v]))
            timings = []
        else:
            raise RuntimeError(line)
    if len(results) != len(QUERIES):
        raise RuntimeError("Incomplete Singular comparison")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.copies < 1 or args.trials < 2:
        parser.error("Use at least one copy and two trials.")
    index_path = ROOT / "web/zoo-data/index.json"
    entries = [e for e in json.loads(index_path.read_bytes())["entries"] if e["dimension"] == 2]
    if not all(e["variables"] == ["x", "y", "z"] for e in entries):
        raise ValueError("This comparison expects surface equations in x, y, z.")
    rows = [dict(e, id=str(i)) for i, e in enumerate(entries * args.copies)]
    start = perf_counter_ns()
    catalogue = _Catalogue(rows)
    load_ms = (perf_counter_ns() - start) / 1e6
    singular = singular_cases(entries, args.copies, args.trials)
    results = {}
    for query, (singular_ms, expected) in zip(QUERIES, singular, strict=True):
        atlas_ms = []
        for trial in range(args.trials + 1):
            start = perf_counter_ns()
            actual = catalogue.search(query)["ids"]
            end = perf_counter_ns()
            if trial:
                atlas_ms.append((end - start) / 1e6)
            if actual != expected:
                raise AssertionError(f"Different rows for {query}")
        results[query] = {"matches": len(expected), "atlas_ms": atlas_ms, "singular_ms": singular_ms,
                          "atlas_median_ms": median(atlas_ms), "singular_median_ms": median(singular_ms)}
    cpuinfo = Path("/proc/cpuinfo")
    cpu_lines = cpuinfo.read_text().splitlines() if cpuinfo.exists() else ()
    report = {"conditions": "One warmup per query, same rational surface equations in both systems, in-memory "
                            "catalogues. Atlas timings include ctypes transport, parsing and the returned list of IDs. "
                            "Singular times exclude parsing the equations, preparation of all six variable "
                            "permutations, and printing the result bit vector. Its loop checks the term counts after "
                            "adding and subtracting each remapped query. Ordered row IDs are compared for equality.",
              "cpu": next((line.partition(":")[2].strip() for line in cpu_lines
                           if line.startswith("model name")), platform.machine()),
              "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "corpus_sha256": sha256(index_path.read_bytes()).hexdigest(), "surface_equations": len(entries),
              "copies": args.copies, "rows": len(rows), "trials": args.trials, "atlas_load_ms": load_ms,
              "queries": results}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
