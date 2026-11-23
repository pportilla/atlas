"""Compare catalogue ordering and interactive monomial searches with Singular.

Run python3 benchmarks/zoo_search.py --copies 20 --trials 5 --output results.json.
Each copy gets a distinct row ID. The first comparison times construction of
the ordered catalogue. Interactive queries then reuse that index.
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
from engine.zoo_data import catalogue_file
from engine.zoo_search import _Catalogue

QUERIES = ("x*y^2", "1/2*x*y^3 - x^4", "x^2 + y^3 + z^5")

# Singular compares polynomials by their leading terms. Compare the remaining
# terms too, using lp for monomials and QQ for exact signed coefficients.
SINGULAR_ORDER = """
proc rowLess(int i, int j) {
    poly f = rows[i];
    poly g = rows[j];
    while (f != 0 && g != 0) {
        if (leadmonom(f) != leadmonom(g)) { return(leadmonom(f) < leadmonom(g)); }
        if (leadcoef(f) != leadcoef(g)) { return(leadcoef(f) < leadcoef(g)); }
        f = f - lead(f);
        g = g - lead(g);
    }
    if (f != 0 || g != 0) { return(f == 0); }
    return(string(i-1) < string(j-1));
}
proc orderedRows(int n) {
    intvec order = 1..n;
    intvec merged = order;
    int step, start, mid, stop, left, right, out, fromLeft;
    for (step=1; step<n; step=step*2) {
        for (start=1; start<=n; start=start+2*step) {
            mid = start+step;
            stop = mid+step;
            if (stop>n+1) { stop=n+1; }
            left=start; right=mid;
            for (out=start; out<stop; out++) {
                fromLeft=(right>=stop);
                if (!fromLeft && left<mid) { fromLeft=rowLess(order[left],order[right]); }
                if (fromLeft) { merged[out]=order[left]; left++; }
                else { merged[out]=order[right]; right++; }
            }
        }
        order=merged;
    }
    return(order);
}
"""


def singular_cases(entries, copies, trials):
    # All query terms have distinct powers. |f-q| = |f|-|q| exactly when
    # every term of q occurs in f with its coefficient. Test the opposite sign too.
    lines = ["ring R=0,(x,y,z),lp;", "ideal surfaces;", "int i,k,a,t,j,elapsed,matched;",
             "intvec hits; poly f,q;",
             "proc matches(poly f, ideal patterns) { int k,goal; poly q; "
             "for(k=1;k<=size(patterns);k++){ q=patterns[k]; goal=size(f)-size(q); "
             "if(size(f-q)==goal || size(f+q)==goal){return(1);} } return(0); }"]
    for i, e in enumerate(entries, 1):
        lines.append(f'surfaces[{i}]={e["equation"]};')
    count = len(entries) * copies
    # Load and parse outside the measured loops.
    lines.append(f"ideal rows; for(i=1;i<={count};i++) {{rows[i]=surfaces[1+(i-1)%{len(entries)}];}}")
    lines.extend([SINGULAR_ORDER, "intvec order;",
                  f"for(t=0;t<={trials};t++) {{ a=rtimer; order=orderedRows({count}); "
                  'elapsed=rtimer-a; if(t>0){print("time "+string(elapsed));} }',
                  'print("order "+string(order));'])
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
    ordering = None
    results = []
    timings = []
    for line in proc.stdout.splitlines():
        if line.startswith("time "):
            timings.append(int(line[5:]) / 1000)
        elif line.startswith("order ") and len(timings) == trials:
            ids = [str(int(index)-1) for index in line[6:].split(",")]
            if len(ids) != count or len(set(ids)) != count:
                raise RuntimeError("Singular returned an incomplete catalogue ordering")
            ordering = timings, ids
            timings = []
        elif line.startswith("hits ") and len(timings) == trials:
            flags = list(map(int, line[5:].split(",")))
            if len(flags) != count:
                raise RuntimeError("Singular returned the wrong number of rows")
            results.append((timings, [str(i) for i, v in enumerate(flags) if v]))
            timings = []
        else:
            raise RuntimeError(line)
    if ordering is None or len(results) != len(QUERIES):
        raise RuntimeError("Incomplete Singular comparison")
    return ordering, results


def comparison(atlas_ms, singular_ms):
    return {"atlas_ms": atlas_ms, "singular_ms": singular_ms,
            "atlas_median_ms": median(atlas_ms), "singular_median_ms": median(singular_ms)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=20)
    parser.add_argument("--limit-surfaces", type=int, help="Use the first N surfaces in the index")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.copies < 1 or args.trials < 2 or (args.limit_surfaces is not None and args.limit_surfaces < 1):
        parser.error("Use at least one surface, one copy and two trials.")
    index_path = catalogue_file("index.json")
    entries = [e for e in json.loads(index_path.read_bytes())["entries"] if e["dimension"] == 2]
    entries = entries[:args.limit_surfaces]
    if not all(e["variables"] == ["x", "y", "z"] for e in entries):
        raise ValueError("This comparison expects surface equations in x, y, z.")
    rows = [dict(e, id=str(i)) for i, e in enumerate(entries * args.copies)]
    (singular_order_ms, expected_order), singular = singular_cases(entries, args.copies, args.trials)
    atlas_order_ms = []
    for trial in range(args.trials + 1):
        start = perf_counter_ns()
        catalogue = _Catalogue(rows)
        actual = catalogue.search("")["ids"]
        elapsed = (perf_counter_ns() - start) / 1e6
        if actual != expected_order:
            raise AssertionError("Atlas and Singular returned different catalogue orderings")
        if trial:
            atlas_order_ms.append(elapsed)
        if trial < args.trials:
            del catalogue
    results = {}
    for query, (singular_ms, expected) in zip(QUERIES, singular, strict=True):
        matching = set(expected)
        expected = [ident for ident in expected_order if ident in matching]
        atlas_ms = []
        for trial in range(args.trials + 1):
            start = perf_counter_ns()
            actual = catalogue.search(query)["ids"]
            end = perf_counter_ns()
            if trial:
                atlas_ms.append((end - start) / 1e6)
            if actual != expected:
                raise AssertionError(f"Different rows for {query}")
        results[query] = {"matches": len(expected), **comparison(atlas_ms, singular_ms)}
    cpuinfo = Path("/proc/cpuinfo")
    cpu_lines = cpuinfo.read_text().splitlines() if cpuinfo.exists() else ()
    report = {"conditions": "One warmup per operation, same rational surface equations and string row IDs in both "
                            "systems. Catalogue ordering in Atlas includes encoding, native parsing, sorting, building "
                            "the search index and retrieving all IDs. Singular merge-sorts the parsed polynomials "
                            "using successive lp leading monomials, exact QQ coefficients and the string row ID. "
                            "Searches use the resident catalogues. Atlas query timings include ctypes transport, "
                            "query parsing and the returned list of IDs. "
                            "Singular times exclude parsing the equations, preparation of all six variable "
                            "permutations, and printing the result bit vector. Its loop checks the term counts after "
                            "adding and subtracting each remapped query. Every sorted ID is compared between systems. "
                            "Each query's matching IDs are then checked in that independently verified order.",
              "cpu": next((line.partition(":")[2].strip() for line in cpu_lines
                           if line.startswith("model name")), platform.machine()),
              "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
              "corpus_sha256": sha256(index_path.read_bytes()).hexdigest(), "surface_equations": len(entries),
              "copies": args.copies, "rows": len(rows), "trials": args.trials,
              "ordering": comparison(atlas_order_ms, singular_order_ms),
              "queries": results}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
