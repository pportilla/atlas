"""Time resolve(method="weighted") against resolve(method="auto") on the same inputs.

Each request runs in Atlas's isolated resolution worker, as at the prompt, so
both methods pay the same process costs. The table reports the solver's own
elapsed time (the record's elapsed_ms), median over the trials, and whether
each method returned a certified resolution. Atlas is its own benchmark here:
the weighted engine is compared with the engines it can replace.

    python3 benchmarks/weighted_resolution.py [--trials 3] [--timeout 60] [--json out.json]
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = [
    # name, polynomial, variables
    ("E8", "x^2+y^3+z^5", "x,y,z"),
    ("A20", "x^2+y^2+z^21", "x,y,z"),
    ("T444", "x^4+y^4+z^4+x*y*z", "x,y,z"),
    ("canon_x2y_y4_z3", "x^2*y+y^4+z^3", "x,y,z"),
    ("whitney", "x^2-y^2*z", "x,y,z"),
    ("double_conic", "(x*y-z^2)^2+x^5+y^5", "x,y,z"),
    ("double_conic_norat", "(x^2+y^2+z^2)^2+x^5", "x,y,z"),
    ("double_elliptic", "(x^3+y^3+z^3)^2+x^7", "x,y,z"),
    ("triple_conic", "(x^2+y^2-z^2)^3+x^7", "x,y,z"),
    ("binomial4", "x*y^2*z^3-t^6", "x,y,z,t"),
    ("double_quadric3", "(x*y-z*w)^2+x^5+y^5+z^5+w^5", "x,y,z,w"),
    ("B9", "z^2 - (x^2*(x+2*y)^2 + 4*y^3*(x+y)^2)*(x^2*(x+2*y)^2 + 4*y*(x+y)^2)", "x,y,z"),
]


def run(polynomial, variables, method, timeout):
    from engine.resolve_cli import run_job
    request = {"polynomial": polynomial, "variables": variables.split(","), "mode": "embedded",
               "search": "exhaustive"}
    if method != "auto":
        request["method"] = method
    start = time.perf_counter()
    status, record = run_job(request, timeout=timeout)
    wall = time.perf_counter() - start
    certified = (status == 200 and record.get("status") == "resolved"
                 and (record.get("engine") != "weighted" or record["certificate"]["certified"]))
    if status == 200 and record.get("status") == "inconclusive":
        return {"ok": False, "wall_s": wall, "note": "inconclusive or timed out"}
    if status != 200:
        return {"ok": False, "wall_s": wall, "note": record.get("error", "")[:120]}
    return {"ok": certified, "wall_s": wall, "elapsed_ms": record.get("elapsed_ms"), "engine": record.get("engine")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--only", default="")
    parser.add_argument("--json")
    args = parser.parse_args()
    only = set(filter(None, args.only.split(",")))
    rows = []
    for name, polynomial, variables in CASES:
        if only and name not in only:
            continue
        row = {"case": name, "polynomial": polynomial}
        for method in ("weighted", "auto"):
            runs = []
            for _ in range(args.trials):
                runs.append(run(polynomial, variables, method, args.timeout))
                if not runs[-1]["ok"]:
                    break  # one failure, such as a timeout, settles the case
            done = [r for r in runs if r["ok"]]
            row[method] = {"certified": bool(done) and len(done) == len(runs),
                           "median_ms": statistics.median(r["elapsed_ms"] for r in done) if done else None,
                           "trials_ms": [r.get("elapsed_ms") for r in runs],
                           "engine": done[0]["engine"] if done else None,
                           "note": next((r["note"] for r in runs if not r["ok"]), "")}
        rows.append(row)
        w, a = row["weighted"], row["auto"]
        fmt = lambda m: f"{m['median_ms']:10.1f}" if m["median_ms"] is not None else f"{'—':>10s}"
        print(f"{name:20s} weighted {fmt(w)} ms {'certified' if w['certified'] else w['note'][:40]:12s} "
              f"auto {fmt(a)} ms {('certified (' + str(a['engine']) + ')') if a['certified'] else a['note'][:60]}",
              flush=True)
    conditions = {"date": time.strftime("%Y-%m-%d"), "machine": platform.processor() or platform.machine(),
                  "python": platform.python_version(), "cpus": os.cpu_count(), "trials": args.trials,
                  "timeout_s": args.timeout, "search": "exhaustive", "mode": "embedded"}
    if args.json:
        Path(args.json).write_text(json.dumps({"conditions": conditions, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
