#!/usr/bin/env python3
"""Run only Atlas on two published Singular inputs and a four-variable target.

Reference measurements are transcribed from their sources. No other solver
is invoked. Every sample starts a fresh CLI and includes its JSON output.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "x^5+z*y^3+z^15+t^4*x+t^8"
CASES = {
    "a6": ("x^7+y^2-z^2", "x,y,z", "intrinsic"),
    "binomial": ("x*y^2*z^3-t^6", "x,y,z,t", "embedded"),
    "binomial_intrinsic": ("x*y^2*z^3-t^6", "x,y,z,t", "intrinsic"),
    "target_intrinsic": (EXAMPLE, "x,y,z,t", "intrinsic"),
    "target_embedded": (EXAMPLE, "x,y,z,t", "embedded"),
}
PUBLISHED = {
    "a6": {
        "source": "Singular tutorial, Resolution of singularities",
        "url": "https://www.singular.uni-kl.de/ftp/pub/Math/Singular/doc/tutor.pdf",
        "reported_result": "Six genus-zero exceptional curves of self-intersection -2 in a path.",
    },
    "binomial": {
        "source": "Blanco and Frühbis-Krüger (2011), section 5, Table 1, fourth row",
        "url": "https://arxiv.org/pdf/1109.1655#page=10",
        "hardware": "Intel Core i3-330M at 2.13 GHz, 4 GB memory",
        "reported_result": {"resolve": {"time": ">7200 s", "final_charts": None},
                            "resbinomial": {"time": "9 s", "final_charts": 240}},
    },
}


def cpu_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def geometry(result):
    fields = ("variables", "polynomial", "mode", "charts", "dual_complex", "fan",
              "root_families", "coefficient_field", "algebraic_numbers")
    selected = {k: result[k] for k in fields if k in result}
    return hashlib.sha256(json.dumps(selected, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_a6(result):
    dual = result["dual_complex"]
    vertices = dual["vertices"]
    edges = dual["cells"]
    adj = {v["id"]: set() for v in vertices}
    for e in edges:
        a, b = e["vertices"]
        adj[a].add(b)
        adj[b].add(a)
    if not adj:
        raise RuntimeError("Missing A6 graph.")
    stack, seen = [next(iter(adj))], set()
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(adj[v]-seen)
    if not (len(vertices) == len(seen) == 6 and len(edges) == 5 and
            sorted(map(len, adj.values())) == [1,1,2,2,2,2] and
            all(v.get("genus") == 0 and v.get("self_intersection") == -2 for v in vertices)):
        raise RuntimeError("Atlas's A6 graph differs from the published intersection matrix.")


def sample(case, policy):
    eq, names, mode = CASES[case]
    command = [str(ROOT/"atlas"), "resolve", eq, "--vars", names, "--mode", mode,
               "--compute", policy, "--quick", "--json"]
    start = perf_counter_ns()
    proc = subprocess.run(command, capture_output=True, timeout=90)
    elapsed = (perf_counter_ns()-start)/1e6
    r = json.loads(proc.stdout)
    if proc.returncode not in (0,3) or r.get("status") not in ("resolved","inconclusive"):
        raise RuntimeError(f"{case}: {r.get('error', r.get('status', proc.stderr))}")
    if r["status"] != "resolved":
        raise RuntimeError(f"{case} yielded no verified resolution.")
    if case == "a6":
        validate_a6(r)
    batch = r.get("certificate", {}).get("chart_batch", {})
    if policy == "gpu" and batch.get("backend") != "opencl":
        raise RuntimeError(f"{case}: requested GPU was not used; no GPU timing recorded.")
    dual = r["dual_complex"]
    digest = geometry(r) if r["status"] == "resolved" else None
    counts = Counter(len(c["vertices"])-1 for c in dual["cells"])
    record = {"input": eq, "variables": names.split(","), "mode": mode,
              "search": "quick", "compute": policy, "status": r["status"],
              "engine": r["engine"], "dual_status": dual["status"], "charts": len(r["charts"]),
              "vertices": len(dual["vertices"]), "cells_by_dimension": dict(sorted(counts.items())),
              "geometry_sha256": digest}
    measurement = {"fresh_cli_ms": round(elapsed,3), "solver_ms": r["elapsed_ms"],
                   "json_bytes": len(proc.stdout)}
    if batch:
        record["chart_batch"] = {k: batch[k] for k in ("backend","dimension","chart_term_pairs") if k in batch}
        if batch.get("device"): record["device"] = batch["device"]
        if "native_elapsed_ms" in batch: measurement["native_batch_ms"] = round(batch["native_elapsed_ms"],3)
    return record, measurement


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--gpu-target", action="store_true",
                    help="Also time actual OpenCL chart transforms on the four-variable case.")
    ap.add_argument("--output", type=Path)
    args=ap.parse_args()
    if not 1 <= args.repeats <= 50:
        ap.error("--repeats must be between 1 and 50")
    data = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": {"cpu": cpu_name(), "os": platform.platform(), "python": platform.python_version()},
        "method": "One untimed warm-up per case/backend, then fresh CLI + worker for every measured "
                  "run including complete JSON output. No cache flush. Cases interleaved, alternating order. "
                  "Local germ at the origin. Published timings were obtained separately on other hardware.",
        "repeats": args.repeats, "published_references": PUBLISHED, "atlas": {},
    }
    jobs=[(c,"cpu") for c in CASES]
    if args.gpu_target: jobs.extend((c,"gpu") for c in CASES if c.startswith("target_"))
    reference = {}
    for c,policy in jobs:
        record,_ = sample(c,policy)
        key=f"{c}/{policy}"
        if c in reference and reference[c] != record["geometry_sha256"]:
            raise RuntimeError(f"{key}: CPU and GPU geometries differ.")
        reference[c] = record["geometry_sha256"]
        data["atlas"][key] = record | {"samples": []}
    for i in range(args.repeats):
        for c,policy in jobs if i%2 == 0 else reversed(jobs):
            record,measurement = sample(c,policy)
            prior = data["atlas"][f"{c}/{policy}"]
            if record["status"] != prior["status"] or record["geometry_sha256"] != prior["geometry_sha256"]:
                raise RuntimeError(f"{c}/{policy}: outcomes differed between repetitions.")
            prior["samples"].append(measurement)
    for key,item in data["atlas"].items():
        item["median_fresh_cli_ms"]=round(statistics.median(s["fresh_cli_ms"] for s in item["samples"]),3)
        item["median_solver_ms"]=round(statistics.median(s["solver_ms"] for s in item["samples"]),3)
        print(f"{key}: {item['status']}, median fresh CLI {item['median_fresh_cli_ms']:.1f} ms")
    if args.output:
        args.output.write_text(json.dumps(data,indent=2)+"\n")


if __name__ == "__main__":
    main()
