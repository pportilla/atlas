"""Time Jung's method next to `atlas resolve` on the test surfaces and check Laufer's formula.

Run from the repository root: python3 benchmarks/jung_timings.py [--trials N] [--cap SECONDS]
Each Jung run is checked against the Milnor number from Atlas's local algebra:
mu = 12 p_g + K^2 + chi(E) - 1. `atlas resolve` runs are capped (default 40 s).
"""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATLAS = str(ROOT / "atlas")
SURFACES = [
    ("E8", "x^2+y^3+z^5"),
    ("A20", "x*y-z^21"),
    ("x^3+y^3+z^4", "x^3+y^3+z^4"),
    ("cusp T444", "x*y*z+x^4+y^4+z^4"),
    ("D4", "(x+y)^2+z^3+x^3"),
    ("order 4", "(x*y-z^2)^2+x^5+y^5"),
    ("E6", "x^2+(y+z)^3+y^4"),
    ("x^2y+y^4+z^3", "x^2*y+y^4+z^3"),
    ("E6'", "(x+2*y)^2+(y+z)^3+z^4"),
    ("key", "z^2 - (x^2*(x+2*y)^2 + 4*y^3*(x+y)^2)*(x^2*(x+2*y)^2 + 4*y*(x+y)^2)"),
    ("non-isolated", "(x+y+z)^2-z*(x-y)^2"),
]


def timed(args, cap):
    start = time.perf_counter()
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=cap, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None, None
    return time.perf_counter() - start, done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cap", type=float, default=40)
    args = parser.parse_args()
    rows = []
    for label, f in SURFACES:
        mu_text = subprocess.run([ATLAS, "algebra", "milnor-number", "--vars", "x,y,z", "--poly", f],
                                 capture_output=True, text=True, cwd=ROOT).stdout
        mu = next((line.split(":")[1].strip() for line in mu_text.splitlines() if line.startswith("Milnor")), "?")
        jung_wall, jung_native, record = [], [], None
        for _ in range(args.trials):
            seconds, done = timed([ATLAS, "jung", f, "--json"], args.cap)
            if done is None or done.returncode:
                record = {"error": (done.stderr.strip() if done else "timeout")}
                break
            record = json.loads(done.stdout)
            jung_wall.append(seconds)
            jung_native.append(record["elapsed_ms"])
        resolve = []
        for _ in range(min(args.trials, 3)):
            seconds, done = timed([ATLAS, "resolve", f, "--vars", "x,y,z", "--json"], args.cap)
            if done is None:
                resolve.append(None)
                break
            status = json.loads(done.stdout).get("status") if done.returncode == 0 and done.stdout.startswith("{") else "error"
            resolve.append((seconds, status))
            if status != "resolved":
                break
        row = {"label": label, "f": f, "mu": mu}
        if "error" in record:
            row["jung"] = record["error"]
        else:
            inv = record["minimal_good_resolution"]["invariants"]
            row.update(curves=len(record["minimal_good_resolution"]["vertices"]),
                       jung_curves=len(record["graph"]["vertices"]), K2=inv["K2"],
                       chi=inv["euler_characteristic"], pg=record["invariants"]["geometric_genus"],
                       laufer=record["invariants"]["milnor_number_from_laufer"],
                       jung_cli_ms=round(1000 * statistics.median(jung_wall)),
                       jung_native_ms=statistics.median(jung_native), trials=len(jung_wall),
                       global_tests=record["global_monodromy_tests"], cyclic=record["cyclic_cover"])
            row["laufer_ok"] = row["laufer"] == mu
        if resolve and resolve[-1] is None:
            row["resolve"] = f"no result within {args.cap:g} s"
        else:
            times = [s for s, status in resolve if status == "resolved"]
            row["resolve"] = (f"{round(1000 * statistics.median(times))} ms (median of {len(times)})" if times
                              else f"status {resolve[-1][1]}")
        rows.append(row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
