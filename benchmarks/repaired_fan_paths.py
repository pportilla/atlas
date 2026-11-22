"""Compare public fan resolutions before and after repair changes, and replay the saved boundary graphs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from time import monotonic

from zoo_resolution import oracle

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("repaired-cusp", "xyz", "x^3+(z^2-y^3)^2+2*y^5*z"),
    ("two-conics", "xyz", "(x*z-2*y^2)*(y*z-x^2)+(x+3*y+7*z)^5"),
    ("four-variable-chain", "xyzt", "x^2*y+2*y^3*z+3*z^2*t+t^3"),
)


def run(command, timeout, source=""):
    start = monotonic()
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True) as child:
        try:
            out, err = child.communicate(source, timeout=timeout)
        except subprocess.TimeoutExpired:
            # The public resolution supervisor handles TERM and reaps all its workers.
            child.terminate()
            child.communicate()
            return {"status": "timeout", "wall_s": round(monotonic()-start, 4)}, None
    if child.returncode and not (len(command) > 1 and command[1] == "resolve" and child.returncode == 3):
        raise RuntimeError(f"{command}: {err[-1000:]} {out[-800:]}")
    return {"status": "complete", "wall_s": round(monotonic()-start, 4)}, out


def resolution(cli, mode, eq, names, numbers, timeout):
    command = [cli, "resolve", eq, "--vars", ",".join(names), "--mode", mode, "--quick", "--compute", "cpu", "--json"]
    if mode == "embedded":
        command.append("--monodromy")
    row, out = run(command, timeout)
    if out is None:
        return row
    record = json.loads(out)
    graph = record.pop("dual_complex")
    if record["status"] != "resolved" or graph["status"] != "complete":
        row.update(status="incomplete", solver=record["status"], graph=graph["status"])
        return row
    if mode == "embedded":
        if (record["topological_zeta"]["status"] != "computed" or record["monodromy_zeta"]["status"] != "computed"
                or record["monodromy_zeta"]["milnor_fiber_euler_characteristic"] != numbers[2]):
            raise ValueError("Embedded zeta functions failed the independent Milnor fibre Euler check.")
        row.update(topological_zeta=record["topological_zeta"]["expression"],
                   monodromy_zeta=record["monodromy_zeta"]["expression"])
    replay, text = run([cli, "dual-complex", "-"], timeout, json.dumps(record))
    if text is None or json.loads(text) != graph:
        raise ValueError(f"Replay of the saved {mode} graph failed for {eq}.")
    row.update(engine=record["engine"], charts=len(record["charts"]), repair_charts=sum(
        bool(chart.get("repair_projection")) for chart in record.get("atlas", [])),
        vertices=len(graph["vertices"]), cells=len(graph["cells"]), replay_s=replay["wall_s"],
        graph_sha256=hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest())
    return row


def compare(ident, names, eq, args):
    numbers = oracle(dict(id=ident, variables=names, equation=eq, dimension=len(names)-1, labels=[]))
    for op, value in zip(("milnor-number", "tjurina-number"), numbers):
        _, out = run([ROOT/"atlas", "algebra", op, "--vars", ",".join(names), "--poly="+eq, "--compute", "cpu",
                      "--json"], args.timeout)
        result = json.loads(out)["result"]
        if not result["finite"] or result["value"] != value:
            raise ValueError(f"{op} disagrees with Singular for {ident}.")
    result = dict(equation=eq, variables=list(names), mu_tau_fibre_euler=numbers, trials={})
    tools = ([("baseline", args.baseline.resolve())] if args.baseline else []) + [("final", ROOT/"atlas")]
    for mode in ("embedded", "intrinsic"):
        runs = {name: [] for name, cli in tools}
        for trial in range(args.trials):
            for name, cli in (tools if trial % 2 == 0 else reversed(tools)):
                if runs[name] and runs[name][-1]["status"] != "complete":
                    continue
                row = resolution(cli, mode, eq, names, numbers, args.timeout)
                if name == "final" and row["status"] != "complete":
                    raise ValueError(f"Final {mode} {ident} failed: {row}")
                if runs[name] and row.get("graph_sha256") != runs[name][0].get("graph_sha256"):
                    raise ValueError(f"Fresh runs disagree on {ident}.")
                runs[name].append(row)
        result["trials"][mode] = runs
    program = (f'LIB "resolve.lib";ring r=0,({",".join(names)}),dp;poly f={eq};'
               'list re=resolve(ideal(f),1,"L","A");print("CHARTS "+string(size(re[1])));quit;')
    singular = result["trials"]["Singular_resolve"] = []
    for _ in range(args.trials):
        row, out = run(["Singular", "-q", "--no-rc"], args.timeout, program)
        singular.append(row)
        if out is None:
            break
        count = re.search(r"^CHARTS (\d+)$", out, re.M)
        if not count or "result is o.k." not in out or re.search(r"^\s*\?", out, re.M):
            raise ValueError(f"Unchecked Singular resolution of {ident}.")
        row["charts"] = int(count[1])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Atlas CLI from a separately built baseline checkout")
    parser.add_argument("--baseline-revision", help="Revision of the supplied, independently built CLI")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 2 or args.timeout <= 0:
        parser.error("Use multiple trials and a positive timeout.")
    report = dict(conditions="Fresh public CLI processes, CPU, QQ at the origin, quick. Independent 30-second "
                  "default per command. Successful baseline and final resolutions replay their complete graphs "
                  "from the saved records after removing dual_complex. Replay and local invariant checks run "
                  "separately from resolution. Timings include process startup. Singular resolve.lib checks all "
                  "charts of its embedded resolution. Scalar agreement uses Singular local ds Jacobian lengths "
                  "and 1+(-1)^dim*mu for isolated germs. Other validation shared the host.",
                  cpu_affinity=sorted(os.sched_getaffinity(0)), baseline=args.baseline_revision,
                  singular=subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
                  timeout_s=args.timeout, trials=args.trials, cases={})
    for ident, names, eq in CASES:
        report["cases"][ident] = compare(ident, names, eq, args)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(ident, flush=True)
