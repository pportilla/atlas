"""Compare fresh Atlas and Singular embedded resolutions of related pinch-point germs over QQ.

Time both Atlas roots with and without both zeta functions, and time Singular
resolve.lib's default and checked resolution separately. Chart and fibre
oracles and saved-record replay run outside all timers.
"""
import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys

from plane_conjugate_resolution import measure

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    polynomial: str
    variables: str
    square: str
    partner: str


CASES = {
    "whitney": Case("x^2-y^2*(z+t)", "xyzt", "x", "y"),
    "mixed": Case("y^2*t+z^2+x*y*z", "xyzt", "z", "y"),
    "three": Case("x^2-y^2*z", "xyz", "x", "y"),
    "permuted": Case("z^2-y^2*(x+t)", "tyxz", "z", "y"),
    "rational": Case("(2/17)*x^2-(9/11)*y^2*(z-t)", "tyxz", "x", "y"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def singular_script(case, *, checked=True, fibre=False):
    n = len(case.variables)
    program = [
        'LIB "resolve.lib";', f'ring source=0,({",".join(case.variables)}),dp;',
        f'poly original={case.polynomial};',
        f'list resolved=resolve(ideal(original),{("1," if checked else "")}"L","A");',
        'print("COUNT "+string(size(resolved[1])));',
    ]
    if fibre:
        i = case.variables.index(case.square) + 1
        j = case.variables.index(case.partner) + 1
        frame = ",".join(f"projection[{k}]" for k in range(1, n + 1) if k not in (i, j))
        program.append(f"""
int ix,j,k,matches,ok,pivot;
for(ix=1;ix<=size(resolved[1]);ix++){{
    def chartRing=resolved[1][ix]; setring chartRing;
    ideal projection=BO[5]; map projectionMap=source,projection;
    poly exc=BO[4][1][1]; poly strict=BO[2][1]; poly total=exc^2*strict;
    poly pull=projectionMap(original); poly ratio=var({n});
    ok=(nvars(basering)=={n} && size(BO[1])==0 && size(BO[2])==1 && size(BO[4])==1);
    ok=ok && pull!=0 && total!=0 && leadcoef(total)*pull==leadcoef(pull)*total;
    matrix M[{n}][{n}];
    for(j=1;j<={n};j++){{for(k=1;k<={n};k++){{M[j,k]=diff(projection[j],var(k));}}}}
    poly jac=det(M);
    ok=ok && jac!=0 && leadcoef(exc)*jac==leadcoef(jac)*exc;
    pivot=0;
    if(projection[{i}]==exc && projection[{j}]==exc*ratio){{pivot=1;}}
    if(projection[{j}]==exc && projection[{i}]==exc*ratio){{pivot=2;}}
    ideal frame=exc,{frame},ratio;
    for(j=1;j<={n};j++){{matches=0;for(k=1;k<={n};k++){{if(var(j)==frame[k]){{matches++;}}}}
        ok=ok && matches==1;}}
    ideal fibre=std(projection); poly restriction=reduce(strict,fibre);
    if(pivot==1){{ok=ok && restriction!=0 && deg(restriction)==0;}}
    if(pivot==2){{
        ideal divs=exc,strict;
        ideal bad=std(fibre+divs+minor(jacob(divs),2));
        ok=ok && restriction!=0 && restriction==leadcoef(restriction)*ratio^2 && bad[1]==1;
    }}
    print("STANDARD_CHART "+string(pivot)+" "+string(ok));
    setring source;kill chartRing;
}}
""")
    return "\n".join([*program, "quit;", ""])


def require_singular(output, checked):
    lines = output.splitlines()
    if ("COUNT 2" not in lines or any(line.lstrip().startswith("?") for line in lines)
            or (checked and "result is o.k." not in output)):
        raise ValueError(f"Singular failed to certify the two expected charts.\n{output[-1800:]}")


def singular_fibre(case, singular, timeout):
    _, output = measure(singular, ROOT, timeout, singular_script(case, fibre=True))
    require_singular(output, True)
    checks = sorted(row for row in output.splitlines() if row.startswith("STANDARD_CHART "))
    if checks != ["STANDARD_CHART 1 1", "STANDARD_CHART 2 1"]:
        raise ValueError(f"Singular fibre oracle failed.\n{output[-1800:]}")
    # The verified projections identify the two pivot ratios as reciprocals.
    # The affine lines therefore glue to P^1. The strict cut, with reduced
    # support one point, is transverse there by the preceding minors.
    return {
        "origin_fibre": "P^1", "centre_generators": [case.square, case.partner],
        "exceptional_N_nu": [2, 2], "strict_N_nu": [1, 1],
        "euler_exceptional_open": 1, "euler_exceptional_strict": 1,
        "monodromy": "1/(1-T^2)", "milnor_fibre_euler": 2,
        "topological": "(s + 2)/(2*(s + 1)^2)",
    }


def chart_expression(text, names):
    if (not re.fullmatch(r"[a-zA-Z0-9_+*^() /-]+", text)
            or not set(re.findall(r"[a-zA-Z_]\w*", text)) <= set(names)):
        raise ValueError(f"Unexpected Atlas coordinate expression {text!r}")
    return f"({text})"


def check_atlas_charts(record, case, singular, timeout):
    # These cases all have free affine charts. The owned fibre is V(projection,
    # region). Showing the minors generate the unit ideal there verifies SNC
    # for each intersection of coordinate divisors with the strict transform.
    n = len(case.variables)
    program = [f'ring source=0,({",".join(case.variables)}),dp;', f'poly original={case.polynomial};']
    checked = []
    for index, chart in enumerate(record["atlas"], 1):
        names = chart["coordinates"]
        if (names != [f"u{i}" for i in range(1, n + 1)] or chart["ambient_ideal"] or chart["removed"]):
            raise ValueError("Expected a free affine toric chart")
        def eq(text):
            return chart_expression(text, names)

        projection = ",".join(eq(chart["map"][v]) for v in case.variables)
        region = ",".join(map(eq, chart["region"]))
        base = "image" + ("," + region if region else "")
        components = chart["components"]
        if [c["id"] for c in components].count("strict") != 1 or any(len(c["ideal"]) != 1 for c in components):
            raise ValueError("Expected principal components including one strict divisor")
        total = "*".join(f'{eq(c["ideal"][0])}^{c["multiplicity"]}' for c in components)
        volume = "*".join(f'{eq(c["ideal"][0])}^{c["nu"]-1}' for c in components)
        program.extend([
            f"ring atlas{index}=0,({','.join(names)}),dp;",
            f"ideal image={projection}; map chartMap=source,image;",
            f"poly pulled=chartMap(original); poly total={total}; poly volume={volume};",
            f"matrix M[{n}][{n}]; int j,k;",
            f"for(j=1;j<={n};j++){{for(k=1;k<={n};k++){{M[j,k]=diff(image[j],var(k));}}}}",
            "poly jac=det(M);",
            "int ok=(pulled!=0 && total!=0 && leadcoef(total)*pulled==leadcoef(pulled)*total);",
            "ok=ok && volume!=0 && jac!=0 && leadcoef(volume)*jac==leadcoef(jac)*volume;",
            f"ideal owned={base};",
        ])
        for count in range(1, len(components) + 1):
            for group in combinations(range(len(components)), count):
                name = "_".join(map(str, group))
                equations = ",".join(eq(components[i]["ideal"][0]) for i in group)
                program.extend([f"ideal cut{name}={equations};",
                                f"ideal bad{name}=std(owned+cut{name}+minor(jacob(cut{name}),{count}));",
                                f"ok=ok && bad{name}[1]==1;"])
        program.append(f'print("ATLAS_CHART {index} "+string(ok));')
        checked.append(f"ATLAS_CHART {index} 1")
    _, output = measure(singular, ROOT, timeout, "\n".join([*program, "quit;"]))
    if (any(line.lstrip().startswith("?") for line in output.splitlines())
            or [l for l in output.splitlines() if l.startswith("ATLAS_CHART ")] != checked):
        raise ValueError(f"Independent Atlas chart verification failed.\n{output[-2000:]}")
    return checked


def check_record(raw, case, zetas, oracle):
    record = json.loads(raw)
    if (record["status"] != "resolved" or record["mode"] != "embedded" or record["variables"] != list(case.variables)
            or record["dual_complex"]["status"] != "complete" or not record["atlas"]
            or len(record["atlas"]) != len(record["charts"])
            or record["certificate"]["atlas_verification"] != "Embedded normal crossings on the owned origin fibre"):
        raise ValueError("Expected a complete verified embedded resolution")
    if zetas:
        mono, topo = record["monodromy_zeta"], record["topological_zeta"]
        if (mono["status"] != "computed" or topo["status"] != "computed"
                or mono["expression"] != oracle["monodromy"] or topo["expression"] != oracle["topological"]
                or mono["milnor_fiber_euler_characteristic"] != oracle["milnor_fibre_euler"]
                or mono["factors"] != [{"multiplicity": oracle["exceptional_N_nu"][0], "exponent": -1}]):
            raise ValueError("Atlas zeta functions disagree with the independently certified Singular fibre")
    # The binomial producer repeats its measured chart-batch latency in the
    # certificate and execution report. Preserve every timing in the receipt.
    timings = {"elapsed_ms": record.pop("elapsed_ms")}
    for path, section in [("certificate.chart_batch", record["certificate"].get("chart_batch", {})),
                          *((f"execution.stages.{i}", stage) for i, stage in enumerate(record["execution"]["stages"]))]:
        for key in ("elapsed_ms", "native_elapsed_ms"):
            if key in section:
                timings[f"{path}.{key}"] = section.pop(key)
    return record, timings


def replay(root, record, timeout):
    archived = {k: record[k] for k in ("status", "mode", "engine", "variables", "polynomial", "atlas")}
    source = json.dumps(archived)
    _, mono = measure(["./atlas", "monodromy", "-", "--json"], root, timeout, source)
    script = ("import json, sys\nfrom engine.monodromy import monodromy_zeta\n"
              "from engine.topological import topological_zeta\n"
              "r=json.load(sys.stdin)\nprint(json.dumps(topological_zeta(r, monodromy=monodromy_zeta(r))))")
    _, topo = measure([sys.executable, "-c", script], root, timeout, source)
    if json.loads(mono) != record["monodromy_zeta"] or json.loads(topo) != record["topological_zeta"]:
        raise ValueError(f"Saved atlas replay changed the zeta record in {root}")


def run(args, *, cases=CASES, fibre_oracle=singular_fibre, chart_oracle=check_atlas_charts, driver=__file__,
        chart_changes=False, enforce_equal=False):
    roots = {"before": args.baseline.resolve(), "after": args.atlas_root.resolve()}
    python_command = shutil.which("python3") or sys.executable
    exe = shutil.which("Singular")
    if not exe:
        raise SystemExit("Install Singular with resolve.lib")
    singular = [exe, "-q", "--no-rc"]
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    def hashes():
        binaries = ("libgroebner.so", "libresolution.so", "libchain.so", "newton_fan", "libsmall_newton_fan.so",
                    "resolve_supervisor", "graph_verify", "face_gb", "toricfan", "plane", "libmonomial*.so", "libcertified*.so")
        manifests = {}
        for name, root in roots.items():
            files = [root/"atlas", *(root/"engine").glob("*.py")]
            for pattern in binaries:
                files.extend((root/"native/build").glob(pattern))
            manifests[name] = {str(f.relative_to(root)): digest(f.read_bytes()) for f in sorted(files)}
        return manifests
    report = {"conditions": args.__dict__.get("conditions", __doc__), "trials": args.trials, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "search": args.search, "platform": platform.platform(), "python": python_command, "singular":
              subprocess.check_output([exe, "--version"], text=True).splitlines()[0],
              "roots": {name: str(root) for name, root in roots.items()}, "sha256": hashes(),
              "benchmark_sha256": digest(Path(driver).read_bytes()), "cases": {}}
    for label in args.cases:
        case = cases[label]
        oracle = fibre_oracle(case, singular, args.timeout)
        result = report["cases"][label] = {"polynomial": case.polynomial, "variables": list(case.variables),
                                            "singular_oracle": oracle, "runs": [], "agreed_records": {}}
        labels = ["before", "after", "singular_default", "singular_checked", "before_zetas", "after_zetas"]
        for trial in range(-1, args.trials):
            for solver in (labels if trial % 2 == 0 else reversed(labels)):
                if solver.startswith("singular"):
                    checked = solver == "singular_checked"
                    row, raw = measure(singular, ROOT, args.timeout, singular_script(case, checked=checked))
                    require_singular(raw, checked)
                else:
                    root_label, _, extra = solver.partition("_")
                    zetas = bool(extra)
                    cmd = ["./atlas", "resolve", case.polynomial, "--vars", ",".join(case.variables),
                           "--mode", "embedded", "--compute", "cpu", "--json"]
                    if args.search != "default":
                        cmd.append("--" + args.search)
                    if zetas:
                        cmd.append("--monodromy")
                    row, raw = measure(cmd, roots[root_label], args.timeout)
                    if row["status"] != "complete":
                        raise TimeoutError(f"{solver} did not return a complete Atlas resolution of {label}")
                    record, row["reported_ms"] = check_record(raw, case, zetas, oracle)
                    mode = "zetas" if zetas else "geometry"
                    record_key = f"{root_label}_{mode}" if chart_changes else mode
                    agreed = result["agreed_records"].setdefault(record_key, record)
                    if record != agreed:
                        raise ValueError(f"Complete Atlas record differs in {label}, {solver}, trial {trial + 1}")
                    row["record_sha256"] = digest(json.dumps(record, sort_keys=True).encode())
                    if trial < 0 and zetas:
                        replay(roots[root_label], record, args.timeout)
                        if chart_changes:
                            result.setdefault("singular_checks_on_atlas", {})[root_label] = chart_oracle(
                                record, case, singular, args.timeout)
                        elif "singular_checks_on_atlas" not in result:
                            result["singular_checks_on_atlas"] = chart_oracle(record, case, singular, args.timeout)
                if trial >= 0:
                    result["runs"].append(dict(trial=trial+1, solver=solver, **row))
        for prefix in (("before_", "after_") if chart_changes else ("",)):
            records = result["agreed_records"]
            with_zetas = dict(records[prefix+"zetas"])
            for key in ("monodromy_zeta", "topological_zeta"):
                del with_zetas[key]
            geometry = records[prefix+"geometry"]
            if with_zetas != geometry:
                # Canonical blowups may attach separately computed strata to their certificate.
                with_zetas["certificate"] = {**with_zetas["certificate"]}
                for key in ("numerical_data", "strata"):
                    if key not in geometry["certificate"]:
                        with_zetas["certificate"].pop(key, None)
                if with_zetas != geometry:
                    raise ValueError(f"Requesting zeta functions changed the geometry of {label}")
        if chart_changes:
            equal = {mode: records["before_"+mode] == records["after_"+mode] for mode in ("geometry", "zetas")}
            result["identical_full_record_between_roots"] = equal
            if enforce_equal and not all(equal.values()):
                changed = {mode: sorted(k for k in records["before_"+mode].keys() | records["after_"+mode].keys()
                                        if records["before_"+mode].get(k) != records["after_"+mode].get(k))
                           for mode, same in equal.items() if not same}
                raise ValueError(f"Pre/post resolution records differ for {label}: {changed}")
        result["median_s"] = {solver: {key: statistics.median(row[key] for row in result["runs"]
                              if row["solver"] == solver) for key in ("wall_s", "cpu_s")} for solver in labels}
        result["paired_speedup"] = {}
        for suffix in ("", "_zetas"):
            before, after = ([row["wall_s"] for row in result["runs"] if row["solver"] == name+suffix]
                             for name in ("before", "after"))
            result["paired_speedup"]["resolution"+suffix] = statistics.median(b/a for b, a in zip(before, after))
        print(label, {solver: round(values["wall_s"]*1000, 1) for solver, values in result["median_s"].items()}, flush=True)
        if hashes() != report["sha256"]:
            raise RuntimeError("A measured executable or source changed during the benchmark. Use frozen roots.")
        args.output.write_text(json.dumps(report, indent=2)+"\n")


def arguments(cases, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--atlas-root", default=ROOT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=cases, default=list(cases))
    parser.add_argument("--search", choices=("default", "quick", "exhaustive"), default="default")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout <= 0:
        parser.error("Use a positive trial count and timeout")
    args.conditions = description
    return args


if __name__ == "__main__":
    run(arguments(CASES, __doc__))
