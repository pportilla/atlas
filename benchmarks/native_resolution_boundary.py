"""Fresh CLI timing and whole-record comparison for graph, canonical, Newton and coordinate resolutions."""
import argparse
import gzip
import hashlib
import json
from itertools import combinations
import os
from pathlib import Path
import platform
import shutil
import statistics
import tempfile
import sys
import subprocess

from plane_conjugate_resolution import measure

CASES = {
    "whitney": ("x^2-y^2*(z+t)", "xyzt", "embedded", "verified-monomial-modification", 3),
    "graph": ("(x+y+z)^2-z*(x-y)^2", "xyz", "embedded", "verified-ideal-blowup", 2),
    "centre": ("(x+y)^2+z^3+x^3", "xyz", "embedded", "canonical-blowups", 15),
    "newton": ("x*y*z+x^4+y^4+z^4", "xyz", "intrinsic", "newton-polyhedron", 9),
}
DEFAULT_CASES = tuple(CASES)
CASES.update({
    "tangent": ("(x*y-z^2)^2+x^5+y^5", "xyz", "embedded", "canonical-blowups", 18),
    "double_cone": ("x^2+(y+z)^3+y^4", "xyz", "embedded", "canonical-blowups", 19),
})


def without_timings(result):
    if isinstance(result, list):
        return [without_timings(value) for value in result]
    if isinstance(result, dict):
        return {key: without_timings(value) for key, value in result.items()
                if key not in {"elapsed_ms", "native_elapsed_ms"}}
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def saved_outputs(tree, record, timeout, expected_complex=None):
    saved = {k: v for k, v in record.items()
             if k not in ("dual_complex", "monodromy_zeta", "topological_zeta", "monodromy_check")}
    cli = tree / "atlas"
    arg = json.dumps(saved)
    dual = json.loads(subprocess.check_output([cli, "dual-complex", "-"], input=arg, text=True, timeout=timeout))
    mono = json.loads(subprocess.check_output([cli, "monodromy", "-", "--json"], input=arg, text=True, timeout=timeout))
    # The prompt and HTTP replay service additionally exposes the topological zeta and pole witnesses.
    script = ("from engine.resolution_zetas import replay_zetas\nimport json,sys\n"
              "print(json.dumps(replay_zetas(dict(record=json.load(sys.stdin),zetas=['monodromy','topological']))))")
    zetas = json.loads(subprocess.check_output([shutil.which("python3"), "-c", script], input=arg, text=True,
                                               cwd=tree, timeout=timeout))
    expected = record["dual_complex"] if expected_complex is None else expected_complex
    if dual != expected:
        # Fresh toric records carry optional ray and root annotations which
        # the saved-chart command has always omitted. Compare every labelled
        # incidence, face, dimension and cell ID before accepting this case.
        def incidence(complex):
            return {**complex,
                    "vertices": sorted(({k: v[k] for k in ("id", "dimension")} for v in complex["vertices"]),
                                       key=lambda v: v["id"]),
                    "cells": [{k: c[k] for k in ("id", "dimension", "vertices", "faces")} for c in complex["cells"]]}
        if record["engine"] != "newton-polyhedron" or incidence(dual) != incidence(expected):
            raise ValueError("Saved dual complex differs from the fresh chart incidence")
    keys = ["monodromy_zeta", "topological_zeta"]
    if "monodromy_check" in record:
        keys.append("monodromy_check")
    if mono != record["monodromy_zeta"] or any(zetas[k] != record[k] for k in keys):
        raise ValueError("Saved zeta or pole witnesses differ from the fresh resolution")
    return {"dual_complex": dual, **zetas}


def singular_chart_checks(record, singular, timeout):
    """Check each principal chart over its owned fibre, including complete-intersection ambients."""
    from whitney_related import chart_expression

    program = ['LIB "elim.lib";', 'proc saturatedIdeal(ideal i,ideal j){def result=sat(i,j);'
               'if(typeof(result)=="list"){return(result[1]);}return(result);}', 'proc same(ideal a,ideal b){return(size(reduce(a,std(b)))==0 '
               '&& size(reduce(b,std(a)))==0);}', 'ring source=0,(x,y,z),dp;',
               f'poly equation={chart_expression(record["polynomial"], record["variables"])};']
    expected = []
    for index, (chart, public) in enumerate(zip(record["atlas"], record["charts"], strict=True), 1):
        names = chart["coordinates"]
        if not all(v.isascii() and v.isidentifier() for v in names):
            raise ValueError("Unexpected chart coordinates")
        def equation(value):
            return chart_expression(value, names)
        def ideal(values):
            return ",".join(map(equation, values)) or "0"
        if names != public["coordinates"] or any(len(c["ideal"]) != 1 for c in chart["components"]):
            raise ValueError("Expected principal components in the submitted coordinates")
        fields = record["variables"]
        program += [f'ring C{index}=0,({",".join(names)}),dp;',
                    f'ideal A=std(ideal({ideal(chart["ambient_ideal"])})); int cod=size(A);',
                    f'ideal image={ideal([chart["map"][v] for v in fields])}; map phi=source,image;',
                    f'ideal fibre={ideal(chart["fibre"])};ideal owned=A+fibre+ideal({ideal(chart["region"])});',
                    f'ideal projection={ideal([public["map"][v] for v in fields])};',
                    'int ok=same(A+fibre,A+image);int j;'
                    'for(j=1;j<=3;j++){ok=ok && reduce(projection[j]-image[j],A)==0;}',
                    'poly total=1;poly volume=1;']
        pieces = []
        for j, c in enumerate(chart["components"]):
            var = 'piece' + str(j)
            pieces.append(var)
            program += [f'poly {var}={equation(c["ideal"][0])};',
                        f'total=total*{var}^{c["multiplicity"]};volume=volume*{var}^{c["nu"]-1};']
            if c["id"] == "strict":
                rep = public["strict_transform"]
                program.append(f'ok=ok && same(A+ideal({var}),A+ideal({ideal(rep if isinstance(rep, list) else [rep])}));')
        program += ['ideal chartMap=A+image; if(cod==0){chartMap=image;}',
                    'ideal vol=minor(jacob(chartMap),nvars(basering));',
                    'ideal pull=A+ideal(phi(equation));ideal submitted=A+ideal(total);',
                    'ideal form=A+vol;ideal logForm=A+ideal(volume);']
        def empty(bad):
            program.append(f'ideal check={bad};')
            for removed in chart["removed"]:
                program.append(f'check=saturatedIdeal(check,ideal({ideal(removed)}));')
            program.append('ok=ok && std(check)[1]==1;kill check;')
        # Two principal ideals agree near an owned point iff both ideal quotients are units there.
        for a, b in (('pull','submitted'),('form','logForm')):
            empty(f'owned+quotient({a},{b})*quotient({b},{a})')
        for n in range(len(pieces) + 1):
            for group in combinations(pieces, n):
                if not n and len(names) == 3:
                    continue
                operands = ",".join(group) or "0"
                program.append(f'ideal cut=std(A+ideal({operands}));')
                rank = len(names) - 3 + n
                program.append(f'ideal bad=owned+cut;')
                if rank <= len(names):
                    program.append(f'if(size(cut)>={rank}){{bad=bad+minor(jacob(cut),{rank});}}')
                empty('bad')
                program.append('kill cut,bad;')
        program.append(f'print("CHART {index} "+string(ok));setring source;')
        expected.append(f'CHART {index} 1')
    _, output = measure(singular, Path.cwd(), timeout, "\n".join([*program, "quit;"]))
    observed = [s for s in output.splitlines() if s.startswith("CHART ")]
    errors = [s for s in output.splitlines() if s.lstrip().startswith("?")]
    if observed != expected or errors:
        raise ValueError(f"Independent Singular chart checks failed. {observed}, {errors[:8]}")
    return observed


def independent_checks(record, name, singular, timeout):
    cmd = [singular, "-q", "--no-rc"]
    if name == "whitney":
        from whitney_related import CASES as family, check_atlas_charts, check_record, singular_fibre
        control = family[name]
        fibre = singular_fibre(control, cmd, timeout)
        check_record(json.dumps(dict(record, elapsed_ms=0)), control, True, fibre)
        return dict(charts=check_atlas_charts(record, control, cmd, timeout), fibre=fibre)
    if name == "graph":
        from reduced_centre_resolution import CASES as lines, verify, radical_oracle, fibre_oracle
        line = lines["inclined"]
        fibre = fibre_oracle(line, cmd, timeout)
        if any(record[key + "_zeta"]["expression"] != fibre[key] for key in ("monodromy", "topological")):
            raise ValueError("Singular graph fibre invariants disagree with the record")
        return dict(overlaps=verify(record, line, cmd, timeout), centre=radical_oracle(line, cmd, timeout), fibre=fibre)
    charts = singular_chart_checks(record, cmd, timeout)
    results = {"chart_certificates": charts}
    if name == "newton":
        from resolution_zeta import Solver, check_expressions
        solver = Solver("singular", None, singular)
        _, answer = solver.fresh(record["polynomial"], os.environ, timeout)
        check_expressions(solver, answer, [record["topological_zeta"]["expression"],
                                           record["monodromy_zeta"]["expression"]])
        results["singular_local_zetas"] = answer
    return results


def check_archived_graph(roots, timeout):
    """Replay the previous 36-chart modification of the same inclined surface."""
    fixture = Path(__file__).with_name("reduced-centre-resolution-ryzen-6800h.json.gz")
    with gzip.open(fixture, "rt") as source:
        archived = json.load(source)["cases"]["inclined"]["agreed_records"]["before_zetas"]
    if len(archived["atlas"]) != 36:
        raise ValueError("Expected the archived 36-chart resolution")
    stored = archived["dual_complex"]
    duplicate = {"id": "CB1-B9.2", "dimension": 1, "vertices": ["B1", "B9"], "faces": ["B1", "B9"]}
    if stored["cells"].count(duplicate) != 1:
        raise ValueError("The historical incidence fixture changed")
    verified = {**stored, "cells": [cell for cell in stored["cells"] if cell != duplicate]}
    results = [saved_outputs(root, archived, timeout, verified) for root in roots.values()]
    if results[0] != results[1]:
        raise ValueError("Archived 36-chart replay differs")
    return {"charts": 36, "archived_cell_count": len(stored["cells"]), "recomputed_cell_count": len(verified["cells"]),
            "archived_duplicate": duplicate["id"], "record_sha256": digest(archived), "replay_sha256": digest(results[0])}


def toric_control(roots, args):
    polynomial = "x^2+y^19+z^37+x^3*(x+y+z)^27"
    command = ["resolve", polynomial, "--vars", "x,y,z", "--mode", "intrinsic", "--compute", "cpu", "--json"]
    agreed = None
    samples = {key: [] for key in roots}
    for trial in range(args.trials + 1):
        for key in (list(roots) if trial % 2 else reversed(roots)):
            root = roots[key]
            timing, output = measure([str(root / "atlas"), *command], root, args.timeout)
            if timing["status"] != "complete":
                raise ValueError("The 191-chart resolution timed out")
            record = without_timings(json.loads(output))
            if (record["status"], record["engine"], len(record["charts"])) != ("resolved", "native-toric", 191):
                raise ValueError("The 191-chart control changed route or chart count")
            if agreed is not None and record != agreed:
                raise ValueError("The full toric resolution differs")
            agreed = record
            if trial:
                samples[key].append(timing)
    dual = agreed["dual_complex"]
    singular_input = (f'LIB "resolve.lib";ring source=0,(x,y,z),dp;poly f={polynomial};'
                      'list rr=resolve(ideal(f),"L","A");print("FINAL_CHARTS "+string(size(rr[1])));quit;')
    timing, output = measure([args.singular, "-q", "--no-rc"], roots["after"], args.singular_timeout, singular_input)
    counts = [int(s.split()[1]) for s in output.splitlines() if s.startswith("FINAL_CHARTS ")]
    if timing["status"] == "complete" and (len(counts) != 1 or any(s.lstrip().startswith("?") for s in output.splitlines())):
        raise ValueError("Singular did not complete the toric control successfully")
    return dict(polynomial=polynomial, mode="intrinsic", full_record_sha256=digest(agreed), samples=samples,
                charts=191, vertices=len(dual["vertices"]), cells=len(dual["cells"]),
                median_wall_ms={k: round(1000 * statistics.median(row["wall_s"] for row in v), 2) for k,v in samples.items()},
                singular_default=timing, singular_chart_count=counts[0] if counts else None)


def run_case(name, roots, args):
    case_name = name
    polynomial, variables, mode, engine, charts = CASES[case_name]
    operations = [(name + suffix, name, zetas) for name in roots
                  for suffix, zetas in [("", False), ("_zetas", True)]]
    results, samples = {}, {key: [] for key, _, _ in operations}
    for trial in range(args.trials + 1):
        for key, name, zetas in (operations if trial % 2 else reversed(operations)):
            tree = roots[name]
            command = [str(tree / "atlas"), "resolve", polynomial, "--vars", ",".join(variables), "--mode", mode,
                       "--compute", "cpu", "--json"]
            if zetas:
                command.append("--monodromy")
            timing, output = measure(command, tree, args.timeout)
            if timing["status"] != "complete":
                raise RuntimeError(f"Atlas timed out for {key}/{polynomial}")
            record = without_timings(json.loads(output))
            if (record["status"], record["engine"], len(record["charts"])) != ("resolved", engine, charts):
                raise ValueError(f"Unexpected resolution for {key}/{polynomial}: {record['status']}, {record['engine']}, {len(record['charts'])} charts")
            if results.setdefault(zetas, record) != record:
                raise ValueError(f"Full resolution differs for {key}/{polynomial}")
            if trial:
                samples[key].append(timing)
    replays = [saved_outputs(tree, results[True], args.timeout) for tree in roots.values()]
    if replays[0] != replays[1]:
        raise ValueError("The new solver changed the independently replayed record.")
    singular_counts = {}
    for flag in ("default", "checked"):
        key = "singular_" + flag
        samples[key] = []
        script = (f'LIB "resolve.lib";ring source=0,({",".join(variables)}),dp;poly f=' + polynomial +
                  ';list rr=resolve(ideal(f),' + ('1,' if flag == 'checked' else '') +
                  '"L","A");print("FINAL_CHARTS "+string(size(rr[1])));quit;\n')
        for trial in range(args.trials + 1):
            timing, text = measure([args.singular, "-q", "--no-rc"], roots["after"], args.singular_timeout, script)
            if timing["status"] == "timeout":
                samples[key].append(timing)
                break
            counts = [int(s.split()[1]) for s in text.splitlines() if s.startswith("FINAL_CHARTS ")]
            if len(counts) != 1 or any(s.lstrip().startswith("?") for s in text.splitlines()):
                raise ValueError("Singular did not certify its output: " + text[-1200:])
            if singular_counts.setdefault(key, counts[0]) != counts[0]:
                raise ValueError("Singular's chart count changed between trials")
            if trial:
                samples[key].append(timing)
    medians = {k: round(1000 * statistics.median(s["wall_s"] for s in rows), 2)
               for k, rows in samples.items() if all(s["status"] == "complete" for s in rows)}
    print(engine, charts, medians, singular_counts, flush=True)
    independent = independent_checks(results[True], case_name, args.singular, args.timeout)
    if case_name == "graph":
        independent["archived"] = check_archived_graph(roots, args.timeout)
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "singular.json"
            subprocess.run([sys.executable, Path(__file__).with_name("historic_inclined_edge.py"),
                            "--singular", args.singular, "--output", receipt], check=True, timeout=args.timeout)
            independent["archived"]["singular"] = json.loads(receipt.read_text())
    return dict(polynomial=polynomial, variables=list(variables), mode=mode, samples=samples, median_wall_ms=medians,
                independent=independent,
                singular_charts=singular_counts, replay_sha256=digest(replays[0]),
                record_sha256={str(k): digest(v) for k, v in results.items()},
                record=results[True], resolution_only=results[False])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--singular-timeout", type=float, default=120)
    parser.add_argument("--singular", default=shutil.which("Singular"))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=DEFAULT_CASES)
    parser.add_argument("--toric-control", action="store_true", help="Also compare all 191 published toric charts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or not args.singular:
        parser.error("Select at least one trial and an installed Singular.")
    os.sched_setaffinity(0, {args.cpu})
    roots = dict(before=args.before.resolve(), after=args.after.resolve())
    report = dict(cpu=args.cpu, trials=args.trials, singular_timeout_s=args.singular_timeout,
                  installed_hashes={key: json.loads((root / "source-hashes.json").read_text())
                                    for key, root in roots.items() if (root / "source-hashes.json").is_file()},
                  machine=platform.machine(), python=platform.python_version(),
                  singular=subprocess.check_output([args.singular, "--version"], text=True).splitlines()[0],
                  cases={})
    for name in args.cases:
        report["cases"][name] = run_case(name, roots, args)
        args.output.write_bytes(gzip.compress(json.dumps(report, indent=2).encode(), mtime=0))
    if args.toric_control:
        report["toric_control"] = toric_control(roots, args)
        args.output.write_bytes(gzip.compress(json.dumps(report, indent=2).encode(), mtime=0))


if __name__ == "__main__":
    main()
