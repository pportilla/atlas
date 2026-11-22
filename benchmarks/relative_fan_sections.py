"""Compare intrinsic relative-fan sections against affine factorization and fibre incidence in Singular.

Run python3 benchmarks/relative_fan_sections.py --trials 3 --output report.json.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.atlas_torus import section_complex
from engine.fan_surface import surface_complex
from engine.polynomial import parse_expression
from engine.quadratic_fan import translations

PINCH = "x^2-(y^2+z^3)*(y-z^2)^2"
POLYNOMIAL_GRAPH = "x^2-(y^2+z^3)*(y-z^2-z^3)^2"
CASES = ("x^4+y^5+z*x^2*y^2", "z^4+x^2*y^2*(x^3+y^4)", "y*(x^2+y^3+y*z^3)", PINCH,
         "(x-y*z)^2+y^2*(y^3+z^5)")
THREEFOLDS = ("y*(x^2+y^3+y*z^3)", "y*(x^2+y^3+y*z^3+y*t^3)", "y*(x^2+y^3+y*z^3+y*z*t)")


def ramification_timings(equation, trials):
    from engine.polynomial import format_poly
    names = list("xyz")
    f = parse_expression(equation, names)
    code = ('ring R=0,(x,y,z),dp;poly f='+equation+';int t=rtimer;'
            'list fs=sqrfree(subst(f,x,0));int j;poly graph;'
            'for(j=1;j<=size(fs[1]);j++){if(fs[2][j]>1){graph=fs[1][j]/diff(fs[1][j],y);}}'
            'poly adapted=subst(f,y,2*y-graph);print("TIME "+string(rtimer-t));')
    times = {"atlas_s": [], "singular_s": []}
    for _ in range(trials):
        start = perf_counter()
        choice, = translations(f, names, ramification=True)
        times["atlas_s"].append(perf_counter()-start)
        change, adapted = choice
        if change["coordinate"] != "y":
            raise AssertionError("Wrong ramification branch.")
        checks = (f'if(graph!=y-({change["offset"]}) || adapted!={format_poly(adapted,names)})'
                  '{print("BAD");}quit;')
        proc = subprocess.run(["Singular","-q","--no-rc","--ticks-per-sec=1000000"], input=code+checks,
                              capture_output=True, text=True, check=True)
        lines = (proc.stdout+proc.stderr).splitlines()
        if any(l.lstrip().startswith(("?", "BAD")) for l in lines):
            raise AssertionError(lines)
        timing, = [l for l in lines if l.startswith("TIME ")]
        times["singular_s"].append(int(timing.split()[1])/1000000)
    return times


def atlas_inputs(record):
    rays = record["fan"]["rays"]
    terms, names = [], []
    for chart in record["charts"]:
        coordinates = chart["coordinates"]
        cone = tuple(tuple(rays[i]) for i in chart["ray_indices"])
        boundary = {coordinates.index(v["equation"]): v["id"] for v in chart["ambient_components"]}
        strict = tuple(parse_expression(chart["strict_transform"], coordinates).items())
        terms.append((cone, boundary, strict))
        names.append(coordinates)
    owned = [tuple(parse_expression(f, chart["coordinates"], as_row=True)
                   for f in chart["fibre"] + chart["region"]) for chart in record["atlas"]]
    return terms, names, [(*entry, fibre) for entry, fibre in zip(terms, owned)]


def singular_input(record):
    # Factor each affine restriction over its splitting field, then intersect
    # all subsets of the resulting divisors with the origin fibre.
    families = record.get("root_families", {})
    cubic = any(v["polynomial"] == "T^3+1" for v in families.values())
    roots = {"-T^2+1": ("(-1)", "1")}
    if cubic:
        roots["T^3+1"] = ("(-1)", "((1-a)/2)", "((1+a)/2)")
    else:
        roots["T^2+1"] = ("(-a)", "a")
    if any(v["polynomial"] not in roots for v in families.values()):
        raise ValueError("The comparison field must contain all displayed roots.")
    coordinates = record["charts"][0]["coordinates"]
    lines = [f'ring R=(0,a),({",".join(coordinates)}),dp;', f'minpoly=a^2+{3 if cubic else 1};',
             'poly f,g,h; ideal fibre,fs; int j,hit; int start=rtimer;']
    def literal(text):
        for name, family in families.items():
            for j, root in enumerate(roots[family["polynomial"]], 1):
                text = text.replace(f"{name}_{j}", root)
        return text
    for chart in record["charts"]:
        lines.append(f'f={chart["strict_transform"]};fibre={",".join(chart["map"].values())};')
        for ambient in chart["ambient_components"]:
            axis = ambient["equation"]
            candidates = [v for v in chart["exceptional"] if v["ambient_divisor"] == ambient["id"]]
            lines.append(f'g=subst(f,{axis},0);fs=factorize(g,1);')
            for v in candidates:
                h = literal(v['ideal'][1])
                lines.append(f'if(reduce(g,std(ideal({h})))!=0){{print("BAD displayed factor");}}')
            lines.append(f'for(j=1;j<=size(fs);j++){{h=fs[j];'
                         f'if(deg(h)>0 && reduce(1,std(fibre+ideal({axis},h)))!=0){{hit=0;')
            for v in candidates:
                h = literal(v['ideal'][1])
                lines.append(f'if(h/leadcoef(h)==({h})/leadcoef({h})){{hit++;print("V {v["id"]}");}}')
            lines.append('if(hit!=1){print("BAD fibre factor");}}}')
        for size in range(2, len(coordinates)+1):
            for pieces in combinations(chart["exceptional"], size):
                equations = ",".join(literal(f) for piece in pieces for f in piece["ideal"])
                labels = " ".join(sorted(piece["id"] for piece in pieces))
                lines.append(f'if(reduce(1,std(fibre+ideal({equations})))!=0){{print("C {labels}");}}')
    lines.append('print("TIME "+string(rtimer-start));quit;')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.trials < 2:
        p.error("Use at least two trials.")
    report = {"conditions": "Identical precomputed affine Newton charts. Atlas computes the labelled complex and "
              "local ideals via the orbit plan or affine surface sections. Singular factors each affine restriction over Q(i) "
              "or Q(sqrt(-3)) and checks visible primes and their intersections via Groebner bases of their origin fibres. Atlas wall "
              "time includes transport; Singular rtimer excludes process startup. Load uncontrolled.",
              "ramification_branch_and_substitution": ramification_timings(PINCH, args.trials),
              "polynomial_graph_branch_and_substitution": ramification_timings(POLYNOMIAL_GRAPH, args.trials),
              "cases": []}
    for variables, f in [*(("x,y,z", f) for f in CASES), *(("x,y,z,t", f) for f in THREEFOLDS)]:
        record = json.loads(subprocess.check_output([ROOT/"atlas","resolve","--mode","intrinsic","--vars",
            variables,"--compute","cpu","--quick","--json","--",f]))
        dimension = len(record["variables"])
        terms, names, charts = atlas_inputs(record)
        code = singular_input(record)
        times = {"atlas_s": [], "singular_s": []}
        for _ in range(args.trials):
            start = perf_counter()
            answer = section_complex(terms, dimension, True, common_source=True, embedded_snc=True, local_coordinates=names,
                                     defer_relative_coordinates=dimension == 3)
            if answer is None and dimension == 3:
                answer = surface_complex(charts, local_coordinates=names)
            times["atlas_s"].append(perf_counter()-start)
            vertices = {v["id"] for v in answer["vertices"]}
            cells = {tuple(sorted(c["vertices"])) for c in answer["cells"]}
            if len(cells) != len(answer["cells"]):
                raise AssertionError("These controls have one simplex per nonempty intersection.")
            labels = {c["id"]: tuple(sorted(c["vertices"])) for c in answer["cells"]}
            labels.update({v:(v,) for v in vertices})
            for c in answer["cells"]:
                facets = {labels[name] for name in c["faces"]}
                if facets != set(combinations(labels[c["id"]], c["dimension"])):
                    raise AssertionError("The facet attachments disagree with the labelled simplex.")
            out = subprocess.run(["Singular","-q","--no-rc","--ticks-per-sec=1000000"], input=code, text=True,
                                 check=True, capture_output=True)
            lines = (out.stdout+out.stderr).splitlines()
            if any(l.lstrip().startswith(("?", "BAD")) for l in lines):
                raise RuntimeError('\n'.join(lines[:80]))
            if vertices != {l[2:] for l in lines if l.startswith("V ")} or cells != {
                    tuple(l.split()[1:]) for l in lines if l.startswith("C ")}:
                raise AssertionError("Singular disagrees with the labelled vertices or intersections.")
            timing, = [l for l in lines if l.startswith("TIME ")]
            times["singular_s"].append(int(timing.split()[1])/1000000)
        report["cases"].append(dict(equation=f, variables=record["variables"], charts=len(terms), vertices=len(vertices),
                                    edges=sum(len(face)==2 for face in cells), triangles=sum(len(face)==3 for face in cells), **times))
    serialized = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
