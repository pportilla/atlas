"""Check geometric section decompositions against Singular's absPrimdecGTZ over QQbar."""
import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import native_polynomial as np
from engine.absolute_sections import ExtendSectionField, absolute_primes, extend_section_field
from engine.affine_kernel import sparse_terms
from engine.local_blowups import Chart
from engine.native_library import _LIBRARY
from engine.plane_backend import _WORKER
from engine.polynomial import format_poly, parse_expression


@dataclass(frozen=True)
class Example:
    label: str
    axes: str
    equations: tuple
    dimension: int
    marked: bool = False


def corpus():
    return [
        Example("irreducible_torus_curve", "xy", ("x*y-1",), 1),
        Example("coupled_conjugate_curves", "xyz", ("x^2+y^2", "z*x-1"), 1),
        Example("smooth_coupled_surface", "xyzw", ("x*y-1", "z-w^3"), 2),
        Example("correlated_four_points", "xyz", ("x^2+1", "y^2+1", "z-x*y"), 0),
        Example("two_distinct_quadratics", "xyz", ("(2*x^2+1)*(3*x^2+1)", "y-x^2", "z*x-1"), 0),
        Example("six_conjugate_coupled_points", "xyz", ("x^3-2", "y^2-x", "z-y"), 0),
        Example("nonreduced_conjugate_lines", "xyz", ("(x^2+y^2)^2", "z-x-y"), 1),
        Example("wide_coefficient_curve", "xy", (f"{2**150+1}*x*y-1",), 1),
        Example("marked_geometric_line", "xyt", ("t*x-y",), 1, True),
        Example("marked_split_points", "xyt", ("x^2-2", "y-x^2"), 0, True),
        Example("marked_extension_required", "xyt", ("x^2-t", "y-x^2"), 0, True),
    ]


def prepare(example):
    variables = tuple(map(np.Symbol, example.axes))
    def parse(source):
        return np.from_terms(variables, parse_expression(source, list(example.axes)))
    field = (variables[-1], parse("t^2-2"), 2, "root 1 (FLINT order)") if example.marked else ()
    return Chart(variables, (), (("D", 1, 1, tuple(map(parse, example.equations))),),
                 (), (), (), (), coefficient_field=field)


def atlas(chart, dimension):
    while True:
        try:
            primes = absolute_primes(chart, chart.components[0][3], dimension)
        except ExtendSectionField as exc:
            chart, = extend_section_field([chart], exc.args[0])
            continue
        if primes is None:
            raise AssertionError("An equidimensional ideal was left unresolved")
        return chart, primes


def singular(code, seconds=120):
    proc = subprocess.run(["Singular", "-q", "--no-rc", "--random=13", "--ticks-per-sec=1000000"],
                          input=code + "\nquit;\n", capture_output=True, text=True, check=True, timeout=seconds)
    text = proc.stdout + proc.stderr
    if any(line.lstrip().startswith("?") for line in text.splitlines()):
        raise AssertionError(text)
    return text


def reference(e, trials, *, include_first=False):
    names = list(e.axes)
    equations = list(e.equations)
    if e.marked:
        # absPrimdecGTZ documents that minpoly should be adjoined as an extra variable.
        equations.append("t^2-2")
    code = ['LIB "primdec.lib";', f'ring R=0,({",".join(names)}),dp;',
            f'ideal I={",".join(equations)};']
    for i in range(trials + 1):
        code.extend([f'int start{i}=rtimer;', f'def S{i}=absPrimdecGTZ(I);',
                     f'print("TIME "+string(rtimer-start{i}));', f'setring S{i};',
                     'int howmany=0; int j;',
                     'for(j=1;j<=size(absolute_primes);j++){howmany=howmany+absolute_primes[j][2];}',
                     'print("COUNT "+string(howmany));', f'setring R;kill S{i};'])
    out = singular("\n".join(code))
    counts = [int(x.split()[1]) for x in out.splitlines() if x.startswith("COUNT ")]
    if len(counts) != trials + 1 or len(set(counts)) != 1 or counts[0] % (2 if e.marked else 1):
        raise AssertionError(out)
    times = [int(x.split()[1])/1e6 for x in out.splitlines() if x.startswith("TIME ")]
    if len(times) != trials + 1:
        raise AssertionError(out)
    return counts[0] // (2 if e.marked else 1), times if include_first else times[1:]


def verify(chart, primes, expected):
    """Independent ideal arithmetic in Atlas's returned field, in one chosen embedding."""
    if len(primes) != expected:
        raise AssertionError(f"Singular reports {expected} components, Atlas returned {len(primes)}")
    axes = chart.variables
    field = chart.coefficient_field
    vars_ = [v.name for v in axes]
    coeff = "0"
    if field:
        if axes[-1] != field[0]:
            raise AssertionError("Coefficient field parameter must be last")
        vars_ = vars_[:-1] + ["alpha"]
        coeff = "(0,alpha)"
    geometric = vars_[:-1] if field else vars_
    def text(f):
        return format_poly(dict(sparse_terms(axes, f)), vars_)
    def ideal(entries):
        return "ideal(" + (",".join(map(text, entries)) or "0") + ")"
    code = ['LIB "primdec.lib";', f'ring R={coeff},({",".join(geometric)}),dp;']
    if field:
        code.append(f'minpoly={text(field[1])};')
    code.extend([f'ideal src={ideal(chart.components[0][3])};',
                 'ideal support=std(radical(src));', 'list parts;ideal covered=1;', 'int j,k;',
                 'proc equalIdeal(ideal a,ideal b){return(size(reduce(a,std(b)))==0 && size(reduce(b,std(a)))==0);}'])
    code += [f'parts[{i}]={ideal(p)};' for i, p in enumerate(primes, 1)]
    code.extend(['int ok=1;list dec;',
                 'for(j=1;j<=size(parts);j++){covered=intersect(covered,parts[j]);'
                 'dec=primdecGTZ(parts[j]);if(size(dec)!=1){ok=0;}'
                 'else{if(!equalIdeal(dec[1][2],parts[j])){ok=0;}}'
                 'for(k=1;k<j;k++){if(equalIdeal(parts[j],parts[k])){ok=0;}}}',
                 'if(!equalIdeal(support,covered)){ok=0;}print("VERIFIED "+string(ok));'])
    out = singular("\n".join(code))
    if "VERIFIED 1" not in out.splitlines():
        raise AssertionError(out)


def run_case(e, trials):
    chart = prepare(e)
    count, delays = reference(e, trials)
    results = []
    for i in range(trials+1):
        start = perf_counter()
        extended, primes = atlas(chart, e.dimension)
        elapsed = perf_counter()-start
        if i:
            results.append(round(elapsed, 7))
        verify(extended, primes, count)
    return dict(case=e.label, variables=list(e.axes), generators=list(e.equations),
                base_mark="t^2-2, root 1 (FLINT ordering)" if e.marked else "QQ", dimension=e.dimension,
                components=count, common_field_degree=extended.coefficient_field[2] if extended.coefficient_field else 1,
                membership_distinctness_primality_and_radical_intersection_checked=True,
                atlas_seconds=results, singular_seconds=delays, atlas_median=statistics.median(results),
                singular_median=statistics.median(delays))


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output", required=True, type=Path)
    args.add_argument("--trials", type=int, default=3)
    args.add_argument("--case", action="append", default=[])
    opt = args.parse_args()
    if opt.trials < 2:
        args.error("At least two trials are required")
    cpu = platform.machine()
    info = Path("/proc/cpuinfo")
    if info.is_file():
        cpu = next((s.split(":", 1)[1].strip() for s in info.read_text().splitlines()
                    if s.startswith("model name")), cpu)
    result = {"cpu": cpu, "singular": subprocess.check_output(["Singular", "--version"],
              text=True).splitlines()[0], "trials": opt.trials,
              "kernel_sha256": sha256(_LIBRARY.read_bytes()).hexdigest(),
              "plane_worker_sha256": sha256(_WORKER.read_bytes()).hexdigest(),
              "conditions": "Warm Atlas worker and Singular interpreter, exact characteristic 0, one warm-up "
                 "discarded per system. Atlas elapsed wall seconds include field construction, extension and "
                 "every individual component ideal in one common marked field. Singular rtimer seconds for "
                 "absPrimdecGTZ exclude process startup and return one representative per conjugacy class. "
                 "For marked inputs we adjoin the original minimal polynomial and divide the absolute count by "
                 "its degree, as each embedding has the same count. Outside timing, Singular checks that "
                 "returned ideals are pairwise distinct and prime over the declared common field and that "
                 "their intersection equals the radical of the input on the chosen embedding. Together with "
                 "the independent absolute component count, this certifies geometric primality.", "cases": []}
    for e in corpus():
        if opt.case and e.label not in opt.case:
            continue
        result["cases"].append(run_case(e, opt.trials))
        print(e.label, result["cases"][-1]["atlas_median"], result["cases"][-1]["singular_median"], flush=True)
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
