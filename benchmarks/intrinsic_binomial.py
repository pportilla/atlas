"""Compare intrinsic cusp cylinders over QQ with both charts of Singular's blowup."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SINGULAR = """
LIB "resolve.lib";
ring R=0,(x,y,z),dp;
int d=DEGREE;
poly f=x^d-y^(d+1);
ideal center=radical(slocus(ideal(f)));
ideal ambient;
list boundary;
intvec v;
intvec w=-1;
matrix intersections;
list BO=ambient,ideal(f),v,boundary,maxideal(1),v,w,intersections;
list charts=blowUpBO(BO,center,2);
int mask=0;
int ok;
for(int i=1;i<=size(charts);i++){
    def C=charts[i];setring C;
    poly h=BO[2][1];poly e=BO[4][1][1];
    poly x0=BO[5][1];poly y0=BO[5][2];poly z0=BO[5][3];
    poly total=x0^d-y0^(d+1);
    ok=(size(BO[1])==0 && nvars(C)==3 && z0==var(2));
    ok=ok && ((total==e^d*h) || (total==-e^d*h));
    ok=ok && (std(jacob(h)+ideal(h))[1]==1);
    if(ok && y0==e && x0==e*var(3)){
        ideal section=std(radical(ideal(h,e)));
        ideal line=std(ideal(e,var(3)));
        if(size(reduce(section,line))==0 && size(reduce(line,section))==0){mask=mask+1;}
    }
    if(ok && x0==e && y0==e*var(3) && std(ideal(h,e))[1]==1){mask=mask+2;}
    setring R;kill C;
}
print("CHECK "+string(size(charts))+" "+string(mask));
quit;
"""


SHEARED = "(x+y+z)^2-(x-y)^3"
SHEARED_SINGULAR = """
LIB "resolve.lib";
ring R=0,(x,y,z),dp;
poly f=(x+y+z)^2-(x-y)^3;
ideal center=std(radical(slocus(ideal(f))));
ideal expected=std(ideal(x+y+z,x-y));
int ok=(size(reduce(center,expected))==0 && size(reduce(expected,center))==0);
ideal ambient;list boundary;intvec v;intvec w=-1;matrix intersections;
list BO=ambient,ideal(f),v,boundary,maxideal(1),v,w,intersections;
list charts=blowUpBO(BO,center,2);
for(int i=1;i<=size(charts);i++){
    def C=charts[i];setring C;
    poly h=BO[2][1];poly e=BO[4][1][1];
    poly x0=BO[5][1];poly y0=BO[5][2];poly z0=BO[5][3];
    poly total=(x0+y0+z0)^2-(x0-y0)^3;
    poly pulled=e^2*h;
    ok=ok && (size(BO[1])==0 && nvars(C)==3 && z0==var(2));
    ok=ok && (total*leadcoef(pulled)==pulled*leadcoef(total));
    ok=ok && (std(jacob(h)+ideal(h))[1]==1);
    ideal section=std(radical(ideal(h,e)));
    ideal line=std(ideal(e,var(3)+1));
    ok=ok && (size(reduce(section,line))==0 && size(reduce(line,section))==0);
    setring R;kill C;
}
print("CHECK "+string(size(charts))+" "+string(ok));
quit;
"""


def run(degree, system):
    if system == "Singular":
        command = ["Singular", "-q", "--no-rc"]
        source = SHEARED_SINGULAR if degree is None else SINGULAR.replace("DEGREE", str(degree))
    else:
        polynomial = SHEARED if degree is None else f"x^{degree}-y^{degree+1}"
        command = [ROOT / "atlas", "resolve", polynomial, "--vars", "x,y,z",
                   "--mode", "intrinsic", "--compute", "cpu", "--json"]
        if degree is None:
            command.append("--quick")
        source = None
    start = perf_counter()
    output = subprocess.run(command, cwd=ROOT, input=source, text=True, capture_output=True,
                            check=True, timeout=60).stdout
    elapsed = 1000 * (perf_counter() - start)
    if system == "Singular":
        assert output.strip() == ("CHECK 2 1" if degree is None else "CHECK 2 3"), output
    else:
        record = json.loads(output)
        assert record["status"] == "resolved" and len(record["steps"]) == 1 and len(record["charts"]) == 2
        if degree is None:
            assert record["engine"] == "canonical-blowups"
            assert record["steps"][0]["center_charts"][0]["ideal"] == ["x+1/2*z", "y+1/2*z"]
        else:
            assert record["steps"][0]["image_zero_coordinates"] == ["x", "y"]
        assert record["dual_complex"]["status"] == "complete" and not record["dual_complex"]["cells"]
        vertex, = record["dual_complex"]["vertices"]
        assert vertex["dimension"] == 1
        if degree is not None:
            assert vertex["ambient_order"] == degree
        else:
            assert all([e["total_multiplicity"] for e in chart["exceptional"]] == [2] for chart in record["charts"])
    return elapsed


def compare_fan(degree, system):
    from engine.binomial_fan import build_binomial_fan
    from engine.resolve import resolve

    def embedded_fan(*args, **options):
        return build_binomial_fan(*args, **dict(options, intrinsic=False))

    planner = build_binomial_fan if system == "intrinsic fan" else embedded_fan
    request = dict(polynomial=f"x^{degree}-y^{degree+1}", variables=list("xyz"), mode="intrinsic",
                   zetas=[], compute="cpu")
    with patch("engine.binomial_backend.build_binomial_fan", new=planner):
        start = perf_counter()
        record = resolve(request)
        elapsed = 1000 * (perf_counter() - start)
    assert record["status"] == "resolved" and record["dual_complex"]["status"] == "complete"
    assert [v["dimension"] for v in record["dual_complex"]["vertices"]] == [1]
    assert not record["dual_complex"]["cells"]
    assert len(record["steps"]) == (1 if system == "intrinsic fan" else degree+1)
    return elapsed


def compare_canonical(system):
    from engine.origin_backend import _resolve, certified_outputs
    from engine.polynomial import parse
    names = list("xyz")
    poly = parse(SHEARED, names)
    # Keep exactly the same centres, integration setting and final verifier.
    # Skipping only the intrinsic stopping test reproduces the previous path.
    old = system == "embedded stopping"
    with patch("engine.local_blowups._smooth_strict", return_value=False) if old else nullcontext():
        start = perf_counter()
        record = _resolve(poly, names, "intrinsic", seconds=None, defer_outputs=True)
        _, outputs = certified_outputs(record, names, "intrinsic", False)
        elapsed = 1000 * (perf_counter() - start)
    assert len(record["steps"]) == (3 if old else 1)
    dual = outputs["dual_complex"]
    assert dual["status"] == "complete" and not dual["cells"]
    assert [v["dimension"] for v in dual["vertices"]] == [1]
    return elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compare-canonical", action="store_true", help="time both Atlas canonical stopping rules")
    modes.add_argument("--sheared", action="store_true", help="compare the canonical sheared cusp with Singular")
    modes.add_argument("--compare-fans", action="store_true", help="time both Atlas planners in a warm process")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    for degree in ((None,) if args.sheared or args.compare_canonical else (2, 5, 17)):
        if args.compare_canonical:
            times = {"intrinsic stopping": [], "embedded stopping": []}
        else:
            times = {"intrinsic fan": [], "embedded fan": []} if args.compare_fans else {"Atlas": [], "Singular": []}
        for trial in range(-1, args.trials):
            for system in (times if trial % 2 == 0 else reversed(times)):
                if args.compare_canonical:
                    elapsed = compare_canonical(system)
                else:
                    elapsed = compare_fan(degree, system) if args.compare_fans else run(degree, system)
                if trial >= 0:
                    times[system].append(elapsed)
        for system, samples in times.items():
            case = "sheared cusp" if degree is None else f"d={degree}"
            print(f"{case} {system} median {median(samples):.2f} ms; trials {[round(t, 2) for t in samples]}")
