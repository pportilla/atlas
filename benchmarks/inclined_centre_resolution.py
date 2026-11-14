"""Compare complete embedded resolutions of three inclined Whitney germs and an axis-aligned control.

Each fresh Atlas CLI prints the resolution JSON, optionally with both zetas.
Each fresh Singular process runs default or checked resolve.lib on the same
polynomial over QQ and prints the number of charts. Outside the timers,
Singular checks every Atlas chart and its own two maps of the blowup.
Replay both zeta functions from the saved Atlas charts alone.
"""
from copy import deepcopy
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from plane_conjugate_resolution import measure
from sheared_centre_resolution import atlas_geometry
from whitney_related import ROOT, arguments, require_singular, run, singular_script


@dataclass(frozen=True)
class InclinedCentre:
    variables: str
    tangent: str
    partner: str
    parameter: str
    pivots: tuple[str, str]

    @property
    def polynomial(self):
        return f"({self.tangent})^2-{self.parameter}*({self.partner})^2"


CASES = {
    "inclined": InclinedCentre("xyz", "x+y+z", "x-y", "z", ("2*x+z", "2*y+z")),
    "rational": InclinedCentre("xyz", "x/2+2*y/3-z/5", "3*x/4-y/5", "z", ("15*x-z", "4*y-z")),
    "permuted": InclinedCentre("yzx", "x+y+z", "y-z", "x", ("2*y+x", "2*z+x")),
    "unsheared": InclinedCentre("xyz", "x", "y", "z", ("x", "y")),
}


def singular_fibre(case, singular, timeout):
    p, q = case.pivots
    script = singular_script(case).removesuffix("quit;\n") + f"""
poly a={case.tangent};poly b={case.partner};poly t={case.parameter};
poly p={p};poly q={q};
ideal centre=std(ideal(p,q));ideal pencil=std(ideal(a,b));
int independent=(deg(det(jacob(ideal(a,b,t))))==0 && det(jacob(ideal(a,b,t)))!=0);
print("CENTRE "+string(independent && size(reduce(centre,pencil))==0 && size(reduce(pencil,centre))==0));
int i,j,k,ok,pivot;
for(i=1;i<=size(resolved[1]);i++){{
    def C=resolved[1][i];setring C;
    ideal image=BO[5];map phi=source,image;poly exc=BO[4][1][1];poly h=BO[2][1];
    poly pull=phi(original);poly total=exc^2*h;
    ok=(nvars(basering)==3 && size(BO[1])==0 && size(BO[2])==1 && size(BO[4])==1);
    ok=ok && pull!=0 && total!=0 && leadcoef(pull)*total==leadcoef(total)*pull;
    poly jac=det(jacob(image));ok=ok && jac!=0 && exc!=0 && leadcoef(jac)*exc==leadcoef(exc)*jac;
    poly cp=phi(p);poly cq=phi(q);poly ratio=var(3);poly axial=phi(t);
    pivot=0;
    if(cp*ratio==cq && cp!=0 && leadcoef(cp)*exc==leadcoef(exc)*cp){{pivot=1;}}
    if(cq*ratio==cp && cq!=0 && leadcoef(cq)*exc==leadcoef(exc)*cq){{pivot=2;}}
    ideal frame=exc,axial,ratio;poly frameJac=det(jacob(frame));
    ok=ok && deg(exc)==1 && deg(axial)==1 && deg(frameJac)==0 && frameJac!=0;
    ideal fibre=std(image);ideal line=std(ideal(exc,axial));
    ok=ok && size(reduce(fibre,line))==0 && size(reduce(line,fibre))==0;
    poly first=phi(a)/exc;poly onFibre=reduce(first,fibre);poly restriction=reduce(h,fibre);
    ok=ok && exc*first==phi(a) && onFibre!=0 && deg(onFibre)<=1 && restriction!=0;
    ok=ok && leadcoef(restriction)*onFibre^2==leadcoef(onFibre)^2*restriction;
    ideal cut=exc,h;ideal bad=std(fibre+cut+minor(jacob(cut),2));
    print("PIVOT "+string(pivot)+" "+string(ok && bad[1]==1));
    setring source;kill C;
}}
quit;
"""
    _, raw = measure(singular, ROOT, timeout, script)
    require_singular(raw, True)
    checks = sorted(line for line in raw.splitlines() if line.startswith(("CENTRE ", "PIVOT ")))
    if checks != ["CENTRE 1", "PIVOT 1 1", "PIVOT 2 1"]:
        raise ValueError(f"Singular did not verify the centre, blowup maps and fibre.\n{raw[-2500:]}")
    # The reciprocal ratios give the standard projective line, and the strict
    # transform cuts it in the square of one nonzero projective linear form.
    return dict(centre_generators=case.pivots, centre_checks=checks, origin_fibre="P^1",
                strict_fibre_equation=f"({case.tangent})^2 in the homogeneous centre coordinates",
                exceptional_N_nu=[2, 2], strict_N_nu=[1, 1],
                euler_exceptional_open=1, euler_exceptional_strict=1,
                monodromy="1/(1-T^2)", milnor_fibre_euler=2, topological="(s + 2)/(2*(s + 1)^2)")


def compact_report(args):
    target = args.output
    with tempfile.TemporaryDirectory() as directory:
        args.output = Path(directory) / "full.json"
        run(args, cases=CASES, fibre_oracle=singular_fibre, chart_oracle=atlas_geometry, driver=__file__,
            chart_changes=True, enforce_equal=True)
        report = json.loads(args.output.read_text())
    for result in report["cases"].values():
        records = result["agreed_records"]
        result["agreed_records"] = {mode: records["before_"+mode] for mode in ("geometry", "zetas")}
        charts = result["agreed_records"]["geometry"]["atlas"]
        result["chart_count"] = len(charts)
        result["removed_open_count"] = sum(len(chart["removed"]) for chart in charts)
    if "inclined" in report["cases"]:
        record = report["cases"]["inclined"]["agreed_records"]["geometry"]
        report["chart_tampering_controls"] = check_tampering(record, args.timeout)
    report["benchmark_dependencies_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("sheared_centre_resolution.py", "whitney_related.py", "plane_conjugate_resolution.py")
    }
    report["cpu_model"] = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                               if line.startswith("model name"))
    content = json.dumps(report, indent=2).encode() + b"\n"
    target.write_bytes(gzip.compress(content, mtime=0) if target.suffix == ".gz" else content)


def check_tampering(record, timeout):
    # On canonical chart 9 the strict and exceptional components fail to meet
    # transversely on a line excluded by the second removed ideal.
    expected = {}
    for name in ("multiplicity", "nu", "strict", "projection", "removed_open"):
        altered = deepcopy(record)
        chart = altered["atlas"][8]
        strict, exceptional = chart["components"]
        if name in ("multiplicity", "nu"):
            exceptional[name] += 1
        elif name == "strict":
            strict["ideal"][0] += "+1"
        elif name == "projection":
            chart["map"]["x"] += "+1"
        else:
            chart["removed"].pop()
        try:
            atlas_geometry(altered, CASES["inclined"], [shutil.which("Singular"), "-q", "--no-rc"], timeout)
        except ValueError:
            expected[name] = {"chart": 9, "status": "rejected"}
        else:
            raise ValueError(f"Independent chart checks accepted the altered {name}")
    return expected


if __name__ == "__main__":
    compact_report(arguments(CASES, __doc__))
