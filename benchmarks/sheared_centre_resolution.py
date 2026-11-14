"""Compare the resolution of A(x,y)^d + z*K(x,y) at the origin for transverse homogeneous pencils.

Fresh Atlas processes return the full embedded resolution, optionally with
both zetas. Fresh Singular processes return the default or checked resolution
and the number of charts. Outside the timer, verify both resolutions in
Singular, including their maps, Jacobians, SNC and origin fibre, and replay the
Atlas zetas from the saved atlas. Pin one CPU when comparing complete records.
Under unrestricted affinity the historical exhaustive search can return the
30-chart canonical resolution before the two-chart graph. Both presentations
have the same invariants. Quick search is measured separately.
"""
from dataclasses import dataclass
from itertools import combinations
import re

from plane_conjugate_resolution import measure
from whitney_related import ROOT, arguments, chart_expression, require_singular, run, singular_script


@dataclass(frozen=True)
class Shear:
    a: str
    b: str | None
    degree: int = 2
    coefficient: str | None = None
    variables: str = "xyz"

    @property
    def polynomial(self):
        if self.coefficient is not None:
            return f"({self.a})^{self.degree}+z*({self.coefficient})"
        return f"({self.a})^{self.degree}-z*({self.b})^{self.degree}"


CASES = {
    "sheared": Shear("x+y", "x-y"),
    "rational": Shear("x/2+2*y/3", "3*x/4-y/5"),
    "cubic": Shear("x+y", "x-y", 3),
    "nonpower": Shear("3*x-5*y", None, coefficient="x^2+3*x*y+2*y^2"),
    "unsheared": Shear("x", "y"),
}


def singular_fibre(case, singular, timeout):
    script = singular_script(case).removesuffix("quit;\n") + f'''
poly tangent={case.a};
int ix,i,j,pivot,ok,matched;
for(ix=1;ix<=size(resolved[1]);ix++){{
    def C=resolved[1][ix];setring C;
    ideal image=BO[5];map phi=source,image;poly exc=BO[4][1][1];poly h=BO[2][1];
    poly full=phi(original);poly expectedPull=exc^{case.degree}*h;
    ok=(nvars(basering)==3 && size(BO[1])==0 && size(BO[4])==1 && size(BO[2])==1);
    ok=ok && full!=0 && expectedPull!=0 && leadcoef(expectedPull)*full==leadcoef(full)*expectedPull;
    matrix M[3][3];for(i=1;i<=3;i++){{for(j=1;j<=3;j++){{M[i,j]=diff(image[i],var(j));}}}}
    poly jac=det(M);ok=ok && jac!=0 && leadcoef(exc)*jac==leadcoef(jac)*exc;
    poly ratio=var(3);pivot=0;
    if(image[1]==exc && image[2]==exc*ratio){{pivot=1;}}
    if(image[2]==exc && image[1]==exc*ratio){{pivot=2;}}
    ideal frame=exc,image[3],ratio;
    for(i=1;i<=3;i++){{matched=0;for(j=1;j<=3;j++){{if(var(i)==frame[j]){{matched++;}}}}ok=ok && matched==1;}}
    ideal fibre=std(image);poly onFibre=reduce(h,fibre);
    poly tangentRatio=phi(tangent)/exc;poly expected=reduce(tangentRatio,fibre)^{case.degree};
    ok=ok && onFibre!=0 && expected!=0 && leadcoef(onFibre)*expected==leadcoef(expected)*onFibre;
    ideal cut=exc,h;ideal bad=std(fibre+cut+minor(jacob(cut),2));
    print("CHART "+string(pivot)+" "+string(ok && bad[1]==1));
    setring source;kill C;
}}
quit;
'''
    _, raw = measure(singular, ROOT, timeout, script)
    require_singular(raw, True)
    if sorted(l for l in raw.splitlines() if l.startswith("CHART ")) != ["CHART 1 1", "CHART 2 1"]:
        raise ValueError(f"Singular failed the direct fibre/Jacobian/transversality check.\n{raw[-2400:]}")
    # Both standard opens glue to P^1. A nonzero linear form has one
    # projective zero over Qbar even if the point is in both affine charts.
    d = case.degree
    return dict(origin_fibre="P^1", fibre_strict_equation=f"({case.a})^{d}",
                exceptional_N_nu=[d, 2], strict_N_nu=[1, 1], euler_exceptional_open=1, euler_exceptional_strict=1,
                monodromy=f"1/(1-T^{d})", milnor_fibre_euler=d,
                topological="(s + 2)/(2*(s + 1)^2)" if d == 2 else f"(s + 2)/((s + 1)*({d}*s + 2))")


def atlas_geometry(record, case, singular, timeout):
    script = ['LIB "elim.lib";',
              'proc saturatedIdeal(ideal i, ideal j){def result=sat(i,j);'
              'if(typeof(result)=="list"){return(result[1]);}return(result);}',
              'ring source=0,(x,y,z),dp;', f'poly original={case.polynomial};']
    expected = []
    pivots = set()
    for index, (chart, public) in enumerate(zip(record["atlas"], record["charts"], strict=True), 1):
        names = chart["coordinates"]
        graph = bool(chart["ambient_ideal"])
        if graph:
            pivot = public["pivot_generator"]
            if pivot not in ("x", "y") or pivot in pivots or names[:3] != list("xyz") or len(names) != 4:
                raise ValueError("Expected the two standard graph opens")
            if chart["map"] != dict(zip("xyz", "xyz")) or public["map"] != chart["map"]:
                raise ValueError("The graph projection is not the original affine projection")
            pivots.add(pivot)
            other = "y" if pivot == "x" else "x"
            if public["projective_coordinates"] != (["1", names[3]] if pivot == "x" else [names[3], "1"]):
                raise ValueError("Incompatible projective graph coordinates")
        elif len(names) != 3:
            raise ValueError("Expected a free affine chart")
        def eq(expression):
            value = chart_expression(expression, names)
            if graph:
                value = re.sub(rf"\b{other}\b", f"({pivot}*{names[3]})", value)
            return value
        if graph:
            # Check the original ambient ideal before eliminating the graph coordinate.
            script.extend([f"ring graph{index}=0,({','.join(names)}),dp;",
                           "ideal actual="+",".join(chart_expression(g,names) for g in chart["ambient_ideal"])+";",
                           f"ideal expected={other}-{pivot}*{names[3]};",
                           'print("AMBIENT '+str(index)+' "+string(size(reduce(actual,std(expected)))==0'
                           ' && size(reduce(expected,std(actual)))==0));'])
        free = [v for v in names if not graph or v!=other]
        if any(not v.isidentifier() for v in free):
            raise ValueError("Invalid chart coordinate name")
        script.append(f"ring local{index}=0,({','.join(free)}),dp;")
        image = ",".join(eq(chart["map"][i]) for i in "xyz")
        script.extend([f"ideal image={image};map phi=source,image;poly pull=phi(original);",
                       "matrix M[3][3];int j,k;for(j=1;j<=3;j++){for(k=1;k<=3;k++){M[j,k]=diff(image[j],var(k));}}",
                       "poly jac=det(M);ideal total=1;ideal volume=1;int ok=1;"])
        divisors = []
        for j, c in enumerate(chart["components"]):
            divisor = f"component{j}"
            divisors.append(divisor)
            script.extend([f"ideal {divisor}=std(ideal({','.join(eq(g) for g in c['ideal'])}));",
                           f"ok=ok && size({divisor})==1 && deg({divisor}[1])>0;",
                           f"total=total*{divisor}^{c['multiplicity']};volume=volume*{divisor}^{c['nu']-1};"])
        script.extend(["ok=ok && size(reduce(total,std(ideal(pull))))==0 && reduce(pull,std(total))==0;",
                       "ok=ok && size(reduce(volume,std(ideal(jac))))==0 && reduce(jac,std(volume))==0;",
                       "ideal owned="+",".join(eq(g) for g in [*chart["fibre"],*chart["region"]])+";",
                       "ideal givenFibre="+",".join(eq(g) for g in chart["fibre"])+";",
                       "ok=ok && size(reduce(givenFibre,std(image)))==0 && size(reduce(image,std(givenFibre)))==0;"])
        for count in range(1,len(divisors)+1):
            for ids in combinations(range(len(divisors)), count):
                key = "_".join(map(str,ids))
                script.append(f"ideal cut{key}="+",".join(divisors[j]+"[1]" for j in ids)+";")
                minors = f"+minor(jacob(cut{key}),{count})" if count<=3 else ""
                script.append(f"ideal bad{key}=std(owned+cut{key}{minors});")
                for removed in chart["removed"]:
                    hole = ",".join(eq(g) for g in removed)
                    script.append(f"bad{key}=saturatedIdeal(bad{key},ideal({hole}));")
                script.append(f"ok=ok && std(bad{key})[1]==1;")
        script.append(f'print("ATLAS {index} "+string(ok));')
        if graph:
            expected.append(f"AMBIENT {index} 1")
        expected.append(f"ATLAS {index} 1")
    if pivots and pivots != {"x","y"}:
        raise ValueError("Incomplete graph cover")
    _, raw = measure(singular, ROOT, timeout, "\n".join([*script, "quit;"]))
    checks = [s for s in raw.splitlines() if s.startswith(("AMBIENT ", "ATLAS "))]
    if checks != expected or any(l.lstrip().startswith("?") for l in raw.splitlines()):
        raise ValueError(f"Independent Atlas chart verification failed.\n{raw[-2400:]}")
    return checks


if __name__ == "__main__":
    args = arguments(CASES, __doc__)
    run(args, cases=CASES, fibre_oracle=singular_fibre, chart_oracle=atlas_geometry,
        driver=__file__, chart_changes=True, enforce_equal=True)
