"""Time smooth-point order, directional vanishing and support checks against Singular."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from fractions import Fraction
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import native_polynomial as np
from engine.centres import Frame
from engine.native_library import ideal_request
from engine.polynomial import format_poly, sparse_row
from tests.test_resolution_point_order import direction_request, finite_order_cases, order_request


def singular_ring(variables, order):
    names = list(map(str, variables))
    return f'ring R=0,({",".join(names)}),{order};'


def expr(variables, f):
    return format_poly(sparse_row(np.row(variables, f)), list(map(str, variables)))


def chart_source(variables, equations, point, poly, cap, direction=None):
    # Pull both equations and f back by the same translation. In Singular
    # this is done inside the timed interval, once per trial.
    declared = ",".join(expr(variables, e) for e in equations) or "0"
    shift = ",".join(f"{v}+({q})" for v, q in zip(variables, point))
    lines = [singular_ring(variables, "ds"), f"ideal E={declared};",
             f"poly f={expr(variables, poly)};", "int start=rtimer;",
             f"map toOrigin=R,{shift};ideal I=toOrigin(E);poly g=toOrigin(f);"]
    if equations:
        lines.append(f'if(size(reduce(I,std(maxideal(1)))) || '
                     f'size(reduce(minor(jacob(I),{len(equations)}),std(maxideal(1))))==0)'
                     '{ERROR("expected smooth point");}')
    if direction:
        coords, axis = direction
        coordinates = ",".join(str(variables[i]) for i in coords)
        lines.append(f'ideal closedPoint=I+ideal({coordinates});'
                     f'if(size(reduce(minor(jacob(closedPoint),{len(variables)}),std(maxideal(1))))==0)'
                     '{ERROR("the specified coordinates are not local parameters");}')
        held = [str(variables[i]) for i in coords if i != axis]
        if held:
            lines.append(f'I=I+ideal({",".join(held)});')
        # In this smooth one-dimensional slice, g lies in m^(cap+1)
        # exactly when all its Taylor coefficients through cap vanish.
        lines.extend([f'int vanishes=(reduce(g,std(I+maxideal({cap+1})))==0);',
                      'int finish=rtimer;print("ANSWER "+string(vanishes));'])
    else:
        lines.extend(['int localOrder=-1;int k;ideal G;',
                      f'for(k=1;k<={cap+1};k++)'
                      '{G=std(I+maxideal(k));if(reduce(g,G)!=0){localOrder=k-1;break;}}',
                      'int finish=rtimer;print("ANSWER "+string(localOrder));'])
    return lines


def support_source(variables, equations, units):
    product = "*".join(f"({expr(variables,u)})" for u in units) or "1"
    return ['LIB "primdec.lib";', singular_ring(variables,"lp"),
            f'ideal I={",".join(expr(variables,e) for e in equations)};poly f={product};',
            'int start=rtimer;list C=minAssGTZ(I);ideal P;int i,j;string rows;string p;poly xj;',
            'for(i=1;i<=size(C);i++){P=std(C[i]);if(reduce(f,P)!=0){'
            'if(dim(P)!=0){ERROR("expected finite points");}p="";'
            'for(j=1;j<=nvars(R);j++){xj=reduce(var(j),P);'
            'if(deg(xj)>0){ERROR("expected rational branch");}if(j>1){p=p+" ";}p=p+string(xj);}'
            'rows=rows+"|"+p;}}',
            'int finish=rtimer;print("ANSWER points"+rows);']


def cases():
    for name, variables, equations, point, poly, order in finite_order_cases():
        cap = max(8,order)
        yield (name, "order", order_request(variables, equations, point, poly, cap),
               chart_source(variables, equations, point, poly, cap), [str(order)])
        if name == "b9_chart_23":
            for axis, certified in ((3,True), (5,False)):
                args = direction_request(variables,equations,point,poly,(3,5),axis,6)
                yield (f"{name}_direction_{variables[axis]}", "direction", args,
                       chart_source(variables,equations,point,poly,6,((3,5),axis)), [str(int(certified))])
    x,y = variables = np.symbols("x y")
    for name, equations, units, points in (
            ("translated_linear_point", (x+y-1,x-y-3), (), ["2 -1"]),
            ("nonreduced_two_points", ((x-2)**2*(x+3),y-2*x), (), ["-3 -6","2 4"]),
            ("open_removes_one_point", ((x-2)**2*(x+3),y-2*x), (x+3,), ["2 4"])):
        header = f"frame_points {len(variables)} {len(equations)} {len(units)}"
        yield (name, "support", (header,[np.row(variables,e) for e in (*equations,*units)]),
               support_source(variables,equations,units), ["points",*points])


def measure(case, trials):
    name, operation, native, procedure, expected = case
    script = '\n'.join(['system("--ticks-per-sec",1000000);', *procedure,
                        'print("US "+string(finish-start));quit;'])
    samples = {"Atlas": [], "Singular": []}

    def atlas():
        start = perf_counter_ns()
        answer = ideal_request(*native)
        samples["Atlas"].append((perf_counter_ns()-start)/1e6)
        if operation == "direction":
            answer = [{"yes":"1","unknown":"0"}[answer[0]]]
        if operation == "support":
            answer = [answer[0], *sorted(answer[1:])]
        if answer != expected:
            raise AssertionError((name, answer, expected))

    for trial in range(trials):
        if trial % 2 == 0:
            atlas()
        run = subprocess.run(["Singular","--no-rc","-q"],input=script,text=True,
                             capture_output=True,timeout=60,check=True)
        output = run.stdout.splitlines()
        if len(output) != 2 or not output[0].startswith("ANSWER ") or not output[1].startswith("US "):
            raise RuntimeError((name,run.stdout,run.stderr))
        answer = output[0][len("ANSWER "):].split("|")
        if operation == "support":
            answer = [answer[0], *sorted(answer[1:])]
        if answer != expected:
            raise AssertionError((name,answer,expected))
        samples["Singular"].append(int(output[1][len("US "):])/1000)
        if trial % 2:
            atlas()
    return {"case":name, "operation":operation, "answer":expected, "request":native[0],
            "native_packet_sha256":hashlib.sha256(("\n".join([native[0],*native[1]])+"\n").encode()).hexdigest(),
            "samples_ms":samples, "median_ms":{k:median(v) for k,v in samples.items()}}


def chain_cases():
    x, y, z, u = np.symbols("x y z u")
    yield ("contact_derivatives", (x,y,z), (x,y), (z+z*z-x*y,), z**4+x*y*z, x, 5)
    yield ("translated_contact_derivatives", (x,y,z), (x,y),
           ((z+1)+(z+1)**2-(x-Fraction(3,2))**2-(y+3)**3,), (z+1)**3*(x+1)+(y+3)**5, y, 4)
    yield ("coupled_contact_derivatives", (u,x,z,y), (x,y),
           (u+z+x*u*z-y, u-z+u**2-x*y), u**3*z+x**4-y**4, x, 4)
    name, variables, equations, _, f, order = next(case for case in finite_order_cases()
                                                  if case[0] == "b9_chart_23")
    # The b-direction has a nonzero sixth Taylor coefficient. Its polynomial
    # derivative chain exists on the frame, even though b=0 is excluded by
    # the exceptional open condition for choosing this contact direction.
    yield (name+"_derivative_b", variables, (variables[3],variables[5]),
           equations, f, variables[5], order)


def chain_source(variables, coords, equations, f, axis, degree):
    # Independently construct the scaled implicit field by complementary
    # minors of the Jacobian of the original frame equations.
    dependent = tuple(v for v in variables if v not in coords)
    columns = (*dependent,axis)
    lines = ['system("--ticks-per-sec",1000000);', singular_ring(variables,"dp"), 'option(redSB);',
             f'ideal E={",".join(expr(variables,g) for g in equations)};poly h={expr(variables,f)};',
             'int start=rtimer;ideal G=std(E);']
    for j in range(len(columns)):
        entries = [f'diff(E[{i+1}],{v})' for i in range(len(equations))
                   for k,v in enumerate(columns) if k!=j]
        r = len(equations)
        sign = '-' if (r+j)%2 else ''
        lines += [f'matrix J{j}[{r}][{r}]={",".join(entries)};',
                  f'poly A{j}={sign}det(J{j});']
    deriv = "+".join(f"A{i}*diff(h,{v})" for i,v in enumerate(columns))
    return [*lines, 'list q=h;int k;',
            f'for(k=0;k<{degree};k++){{h=reduce({deriv},G);q[size(q)+1]=h;}}',
            'int finish=rtimer;']


def derivative_chains(trials):
    for name, variables, coords, equations, f, axis, degree in chain_cases():
        def atlas():
            start = perf_counter_ns()
            frame = Frame(variables, coords, equations)
            chain = frame.derivative_chain(f, axis, degree)
            return chain, (perf_counter_ns()-start)/1e6

        reference, _ = atlas()
        # Compare every rational coefficient and monomial after timing; both
        # sides use the reduced grevlex basis and the same ambient coordinates.
        test_rows = [f'poly r{i}={expr(variables,v)};' for i,v in enumerate(reference)]
        checks = [f'if(q[{i+1}]!=r{i}){{ERROR("different derivative");}}' for i in range(degree+1)]
        script = '\n'.join([*chain_source(variables,coords,equations,f,axis,degree), *test_rows, *checks,
                            'print("US "+string(finish-start));quit;'])
        samples = {"Atlas":[], "Singular":[]}
        for trial in range(trials):
            if trial % 2 == 0:
                result, elapsed = atlas()
            run = subprocess.run(["Singular","--no-rc","-q"],text=True,input=script,
                                 capture_output=True,timeout=120,check=True)
            if not run.stdout.startswith("US ") or len(run.stdout.splitlines()) != 1:
                raise RuntimeError((name,run.stdout[:1000],run.stderr[:1000]))
            samples["Singular"].append(int(run.stdout.strip()[3:])/1000)
            if trial % 2:
                result, elapsed = atlas()
            if result != reference:
                raise AssertionError(name)
            samples["Atlas"].append(elapsed)
        sparse_rows = [np.row(variables,g) for g in (*equations,f)]
        yield {"case":name, "operation":"chart_derivative_chain", "degree":degree,
               "variables":list(map(str,variables)), "free_parameters":list(map(str,coords)),
               "direction":str(axis), "input_terms":[r.count(";")+1 for r in sparse_rows],
               "input_sha256":hashlib.sha256(("\n".join(sparse_rows)+"\n").encode()).hexdigest(),
               "answer_terms":[len(np.terms(variables,g)) for g in reference],
               "answer_sha256":hashlib.sha256(("\n".join(np.row(variables,g) for g in reference)+"\n").encode()).hexdigest(),
               "agreement":"All global polynomial derivatives through the stated degree agree coefficient by coefficient.",
               "timing":"Fresh frame basis and signed Jacobian minors, iterated differentiation and normal form. Inputs are loaded in both processes; Atlas includes conversion of returned rows into frame expressions. Comparison and process startup excluded.",
               "samples_ms":samples, "median_ms":{k:median(v) for k,v in samples.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=7)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    data = {"conditions":"Same QQ inputs. Atlas includes decoding sparse input and returning a certificate. Singular enters with the ring and original polynomials loaded, translates them to the point, checks smoothness and computes local standard bases modulo successive powers of the maximal ideal. For directions it verifies the specified parameters and fixes all but the requested one. On support examples it computes minimal primes, filters at the open, and extracts rational coordinates. Process launch is excluded. Trial order alternates.",
            "native_library_sha256":hashlib.sha256((ROOT/"native/build/libgroebner.so").read_bytes()).hexdigest(),
            "cpu_affinity":sorted(os.sched_getaffinity(0)),
            "singular":subprocess.check_output(["Singular","--version"],text=True).splitlines()[0],
            "trials":args.trials, "cases":[*(measure(case,args.trials) for case in cases()),
                                            *derivative_chains(args.trials)]}
    args.output.write_text(json.dumps(data,indent=2)+"\n")


if __name__ == "__main__":
    main()
