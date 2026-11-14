"""Time equal prime-localization computations in Atlas and Singular and check their contractions."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.normalization import comparison, ideal
from benchmarks.primary_decomposition import cpu_name
from engine import native_polynomial as np
from engine.native_library import affine_request
from engine.polynomial import parse_expression


@dataclass(frozen=True)
class Case:
    name: str
    variables: tuple
    generators: tuple
    prime: tuple
    numerator: str | None = None
    denominator: str = "1"
    order: str = "grevlex"


def corpus():
    # Fixed before timing either system. Factors visible in these strings are
    # expanded to ordinary polynomials by both engines prior to warm timing.
    cusp = "y^2-x^3"
    curve = ("y^2-x*z","y*z-x^3","z^2-x^2*y")
    prime = ("z-x*y","y^2-x^3")
    cusp3 = "(b^2-a^3)"
    A, B, C = "(x+2*y+z)", "(y-z)", "(x-3*z)"
    return (
        Case("zero_at_generic_point", ("x","y"), (), (), "x", "y"),
        Case("nonzero_at_generic_point", ("u","v"), ("u^2+v^3",), (), "0", "u+v"),
        Case("unit_outside_prime", ("x","y"), ("x*y","y+1"), ("x","y"), "x+2", "1+y"),
        Case("principal_curve", ("x","y"), (f"({cusp})^3*(y+x+1)^2",), (cusp,), cusp, "x"),
        Case("embedded_discarded", ("x","y"), ("x^3","x^2*y"), ("x",), "x^2", "y"),
        Case("embedded_retained", ("x","y"), ("x^3","x^2*y"), ("x","y"), "x^2", "1-x"),
        Case("three_axes", ("x","y","z"), ("x*y","x*z","y*z"), ("x","y"), "x+y", "z"),
        Case("higher_monomial", ("a","b","c","d"), ("a^3*c^2","a*b*d","b^2*c*d^2","a^4*b"),
             ("a","b","c"), "a*b", "d"),
        Case("split_rational_points", ("x","y"), ("x^2*(x-1)^3","y-x"), ("x","y"), "x^2", "x-1", "lex"),
        Case("conjugate_point", ("u","v"), ("(u^2+1)^2*(u-1)","v-u"), ("u^2+1","v-u"),
             "u^2+1", "u-1", "lex"),
        Case("movable_prime", ("u","x","y"), ("(x^2-u)*(x+3)", "(y-u)*(x+3)"), ("y-u","x^2-u"),
             "y-x^2", "x+3"),
        Case("nonlinear_height_two", ("x","y","z"), tuple(f"({f})*(x+1)" for f in prime), prime,
             "z^2-x^5","x+1"),
        Case("space_curve_prime", ("x","y","z"), tuple(f"({f})*(x-y+1)" for f in curve), curve,
             "x*z-y^2", "x-y+1"),
        Case("intersecting_curves", ("x","y","z"), ("x*y","z*x","z*y","z^2"), ("x","y","z"),
             "z", "x+1"),
        Case("finite_thick_fiber", ("x","y"), ("(x^2+1)^2","(y^2+1)^2"), ("x^2+1","y-x"),
             "(y-x)^2", "x+y"),
        Case("surface_and_embedded", ("a","b","c"), ("a^2*(b-a)^2","a*(b-a)*c","a*(b-a)^3"),
             ("b-a","c"), "b-a", "a", "lex"),
        # Retained after profiling the general algorithm.
        Case("hidden_primary_unit", ("a","b","c"),
             (f"{cusp3}^2*(a+1)+{cusp3}*c*(b+1)", f"{cusp3}^2*(a+1)-(a-1)*{cusp3}*c*(b+1)"),
             (cusp3,"c"), cusp3, "a"),
        Case("two_thick_field_orbits", ("a","b","c"), ("(a^2+1)^2","(b^2+1)^2","(c^2+1)^2"),
             ("b-a","c+a","a^2+1"), "c+b", "a"),
        Case("height_one_no_fraction", ("a","b","c"),
             (f"{cusp3}^3*(b-c)", f"{cusp3}^2*(c-a)", f"{cusp3}*(a^2+b^2+c^2)"), (cusp3,)),
        Case("dense_linear_change", ("x","y","z"),
             (f"{A}^2*({C}+1)+{B}^2*({C}-2)", f"{A}^2*({C}+1)*({C}+2)+{B}^2*({C}-2)*({C}+3)"),
             (A,B),f"{A}*{B}",C),
        Case("complex_curve_family", ("x","y","z","w"),
             ("(x^2+1)^2-(y-z)^2","x^2*y+y-z^2","x*y*w+z"),
             ("x^2+1","y","z","w"), "(x-y)*w", "x-1"),
        Case("artinian_translation", ("x","y","z"),
             tuple(f"({f})*({g})" for f in ("(x-1)^3","y^3","z^2") for g in ("(x+2)^2","y^2+z","z^2")),
             ("x-1","y","z"), "x-1", "x+2"),
    )


def native_input(c):
    symbols = tuple(map(np.Symbol,c.variables))
    def row(f):
        return np.row(symbols, np.from_terms(symbols, parse_expression(f, c.variables)))
    has_query = c.numerator is not None
    rows = (*c.generators, *c.prime, *((c.numerator,c.denominator) if has_query else ()))
    return ["localization"," ".join(c.variables),c.order,
            f"{len(c.generators)} {len(c.prime)} {int(has_query)}",*(row(f) for f in rows)]


def localize(c):
    line, = affine_request(native_input(c))
    return json.loads(line)


def declarations(c):
    order = {"grevlex":"dp","lex":"lp"}[c.order]
    return (f'ring R=0,({",".join(c.variables)}),{order};\n'
            f'ideal I={ideal(c.generators)}; ideal P={ideal(c.prime)};\n'
            f'poly f={c.numerator or "0"}; poly s={c.denominator};\n')


# Ishihara's 2021 locPrime library returns either a contracted ideal or a
# list of primary components. Keep it optional so no network is needed when
# only the two installed primdec.lib methods are desired.
LOCPRIME_REV = "5624175785c7982358d9d31f89d29345bad1680b"


def procedures(method, locprime=None):
    # Each timed routine validates P. A proper linear ideal is prime, and a
    # principal generator can be checked by ordinary factorization in the UFD.
    decompose = "primdecSY" if method == "SY" else "primdecGTZ"
    certify = f"""
proc contained(ideal A, ideal G) {{ return(size(reduce(A,G))==0); }}
proc properPrime(ideal G) {{
  if (size(G)==0) {{return(1);}}
  if (deg(G[1])==0) {{return(0);}}
  if (deg(G)==1) {{return(1);}}
  if (size(G)==1) {{
    list F=factorize(G[1]); return(size(F[1])==2 && F[2][2]==1);
  }}
  if(!defined({decompose})) {{ LIB "primdec.lib"; }}
  list D={decompose}(G);
  if(size(D)!=1) {{return(0);}}
  return(contained(D[1][2],G));
}}
"""
    if method == "locPrime":
        contraction = f'if(!defined(locPrime)){{ LIB "{locprime}"; }}' + """
    def D=locPrime(I,G);
    if (typeof(D)=="ideal") { J=D; }
    else { for(i=1;i<=size(D);i++){J=intersect(J,D[i]);} }"""
    else:
        contraction = f"""
    if(!defined({decompose})) {{ LIB "primdec.lib"; }}
    list D={decompose}(I);
    for(i=1;i<=size(D);i++) {{
      if (contained(D[i][2],G)) {{J=intersect(J,D[i][1]);}}
    }}"""
    return certify + """
proc localization(ideal I, ideal P, poly f, poly s, int query) {
  option(redSB); ideal G=std(P);
  if(!properPrime(G)){ERROR("P must be prime");}
  ideal J=1; int i;
  if(size(I)==0){J=0;}
  else {if(contained(I,G)){
    if(size(G)==1) {
      // At a height-one prime in a UFD the contraction is (p^min ord_p(g)).
      poly h=0; for(i=1;i<=size(I);i++){h=gcd(h,I[i]);}
      poly localFactor=1;
      while(h%G[1]==0){localFactor=localFactor*G[1]; h=h/G[1];}
      J=ideal(localFactor);
    }
    else {
      if(size(I)==1) {
        list F=factorize(I[1]); poly part=1;
        for(i=2;i<=size(F[1]);i++){
          if(reduce(F[1][i],G)==0){part=part*F[1][i]^F[2][i];}
        }
        J=ideal(part);
      }
      else {
        int isMonomial=1;
        for(i=1;i<=size(I);i++){if(size(I[i])>1){isMonomial=0; break;}}
        if(isMonomial){
          J=I;
          for(i=1;i<=nvars(basering);i++){
            if(reduce(var(i),G)!=0){J=subst(J,var(i),1);}
          }
        } else {
          if(dim(G)==0 && dim(std(I))==0){
            // In an Artinian quotient all associated primes are maximal.
            // The first saturation removes the P-primary component and
            // the second removes exactly its complement.
            if(!defined(sat)){ LIB "elim.lib"; }
            J=sat(I,sat(I,G));
          } else {""" + contraction + """
          }
        }
      }
    }
  }}
  J=std(J);
  if(!query){return(list(J,G));}
  if(reduce(s,G)==0){ERROR("denominator in prime");}
  return(list(J,G,f,s,int(reduce(f,J)==0),int(reduce(f,G)!=0)));
}
"""


def run_singular(c, method, seed, timeout, warm, locprime, check=None, batch=1):
    source = declarations(c) + procedures(method,locprime)
    call = f'localization(I,P,f,s,{int(c.numerator is not None)})'
    if warm:
        source += (f'list D={call}; int step; int start=rtimer; '
                   f'for(step=0;step<{batch};step++){{D={call};}} '
                   'int elapsed=rtimer-start; print("TIME "+string(elapsed));\n')
    else:
        source += f'list D={call}; print(D);\n'
    if check is not None:
        source += 'LIB "primdec.lib";\n' + make_checks(c, check)
    cmd = ["Singular","-q","--no-rc",f"--random={seed}","--ticks-per-sec=1000000"]
    start = perf_counter_ns()
    p = subprocess.run(cmd,input=source+"quit;\n",text=True,capture_output=True,timeout=timeout,check=True)
    elapsed=(perf_counter_ns()-start)/1e6
    lines = (p.stdout+p.stderr).splitlines()
    if any(x.lstrip().startswith("?") for x in lines) or (check is not None and "VERIFIED 1" not in lines):
        raise ValueError(f"Singular {method} failed on {c.name}: "+p.stdout+p.stderr)
    if warm:
        micros, = [int(x[5:]) for x in lines if x.startswith("TIME ")]
        return micros/1000/batch
    return elapsed


def make_checks(c, result):
    if result["ring"] != {"variables":list(c.variables),"order":c.order,"field":"QQ"} or result["type"]!="localization":
        raise ValueError(f"Wrong ring/result type on {c.name}")
    src = (f'ideal A=std({ideal(result["generators"])}); ideal B=std({ideal(result["prime"])});\n'
           'int ok=contained(A,D[1]) && contained(D[1],A) && contained(B,D[2]) && contained(D[2],B);\n'
           # For every returned generator g, membership in IR_P is also
           # equivalent to the colon I:g containing an element outside P.
           'int j; for(j=1;j<=size(A);j++){ if(contained(quotient(I,ideal(A[j])),B)){ok=0;}}\n'
           # Conversely, if I is contained in A and all associated primes of
           # A are contained in P, A is saturated by the complement of P.
           'if(!contained(I,A)){ok=0;}\n'
           'if(size(A)>0 && deg(A[1])>0){list associated=primdecSY(A); '
           'for(j=1;j<=size(associated);j++){if(!contained(associated[j][2],B)){ok=0;}}}\n')
    if c.numerator is not None:
        e = result["element"]
        src += (f'if(f!=({e["numerator"]}) || s!=({e["denominator"]})){{ok=0;}}\n'
                f'if(D[5]!={int(e["member"])} || D[6]!={int(e["unit"])}){{ok=0;}}\n'
                f'if(int(!contained(quotient(I,ideal(f)),B))!={int(e["member"])}){{ok=0;}}\n')
    elif "element" in result:
        raise ValueError("Unexpected query result")
    return src+'print("VERIFIED "+string(ok));\n'


def run_cli(c, timeout, python_cli):
    cmd = ([sys.executable] if python_cli else []) + [
        str(ROOT/"atlas"),"algebra","localization","--vars",",".join(c.variables),"--order",c.order,
        "--compute","cpu","--json"]
    for g in c.generators: cmd += ["--gen",g]
    for g in c.prime: cmd += ["--prime",g]
    if c.numerator is not None: cmd += ["--numerator",c.numerator,"--denominator",c.denominator]
    start = perf_counter_ns()
    p = subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=True)
    return (perf_counter_ns()-start)/1e6,json.loads(p.stdout)["result"]


def prepare_locprime(path, destination, report):
    # Singular 4.2's sat() returned an ideal/exponent pair. The corresponding
    # operation in Singular 4.4 is sat_with_exp().
    original = path.read_bytes()
    source_hash = hashlib.sha256(original).hexdigest()
    if source_hash != "cd02a5f31c5c4fab4b9c8915487250212dbf09febff3d00304cf94992ec79961":
        raise ValueError("Use the pinned revision of Ishihara's locprime.lib")
    patched = destination / "locprime.lib"
    patched.write_bytes(original.replace(b"sat(", b"sat_with_exp("))
    report.update(
        locprime_source_sha256=source_hash,
        locprime_compatibility="Use sat_with_exp for the Singular 4.4 list-returning API",
        locprime_sha256=hashlib.sha256(patched.read_bytes()).hexdigest(),
        locprime_source=f"https://github.com/IshiharaYuki/localprime/blob/{LOCPRIME_REV}/locprime.lib",
    )
    return patched


def time_case(c, args, methods, locprime):
    active_methods = methods.copy()
    record = asdict(c)
    if locprime:
        cutoff = min(args.timeout, 5)
        try:
            run_singular(c, "locPrime", 17, cutoff, False, locprime)
        except subprocess.TimeoutExpired:
            record["locPrime_timeout_seconds"] = cutoff
            active_methods.remove("locPrime")

    data = None if args.singular_only else native_input(c)
    result = None
    for warm, repeats, field in (
        (True, args.repeats, "warm"),
        (False, args.cold_repeats, "fresh_command"),
    ):
        if not repeats:
            continue
        samples = {m: [] for m in active_methods}
        for trial in range(repeats):
            offset = trial % len(active_methods)
            for method in active_methods[offset:] + active_methods[:offset]:
                if method != "atlas":
                    elapsed = run_singular(c, method, 17 + trial, args.timeout, warm,
                                           locprime, batch=args.batch)
                elif warm:
                    affine_request(data)
                    start = perf_counter_ns()
                    for _ in range(args.batch):
                        line, = affine_request(data)
                    elapsed = (perf_counter_ns() - start) / 1e6 / args.batch
                    result = json.loads(line)
                else:
                    elapsed, answer = run_cli(c, args.timeout, args.python_cli)
                    if answer != result:
                        raise ValueError("CLI/kernel disagreement on " + c.name)
                samples[method].append(elapsed)
        record[field] = comparison(samples)
        if "atlas" in active_methods:
            record[field]["wins_vs_all"] = record[field].pop("wins_vs_both")

    if result is not None:
        for method in active_methods:
            if method != "atlas":
                run_singular(c, method, 33, args.timeout, False, locprime, check=result)
        record["verified"] = True
    print(c.name, *(f"{m} {record['warm'][m]['median_ms']:.3f}ms" for m in active_methods), flush=True)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--cold-repeats", type=int, default=7)
    ap.add_argument("--batch", type=int, default=5, help="Full recomputations per warm sample, in each system")
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--case", action="append")
    ap.add_argument("--locprime", type=Path, help="Optional path to Ishihara's pinned locprime.lib")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--singular-only", action="store_true")
    ap.add_argument("--python-cli", action="store_true", help="Use the Python launcher to measure the baseline")
    args = ap.parse_args()
    if args.repeats <= 0 or args.cold_repeats < 0 or args.timeout <= 0 or args.batch <= 0:
        ap.error("Invalid timing options")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    cases = corpus()
    if args.case:
        cases = [c for c in cases if c.name in args.case]
        if {c.name for c in cases} != set(args.case):
            ap.error("Unknown case")

    methods = ["GTZ", "SY"]
    if args.locprime:
        methods.append("locPrime")
    report = {
        "cpu": cpu_name(),
        "cpu_affinity": args.cpu,
        "singular": subprocess.check_output(["Singular", "--version"], text=True).splitlines()[0],
        "configuration": "pre-optimization native library and Python CLI" if args.python_cli else "native library and public CLI",
        "units": "wall-clock milliseconds per computation",
        "repeats": args.repeats,
        "cold_repeats": args.cold_repeats,
        "warm_batch_size": args.batch,
        "scope": (
            "Warm Atlas sparse native request including parsing rows and serializing the answer. Singular starts "
            "with a loaded ring and parsed inputs. Each procedure verifies primality, computes the contraction and "
            "checks the supplied fraction. All Singular variants use zero/unit shortcuts, height-one valuation via "
            "polynomial gcd, factorization of principal inputs, removal of invertible variables from monomial ideals, "
            "and double saturation of zero-dimensional ideals at maximal primes. Singular libraries are loaded only "
            "when required. One discarded warm-up precedes each warm trial. Fresh CLI times include parsing, startup "
            "and complete answer output. Execution order rotates and Singular seeds are 17 + trial index."
        ),
        "methods": methods.copy(),
        "cases": [],
    }
    if not args.singular_only:
        from engine.native_library import _LIBRARY
        paths = {"atlas_library": _LIBRARY, "atlas_launcher": ROOT / "atlas"}
        cli = ROOT / "native/build/localization"
        if cli.exists():
            paths["atlas_native_cli"] = cli
        report.update({k + "_sha256": hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()})
        methods.insert(0, "atlas")

    with tempfile.TemporaryDirectory(prefix="atlas-singular-localization-") as temp:
        locprime = prepare_locprime(args.locprime, Path(temp), report) if args.locprime else None
        for c in cases:
            report["cases"].append(time_case(c, args, methods, locprime))
            if args.output:
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
