"""Compare polynomial arithmetic used in resolution and marked chart replay."""

import argparse
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def factor_gcd(exponent):
    split = f"(x+1)*(y^{exponent}+z)"
    common = f"x^{exponent}+x*y^2+z^2"
    atlas = f"""x,y,z=coordinates("x","y","z",coeffs=QQ)
h={common}
p={split}
F=factor(p,order="lex")
product=F.unit
for entry in F.factors:
    product=product*entry.polynomial^entry.multiplicity

print(len(F.factors), product==p, gcd(h*(x+y),h*(x+z),order="lex")==h)
"""
    singular = f"""ring R=0,(x,y,z),lp;
poly h={common}; poly p={split};
list L=factorize(p);
poly product=1; int nonconstant=0;
for(int i=1;i<=size(L[1]);i++){{product=product*L[1][i]^L[2][i];if(deg(L[1][i])>0){{nonconstant++;}}}}
poly g=gcd(h*(x+y),h*(x+z));
print(string(nonconstant)+" "+string(product==p)+" "+string(g/leadcoef(g)==h));
quit;
"""
    return [
        ("Atlas", [ROOT / "atlas", "run", "-"], atlas, "2 True True"),
        ("Singular", ["Singular", "-q", "--no-rc"], singular, "2 1 1"),
    ]


def marked_power(exponent):
    # Both benchmark exponents are 5 modulo 6, so (1+t)^n=-t.
    atlas = f"""from engine.polynomial import parse_expression
answer = parse_expression("(t+1)^{exponent}", ["x", "t"], coefficient_modulus="t^2+t+1")
print(answer == {{(0,1): -1}})
"""
    singular = f"""ring R=(0,t),(x),lp;
minpoly=t^2+t+1;
number a=t+1; number b=1; bigint n={exponent};
while (n>0) {{ if (n%2==1) {{ b=b*a; }} n=n div 2; if (n>0) {{a=a*a;}} }}
print(b==-t);
quit;
"""
    return [
        ("Atlas", [sys.executable, "-"], atlas, "True"),
        ("Singular", ["Singular", "-q", "--no-rc"], singular, "1"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--workload", choices=("factor-gcd", "marked-power"), default="factor-gcd"
    )
    args = parser.parse_args()
    for exponent in (47, 2**64 + 1):
        samples = {"Atlas": [], "Singular": []}
        for trial in range(args.repeats):
            solvers = (factor_gcd if args.workload == "factor-gcd" else marked_power)(
                exponent
            )
            for label, command, source, expected in (
                solvers if trial % 2 == 0 else reversed(solvers)
            ):
                start = perf_counter()
                result = subprocess.run(
                    command,
                    input=source,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=ROOT,
                )
                elapsed = (perf_counter() - start) * 1000
                if (
                    args.workload == "factor-gcd"
                    and label == "Singular"
                    and exponent > 2**64
                    and "`poly` ^ `bigint` failed" in result.stdout
                ):
                    continue
                if (
                    result.returncode
                    or expected not in result.stdout.splitlines()
                    or "   ?" in result.stdout
                ):
                    raise RuntimeError(result.stderr or result.stdout)
                samples[label].append(elapsed)
        for label, times in samples.items():
            detail = (
                f"{[round(t,3) for t in times]} ms, median {median(times):.3f} ms"
                if times
                else "exponent rejected"
            )
            print(exponent, label, detail)


if __name__ == "__main__":
    main()
