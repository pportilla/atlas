"""Compare embedded plane resolution with the identity test over Q(i)."""
import argparse
from contextlib import ExitStack
import os
import shutil
from statistics import median
import sys
from time import perf_counter_ns
from unittest.mock import patch

from resolution_zeta import ROOT, Session

sys.path.insert(0, str(ROOT))
from engine.resolve import resolve
from engine.kernel_json import kernel_json
from engine.polynomial import encode_sparse_row, parse


# Name, ordered variables, input, multiplicity of each transverse branch
# (zero for controls), steps without the conjugate identity trial.
CASES = [
    ("lines", "xy", "x^2+y^2", 1, 1),
    ("rational coefficients", "xy", "(3*x/7+5*y/11)^2+(2*x/13-y/17)^2", 1, 1),
    ("curved branches", "xy", "(y-x^2)^2+x^2", 1, 1),
    ("powers with unit", "xy", "(1+x)*((x-y^2)^2+2*(x-y^2)*y+2*y^2)^3", 3, 1),
    ("irreducible node", "xy", "y^2+x^2*(1+x^2)", 0, 1),
    ("cusp", "xy", "y^2-x^3", 0, 3),
    ("cylinder", "xzy", "x^2+y^2", 1, 1),
]


class SingularCrossing:
    label = "Singular"
    procedure = '''minpoly=i^2+1;
proc crossing(poly f) {
    list fs=factorize(f);
    ideal parts=fs[1]; intvec powers=fs[2];
    poly product=1; ideal active; intvec weights; int n,j;
    for (j=1; j<=size(parts); j++) {
        product=product*parts[j]^powers[j];
        if (jet(parts[j],0)==0) { n++; active[n]=parts[j]; weights[n]=powers[j]; }
    }
    if (product!=f) { ERROR("factor product differs from input"); }
    int branchWeight=0;
    if (n==2) {
        poly gradientMinor=diff(active[1],x)*diff(active[2],y)-diff(active[1],y)*diff(active[2],x);
        if (weights[1]==weights[2] && jet(gradientMinor,0)!=0) { branchWeight=weights[1]; }
    }
    print("ATLASBENCH "+string(branchWeight));
}
print("ATLASBENCH ready");
'''

    def __init__(self, executable, variables):
        self.command = [executable, "-q", "--no-rc"]
        self.setup = f"ring R=(0,i),({','.join(variables)}),dp;\n" + self.procedure

    def job(self, polynomial):
        return f"crossing({polynomial});\n"


def measured_resolution(equation, names, weight, steps, baseline):
    start = perf_counter_ns()
    record = resolve(dict(polynomial=equation, variables=list(names), mode="embedded",
                          search="exhaustive", compute="cpu", zetas=[]))
    elapsed = (perf_counter_ns() - start) / 1e6
    expected = steps if baseline or weight == 0 else 0
    if (record["status"] != "resolved" or len(record["steps"]) != expected or
            record["dual_complex"]["status"] != "complete"):
        raise ValueError(f"Unexpected resolution for {equation}")
    if expected == 0:
        chart, = record["atlas"]
        if (chart["coefficient_field"]["degree"] != 2 or chart["map"] != dict(zip(names, names)) or
                [p["multiplicity"] for p in chart["components"]] != [weight, weight]):
            raise ValueError(f"Unexpected marked identity for {equation}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()
    executable = shutil.which("Singular")
    if not executable or args.trials < 2:
        parser.error("Install Singular and select at least two trials.")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with ExitStack() as stack:
        sessions = {names: stack.enter_context(Session(SingularCrossing(executable, names), env, timeout=30))
                    for names in {case[1] for case in CASES}}
        print("Case | Atlas ms | Atlas without identity trial ms | Atlas criterion ms | Singular criterion ms")
        for name, names, equation, weight, steps in CASES:
            samples = [[], [], [], []]
            for trial in range(args.trials + 1):
                for route in (range(4) if trial % 2 else reversed(range(4))):
                    if route == 0:
                        elapsed = measured_resolution(equation, names, weight, steps, False)
                    elif route == 1:
                        with patch("engine.plane_identity.conjugate_plane_identity", lambda *a, **kw: None):
                            elapsed = measured_resolution(equation, names, weight, steps, True)
                    elif route == 2:
                        start = perf_counter_ns()
                        terms = parse(equation, list(names), bounded=False)
                        found = kernel_json("atlas_conjugate_plane_crossing", "recognize",
                                            width=len(names), polynomial=encode_sparse_row(terms))
                        elapsed = (perf_counter_ns() - start) / 1e6
                        if (int(found["multiplicity"]) if found else 0) != weight:
                            raise ValueError(f"Native crossing criterion disagrees on {name}")
                    else:
                        elapsed, result = sessions[names].warm(equation)
                        if int(result) != weight:
                            raise ValueError(f"Singular disagrees on {name}: {result}")
                    if trial:
                        samples[route].append(elapsed)
            print(f"{name} | " + " | ".join(f"{median(sample):.3f}" for sample in samples))


if __name__ == "__main__":
    main()
