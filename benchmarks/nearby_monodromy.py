#!/usr/bin/env python3
"""Compare nearby local zetas with Singular's monodromy on transverse slices."""
import argparse
from collections import Counter
from fractions import Fraction
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

from resolution_zeta import ROOT, Session

sys.path.insert(0, str(ROOT))
from engine.monodromy import _nearby_strata
from engine.polynomial import parse
from engine.resolve import resolve


# Singular's slice sets the listed coordinates free and all others to 1.
# The last field selects Atlas's transverse composition row instead of the
# row on that coordinate locus. The quadric locus has no rational point
# except the origin, so Singular receives the certified germ G itself.
CASES = [
    ("Repeated component", "x^4*y^3", "xy", "x", False),
    ("Plane binomial", "x^3+y^4*z", "xyz", "xy", False),
    ("Even Morse", "x^2+y^2*z+y^3", "xyz", "xy", False),
    ("Surface binomial", "x*y+z^5*t", "xyzt", "xyz", False),
    ("Odd Morse", "x^2+y^2+t*z^2+z^3", "xyzt", "xyz", False),
    ("Cusp cylinder", "x^2+y^3+y^4", "xyz", "xy", False),
    ("Coordinate composition", "(x+y*z)^2+x^3", "xyz", "xy", True),
    ("Quadric composition", "(x^2+y^2+z^2)^2+x^5", "xyz", None, True),
]

CONTACT_CASES = [
    ("Two osculating conics", "(x*z-y^2)*(x*z+y*z-y^2)", "xyz", "yz", False),
    ("E7 hyperplane cylinder", "y*(x^2+y^3+y*z^3)", "xyzt", "xy", False),
    ("Cubic parameters", "y*(x^2+y^3+y*z^3+y*t^3)", "xyzt", "xy", False),
    ("Mixed parameters", "y*(x^2+y^3+y*z^3+y*z*t)", "xyzt", "xy", False),
]

MEASURE = '''proc measure(poly f) {
    list blocks=monodromy(f);
    string result="";
    for (int i=1; i<=size(blocks[2]); i++) {
        result=result+string(blocks[1][i])+","+string(blocks[2][i])+","+string(blocks[3][i])+" ";
    }
    print("ATLASBENCH "+result);
}
print("ATLASBENCH ready");
'''


class SingularSlice:
    label = "Singular"

    def __init__(self, executable, variables, normal):
        self.command = [executable, "-q", "--no-rc"]
        images = ",".join(name if name in normal else "1" for name in variables)
        self.setup = f'''LIB "gmssing.lib";
ring sourceRing=0,({",".join(variables)}),ds;
ring normalRing=0,({",".join(normal)}),ds;
map transverse=sourceRing,{images};
''' + MEASURE

    def job(self, polynomial):
        return (f"setring sourceRing; poly f={polynomial}; setring normalRing; "
                "measure(transverse(f)); setring sourceRing; kill f;\n")


class SingularGerm:
    label = "Singular"

    def __init__(self, executable, names):
        self.command = [executable, "-q", "--no-rc"]
        self.setup = f'''LIB "gmssing.lib";
ring normalRing=0,({",".join(names)}),ds;
''' + MEASURE

    def job(self, polynomial):
        return f"poly g={polynomial}; measure(g); kill g;\n"


def atlas_valuations(row):
    # Orders at individual roots exp(2*pi*i*r), including the trivial root.
    result = Counter()
    for factor in row["factors"]:
        n = factor["multiplicity"]
        for k in range(n):
            result[Fraction(k, n)] += factor["exponent"]
    return result


def singular_valuations(answer, dimension):
    # Unreduced H^0 contributes -1. Reduced cohomology lives in degree dimension-1.
    result = Counter({Fraction(0): -1})
    for block in answer.split():
        residue, size, count = block.split(",")
        result[-Fraction(residue) % 1] += (-1)**dimension * int(size) * int(count)
    return result


def measure(case, executable, trials):
    name, equation, variables, normal, composition = case
    variables = list(variables)
    terms = parse(equation, variables)
    # The origin resolution is outside the timed interval. On one logical
    # CPU the quick budget can expire, so continue with the exhaustive search.
    for search in ("quick", "exhaustive"):
        record = resolve(dict(polynomial=equation, variables=variables, search=search, monodromy=True, compute="cpu"))
        if record["status"] == "resolved":
            break
    topological = record["topological_zeta"]
    inputs = (terms, variables, topological["numerical_data"], topological["strata"],
              record["monodromy_zeta"]["factors"])
    if composition:
        def selected(r):
            return r["model"] == "transverse composition"
        row, = filter(selected, _nearby_strata(*inputs))
        locus = " = ".join(row["coordinates"]) + " = 0"
    else:
        def selected(r):
            return set(r.get("coordinates", [])) == set(normal)
        locus = normal
    if normal is None:
        solver, source = SingularGerm(executable, row["transverse_coordinates"]), row["transverse_germ"]
        dimension = len(row["transverse_coordinates"])
    else:
        solver, source, dimension = SingularSlice(executable, variables, normal), equation, len(normal)
    times = {"Atlas": [], "Singular": []}
    with Session(solver, None, timeout=30) as singular:
        for trial in range(trials + 1):
            for system in (times if trial % 2 else reversed(times)):
                if system == "Atlas":
                    start = perf_counter_ns()
                    rows = _nearby_strata(*inputs)
                    elapsed = (perf_counter_ns() - start) / 1e6
                    row, = filter(selected, rows)
                    atlas = atlas_valuations(row)
                else:
                    elapsed, answer = singular.warm(source)
                    reference = singular_valuations(answer, dimension)
                if trial:
                    times[system].append(elapsed)
            if atlas != reference:
                raise ValueError(f"{name}: Atlas {row} disagrees with Singular {answer}")
    print(f"{name:22} {locus:20} {row['expression']:30}  "
          f"Atlas {median(times['Atlas']):8.3f} ms   Singular {median(times['Singular']):8.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--contacts-only", action="store_true", help="compare the two-smooth-branch cases")
    args = parser.parse_args()
    singular = shutil.which("Singular")
    if not singular or args.trials < 2:
        parser.error("Install Singular and select at least two trials.")
    print(subprocess.check_output([singular, "--version"], text=True).splitlines()[0])
    for case in CONTACT_CASES if args.contacts_only else CASES + CONTACT_CASES:
        measure(case, singular, args.trials)
