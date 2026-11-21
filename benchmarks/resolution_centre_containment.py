"""Compare source singular-locus checks for a sheared line in fan and graph replay."""
import argparse
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import fan_centres
from engine.graph_record import _graph_request
from engine.strata import InvalidChart


NAMES = tuple("xyz")
FRAME = ((1, 0, 1), (0, 1, 0), (0, 0, 1))
CASES = [
    ("singular line", "(x-z)^2-y^2*z", True),
    ("generically smooth line", "(x-z)^2+y*z", False),
    ("repeated smooth plane", "(x-z)^7", False),
]


def fan(equation):
    return fan_centres.singular_ray_images(equation, NAMES, [(1, 1, 0)], FRAME) is not None


def graph(equation):
    try:
        return _graph_request("verify_centre", NAMES, centre=["x-z", "y"], relation=None, equation=equation)
    except InvalidChart as exc:
        if "meets the smooth locus" not in str(exc):
            raise
        return False


def atlas_trial(method, equation, expected, batch):
    elapsed = 0
    for _ in range(batch):
        if method is fan:
            fan_centres._in_frame.cache_clear()
            fan_centres._reduced_equation.cache_clear()
        start = perf_counter_ns()
        answer = method(equation)
        elapsed += perf_counter_ns() - start
        if answer is not expected:
            raise AssertionError((method.__name__, equation, answer, expected))
    return elapsed / batch / 1000


def singular_trial(method, equation, expected, batch):
    test = ("ok=(reduce(h,std(C^2))==0);" if method == "Singular fan" else
            "ok=(size(reduce(ideal(h)+jacob(h),std(C)))==0);")
    source = f"""
system("--ticks-per-sec",1000000);
ring R=0,(x,y,z),lp;
poly f,h;
ideal C,factors;
int ok,i,j,total;
int start=rtimer;
for(i=0;i<{batch};i++){{
    f={equation};
    C=ideal(x-z,y);
    factors=sqrfree(f,1);
    h=1;
    for(j=1;j<=size(factors);j++){{h=h*factors[j];}}
    {test}
    total=total+ok;
}}
print("RESULT "+string(total)+" "+string(rtimer-start));
quit;
"""
    result = subprocess.run(["Singular", "-q", "--no-rc"], text=True, input=source, capture_output=True,
                            check=True, timeout=60)
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    if len(rows) != 1 or int(rows[0][1]) != batch * int(expected) or "   ?" in result.stdout:
        raise AssertionError(result.stdout + result.stderr)
    return int(rows[0][2]) / batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if min(args.trials, args.batch) < 1:
        parser.error("Use positive trial and batch counts.")
    for label, equation, expected in CASES:
        systems = ("Atlas fan", "Atlas graph", "Singular fan", "Singular graph")
        samples = {system: [] for system in systems}
        for trial in range(args.trials + 1):
            shift = trial % len(systems)
            for system in systems[shift:] + systems[:shift]:
                if system.startswith("Singular"):
                    elapsed = singular_trial(system, equation, expected, args.batch)
                else:
                    elapsed = atlas_trial(fan if system == "Atlas fan" else graph, equation, expected, args.batch)
                if trial:
                    samples[system].append(elapsed)
        for system, times in samples.items():
            print(f"{label}, {expected}, {system}: median {median(times):.2f} us "
                  f"({', '.join(f'{t:.2f}' for t in times)})")


if __name__ == "__main__":
    main()
