#!/usr/bin/env python3
"""Time exhaustive public zeta calculations against Singular on identical QQ inputs."""
import argparse
import json
from statistics import median

import sympy as sp

from three_graph_contact import POLYNOMIAL, ROOT, SINGULAR_RESOLUTION, trial


CASES = {
    # Polynomial, coordinates, Atlas resolution method, Singular specialized calculation,
    # expected topological zeta and alternating monodromy zeta in the Atlas convention.
    "cusp": ("y^2-x^3", "x,y", "auto", "cuspZetas",
             "(4*s+5)/((s+1)*(6*s+5))", "(1-s^6)/((1-s^2)*(1-s^3))"),
    "pinch": ("(1+z)*(x^2-y^2*z)", "x,y,z", "auto", "pinchZetas",
              "(s+2)/(2*(s+1)^2)", "1/(1-s^2)"),
    "weighted_quadric": ("x^2+y^2+z^2", "x,y,z", "weighted", None,
                         "(s+3)/((s+1)*(2*s+3))", "1/(1-s^2)"),
    "weighted_e8": ("x^2+y^3+z^5", "x,y,z", "weighted", "brieskornZetas",
                    "(22*s+31)/((s+1)*(30*s+31))",
                    "(1-s^6)*(1-s^10)*(1-s^15)/((1-s^2)*(1-s^3)*(1-s^5)*(1-s^30))"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=100)
    parser.add_argument("--case", choices=CASES, action="append")
    args = parser.parse_args()
    if args.repeats < 2 or args.timeout <= 0:
        parser.error("Supply at least two trials and a positive timeout.")
    specialized = (ROOT / "benchmarks" / "zeta_completion_charts.sing").read_text()
    for name in args.case or CASES:
        f, names, method, calculation, top, mon = CASES[name]
        expected = tuple(map(sp.sympify, (top, mon)))
        jobs = [("Atlas from polynomial", [ROOT / "atlas", "resolve", f, "--vars", names, "--compute", "cpu",
                                          "--exhaustive", "--method", method, "--zetas-only", "--json"], None)]
        if method == "weighted":
            # zetaDL computes surface zetas. For the pinch-point control its
            # local Euler routine errors, so the chart procedure below is used.
            jobs.append(("Singular from polynomial", ["Singular", "-q", "--no-rc"],
                         SINGULAR_RESOLUTION.replace(POLYNOMIAL, f)))
        if calculation:
            src = specialized + f"\nring R=0,({names}),dp;\nprint({calculation}({f}));\n"
            jobs.append(("Singular specialized", ["Singular", "-q", "--no-rc"], src))
        samples = {label: [] for label, _, _ in jobs}
        for repeat in range(args.repeats):
            for label, command, src in (jobs if repeat % 2 == 0 else reversed(jobs)):
                samples[label].append(trial(label, command, src, args.timeout, expected))
        print(json.dumps({"case": name, "samples_ms": samples,
                          "median_ms": {k: median(v) if None not in v else None for k, v in samples.items()}}), flush=True)


if __name__ == "__main__":
    main()
