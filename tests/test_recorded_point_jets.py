"""Specialize every archived implicit jet on the integer grid of its open frame."""
from fractions import Fraction
import gzip
from itertools import product
import json
from math import prod
from pathlib import Path
import subprocess
import unittest

from engine import native_polynomial as np
from engine.affine_kernel import frame_minor

ROOT = Path(__file__).resolve().parents[1]


def terms(row):
    result = []
    for term in row.split(";"):
        if term.strip():
            coefficient, *exponents = term.split()
            result.append((Fraction(coefficient), tuple(map(int, exponents))))
    return result


def evaluate(polynomial, point):
    return sum(c * prod(x**n for x, n in zip(point, e)) for c, e in polynomial)


class RecordedPointJets(unittest.TestCase):
    def test_every_recorded_implicit_jet(self):
        runner = ROOT / "native/build/bench_point_jet"
        if not runner.is_file():
            self.skipTest("Build the point jet runner with make -C native -f groebner.mk test-verify.")
        count = 0
        with gzip.open(ROOT / "benchmarks/canonical-requests.jsonl.gz", "rt") as records:
            for line in records:
                record = json.loads(line)
                if record.get("symbol") != "atlas_centre_maximum":
                    continue
                request = json.loads(record["request"])
                if request.get("kind") != "implicit_jet":
                    continue
                with self.subTest(request=request["names"], order=request["order"]):
                    result = json.loads(record["response"])["result"]
                    self.check_frame(request, result, runner)
                count += 1
        self.assertGreater(count, 0)

    def check_frame(self, request, result, runner):
        # The archived requests go through order one in h. For a free axis x,
        # the archived field evaluates to det(J_dep) * d/dx on the frame.
        # Compare its scale as well as every coefficient of the truncation.
        degree = request["order"] - 1
        self.assertIn(degree, (0, 1))
        axes = request["coordinates"]
        names = request["names"]
        dependent = [i for i in range(len(names)) if i not in axes]
        variables = np.symbols(" ".join(names))
        equations = [np.from_row(variables, row) for row in request["equations"]]
        jacobian = frame_minor(variables, equations, [variables[i] for i in dependent])
        minor = terms(np.row(variables, jacobian))
        rows = list(map(terms, request["equations"]))
        units = list(map(terms, request["extra"]))
        expected = list(map(terms, result))
        self.assertEqual(len(expected), 1 + degree * len(axes))
        seen = 0
        for point in product(range(-4, 5), repeat=len(names)):
            if any(evaluate(row, point) for row in rows) or not all(evaluate(u, point) for u in units):
                continue
            scale = evaluate(minor, point)
            if not scale:
                continue
            case = dict(variables=names, dependent=dependent, rows=request["equations"],
                        f=request["poly"], mu=[0, 1], point=[[v] for v in point],
                        degree=degree, repeats=1, encoded_rows=True)
            proc = subprocess.run([runner], input=json.dumps(case), capture_output=True,
                                  text=True, check=True, timeout=10)
            jet = {tuple(item["powers"]): list(map(Fraction, item["coefficient"]))
                   for item in json.loads(proc.stdout)["terms"]}
            zero = (0,) * len(axes)
            self.assertEqual(jet.get(zero, [0]), [evaluate(expected[0], point)])
            for j, symbolic in enumerate(expected[1:]):
                power = tuple(int(i == j) for i in range(len(axes)))
                self.assertEqual(jet.get(power, [0]), [evaluate(symbolic, point) / scale])
            seen += 1
        self.assertGreater(seen, 0)
