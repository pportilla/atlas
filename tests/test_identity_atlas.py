import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from engine.monodromy import monodromy_zeta
from engine.resolve import resolve
from engine.strata import resolution_strata
from engine.topological import topological_zeta


ROOT = Path(__file__).resolve().parents[1]


class RationalCharts(unittest.TestCase):
    def test_cli_and_saved_atlas_without_sympy(self):
        # sitecustomize runs in the actual CLI's resolution worker as well as
        # in the command reading a previously serialized chart.
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sitecustomize.py").write_text(
                "import sys\n"
                "class Guard:\n"
                "    def find_spec(self, name, *args):\n"
                "        if name == 'sympy' or name.startswith('sympy.'):\n"
                "            raise AssertionError('unexpected import of ' + name)\n"
                "sys.meta_path.insert(0, Guard())\n")
            env = dict(os.environ, ATLAS_GROEBNER="native", PYTHONPATH=os.pathsep.join((folder, str(ROOT))))
            for polynomial, variables, mode, expected in (
                    ("x+y^5", "x,y,z", "intrinsic", ("1/(1-T)", "1/(s + 1)")),
                    ("2/3*(I+class)^2*(z+class)*(1+I)", "I,class,z", "embedded",
                     ("1", "1/((s + 1)*(2*s + 1))")),
                    ("x^2+y^2+z^2", "x,y,z", "intrinsic",
                     ("1/(1-T^2)", "(s + 3)/((s + 1)*(2*s + 3))")),
                    ("x^3+y^3+z^3", "x,y,z", "embedded", ("1/(1-T^3)^3", "1/(s + 1)")),
                    ("x^2*y-z^3", "x,y,z", "embedded", ("1/(1-T^3)", "(2*s + 5)/((s + 1)*(6*s + 5))")),
                    ("(x+y)^2+z^3+x^3", "x,y,z", "embedded",
                     ("(1-T^3)/((1-T^2)*(1-T^6))", "(2*s + 7)/((s + 1)*(6*s + 7))")),
                    ("(x*y-z^2)^2+x^5+y^5", "x,y,z", "embedded",
                     ("(1-T^5)^8/((1-T^4)*(1-T^10)^8)",
                      "-(22*s^2 - 11*s - 21)/((s + 1)*(4*s + 3)*(10*s + 7))")),
                    ("z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2", "x,y,z", "intrinsic",
                     ("(1-T^6)^2/((1-T^3)*(1-T^4)*(1-T^12)^2*(1-T^13))",
                      "(77*s^2 + 175*s + 90)/(3*(s + 1)*(4*s + 3)*(13*s + 10))")),
                    ("x^2+y^3+z^5", "x,y,z", "intrinsic",
                     ("(1-T^6)*(1-T^10)*(1-T^15)/((1-T^2)*(1-T^3)*(1-T^5)*(1-T^30))",
                      "(22*s + 31)/((s + 1)*(30*s + 31))")),
                    ("x^2+y^3+z^7+w^2", "x,y,z,w", "intrinsic",
                     ("(1-T^21)/((1-T^3)*(1-T^7))", "(9*s + 31)/((s + 1)*(21*s + 31))")),
                    ("(x1+x0^10000000000)", ",".join(f"x{i}" for i in range(34)), "embedded",
                     ("1/(1-T)", "1/(s + 1)"))):
                with self.subTest(polynomial=polynomial):
                    command = [sys.executable, str(ROOT / "atlas"), "resolve", polynomial, "--vars",
                               variables, "--mode", mode, "--monodromy", "--json"]
                    answer = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
                    self.assertEqual(answer.returncode, 0, answer.stderr or answer.stdout)
                    record = json.loads(answer.stdout)
                    self.assertEqual((record.pop("monodromy_zeta")["expression"],
                                      record.pop("topological_zeta")["expression"]), expected)
                    record["engine"] = "serialized-chart"
                    for key in ("fan", "certificate", "dual_complex"):
                        record.pop(key, None)
                    archived = subprocess.run([sys.executable, "-c",
                        "import json, sys; from engine.monodromy import monodromy_zeta; "
                        "from engine.topological import topological_zeta; r = json.load(sys.stdin); "
                        "print(json.dumps([f(r)['expression'] for f in (monodromy_zeta, topological_zeta)]))"],
                        cwd=ROOT, input=json.dumps(record), capture_output=True, text=True, env=env, timeout=30)
                    self.assertEqual(archived.returncode, 0, archived.stderr)
                    self.assertEqual(tuple(json.loads(archived.stdout)), expected)

    def test_alternate_unit_generators_and_fibre_ideal(self):
        record = resolve(dict(polynomial="(x+y)^2*(y+z)*(1+x)", variables=list("xyz")))
        chart, = record["atlas"]
        chart["fibre"] = ["x+y", "z", "y^2"]
        chart["components"][0]["ideal"] = ["3*(y+z)*(1-z)"]
        data, strata, _ = resolution_strata(record)
        self.assertEqual([r["multiplicity"] for r in data], [1, 2])
        self.assertEqual(strata, [dict(components=["S1", "S2"], euler_characteristic=1)])

    def test_damaged_identity_cannot_certify_either_zeta(self):
        record = resolve(dict(polynomial="(x+y)^2*(y+z)", variables=list("xyz")))
        for kind in ("weight", "missing branch", "false branch", "off-origin", "dependent", "fibre", "nu"):
            with self.subTest(kind=kind):
                r = copy.deepcopy(record)
                chart = r["atlas"][0]
                branches = chart["components"]
                if kind == "weight":
                    branches[0]["multiplicity"], branches[1]["multiplicity"] = 2, 1
                elif kind == "missing branch":
                    branches.pop()
                elif kind == "false branch":
                    branches[0]["ideal"] = ["z"]
                elif kind == "off-origin":
                    branches[0]["ideal"] = ["1+z"]
                elif kind == "dependent":
                    branches[0]["ideal"] = ["x+y+z^3"]
                elif kind == "fibre":
                    chart["fibre"] = ["x", "y"]
                else:
                    branches[0]["nu"] = 2
                self.assertEqual(monodromy_zeta(r)["status"], "unavailable")
                self.assertEqual(topological_zeta(r)["status"], "unavailable")


    def test_serialized_monomial_charts_verify_geometry_independently(self):
        from engine.native_library import _MODE
        if _MODE == "sympy":
            self.skipTest("The sparse native monomial verifier is disabled in pure reference mode.")
        record = resolve(dict(polynomial="x^2*y-z^3", variables=list("xyz"), monodromy=True))
        for kind in ("cone", "projection", "fibre", "region", "strict", "order", "boundary"):
            with self.subTest(kind=kind):
                r = copy.deepcopy(record)
                charts = r["atlas"]
                if kind == "cone":
                    del charts[1]
                elif kind == "projection":
                    charts[0]["map"]["x"] = "u1+1"
                elif kind == "fibre":
                    charts[0]["fibre"] = []
                elif kind == "region":
                    charts[-1]["region"] = ["1"]
                elif kind == "strict":
                    charts[0]["components"][-1]["ideal"] = ["u1+u2"]
                elif kind == "order":
                    charts[0]["components"][0]["multiplicity"] += 1
                else:
                    charts[0]["components"].pop(0)
                self.assertEqual(topological_zeta(r)["status"], "unavailable")

        # Different generators of the origin fibre and open-chart ownership
        # ideals can define the same underlying constructible sets.
        chart = record["atlas"][-1]
        chart["fibre"] = [f"({s})^2" for s in chart["fibre"]]
        chart["region"] = [f"({s})^2" for s in chart["region"]]
        self.assertEqual(topological_zeta(record)["expression"], "(2*s + 5)/((s + 1)*(6*s + 5))")

    def test_redundant_generators_and_missing_fan_annotations_preserve_order_checks(self):
        from engine.native_library import _MODE
        if _MODE == "sympy":
            self.skipTest("Tests the native stored-chart verifier.")
        record = resolve(dict(polynomial="x^2*y-z^3", variables=list("xyz"), monodromy=True))
        for annotations in ("all", "some", "none"):
            r = copy.deepcopy(record)
            for i, chart in enumerate(r["atlas"]):
                if annotations == "none" or annotations == "some" and i == 0:
                    chart.pop("torus")
                for part in chart["components"]:
                    g, = part["ideal"]
                    part["ideal"] = [g, f"({g})*({g})"]
            self.assertEqual(topological_zeta(r)["expression"], record["topological_zeta"]["expression"])
            for chart in r["atlas"]:
                for part in chart["components"]:
                    if part["id"] == "E1":
                        part["multiplicity"] += 1
            self.assertEqual(topological_zeta(r)["status"], "unavailable")

        changed = copy.deepcopy(record)
        part = next(p for c in changed["atlas"] for p in c["components"] if p["id"] == "E1")
        part.update(id="S7", multiplicity=part["multiplicity"]+1)
        self.assertEqual(topological_zeta(changed)["status"], "unavailable")
