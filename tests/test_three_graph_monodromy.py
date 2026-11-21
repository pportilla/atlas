"""End-to-end regression for the cancellation over the three graph branches."""
import json
import subprocess
import unittest

import sympy as sp

from tests.cli_support import ROOT, run_prompt, run_script


POLYNOMIAL = "z*(z-y^2+x^3)*(z-y^2+x^3-x^4-y^3)"
SPLIT_FACTORS = {3: -1, 6: 1, 8: 1, 9: 1, 11: 1, 18: -1, 22: -1, 32: -1}


class ThreeGraphMonodromyTests(unittest.TestCase):
    def test_pole_witnesses_at_prompt_and_in_script(self):
        source = (
            'x,y,z = coordinates("x","y","z", coeffs=QQ)\n'
            f'C = monodromy_check({POLYNOMIAL})\n'
            'print_json(status=C.status, poles=C.poles, candidates=C.candidate_poles)\n'
        )
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        status, stdout, stderr = run_prompt(source + "quit\n")
        self.assertEqual((status, stderr), (0, ""))
        check = json.loads(script.stdout)
        self.assertEqual(json.loads(stdout.replace("atlas> ", "").strip()), check)
        self.assertEqual(check["status"], "verified")
        self.assertEqual(
            {p["value"]: (p["order"], p["eigenvalue_order"], p["cyclotomic_exponent"], p["witness"])
             for p in check["poles"]},
            {"-1": (2, 1, 0, "origin H^0"),
             "-11/18": (1, 18, -1, "origin zeta"),
             "-13/22": (1, 22, -1, "origin zeta"),
             "-19/32": (1, 32, -1, "origin zeta")},
        )
        candidates = {p["value"]: p for p in check["candidates"]}
        for value, order in (("-2/3", 3), ("-5/8", 8), ("-7/11", 11)):
            row = candidates[value]
            self.assertEqual((row["pole_order"], row["eigenvalue_order"], row["cyclotomic_exponent"]),
                             (0, order, 0))

    def test_exact_zetas_and_replay_of_scaled_reordered_polynomial(self):
        result = subprocess.run(
            [ROOT / "atlas", "resolve", "(-12345678901234567890123456789/7)*" + POLYNOMIAL,
             "--vars", "z,y,x", "--zetas-only", "--json", "--quick"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(result.stdout)
        s = sp.Symbol("s")
        expected = (-356*s**4 + 4912*s**3 + 13699*s**2 + 10903*s + 2717) / (
            (s+1)**2 * (18*s+11) * (22*s+13) * (32*s+19))
        top = record["topological_zeta"]
        self.assertEqual(top["status"], "computed", top)
        self.assertEqual(sp.cancel(sp.sympify(top["expression"]) - expected), 0)
        mon = record["monodromy_zeta"]
        self.assertEqual({f["multiplicity"]: f["exponent"] for f in mon["factors"]}, SPLIT_FACTORS)
        nodes = [row for row in mon["nearby_strata"] if row["model"] == "nearby transverse node"]
        self.assertEqual(len(nodes), 3)
        self.assertTrue(all(row["eigenvalue_orders"] == [1] and row["expression"] == "1" for row in nodes))
        record["monodromy_zeta"] = {**mon, "factors": [], "expression": "1"}
        replay = subprocess.run([ROOT / "atlas", "monodromy", "-", "--json"],
                                input=json.dumps(record), capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(replay.stdout), mon)

    def test_triple_intersection_with_and_without_reduced_plane_contacts(self):
        from engine.resolve import resolve
        for f, top, mon in (
            ("z*(z-x)*(z-y)", "1/(s+1)^3", "1"),
            # Six distinct tangent lines over C, including a conjugate pair.
            ("z*(z-x^2-y^2)*(z-2*x^2-3*x*y-y^2)", "(2*s^2+s+2)/((s+1)^2*(3*s+2))",
             "1/((1-T^3)*(1-T^6)^2)"),
            ("z*(z-x)*(z-2*x)", "(2-s)/((s+1)*(3*s+2))", "(1-T^3)"),
        ):
            with self.subTest(f=f):
                record = resolve(dict(variables=list("xyz"), polynomial=f, search="quick", zetas_only=True))
                self.assertEqual(sp.cancel(sp.sympify(record["topological_zeta"]["expression"]) - sp.sympify(top)), 0)
                self.assertEqual(record["monodromy_zeta"]["expression"], mon)
                self.assertEqual(record["monodromy_check"]["status"], "verified")
                if f == "z*(z-x)*(z-2*x)":
                    self.assertNotIn("contact_certificate", record)
