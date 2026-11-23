"""The public zeta calculation for three polynomial translates of one quadratic."""
import json
import subprocess
import unittest

import sympy as sp

from tests.cli_support import ROOT, run_prompt, run_script


REFERENCE = json.loads((ROOT / "tests" / "fixtures" / "quadratic-translates.json").read_text())
POLYNOMIAL = REFERENCE["polynomial"]
CONICS = "(z^2-x*y)*((z-x-2*y)^2-x*y)*((z-3*x-5*y)^2-x*y)"


class QuadraticTranslateMonodromyTests(unittest.TestCase):
    def test_default_prompt_and_script_verify_all_eight_poles(self):
        source = ('x,y,z=coordinates("x","y","z",coeffs=QQ)\n'
                  f'M=monodromy_check({POLYNOMIAL})\n'
                  'print_json(status=M.status,poles=M.poles,candidates=M.candidate_poles,'
                  'top=M.topological_zeta.expression,mon=M.monodromy_zeta.expression,'
                  'chi=M.monodromy_zeta.milnor_fiber_euler_characteristic)\n')
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        status, out, err = run_prompt(source + 'quit\n')
        self.assertEqual((status, err), (0, ""))
        check = json.loads(script.stdout)
        self.assertEqual(json.loads(out.replace("atlas> ", "").strip()), check)
        self.assertEqual(check["status"], "verified")
        self.assertEqual(sp.cancel(sp.sympify(check["top"]) - sp.sympify(REFERENCE["topological_zeta"])), 0)
        self.assertEqual(check["mon"], REFERENCE["monodromy_zeta"]["expression"])
        self.assertEqual(check["chi"], 415)
        columns = ("order", "eigenvalue_order", "cyclotomic_exponent", "witness")

        def witnesses(rows):
            return {r["value"]: tuple(r[k] for k in columns) for r in rows}

        self.assertEqual(witnesses(check["poles"]), witnesses(REFERENCE["pole_witnesses"]))
        candidate = next(row for row in check["candidates"] if row["value"] == "-14/37")
        self.assertEqual((candidate["pole_order"], candidate["eigenvalue_order"], candidate["cyclotomic_exponent"]),
                         (0, 37, 0))

    def test_translated_scaled_reordered_input_and_saved_record(self):
        # Shift the fibre by a polynomial in its base coordinates and invert
        # its sign. None of the three centres now need to be zero.
        translated = POLYNOMIAL.replace("z", "(x+y^2-z)")
        f = "(-123456789012345678901234567890123456789/13)*" + translated
        proc = subprocess.run([ROOT / "atlas", "resolve", f, "--vars", "z,x,y", "--compute", "cpu", "--quick",
                               "--zetas-only", "--json"], capture_output=True, text=True, check=True, timeout=30)
        record = json.loads(proc.stdout)
        self.assertEqual(record["engine"], "quadratic-translate-contact")
        self.assertEqual(record["monodromy_check"]["status"], "verified")
        self.assertEqual(sp.cancel(sp.sympify(record["topological_zeta"]["expression"])
                                   - sp.sympify(REFERENCE["topological_zeta"])), 0)
        mon = record["monodromy_zeta"]
        self.assertEqual(mon["factors"], REFERENCE["monodromy_zeta"]["factors"])
        self.assertEqual((len(record["contact_certificate"]["divisors"]), len(record["contact_certificate"]["branches"])),
                         (19, 7))
        record["monodromy_zeta"] = {**mon, "expression": "1", "factors": []}
        replay = subprocess.run([ROOT / "atlas", "monodromy", "-", "--json"], input=json.dumps(record),
                                capture_output=True, text=True, check=True, timeout=30)
        self.assertEqual(json.loads(replay.stdout), mon)

    def test_homogeneous_cones_over_three_conics(self):
        from engine.resolve import resolve
        # In P² three smooth conics have 12 transverse pairwise intersections
        # and no triple points. Blowing up the vertex gives Euler coefficients
        # 9 on the complement, -18 on the open conics and 12 on their crossings.
        r = resolve(dict(variables=list("xyz"), polynomial=CONICS, search="quick", zetas_only=True))
        self.assertEqual(r["engine"], "quadratic-translate-contact")
        self.assertEqual(r["monodromy_check"]["status"], "verified")
        self.assertEqual(r["monodromy_zeta"]["factors"], [{"multiplicity": 6, "exponent": -9}])
        self.assertEqual(sp.cancel(sp.sympify(r["topological_zeta"]["expression"])
                                   - sp.sympify("(3*s^2+1)/((2*s+1)*(s+1)^2)")), 0)

    def test_odd_ramification_agrees_with_polynomial_graphs_in_another_coordinate(self):
        from engine.resolve import resolve
        request = dict(variables=list("xyz"), zetas_only=True, search="quick")
        f = "((z+x)^2-(x+x^2))*((z-x-y)^2-(x+x^2))*((z-x-2*y)^2-(x+x^2))"
        ramified = resolve(dict(request, polynomial=f))
        self.assertEqual(ramified["engine"], "quadratic-translate-contact")
        self.assertTrue(any(row["orders"][0] % 2 for row in ramified["contact_certificate"]["divisors"]))
        # The same factors have the form v_i² - d_i*x where every d_i is
        # a unit. Taking X=x*prod(d_i) as the fibre coordinate gives the
        # polynomial graph roots v_i²*prod_{j != i}(d_j).
        a, b, c = "(1-2*z)", "(1+2*(z-y))", "(1+2*(z-2*y))"
        g = f"(x-z^2*{b}*{c})*(x-(z-y)^2*{a}*{c})*(x-(z-2*y)^2*{a}*{b})"
        split = resolve(dict(request, polynomial=g))
        self.assertEqual(split["engine"], "three-graph-contact")
        self.assertEqual(ramified["monodromy_zeta"]["factors"], split["monodromy_zeta"]["factors"])
        self.assertEqual(sp.cancel(sp.sympify(ramified["topological_zeta"]["expression"])
                                   - sp.sympify(split["topological_zeta"]["expression"])), 0)

    def test_unbounded_common_fibre_translation(self):
        # The discriminant and zetas depend on centre differences. A common
        # translation can have exponents larger than an unsigned native word.
        f = CONICS.replace("z", "(z-x^18446744073709551617)")
        proc = subprocess.run([ROOT / "atlas", "resolve", f, "--vars", "x,y,z", "--compute", "cpu",
                               "--exhaustive", "--zetas-only", "--json"],
                              capture_output=True, text=True, check=True, timeout=30)
        record = json.loads(proc.stdout)
        self.assertEqual(record["engine"], "quadratic-translate-contact")
        self.assertEqual(record["monodromy_zeta"]["factors"], [{"multiplicity": 6, "exponent": -9}])
        self.assertEqual(sp.cancel(sp.sympify(record["topological_zeta"]["expression"])
                                   - sp.sympify("(3*s^2+1)/((2*s+1)*(s+1)^2)")), 0)
        self.assertIn("18446744073709551617", str(record["contact_certificate"]["centres"]))
        replay = subprocess.run([ROOT / "atlas", "monodromy", "-", "--json"], input=json.dumps(record),
                                capture_output=True, text=True, check=True, timeout=30)
        self.assertEqual(json.loads(replay.stdout), record["monodromy_zeta"])

    def test_recognizer_declines_inputs_outside_its_hypotheses(self):
        from engine.contact_zetas import contact_data
        from engine.polynomial import parse
        names = list("xyz")
        for f in (
            CONICS.replace("3*x-5*y)^2-x*y", "3*x-5*y)^2-x*(x+y)"),   # different radicands
            "(z^2-x*y)*((z-x)^2-x*y)*((z-2*x)^2-x*y)",               # repeated contact curve
            CONICS + "+x^9007199254740993",                             # wider than the exact integer range of a double
        ):
            with self.subTest(f=f):
                self.assertIsNone(contact_data(parse(f, names, bounded=False), names))
