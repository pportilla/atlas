"""The fourth section changes the plane contact curve and both local zetas."""
import json
import subprocess
import unittest

import sympy as sp

from engine.resolve import resolve
from tests.cli_support import ROOT, run_prompt, run_script


QUARTIC = "z*(z-y^2+x^3)*(z-y^2+x^3-x^4-y^3)*(z-y)"
TOP = ("(-2619*s^4-2996*s^3+8580*s^2+10126*s+2717)"
       "/((s+1)^2*(21*s+11)*(25*s+13)*(36*s+19))")
MON = "(1-T^7)*(1-T^9)/((1-T^21)*(1-T^25)*(1-T^36))"


class FourGraphMonodromyTests(unittest.TestCase):
    def test_witnesses_and_cancelled_orders(self):
        source = ('x,y,z=coordinates("x","y","z",coeffs=QQ)\n'
                  f'M=monodromy_check({QUARTIC})\n'
                  'print_json(status=M.status, poles=M.poles, candidates=M.candidate_poles)\n')
        script = run_script(source)
        status, out, err = run_prompt(source + 'quit\n')
        self.assertEqual((script.returncode, script.stderr, status, err), (0, "", 0, ""))
        result = json.loads(script.stdout)
        self.assertEqual(json.loads(out.replace("atlas> ", "").strip()), result)
        self.assertEqual(result["status"], "verified")
        self.assertEqual({p["value"]: (p["order"], p["eigenvalue_order"]) for p in result["poles"]},
                         {"-1": (2, 1), "-11/21": (1, 21), "-13/25": (1, 25), "-19/36": (1, 36)})
        self.assertTrue(all(p["witness"] == "origin zeta" and p["cyclotomic_exponent"] == -1
                            for p in result["poles"]))
        candidates = {p["value"]: p for p in result["candidates"]}
        for pole, q in (("-4/7", 7), ("-5/9", 9)):
            row = candidates[pole]
            self.assertEqual((row["pole_order"], row["eigenvalue_order"], row["cyclotomic_exponent"]), (0, q, 0))

    def test_cli_and_replay(self):
        result = subprocess.run([ROOT / "atlas", "resolve", "(-12345678901234567890123456789/7)*" + QUARTIC,
                                 "--vars", "z,x,y", "--zetas-only", "--json"],
                                capture_output=True, text=True, check=True, timeout=60)
        record = json.loads(result.stdout)
        self.assertEqual(record["engine"], "four-graph-contact")
        self.assertEqual(sp.cancel(sp.sympify(record["topological_zeta"]["expression"]) - sp.sympify(TOP)), 0)
        mon = record["monodromy_zeta"]
        self.assertEqual((mon["expression"], mon["milnor_fiber_euler_characteristic"]), (MON, 66))
        self.assertEqual(len(record["contact_certificate"]["branches"]), 6)
        self.assertEqual(len(record["contact_certificate"]["divisors"]), 8)
        record["monodromy_zeta"] = {**mon, "expression": "1", "factors": []}
        replay = subprocess.run([ROOT / "atlas", "monodromy", "-", "--json"],
                                input=json.dumps(record), capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(replay.stdout), mon)

    def test_four_generic_planes(self):
        # Blowing up the vertex gives P² with four lines, six intersection
        # points and open complement of Euler characteristic one.
        r = resolve(dict(polynomial="z*(z-x)*(z-y)*(z-2*x-3*y)", variables=list("xyz"),
                         zetas_only=True, search="quick"))
        self.assertEqual(r["engine"], "four-graph-contact")
        self.assertEqual(r["monodromy_zeta"]["expression"], "1/(1-T^4)")
        self.assertEqual(sp.cancel(sp.sympify(r["topological_zeta"]["expression"])
                                   - sp.sympify("(s^2-2*s+3)/((1+s)^2*(3+4*s))")), 0)
