"""Saved local standard bases and weak normal forms in the Atlas session."""
import json
import subprocess
import unittest
from textwrap import dedent

from engine.session import new_session
from engine.session_expressions import SessionError
from tests.cli_support import ROOT, run_prompt, run_script


class LocalStandardBasisSessionTests(unittest.TestCase):
    def test_saved_wide_basis_keeps_a_large_rational_point_through_replay(self):
        q = "1" + "0"*4400 + "7/3"
        n = 2**64+7
        source = f'''
            x,y,z = coordinates("x","y","z",coeffs=QQ)
            I = ideal(polynomial("x^{n}*z-(y-({q}))^2", variables=[x,y,z]))
            B = local_standard_basis(I, point=["1","{q}","0"], compute="cpu")
            R = local_normal_form(z,B,compute="cpu")
            print(R.json)
        '''
        saved = run_script(source)
        self.assertEqual(saved.returncode, 0, saved.stderr)
        record = json.loads(saved.stdout)
        result = record["result"]
        self.assertEqual((result["point"], result["unit"], result["reduction"]), (["1",q,"0"], f"x^{n}", "standard"))
        replay = run_script(record["replay"]["script"])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["result"], result)

    def test_local_membership_and_unit_certificate_at_prompt_and_in_script(self):
        source = dedent("""
            x, y = coordinates("x", "y", coeffs=QQ)
            g = x+1/2
            m = ideal(g, y+1/3)
            I = ideal(g*(x-1), y+1/3, order="lex")
            B = local_standard_basis(I, point=(-1/2, -1/3), compute="cpu")
            print(B.point, B.coordinates == (x,y), B.generators[0]+x-x == B.generators[0])
            localized = saturation(I,x-1)
            h = local_normal_form(g, B, compute="cpu")
            print(h.point == B.point, h.reduction, h.remainder == 0, membership(g,I).value)
            print(normal_form(h.unit,m) == 1, membership(h.unit*g-h.remainder,localized).value)
            r = local_normal_form(1+g, B, point=("-2/4", "-2/6"), compute="cpu")
            print(r.remainder != 0, normal_form(r.unit,m) == 1, membership(r.unit*(1+g)-r.remainder,localized).value)
            print(local_normal_form(1+g, I).remainder == 0)
        """).lstrip()
        expected = ["['-1/2', '-1/3'] True True", "True standard True False", "True True", "True True True", "True"]
        # Comparison mode starts an independent symbolic worker per operation.
        script = subprocess.run([str(ROOT / "atlas"), "run", "-"], input=source, text=True,
                                capture_output=True, timeout=90)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        status, output, error = run_prompt(source + "quit\n")
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_local_basis_requires_its_point_and_original_coordinates(self):
        session = new_session(None)
        list(session.run(dedent("""
            x = coordinates("x")
            s = coordinates("x", fresh=True)
            I = ideal(x*(x-1))
            B = local_standard_basis(I, point=[1])
        """)))
        for statement in ("local_normal_form(x-1, B, point=[0])", "local_normal_form(s-1, B)",
                          "local_standard_basis(I, point=[])", 'local_normal_form(x, I, point=["1/0"])',
                          "membership(x-1, B)", "normal_form(x-1, B)", "intersection(I, B)",
                          "groebner(B)"):
            with self.subTest(statement=statement), self.assertRaises(SessionError):
                list(session.run(statement))
        values = session.variables
        values["B"].point[0] = "0"
        self.assertEqual(values["B"].point, ["1"])
        list(session.run('h = local_normal_form(x-1, B, point=B.point)'))
        self.assertEqual(values["h"].remainder, 0)

    def test_scalar_ring_and_weak_normal_form_fields(self):
        session = new_session(None)
        list(session.run(dedent("""
            B = local_standard_basis(ideal(variables=[]), point=[])
            scalar = local_normal_form(-1/2, B)
            x, y = coordinates("x", "y")
            curve = local_standard_basis(ideal(x-y^2))
            h = local_normal_form(1+x, curve)
        """)))
        values = session.variables
        scalar = values["scalar"]
        self.assertEqual((scalar.coordinates, scalar.point, scalar.reduction), ((), [], "standard"))
        self.assertEqual(2*scalar.remainder, -1)
        self.assertEqual(scalar.unit, 1)
        h = values["h"]
        self.assertEqual(h.reduction, "weak")
        self.assertNotEqual(h.remainder, 0)
        self.assertEqual(h.coordinates, (values["x"], values["y"]))

    def test_saved_local_normal_form_replay_preserves_the_point_and_result(self):
        session = new_session(None)
        list(session.run(dedent("""
            x = coordinates("x")
            B = local_standard_basis(ideal((2*x+1)*(x-3)), point=[-1/2])
            h = local_normal_form(x+1/2, B, compute="cpu")
        """)))
        h = session.variables["h"]
        replay = h.record["replay"]["script"]
        script = run_script(replay)
        self.assertEqual(script.returncode, 0, script.stderr)
        status, output, error = run_prompt(replay + "quit\n")
        self.assertEqual((status, error), (0, ""))
        for rendered in (script.stdout, output.replace("atlas> ", "").strip()):
            row = json.loads(rendered)
            self.assertEqual((row["operation"], row["result"]), ("local-normal-form", h.result))
            self.assertEqual(row["result"]["point"], ["-1/2"])
            self.assertEqual(row["execution"]["requested"], "cpu")
