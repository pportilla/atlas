"""Affine normalization records and coordinate ownership at each public entry point."""
import json
import shlex
from textwrap import dedent
import unittest

from engine.session import SessionError, new_session
from tests.test_cli_session import run_prompt, run_script
from tests.test_http_basis_refs import endpoint, post
from tests.test_point_standard_interface import shell


class NormalizationInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def test_http_cli_maps_and_replay(self):
        n = 2**64 + 17
        d = 2**128 + 7
        for names, generators, reduced, count in (
            (["x", "y"], ["y^2-x^3"], True, 1),
            (["x", "y"], [f"x^{n}*y"], False, 2),
            (["normalization"], [f"normalization-1/{d}"], True, 1),
            ([], [], True, 1), ([], ["1"], True, 0),
        ):
            with self.subTest(generators=generators):
                ring = {"field": "QQ", "variables": names, "order": "lex"}
                status, record = post(self.url, "normalization", {"ideal": generators}, ring, compute="cpu")
                self.assertEqual(status, 200, record)
                result = record["result"]
                self.assertEqual((result["type"], result["ring"], result["input_reduced"]), ("normalization", ring, reduced))
                self.assertEqual(len(result["components"]), count)
                for component in result["components"]:
                    self.assertEqual(component["ring"]["field"], "QQ")
                    self.assertEqual(len(component["images"]), len(names))
                    self.assertEqual(len(component["fractions"]), len(component["ring"]["variables"]))
                args = shlex.split(record["replay"]["command"])
                native = shell(args)
                worker = shell(["python3", *args])
                script = run_script(record["replay"]["script"])
                for run in (native, worker, script):
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(json.loads(run.stdout)["result"], result)
                args.remove("--json")
                native = shell(args)
                self.assertEqual(native.stdout, shell(["python3", *args]).stdout)
                if count:
                    self.assertIn("Normalized algebra:", native.stdout)
                else:
                    self.assertIn("The zero algebra has no components.", native.stdout)
                if not reduced:
                    self.assertIn("nilpotents", native.stdout)

    def test_cusp_images_and_inverse_fractions_at_prompt_and_in_script(self):
        source = dedent("""\
            x, y = coordinates("x", "y", coeffs=QQ)
            I = ideal(y^2-x^3, order="lex")
            N = normalization(I)
            C = N.components[0]
            X, Y = C.images
            print(N.type, N.input_reduced, C.prime.coordinates == I.coordinates)
            print(C.relations.coordinates == C.coordinates, membership(Y^2-X^3, C.relations).value)
            print(C.ring == C.relations.ring, membership(y^2-x^3, C.prime).value)
            for f in C.fractions:
                print(membership(f.denominator, C.prime).value)

            """)
        run = run_script(source)
        self.assertEqual(run.returncode, 0, run.stderr)
        lines = run.stdout.splitlines()
        self.assertEqual(lines[:3], ["normalization True True", "True True", "True True"])
        self.assertTrue(lines[3:])
        self.assertEqual(set(lines[3:]), {"False"})
        code, output, error = run_prompt(source + "quit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").replace("... ", "").strip().splitlines(), lines)

    def test_distinct_source_and_component_spaces(self):
        session = new_session(None)
        list(session.run(dedent("""
            x, y = coordinates("x", "y")
            N = normalization(ideal(x*y))
            A, B = N.components
            a = A.coordinates[0]
            """)))
        with self.assertRaisesRegex(SessionError, "coordinate"):
            list(session.run("membership(a, B.relations)"))
        with self.assertRaisesRegex(SessionError, "coordinate"):
            list(session.run("membership(a, A.prime)"))
