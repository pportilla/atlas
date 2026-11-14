"""Primary decomposition through the HTTP worker, CLI, prompt, scripts and replay."""
import json
import shlex
from textwrap import dedent
import unittest

from engine.session import SessionError, new_session
from tests.test_cli_session import run_prompt, run_script
from tests.test_http_basis_refs import endpoint, post, RING
from tests.test_point_standard_interface import shell


class PrimaryDecompositionInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def test_exact_component_records_and_replay(self):
        n = 2**64 + 17
        denominator = 2**128 + 7
        for names, generators, pairs in (
            (["x", "y"], [f"x^{n}*y^2"], [(f"x^{n}", "x"), ("y^2", "y")]),
            (["primary_decomposition"], [f"primary_decomposition-1/{denominator}"],
             [(f"primary_decomposition-1/{denominator}", f"primary_decomposition-1/{denominator}")]),
        ):
            with self.subTest(names=names):
                ring = dict(RING, variables=names)
                status, record = post(self.url, "primary-decomposition", {"ideal": generators}, ring, compute="gpu")
                self.assertEqual(status, 200, record)
                result = record["result"]
                self.assertEqual((result["type"], result["ring"]), ("primary-decomposition", ring))
                self.assertCountEqual(result["components"], [{"primary": [q], "prime": [p]} for q, p in pairs])
                self.assertEqual(record["execution"]["requested"], "gpu")
                args = shlex.split(record["replay"]["command"])
                for process in (shell(args), run_script(record["replay"]["script"])):
                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertEqual(json.loads(process.stdout)["result"], result)
                args.remove("--json")
                process = shell(args)
                self.assertEqual(process.returncode, 0, process.stderr)
                for i, part in enumerate(result["components"], start=1):
                    self.assertIn(f"Primary component {i}: ({part['primary'][0]})", process.stdout.splitlines())
                    self.assertIn(f"Associated prime: ({part['prime'][0]})", process.stdout.splitlines())

    def test_prompt_and_script_reconstruct_an_ideal_with_an_embedded_prime(self):
        source = dedent("""\
            x, y = coordinates("x", "y", coeffs=QQ)
            I = ideal(x^2, x*y, order="lex")
            D = primary_decomposition(I, compute="cpu")
            a, b = D.components
            print(D.type, D.order, a.primary.coordinates == I.coordinates, b.prime.ring == I.ring)
            print(groebner(intersection(a.primary, b.primary)).generators == groebner(I).generators)
            print(radical(a.primary).generators == groebner(a.prime).generators)
            print(radical(b.primary).generators == groebner(b.prime).generators)
            """)
        expected = ["primary-decomposition lex True True", "True", "True", "True"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), expected)
        code, output, error = run_prompt(source + "quit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)

    def test_component_reuse_in_eliminated_coordinates(self):
        session = new_session(None)
        list(session.run(dedent("""
            t, x, y = coordinates("t", "x", "y")
            E = eliminate(ideal(t-x, x^2, y, order="lex"), t)
            D = primary_decomposition(groebner(E))
            Q, P = D.components[0].primary, D.components[0].prime
            u, v = E.coordinates
            """)))
        variables = session.variables
        for name in ("D", "Q", "P"):
            self.assertEqual(variables[name].coordinates, variables["E"].coordinates)
            self.assertEqual(variables[name].ring, variables["E"].ring)
        list(session.run("in_Q = membership(u, Q).value\nin_P = membership(u, P).value"))
        self.assertEqual((variables["in_Q"], variables["in_P"]), (False, True))
        with self.assertRaisesRegex(SessionError, "coordinate"):
            list(session.run("membership(x, P)"))

    def test_unit_and_zero_are_distinct_in_the_scalar_ring(self):
        for generators, expected, text in (
            (["2/3"], [], "The unit ideal has no primary components."),
            ([], [{"primary": [], "prime": []}], "Associated prime: (0)"),
        ):
            with self.subTest(generators=generators):
                status, record = post(self.url, "primary-decomposition", {"ideal": generators}, {"variables": []})
                self.assertEqual(status, 200, record)
                self.assertEqual(record["result"]["components"], expected)
                args = shlex.split(record["replay"]["command"])
                args.remove("--json")
                displayed = shell(args)
                self.assertEqual(displayed.returncode, 0, displayed.stderr)
                self.assertIn(text, displayed.stdout.splitlines())

    def test_validation(self):
        for ring, operands in ((dict(RING, field="CC"), {"ideal": ["x"]}),
                               (RING, {"ideal": ["x"], "point": [0, 0]}),
                               (RING, {"ideal": ["unknown^2"]})):
            with self.subTest(ring=ring, operands=operands):
                status, record = post(self.url, "primary-decomposition", operands, ring)
                self.assertEqual((status, record["error"]["code"]), (400, "invalid_request"))
