"""The same local ring, extended ideal and fraction through every public frontend."""
import json
import shlex
import subprocess
import unittest
from textwrap import dedent

from engine.session import SessionError, new_session
from tests.cli_support import ROOT, run_prompt, run_script, shell
from tests.http_support import endpoint, post


class LocalizationInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(endpoint())

    def test_results_and_self_contained_replays(self):
        n, c = 2**64 + 13, 2**128 + 21
        for names, generators, prime, query, contracted, flags in (
            (["x", "y"], ["x*y"], ["x"], {"numerator": "-x", "denominator": "y"}, ["x"], (True, False)),
            (["x", "y"], ["x*y"], ["x"], {"denominator": "-y"}, ["x"], (False, True)),
            (["x", "y"], [f"x^{n}*y"], ["x"], {"numerator": f"{c}*x^{n}"}, [f"x^{n}"], (True, False)),
            (["x", "y"], ["x"], [], {}, ["1"], None),
            (["localize"], [], [], {"numerator": "localize"}, [], (False, True)),
            ([], [], [], {"denominator": "2"}, [], (False, True)),
        ):
            ring = {"variables": names, "field": "QQ", "order": "lex"}
            with self.subTest(names=names, query=query):
                status, record = post(self.url, "localization", {"ideal": generators, "prime": prime, **query}, ring)
                self.assertEqual(status, 200, record)
                result = record["result"]
                self.assertEqual((result["type"], result["ring"], result["prime"], result["generators"]),
                                 ("localization", ring, prime, contracted))
                if flags is None:
                    self.assertNotIn("element", result)
                else:
                    self.assertEqual(result["element"], {"numerator": query.get("numerator", "1"),
                                                         "denominator": query.get("denominator", "1"),
                                                         "member": flags[0], "unit": flags[1]})
                args = shlex.split(record["replay"]["command"])
                native = shell(args)
                self.assertEqual(native.returncode, 0, native.stderr)
                native_record = json.loads(native.stdout)
                args[2] = "localize"
                for replay in (shell(args), shell(["python3", *args]), run_script(record["replay"]["script"]),
                               run_script(native_record["replay"]["script"])):
                    self.assertEqual(replay.returncode, 0, replay.stderr)
                    self.assertEqual(json.loads(replay.stdout)["result"], result)
                self.assertEqual(native_record["result"], result)
                args.remove("--json")
                native = shell(args)
                self.assertIn("Contracted ideal:", native.stdout)
                self.assertEqual(native.stdout, shell(["python3", *args]).stdout)

    def test_invalid_prime_and_nonunit_denominator(self):
        for prime, query, detail in ((["1"], {}, "prime"), (["x*y"], {}, "prime"), (["x^2"], {}, "prime"),
                                     (["x"], {"numerator": "x", "denominator": "x"}, "denominator"),
                                     ([], {"denominator": "0"}, "denominator")):
            with self.subTest(prime=prime, query=query):
                status, record = post(self.url, "localization", {"ideal": [], "prime": prime, **query})
                self.assertEqual(status, 400, record)
                self.assertEqual(record["error"]["code"], "invalid_request")
                self.assertIn(detail, record["error"]["message"].lower())
        bad = shell(["./atlas", "algebra", "localization", "--vars", "x", "--prime", "x", "--denominator", "x", "--json"])
        self.assertEqual(bad.returncode, 2, bad.stderr)
        self.assertEqual(json.loads(bad.stdout)["error"]["code"], "invalid_request")

    def test_structured_ring_fractions_and_ideal_at_prompt_and_in_script(self):
        source = dedent("""\
            x, y = coordinates("x", "y", coeffs=QQ)
            P = ideal(x)
            R = localize(P)
            print(R.type, R.contraction.coordinates == P.coordinates)
            print(R.fraction(x, y).member, R.fraction(y).unit, R.fraction(0).member)
            L = localization(ideal(x*y), P, numerator=x, denominator=y, compute="cpu")
            print(L.element.member, L.element.unit, L.element.numerator, L.element.denominator)
            print(L.fraction(x, y).member, L.fraction(y, s=y).unit)
            print(membership(x, L.contraction).value, membership(y, L.prime).value)
            """)
        lines = ["localization True", "False True True", "True False x y", "True True", "True False"]
        script = run_script(source)
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(script.stdout.splitlines(), lines)
        stdin = subprocess.run([str(ROOT / "atlas"), "run", "-"], input=source, text=True,
                               capture_output=True, timeout=15)
        self.assertEqual(stdin.returncode, 0, stdin.stderr)
        self.assertEqual(stdin.stdout.splitlines(), lines)
        code, output, error = run_prompt(source + "quit\n")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), lines)

    def test_coordinates_and_orders_checked_before_localizing(self):
        session = new_session(None)
        list(session.run('x = coordinates("x")\ny = coordinates("x", fresh=True)\nP = ideal(x)\nL = localize(P)'))
        for expression, message in (("localize(P, ideal(y))", "coordinate"), ("L.fraction(y)", "coordinate"),
                                    ("localize(P, ideal(x, order='lex'))", "order")):
            with self.assertRaisesRegex(SessionError, message):
                list(session.run(expression))
