"""Resolution by weighted blowups through every entry point, checked by independent calculations."""
import json
import re
import subprocess
import threading
import unittest
from pathlib import Path

import sympy

from engine.app import calculate_resolution
from engine.session import new_session
from engine.session_values import Resolution
from engine.worker import evaluate
from tests.cli_support import run_prompt, run_script

ROOT = Path(__file__).resolve().parents[1]
DOUBLE_CONIC = "(x*y-z^2)^2+x^5+y^5"
B9 = "z^2 - (x^2*(x+2*y)^2 + 4*y^3*(x+y)^2)*(x^2*(x+2*y)^2 + 4*y*(x+y)^2)"


def built():
    return (ROOT / "native" / "build" / "weighted").is_file()


def _expression(text, names):
    return sympy.sympify(text.replace("^", "**"), locals={name: sympy.Symbol(name) for name in names})


def chart_checks(record, chart):
    """Independent checks of one chart: f vanishes on the strict transform there, and on a
    rational chart without ambient equations f pulls back to the strict transform times each
    exceptional coordinate to exactly the order N of its divisor."""
    names = list(chart["coordinates"])
    field = chart.get("coefficient_field")
    if field:
        names.append(field["generator"])
    original = {v: sympy.Symbol(v) for v in record["variables"]}
    f = _expression(record["polynomial"], record["variables"])
    images = {original[v]: _expression(chart["map"][v], names) for v in record["variables"]}
    pulled = sympy.expand(f.subs(images, simultaneous=True))
    strict = _expression(chart["strict_transform"], names)
    relations = [_expression(a, names) for a in chart.get("ambient_ideal", ())]
    if field:
        relations.append(_expression(field["minimal_polynomial"], names))
    symbols = [sympy.Symbol(name) for name in names]
    basis = sympy.groebner([*relations, strict], *symbols, order="grevlex")
    if basis.reduce(pulled)[1] != 0:
        return f"f does not vanish on the strict transform of {chart['id']}"
    if relations:
        return None
    quotient, remainder = sympy.div(pulled, strict, *symbols)
    if remainder != 0:
        return f"the strict transform does not divide the pullback in {chart['id']}"
    orders = {v["id"]: v["total_multiplicity"] for v in record["dual_complex"]["vertices"]}
    for divisor in chart["exceptional"]:
        t = sympy.Symbol(divisor["equation"])
        order = sympy.Poly(quotient, t).monoms()
        lowest = min(m[0] for m in order)
        if lowest != orders[divisor["id"]]:
            return f"{divisor['id']} has order {lowest}, not {orders[divisor['id']]}, in {chart['id']}"
    return None


@unittest.skipUnless(built(), "Build the native weighted resolution with make.")
class WeightedResolutionTest(unittest.TestCase):
    def setUp(self):
        self.session = new_session(calculate_resolution)

    def run_lines(self, source):
        result = None
        for result in self.session.run(source):
            pass
        return result

    def resolution(self, polynomial, names="x, y, z"):
        coords = ", ".join(f'"{n.strip()}"' for n in names.split(","))
        self.run_lines(f"{names} = coordinates({coords})\nR = resolve({polynomial}, method=\"weighted\")")
        return self.session.variables["R"]

    def assert_charts(self, record):
        for chart in record["charts"]:
            with self.subTest(chart=chart["id"]):
                self.assertIsNone(chart_checks(record, chart))

    def test_a_double_conic_is_blown_up_along_its_curve(self):
        R = self.resolution(DOUBLE_CONIC)
        record = R.record
        self.assertEqual((record["engine"], record["mode"], record["status"]), ("weighted", "embedded", "resolved"))
        self.assertTrue(record["certificate"]["certified"])
        self.assertIn("positive dimension", [step["locus"] for step in record["steps"]])
        self.assertTrue(any(chart.get("ambient_ideal") for chart in record["charts"]))
        self.assert_charts(record)

    def test_numerical_data_of_one_weighted_blowup(self):
        R = self.resolution("x^2+y^3+z^5")
        record = R.record
        self.assertEqual([step["invariant"] for step in record["steps"]], [["2", "3", "5"]])
        (divisor,) = record["dual_complex"]["vertices"]
        # The ray (15, 10, 6): N = 30 and nu = 15 + 10 + 6.
        self.assertEqual((divisor["total_multiplicity"], divisor["nu"]), (30, 31))
        self.assertEqual(sorted(chart.get("orbifold_order", 1) for chart in record["charts"]), [6, 10, 15])
        self.assertEqual([d.id for d in R.divisors], ["E1"])
        self.assert_charts(record)

    def test_wide_coefficients_and_exponents(self):
        R = self.resolution("x^2 + 98765432109876543210/7*y^3 + z^41")
        record = R.record
        self.assertEqual([step["invariant"] for step in record["steps"]], [["2", "3", "41"]])
        self.assertTrue(record["certificate"]["certified"])
        self.assert_charts(record)

    def test_lines_and_a_number_field(self):
        R = self.resolution("x*y^2*z^3-t^6", "x, y, z, t")
        record = R.record
        self.assertTrue(any(chart.get("coefficient_field") for chart in record["charts"]))
        self.assertGreaterEqual(sum(step["locus"] == "positive dimension" for step in record["steps"]), 2)
        self.assert_charts(record)

    def test_the_b9_surface(self):
        record = self.resolution(B9).record
        self.assertTrue(record["certificate"]["certified"])
        self.assert_charts(record)

    def test_a_smooth_germ_gets_the_identity(self):
        R = self.resolution("x + y^2")
        self.assertEqual([chart.id for chart in R.charts], ["identity"])
        self.assertEqual(R.record["steps"], [])

    def test_what_the_method_refuses(self):
        self.run_lines('x, y, z = coordinates("x", "y", "z")')
        cases = {
            'resolve(x^2+y^3+z^5, method="weighted", mode="intrinsic")': "use the embedded mode",
            'resolve(x^2+y^3+z^5, method="weighted", monodromy=True)': "does not compute zeta functions",
            'resolve(x^2+y^3+z^5+1, method="weighted")': "does not pass through the origin",
            'resolve(x^2+y^3+z^5, method="newton")': "Method must be auto or weighted",
        }
        for source, message in cases.items():
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    self.run_lines(source)

    def test_the_worker_and_a_saved_record(self):
        status, record = evaluate({"polynomial": DOUBLE_CONIC, "variables": ["x", "y", "z"], "method": "weighted"})
        self.assertEqual((status, record["engine"]), (200, "weighted"))
        saved = Resolution(json.loads(json.dumps(record)))
        self.assertEqual([c.id for c in saved.charts], [c["id"] for c in record["charts"]])
        # The replayed chart pulls f back by its recorded map; the strict transform divides it.
        chart = next(c for c in saved.charts if "ambient_ideal" not in c._record and "W" in c.id)
        names = [str(t) for t in chart.coordinates]
        pulled = _expression(str(chart.pullback(saved.polynomial)), names)
        strict = _expression(str(chart.strict_transform), names)
        self.assertEqual(sympy.div(pulled, strict, *map(sympy.Symbol, names))[1], 0)

    def test_scripts_and_the_prompt(self):
        done = run_script(f"""
            x, y, z = coordinates("x", "y", "z")
            R = resolve({DOUBLE_CONIC}, method="weighted")
            print_json(engine=R.record["engine"], certified=R.certificate["certified"], steps=len(R.steps))
            """)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), {"certified": True, "engine": "weighted", "steps": 4})
        status, out, err = run_prompt(f"""
            x, y, z = coordinates("x", "y", "z")
            R = resolve({DOUBLE_CONIC}, method="weighted")
            R
            help R
            """)
        self.assertEqual(status, 0, err)
        self.assertIn("by weighted blowups", out)
        self.assertIn("weighted-resolution.md#resolve-with-method-weighted", out)

    def test_the_cli_and_http(self):
        done = subprocess.run([str(ROOT / "atlas"), "resolve", DOUBLE_CONIC, "--vars", "x,y,z",
                               "--method", "weighted", "--json"], capture_output=True, text=True, cwd=ROOT, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        record = json.loads(done.stdout)
        self.assertEqual((record["engine"], record["certificate"]["certified"]), ("weighted", True))
        shown = subprocess.run([str(ROOT / "atlas"), "resolve", "x^2+y^3+z^5", "--vars", "x,y,z",
                                "--method", "weighted"], capture_output=True, text=True, cwd=ROOT, timeout=60)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("Weighted resolution with 1 weighted blowup", shown.stdout)

        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        from engine.app import AtlasServer
        from tests.http_support import QuietHandler
        server = AtlasServer(("127.0.0.1", 0), QuietHandler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.015}, daemon=True)
        thread.start()

        def post(path, body):
            request = Request(f"http://127.0.0.1:{server.server_port}{path}", method="POST",
                              data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=60) as response:
                    return response.status, json.load(response)
            except HTTPError as error:
                return error.code, json.load(error)
        try:
            body = {"polynomial": "x^2+y^3+z^5", "variables": ["x", "y", "z"], "method": "weighted",
                    "search": "quick"}
            status, answer = post("/api/resolve", body)
            self.assertEqual((status, answer["engine"]), (200, "weighted"))
            status, answer = post("/api/resolve", dict(body, mode="intrinsic"))
            self.assertEqual(status, 400)
            status, answer = post("/api/searches", dict(body, method="newton", search="exhaustive"))
            self.assertEqual((status, answer["error"]), (400, "Method must be auto or weighted."))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
