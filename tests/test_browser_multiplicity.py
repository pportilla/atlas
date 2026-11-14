"""Finite-colength quotient multiplicity in Chromium and exported Atlas sessions."""

import json
import shlex
import subprocess
from unittest.mock import patch

from engine.algebra_basis_store import AlgebraBasisStore
from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT


class MultiplicityBrowserTests(AlgebraBrowserCase):
    def compute(self, name=""):
        page = self.page
        page.locator("#result-name").fill(name)
        with page.expect_response(self.url + "/api/algebra") as request:
            page.locator("#run").click()
        response = request.value
        record = json.loads(response.body())
        self.assertEqual(record["status"], "ok", record)
        expect(page.locator("#result")).to_be_visible()
        return response.request.post_data_json["operands"], record

    def replay(self, script):
        run = subprocess.run(["./atlas", "run", "-"], cwd=ROOT, input=script, capture_output=True,
                             text=True, timeout=65)
        self.assertEqual(run.returncode, 0, run.stderr)
        return [json.loads(line)["result"] for line in run.stdout.splitlines()]

    def test_repeated_and_quadratic_points_saved_ideal_and_basis(self):
        page = self.page
        choose = page.locator("#operation").select_option
        source = page.locator("#ideal-source")
        choose("ideal")
        self.rows("#generators", "#add-generator", ["(x-1)^2*(x^2+1)", "2*y-x"])
        _, ideal = self.compute("I")
        choose("groebner")
        source.select_option("I")
        _, basis = self.compute("B")
        choose("multiplicity")
        expect(page.locator("#local-point")).to_be_hidden()
        expect(page.locator("#degrees")).to_be_hidden()
        operands, direct = self.compute()
        self.assertEqual(operands, {"ideal": ideal["result"]["generators"]})
        source.select_option("B")
        operands, record = self.compute("multiplicity")
        self.assertEqual(operands, {"basis_ref": basis["basis_ref"]})
        expect(page.locator("#execution")).to_contain_text("B reused")
        result = record["result"]
        self.assertEqual(result, direct["result"])
        self.assertEqual((result["type"], result["scope"], result["finite"], result["value"]),
                         ("integer", "global", True, 4))
        expect(page.locator("#result-title")).to_contain_text("Quotient multiplicity")
        expect(page.locator("#result-ring")).to_contain_text("global quotient")
        expect(page.locator("#result-value")).to_have_text("4")
        self.assertEqual(self.replay(self.download("#download-script")), [result])
        args = shlex.split(page.locator("#command").text_content())
        run = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True, timeout=65)
        self.assertEqual(json.loads(run.stdout)["result"], result)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("multiplicity(I,", script)
        self.assertIn("multiplicity(B,", script)
        self.assertEqual(self.replay(script), [basis["result"], result, result])

    def test_basis_rows_when_server_cannot_retain_a_reference(self):
        page = self.page
        page.locator("#operation").select_option("groebner")
        with patch.object(AlgebraBasisStore, "_remember", return_value=None):
            _, basis = self.compute("G")
        self.assertNotIn("basis_ref", basis)
        page.locator("#operation").select_option("multiplicity")
        page.locator("#ideal-source").select_option("G")
        operands, record = self.compute()
        self.assertEqual(operands, {"ideal": basis["result"]["generators"]})
        self.assertEqual(record["result"]["value"], 2)
        expect(page.locator("#execution")).not_to_contain_text("G reused")

    def test_exact_integer_infinite_colength_and_scalar_ring(self):
        page = self.page
        page.locator("#operation").select_option("multiplicity")
        value = page.locator("#result-value")
        degree = 2**64 + 7
        self.rows("#generators", "#add-generator", [f"x^{degree}", "y^3"])
        _, record = self.compute()
        self.assertEqual(record["result"]["value"], 3 * degree)
        expect(value).to_have_text(str(3 * degree))
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [record["result"]])

        self.rows("#generators", "#add-generator", ["x"])
        _, infinite = self.compute()
        self.assertEqual((infinite["result"]["finite"], infinite["result"]["value"]), (False, None))
        expect(value).to_contain_text("infinite colength")
        self.assertEqual(json.loads(self.download("#export"))["result"], infinite["result"])
        page.locator("#variables").fill("")
        for generators, expected in (([], 1), (["2/3"], 0)):
            self.rows("#generators", "#add-generator", generators)
            _, scalar = self.compute()
            self.assertEqual((scalar["result"]["finite"], scalar["result"]["value"]), (True, expected))
            expect(value).to_have_text(str(expected))
