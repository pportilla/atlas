"""Exercise both GCD operand fields, named polynomial reuse and portable exports."""

import json
import subprocess

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT


LEFT = "(x+y)*(x+1)"
RIGHT = "(x+y)*(y+1)"
RING = {"field": "QQ", "variables": ["x", "y"], "order": "lex"}


class GcdBrowserTests(AlgebraBrowserCase):
    def submit(self, name=""):
        self.page.locator("#result-name").fill(name)
        with self.page.expect_response(self.url + "/api/algebra") as api:
            self.page.locator("#run").click()
        record = api.value.json()
        self.assertEqual(api.value.status, 200, record)
        self.assertEqual(record["result"]["ring"], RING)
        expect(self.page.locator("#result-ring")).to_have_text("ℚ[x, y] · lex")
        expect(self.page.locator("#result-value")).to_have_text(record["result"]["polynomial"])
        return api.value.request.post_data_json["operands"], record

    def run_export(self, command, source=None):
        completed = subprocess.run(command, cwd=ROOT, input=source, capture_output=True,
                                   text=True, timeout=65)
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        return completed.stdout

    def test_literal_gcd_zero_and_stale_ideal_rows(self):
        page = self.page
        page.locator("#operation").select_option("quotient")
        self.rows("#generators", "#add-generator", ["undeclared"])
        self.rows("#second-generators", "#add-second-generator", ["undeclared"])
        page.locator("#operation").select_option("gcd")
        expect(page.locator("#generators")).to_be_hidden()
        expect(page.locator("#second-generators")).to_be_hidden()
        for left, right, expected in ((LEFT, RIGHT, "x+y"), ("0", "0", "0")):
            with self.subTest(result=expected):
                page.locator("#polynomial").fill(left)
                page.locator("#other-polynomial").fill(right)
                operands, record = self.submit()
                self.assertEqual(operands, {"polynomial": left, "other_polynomial": right})
                self.assertEqual(record["result"], {"type": "polynomial", "ring": RING, "polynomial": expected})
                self.assertEqual(json.loads(self.download("#export")), record)
                script = self.download("#download-script")
                repeated = self.run_export(["./atlas", "run", "-"], script)
                self.assertEqual(json.loads(repeated)["result"], record["result"])

    def test_named_polynomials_gcd_and_reuse(self):
        page = self.page
        self.rows("#generators", "#add-generator", ["0"])
        records = []
        for name, literal in (("F", LEFT), ("H", RIGHT)):
            page.locator("#polynomial").fill(literal)
            _, record = self.submit(name)
            records.append(record)

        page.locator("#operation").select_option("gcd")
        page.locator("#polynomial-source").select_option("F")
        page.locator("#other-polynomial-source").select_option("H")
        expect(page.locator("#polynomial")).to_be_hidden()
        expect(page.locator("#other-polynomial")).to_be_hidden()
        operands, divisor = self.submit("D")
        self.assertEqual(operands, {"polynomial": records[0]["result"]["polynomial"],
                                    "other_polynomial": records[1]["result"]["polynomial"]})
        self.assertEqual(divisor["result"]["polynomial"], "x+y")
        self.assertEqual(json.loads(self.download("#export")), divisor)
        page.locator("#copy-command").click()
        expect(page.locator("#copy-status")).to_have_text("Command copied.")
        command = page.evaluate("navigator.clipboard.readText()")
        self.assertEqual(json.loads(self.run_export(["sh", "-c", command]))["result"], divisor["result"])
        records.append(divisor)

        page.locator("#operation").select_option("normal-form")
        page.locator("#polynomial-source").select_option("D")
        self.rows("#generators", "#add-generator", ["x"])
        operands, reduced = self.submit("R")
        self.assertEqual(operands, {"ideal": ["x"], "polynomial": "x+y"})
        self.assertEqual(reduced["result"]["polynomial"], "y")
        records.append(reduced)

        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertRegex(script, r"D\s*=\s*gcd\(F,\s*H\b")
        self.assertRegex(script, r"R\s*=\s*normal_form\(D,")
        replayed = self.run_export(["./atlas", "run", "-"], script)
        self.assertEqual([json.loads(line)["result"] for line in replayed.splitlines()],
                         [item["result"] for item in records])
