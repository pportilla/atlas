"""Factor rational polynomials in the browser and replay named factors in scripts."""

import json
import shlex
import subprocess

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT


class FactorBrowserTests(AlgebraBrowserCase):
    def submit(self, status=200):
        with self.page.expect_response(self.url + "/api/algebra") as pending:
            self.page.locator("#run").click()
        response = pending.value
        record = json.loads(response.text())
        self.assertEqual(response.status, status, record)
        return response.request.post_data_json["operands"], record

    def replay(self, args, script=None):
        completed = subprocess.run(args, cwd=ROOT, input=script, capture_output=True, text=True, timeout=65)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        if script is None:
            return [json.loads(completed.stdout)["result"]]
        return [json.loads(line)["result"] for line in completed.stdout.splitlines()]

    def test_named_factors_and_portable_exports(self):
        page = self.page
        page.locator("#operation").select_option("gcd")
        page.locator("#other-polynomial").fill("undeclared")
        page.locator("#operation").select_option("factor")
        expect(page.locator("#other-polynomial")).to_be_hidden()
        expect(page.locator("#generators")).to_be_hidden()
        source = "-3/2*x^2*(x+y)"
        page.get_by_role("textbox", name="Polynomial to factor").fill(source)
        page.locator("#result-name").fill("factor")
        operands, record = self.submit()
        self.assertEqual(operands, {"polynomial": source})
        result = record["result"]
        self.assertEqual(result["unit"], "-3/2")
        self.assertCountEqual(result["factors"], [
            {"polynomial": "x", "multiplicity": 2}, {"polynomial": "x+y", "multiplicity": 1}])
        expect(page.locator("#result-value")).to_contain_text("-3/2")
        expect(page.get_by_role("columnheader", name="Multiplicity")).to_be_visible()
        for index, entry in enumerate(result["factors"]):
            cells = page.locator("#result-value tbody tr").nth(index).get_by_role("cell")
            expect(cells).to_have_text([entry["polynomial"], str(entry["multiplicity"])])
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(shlex.split(page.locator("#command").text_content())), [result])
        self.assertEqual(self.replay(["./atlas", "run", "-"], self.download("#download-script")), [result])

        # Reuse x+y as the dividend of a normal form. The literal ideal needs
        # coordinates from the factorization's owner when replayed.
        index = next(i for i, entry in enumerate(result["factors"]) if entry["polynomial"] == "x+y")
        field = f".factors[{index}].polynomial"
        page.locator("#operation").select_option("normal-form")
        page.locator("#order").select_option("grevlex")
        option = page.locator(f'#polynomial-source option[value="factor{field}"]')
        expect(option).to_have_js_property("disabled", True)
        page.locator("#order").select_option("lex")
        page.locator("#polynomial-source").select_option("factor" + field)
        page.locator("#result-name").fill("R")
        self.rows("#generators", "#add-generator", ["x"])
        operands, reduced = self.submit()
        self.assertEqual(operands, {"ideal": ["x"], "polynomial": "x+y"})
        self.assertEqual(reduced["result"]["polynomial"], "y")
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn(field, script)
        self.assertEqual(self.replay(["./atlas", "run", "-"], script), [result, reduced["result"]])

    def test_constants_zero_and_exact_multiplicity(self):
        page = self.page
        page.locator("#operation").select_option("factor")
        page.locator("#variables").fill("")
        page.locator("#polynomial").fill("2/7")
        _, record = self.submit()
        self.assertEqual(record["result"]["factors"], [])
        expect(page.locator("#result-value")).to_contain_text("2/7")
        expect(page.locator("#result-value")).to_contain_text("no irreducible factors")
        page.locator("#polynomial").fill("0")
        _, failure = self.submit(400)
        expect(page.locator("#error")).to_contain_text(failure["error"]["message"])
        expect(page.locator("#result")).to_be_hidden()

        page.locator("#variables").fill("x")
        power = 2**64 + 17
        page.locator("#polynomial").fill(f"x^{power}")
        self.submit()
        expect(page.get_by_role("cell", name=str(power), exact=True)).to_be_visible()
        exported = json.loads(self.download("#export"))
        self.assertEqual(exported["result"]["factors"], [{"polynomial": "x", "multiplicity": power}])
        page.locator("#session-replay summary").click()
        replayed = self.replay(["./atlas", "run", "-"], self.download("#download-session"))
        self.assertEqual(replayed, [record["result"], exported["result"]])
