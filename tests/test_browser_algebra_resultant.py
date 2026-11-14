"""Select the resultant variable and reuse the surviving coordinates in Chromium."""

import json
import shlex
import subprocess

from tests.algebra_browser import AlgebraBrowserCase, ROOT, expect


class ResultantBrowserTests(AlgebraBrowserCase):
    def submit(self, name=""):
        self.page.locator("#result-name").fill(name)
        with self.page.expect_response(self.url + "/api/algebra") as pending:
            self.page.locator("#run").click()
        response = pending.value
        record = response.json()
        self.assertEqual(response.status, 200, record)
        expect(self.page.locator("#result")).to_be_visible()
        return response.request.post_data_json["operands"], record

    def replay(self, args, script=None):
        run = subprocess.run(args, cwd=ROOT, input=script, capture_output=True, text=True, timeout=65)
        self.assertEqual(run.returncode, 0, run.stderr)
        if script is None:
            return [json.loads(run.stdout)["result"]]
        return [json.loads(line)["result"] for line in run.stdout.splitlines()]

    def test_change_elimination_variable_and_reuse_its_projected_result(self):
        page = self.page
        page.locator("#variables").fill("x, t, y")
        page.locator("#operation").select_option("resultant")
        selector = page.get_by_role("combobox", name="Variable to eliminate", exact=True)
        expect(selector.locator("option")).to_have_text(["x", "t", "y"])
        expect(page.locator("#generators")).to_be_hidden()
        page.get_by_role("textbox", name="First polynomial").fill("t-x")
        page.get_by_role("textbox", name="Second polynomial").fill("2*t-y")
        records = []
        for variable, survivors, value, name in (
            ("x", ["t", "y"], "2*t-y", ""),
            ("t", ["x", "y"], "2*x-y", "R"),
        ):
            selector.select_option(variable)
            operands, record = self.submit(name)
            self.assertEqual(operands, {"polynomial": "t-x", "other_polynomial": "2*t-y", "variables": [variable]})
            expected = {"type": "polynomial", "ring": {"field": "QQ", "order": "lex", "variables": survivors},
                        "polynomial": value}
            self.assertEqual(record["result"], expected)
            expect(page.locator("#result-value")).to_have_text(value)
            expect(page.locator("#result-ring")).to_have_text("ℚ[" + ", ".join(survivors) + "] · lex")
            records.append(expected)
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(shlex.split(page.locator("#command").text_content())), [expected])
        self.assertEqual(self.replay(["./atlas", "run", "-"], self.download("#download-script")), [expected])

        expect(page.locator('#polynomial-source option[value="R"]')).to_have_js_property("disabled", True)
        page.locator("#variables").fill("x,y")
        expect(selector).to_have_value("x")
        page.locator("#polynomial-source").select_option("R")
        page.locator("#other-polynomial").fill("x-1/2")
        operands, projected = self.submit("Q")
        self.assertEqual(operands, {"polynomial": "2*x-y", "other_polynomial": "x-1/2", "variables": ["x"]})
        self.assertEqual(projected["result"]["polynomial"], "y-1")
        self.assertEqual(projected["result"]["ring"]["variables"], ["y"])
        records.append(projected["result"])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("Q = resultant(R,", script)
        self.assertIn("variables=R.coordinates", script)
        self.assertEqual(self.replay(["./atlas", "run", "-"], script), records)

    def test_selector_follows_declared_variables_and_constants_replay_in_QQ(self):
        page = self.page
        page.locator("#operation").select_option("resultant")
        selector = page.get_by_role("combobox", name="Variable to eliminate", exact=True)
        selector.select_option("y")
        page.locator("#variables").fill("y, x, z")
        expect(selector).to_have_value("y")
        page.locator("#variables").fill("")
        page.locator("#run").click()
        self.assertTrue(selector.evaluate("field => field.validity.valueMissing"))
        page.locator("#variables").fill("resultant")
        expect(selector).to_have_value("resultant")
        page.locator("#polynomial").fill("2/3")
        page.locator("#other-polynomial").fill("5")
        operands, record = self.submit("resultant")
        self.assertEqual(operands["variables"], ["resultant"])
        self.assertEqual(record["result"], {"type": "polynomial", "ring": {"field": "QQ", "order": "lex",
                                                                          "variables": []}, "polynomial": "1"})
        expect(page.locator("#result-ring")).to_have_text("ℚ · lex")
        page.locator("#session-replay summary").click()
        self.assertEqual(self.replay(["./atlas", "run", "-"], self.download("#download-session")), [record["result"]])
