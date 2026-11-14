"""Local standard bases and point-specific normal forms through the guided UI."""

import json
import shlex
import subprocess

from engine.polynomial import parse_expression
from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT


class BrowserLocalStandardBasisTests(AlgebraBrowserCase):
    def submit(self, name=""):
        page = self.page
        page.locator("#result-name").fill(name)
        with page.expect_response(self.url + "/api/algebra", timeout=65000) as api:
            page.locator("#run").click()
        record = json.loads(api.value.body())
        self.assertEqual(record["status"], "ok", record)
        expect(page.locator("#result")).to_be_visible()
        return api.value.request.post_data_json, record

    def replay(self, script):
        result = subprocess.run(["./atlas", "run", "-"], cwd=ROOT, input=script,
                                capture_output=True, text=True, timeout=65)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line)["result"] for line in result.stdout.splitlines()]

    def test_wide_shifted_saved_basis_and_normal_form_replay(self):
        page = self.page
        choose = page.locator("#operation").select_option
        n = 2**64+7
        page.locator("#variables").fill("x, y, z")
        choose("local-standard-basis")
        self.rows("#generators", "#add-generator", [f"x^{n}*z-y^2"])
        page.locator("#local-point").fill("1, 0, 0")
        page.locator("#compute").select_option("cpu")
        _, basis = self.submit("B")
        self.assertTrue(basis["result"]["generators"])
        expect(page.locator("#result-ring")).to_contain_text("at (x = 1, y = 0, z = 0)")
        choose("local-normal-form")
        page.locator("#ideal-source").select_option("B")
        expect(page.locator("#local-point")).to_have_value("1, 0, 0")
        page.locator("#polynomial").fill("z")
        page.locator("#compute").select_option("gpu")
        request, record = self.submit("R")
        self.assertEqual(request["operands"]["ideal"], basis["result"]["generators"])
        result = record["result"]
        self.assertEqual(result["point"], ["1","0","0"])
        # Here x^N*z-y² is the generator. The class of z is y²/x^N.
        self.assertEqual((result["unit"], result["remainder"], result["reduction"]), (f"x^{n}", "y^2", "standard"))
        expect(page.locator(".local-representative")).to_have_text(f"y^2 / x^{n}")
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [result])
        shell = subprocess.run(shlex.split(page.locator("#command").text_content()), cwd=ROOT,
                               capture_output=True, text=True, timeout=65)
        self.assertEqual(shell.returncode, 0, shell.stdout + shell.stderr)
        self.assertEqual(json.loads(shell.stdout)["result"], result)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("local_normal_form(", script)
        self.assertIn("B, point=", script)
        self.assertIn(str(n), script)
        self.assertEqual(self.replay(script), [basis["result"], result])

    def test_fractional_point_local_normal_form_with_nonconstant_unit_and_replay(self):
        page = self.page
        choose = page.locator("#operation").select_option
        choose("local-standard-basis")
        expect(page.locator("#point-label")).to_have_text("Coordinates (x, y)")
        page.locator("#local-point").fill("2/4, -4/2")
        self.rows("#generators", "#add-generator", ["(x-1/2)*(1-(y+2))-(y+2)^2"])
        _, basis = self.submit("B")
        self.assertEqual(basis["result"]["point"], ["1/2", "-2"])
        expect(page.locator("#result-ring")).to_contain_text("at (x = 1/2, y = -2)")
        expect(page.locator("#result-value")).to_contain_text(basis["result"]["generators"][0])

        # A point-local basis is offered only when the operation supports it.
        choose("normal-form")
        self.assertEqual(page.locator('#ideal-source option[value="B"]').count(), 0)
        expect(page.locator("#local-point")).to_be_hidden()
        choose("local-normal-form")
        page.locator("#local-point").fill("")
        page.locator("#ideal-source").select_option("B")
        expect(page.locator("#local-point")).to_have_value("1/2, -2")
        page.locator("#local-point").fill("1, -2")
        requests = []
        page.on("request", lambda req: requests.append(req.url))
        page.locator("#result-name").fill("R")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("was computed at (1/2, -2)")
        self.assertNotIn(self.url + "/api/algebra", requests)
        page.locator("#local-point").fill("002/+004, 4/-2")
        page.locator("#compute").select_option("gpu")
        page.locator("#polynomial").fill("x-1/2")
        request, record = self.submit("R")
        self.assertNotIn("basis_ref", request["operands"])
        self.assertEqual(request["operands"]["ideal"], basis["result"]["generators"])
        self.assertEqual(request["operands"]["point"], ["1/2", "-2"])
        self.assertEqual(record["execution"]["requested"], "gpu")
        expect(page.locator("#execution")).to_contain_text(record["execution"]["device"], ignore_case=True)
        result = record["result"]
        self.assertEqual(result["reduction"], "standard")
        self.assertEqual(parse_expression(result["remainder"], ["x", "y"]), parse_expression("(y+2)^2", ["x", "y"]))
        self.assertEqual(parse_expression(result["unit"], ["x", "y"]), parse_expression("1-(y+2)", ["x", "y"]))
        expect(page.locator(".local-representative")).to_have_text(result["remainder"] + " / " + result["unit"])
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [result])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn('B, point=["1/2","-2"]', script)
        self.assertEqual(self.replay(script), [basis["result"], result])

    def test_origin_weak_normal_form_and_wide_coordinate(self):
        page = self.page
        page.locator("#operation").select_option("local-normal-form")
        self.rows("#generators", "#add-generator", ["x*(1-y)-y^2"])
        page.locator("#polynomial").fill("1+x")
        request, record = self.submit()
        self.assertNotIn("point", request["operands"])
        self.assertEqual(record["result"]["point"], ["0", "0"])
        self.assertEqual(record["result"]["reduction"], "weak")
        expect(page.locator("#result-value")).to_contain_text("Weak normal form")
        expect(page.locator("#result-value")).to_contain_text(record["result"]["remainder"])
        expect(page.locator("#result-ring")).to_contain_text("at (x = 0, y = 0)")

        point = str(2**80 + 1)
        page.locator("#variables").fill("z")
        self.rows("#generators", "#add-generator", [])
        page.locator("#local-point").fill(point)
        page.locator("#polynomial").fill("z")
        request, record = self.submit()
        self.assertEqual(request["operands"]["point"], [point])
        self.assertEqual(record["result"]["point"], [point])
        expect(page.locator("#result-ring")).to_contain_text("at (z = " + point + ")")
        self.assertEqual(self.replay(self.download("#download-script")), [record["result"]])
