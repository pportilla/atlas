"""Standard graded QQ quotients in the guided Algebra browser and exported sessions."""

import json
import shlex
import subprocess

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT


class HilbertSeriesBrowserTests(AlgebraBrowserCase):
    def submit(self, name=""):
        page = self.page
        page.locator("#result-name").fill(name)
        with page.expect_response(self.url + "/api/algebra") as pending:
            page.locator("#run").click()
        response = pending.value
        record = json.loads(response.body())
        self.assertEqual(record["status"], "ok", record)
        expect(page.locator("#result")).to_be_visible()
        return response.request.post_data_json["operands"], record

    def replay(self, script=None, command=None):
        run = subprocess.run(command or ["./atlas", "run", "-"], cwd=ROOT, input=script,
                             capture_output=True, text=True, timeout=65)
        self.assertEqual(run.returncode, 0, run.stderr)
        records = [run.stdout] if command else run.stdout.splitlines()
        return [json.loads(record)["result"] for record in records]

    def test_fresh_example_switches_only_while_the_generators_are_untouched(self):
        page = self.page
        choose = page.locator("#operation").select_option
        rows = page.locator("#generators input")
        choose("hilbert-series")
        operands, _ = self.submit()
        self.assertEqual(operands["ideal"], ["x^2", "y^3"])
        expect(page.locator(".graded-series")).to_have_text("H(t) = 1-t^2-t^3+t^5 / (1-t)^2")
        choose("normal-form")
        expect(rows.nth(0)).to_have_value("x^2-1")
        expect(rows.nth(1)).to_have_value("y-x")
        page.locator("#variables").fill("x, y, z")
        choose("hilbert-series")
        expect(rows.first).to_have_value("x^2-1")
        page.locator("#variables").fill("x, y")
        expect(rows.first).to_have_value("x^2")

        rows.first.fill("x^3")
        choose("groebner")
        expect(rows.first).to_have_value("x^3")
        choose("hilbert-series")
        expect(rows.first).to_have_value("x^3")
        self.rows("#generators", "#add-generator", [])
        choose("normal-form")
        choose("hilbert-series")
        self.assertEqual(self.submit()[0]["ideal"], [])

        page.reload()
        rows.first.fill("x^3")
        rows.first.fill("x^2-1")
        choose("hilbert-series")
        expect(rows.first).to_have_value("x^2-1")

    def test_saved_ideal_and_global_basis_display_and_replay(self):
        page = self.page
        choose = page.locator("#operation").select_option
        source = page.locator("#ideal-source")
        choose("ideal")
        self.rows("#generators", "#add-generator", ["x^2", "x*y", "y^2"])
        _, ideal = self.submit("I")
        choose("groebner")
        source.select_option("I")
        _, basis = self.submit("B")
        choose("hilbert-series")
        expect(page.locator("#local-point")).to_be_hidden()
        expect(page.locator("#degrees")).to_be_hidden()
        operands, direct = self.submit()
        self.assertEqual(operands, {"ideal": ideal["result"]["generators"]})
        source.select_option("B")
        operands, record = self.submit("hilbert_series")
        self.assertEqual(operands, {"ideal": basis["result"]["generators"]})
        result = record["result"]
        self.assertEqual(result, direct["result"])
        self.assertEqual((result["scope"], result["denominator_power"], result["dimension"], result["multiplicity"]),
                         ("global", 2, 0, 3))
        self.assertEqual(result["numerator"], [{"degree": 0, "coefficient": 1}, {"degree": 2, "coefficient": -3},
                                               {"degree": 3, "coefficient": 2}])
        expect(page.locator(".graded-series")).to_have_text("H(t) = 1-3*t^2+2*t^3 / (1-t)^2")
        expect(page.locator("#result-value")).to_contain_text("Multiplicity3")
        expect(page.locator("#result-ring")).to_contain_text("global quotient")
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [result])
        self.assertEqual(self.replay(command=shlex.split(page.locator("#command").text_content())), [result])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("hilbert_series(I,", script)
        self.assertIn("hilbert_series(B,", script)
        self.assertEqual(self.replay(script), [basis["result"], result, result])

    def test_sparse_degree_scalar_and_unit_ideals_and_inhomogeneous_error(self):
        page = self.page
        page.locator("#operation").select_option("hilbert-series")
        page.locator("#variables").fill("x")
        degree = 2**80 + 1
        self.rows("#generators", "#add-generator", [f"x^{degree}"])
        _, wide = self.submit()
        result = wide["result"]
        self.assertEqual((result["numerator"][-1], result["multiplicity"]),
                         ({"degree": degree, "coefficient": -1}, degree))
        expect(page.locator(".graded-series")).to_have_text(f"H(t) = 1-t^{degree} / (1-t)")
        expect(page.locator("#result-value")).to_contain_text(f"Multiplicity{degree}")
        self.assertEqual(json.loads(self.download("#export"))["result"], result)
        self.rows("#generators", "#add-generator", ["x-1"])
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("homogeneous")
        expect(page.locator("#result")).to_be_hidden()

        self.rows("#generators", "#add-generator", ["1"])
        _, unit = self.submit()
        self.assertEqual((unit["result"]["numerator"], unit["result"]["dimension"], unit["result"]["multiplicity"]),
                         ([], -1, 0))
        expect(page.locator(".graded-series")).to_have_text("H(t) = 0 / (1-t)")
        page.locator("#variables").fill("")
        self.rows("#generators", "#add-generator", [])
        _, scalar = self.submit()
        self.assertEqual(scalar["result"]["denominator_power"], 0)
        expect(page.locator(".graded-series")).to_have_text("H(t) = 1")
        self.rows("#generators", "#add-generator", ["2/3"])
        self.submit()
        expect(page.locator(".graded-series")).to_have_text("H(t) = 0")
