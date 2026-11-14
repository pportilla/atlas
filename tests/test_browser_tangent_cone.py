"""Tangent directions, graded series, and saved cone vertex ownership in Chromium."""

import json
import math
import shlex
import subprocess

from engine.polynomial import integer_from_decimal
from tests.algebra_browser import AlgebraBrowserCase, ROOT, expect


class TangentConeBrowserTests(AlgebraBrowserCase):
    def submit(self, name="", compute="cpu"):
        page = self.page
        page.locator("#result-name").fill(name)
        page.locator("#compute").select_option(compute)
        with page.expect_response(self.url + "/api/algebra", timeout=65000) as pending:
            page.locator("#run").click()
        response = pending.value
        answer = json.loads(response.body(), parse_int=integer_from_decimal)
        self.assertEqual(answer["status"], "ok", answer)
        self.assertEqual(answer["execution"]["requested"], compute)
        expect(page.locator("#result")).to_be_visible()
        return response.request.post_data_json["operands"], answer

    def replay(self, script=None, command=None):
        run = subprocess.run(command or ["./atlas", "run", "-"], cwd=ROOT, input=script,
                             capture_output=True, text=True, timeout=65)
        self.assertEqual(run.returncode, 0, run.stderr)
        records = [run.stdout] if command else run.stdout.splitlines()
        return [json.loads(record, parse_int=integer_from_decimal)["result"] for record in records]

    def test_translated_cusp_saved_local_basis_and_cone_vertex(self):
        page = self.page
        choose = page.locator("#operation").select_option
        point = page.locator("#local-point")
        source = page.locator("#ideal-source")
        choose("local-standard-basis")
        self.rows("#generators", "#add-generator", ["(y+2)^2-(x-1/2)^3"])
        point.fill("1/2, -2")
        _, basis = self.submit("B")
        results = [basis["result"]]
        choose("tangent-cone")
        source.select_option("B")
        expect(point).to_have_value("1/2, -2")
        _, record = self.submit("C", "gpu")
        cone = record["result"]
        self.assertEqual((cone["point"], cone["coordinate_system"], cone["generators"]),
                         (["1/2", "-2"], "tangent", ["y^2"]))
        self.assertEqual((cone["dimension"], cone["multiplicity"]), (1, 2))
        expect(page.locator("#result-value")).to_contain_text("tangent directions")
        expect(page.locator("#result-value")).to_contain_text("y^2")
        expect(page.locator("#result-ring")).to_contain_text("at (x = 1/2, y = -2)")
        expect(page.locator("#execution")).to_contain_text(record["execution"]["device"], ignore_case=True)
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [cone])
        results.append(cone)

        choose("tangent-cone-hilbert-series")
        _, direct = self.submit()
        self.assertEqual(direct["result"]["point"], cone["point"])
        results.append(direct["result"])
        source.select_option("C")
        expect(point).to_have_value("")
        expect(page.locator("#point-label")).to_contain_text("Vertex")
        expect(page.locator("#problem-equations")).to_contain_text("C.generators")
        operands, record = self.submit("S")
        self.assertNotIn("point", operands)
        self.assertEqual(operands["ideal"], ["y^2"])
        series = record["result"]
        self.assertEqual(series["scope"], "origin")
        self.assertNotIn("point", series)
        self.assertEqual(series["numerator"], [{"degree": 0, "coefficient": 1}, {"degree": 2, "coefficient": -1}])
        self.assertEqual(series["denominator_power"], 2)
        expect(page.locator(".graded-series")).to_have_text("H(t) = 1-t^2 / (1-t)^2")
        self.assertEqual(self.replay(command=shlex.split(page.locator("#command").text_content())), [series])
        results.append(series)
        point.fill("1/2, -2")
        page.locator("#result-name").fill("")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("all tangent coordinates zero")
        point.fill("0, -0/2")
        _, explicit = self.submit()
        self.assertEqual((explicit["result"]["scope"], explicit["result"]["point"]), ("point", ["0", "0"]))
        results.append(explicit["result"])
        choose("quotient-basis")
        self.assertEqual(source.locator('option[value="C"]').count(), 0)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("tangent_cone(B,", script)
        self.assertIn('tangent_cone_hilbert_series(C, compute="cpu")', script)
        self.assertIn('tangent_cone_hilbert_series(C, point=["0","0"]', script)
        self.assertEqual(self.replay(script), results)

    def test_series_sparse_signs_wide_exponent_and_empty_cone(self):
        page = self.page
        choose = page.locator("#operation").select_option
        choose("tangent-cone-hilbert-series")
        self.rows("#generators", "#add-generator", ["x^2", "y^2"])
        _, complete = self.submit()
        self.assertEqual(complete["result"]["numerator"], [{"degree": 0, "coefficient": 1},
                         {"degree": 2, "coefficient": -2}, {"degree": 4, "coefficient": 1}])
        formula = page.locator(".graded-series")
        expect(formula).to_have_text("H(t) = 1-2*t^2+t^4 / (1-t)^2")
        degree = 2**80 + 1
        self.rows("#generators", "#add-generator", [f"x^{degree}"])
        _, wide = self.submit()
        self.assertEqual(wide["result"]["numerator"][-1], {"degree": degree, "coefficient": -1})
        expect(formula).to_have_text(f"H(t) = 1-t^{degree} / (1-t)^2")
        self.assertEqual(self.replay(self.download("#download-script")), [wide["result"]])

        choose("tangent-cone")
        self.rows("#generators", "#add-generator", ["x-1"])
        _, off_support = self.submit()
        self.assertEqual((off_support["result"]["generators"], off_support["result"]["dimension"],
                          off_support["result"]["multiplicity"]), (["1"], -1, 0))
        expect(page.locator("#result-value")).to_contain_text("Krull dimension-1")
        choose("tangent-cone-hilbert-series")
        _, zero = self.submit()
        self.assertEqual((zero["result"]["numerator"], zero["result"]["denominator_power"]), ([], 2))
        expect(formula).to_have_text("H(t) = 0 / (1-t)^2")
        page.locator("#variables").fill("")
        self.rows("#generators", "#add-generator", [])
        _, field = self.submit()
        self.assertEqual(field["result"]["denominator_power"], 0)
        expect(formula).to_have_text("H(t) = 1")

    def test_exact_large_signed_series_coefficients(self):
        page = self.page
        variables = [f"x{i}" for i in range(60)]
        page.locator("#operation").select_option("tangent-cone-hilbert-series")
        page.locator("#variables").fill(", ".join(variables))
        self.rows("#generators", "#add-generator", variables)
        _, record = self.submit("S")
        # The residue field in 60 variables has numerator (1-t)^60.
        result = record["result"]
        for degree in (29, 30):
            coefficient = (-1)**degree * math.comb(60, degree)
            self.assertEqual(result["numerator"][degree], {"degree": degree, "coefficient": coefficient})
            expect(page.locator(".graded-series")).to_contain_text(f"{coefficient:+d}*t^{degree}")
        self.assertEqual(json.loads(self.download("#export"))["result"], result)
        page.locator("#session-replay summary").click()
        self.assertEqual(self.replay(self.download("#download-session")), [result])
