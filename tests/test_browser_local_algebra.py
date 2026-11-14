"""Run local-algebra calculations in Chromium and replay their exported Atlas scripts."""

import json
import os
import subprocess
import unittest

from engine.polynomial import parse_expression
from tests.algebra_browser import AlgebraBrowserCase, ROOT, expect


class LocalAlgebraBrowserTests(AlgebraBrowserCase):
    def submit(self, name="", compute="cpu"):
        page = self.page
        page.locator("#compute").select_option(compute)
        page.locator("#result-name").fill(name)
        with page.expect_response(self.url + "/api/algebra", timeout=65000) as api:
            page.locator("#run").click()
        record = json.loads(api.value.body())
        self.assertEqual(record["status"], "ok", record)
        expect(page.locator("#result")).to_be_visible()
        self.assertEqual(record["execution"]["requested"], compute)
        if name:
            expect(page.locator("#session-items").get_by_text(name, exact=True)).to_be_visible()
        return record

    def replay(self, script):
        result = subprocess.run(["./atlas", "run", "-"], cwd=ROOT, input=script, capture_output=True,
                                text=True, timeout=65)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_global_and_origin_quotients_replay_the_saved_ideal(self):
        page = self.page
        choose = page.locator("#operation").select_option
        choose("ideal")
        self.rows("#generators", "#add-generator", ["x^2*(x-1)", "y"])
        self.submit("I")
        results = []
        for op, scope, monomials in (("quotient-basis", "global quotient", ["1", "x", "x^2"]),
                                     ("local-quotient-basis", "at the origin", ["1", "x"])):
            choose(op)
            page.locator("#ideal-source").select_option("I")
            expect(page.locator("#polynomial")).to_be_hidden()
            record = self.submit()
            results.append(record["result"])
            self.assertCountEqual(record["result"]["basis"], monomials)
            expect(page.locator("#result-ring")).to_contain_text(scope)
            expect(page.locator("#result-value")).to_contain_text(", ".join(record["result"]["basis"]))
            self.assertEqual(json.loads(self.download("#export")), record)
            self.assertEqual(self.replay(self.download("#download-script"))[0]["result"], record["result"])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("local_quotient_basis(I,", script)
        self.assertEqual([record["result"] for record in self.replay(script)], results)

    def test_named_jacobian_ideal_and_hypersurface_invariants(self):
        page = self.page
        choose = page.locator("#operation").select_option
        self.rows("#generators", "#add-generator", ["0"])
        page.locator("#polynomial").fill("x^2+y^3")
        results = [self.submit("F")["result"]]
        choose("local-jacobian-algebra")
        page.locator("#polynomial-source").select_option("F")
        expect(page.locator("#first-ideal")).to_be_hidden()
        record = self.submit("A", "gpu")
        expect(page.locator("#execution")).to_contain_text(record["execution"]["device"], ignore_case=True)
        value = page.locator("#result-value")
        result = record["result"]
        self.assertEqual(result["dimension"], 2)
        self.assertCountEqual(result["basis"], ["1", "y"])
        expect(value).to_contain_text("Jacobian ideal generators")
        expect(value).to_contain_text(", ".join(result["generators"]))
        results.append(result)
        choose("local-quotient-basis")
        page.locator("#ideal-source").select_option("A")
        expect(page.locator("#problem-equations")).to_contain_text("A.generators")
        from_jacobian = self.submit()["result"]
        self.assertEqual(from_jacobian["basis"], result["basis"])
        results.append(from_jacobian)
        for op in ("milnor-number", "tjurina-number"):
            choose(op)
            result = self.submit()["result"]
            self.assertEqual(result["value"], 2)
            expect(value).to_have_text("2")
            expect(page.locator("#result-ring")).to_contain_text("at the origin")
            results.append(result)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("local_jacobian_algebra(F,", script)
        self.assertIn("local_quotient_basis(ideal(A.generators,", script)
        self.assertEqual([record["result"] for record in self.replay(script)], results)

    def test_zero_and_infinite_germs_display_and_export(self):
        page = self.page
        value = page.locator("#result-value")
        page.locator("#operation").select_option("local-jacobian-algebra")
        page.locator("#polynomial").fill("x")
        self.assertEqual(self.submit()["result"]["basis"], [])
        expect(value).to_contain_text("∅ (empty basis)")
        page.locator("#polynomial").fill("x^2")
        self.assertIsNone(self.submit()["result"]["basis"])
        expect(value).to_contain_text("Infinite")
        expect(value).not_to_contain_text("empty basis")
        for op in ("milnor-number", "tjurina-number"):
            page.locator("#operation").select_option(op)
            expect(page.locator("#polynomial")).to_have_value("x^2")
            self.assertIsNone(self.submit()["result"]["value"])
            expect(value).to_have_text("Infinite")
            self.assertIsNone(json.loads(self.download("#export"))["result"]["value"])
            page.locator("#polynomial").fill("x")
            self.assertEqual(self.submit()["result"]["value"], 0)
            expect(value).to_have_text("0")
            page.locator("#polynomial").fill("x^2")

    def test_translated_hypersurface_invariants_and_named_derivatives(self):
        page = self.page
        choose = page.locator("#operation").select_option
        point = page.locator("#local-point")
        polynomial = "(x-1/2)^3+(y+2)^2"
        results = []
        choose("local-jacobian-algebra")
        point.fill("2/4, -4/2")
        page.locator("#polynomial").fill(polynomial)
        record = self.submit("A")
        jacobian = record["result"]
        results.append(jacobian)
        self.assertEqual((jacobian["scope"], jacobian["point"]), ("point", ["1/2", "-2"]))
        self.assertEqual([parse_expression(f, ["x", "y"]) for f in jacobian["generators"]],
                         [parse_expression(f, ["x", "y"]) for f in ["3*(x-1/2)^2", "2*(y+2)"]])
        self.assertEqual(jacobian["basis"], ["1", "x-1/2"])
        expect(page.locator("#result-value")).to_contain_text("1, x-1/2")
        expect(page.locator("#result-ring")).to_contain_text("at (x = 1/2, y = -2)")
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script"))[0]["result"], jacobian)

        for op in ("milnor-number", "tjurina-number"):
            choose(op)
            expect(point).to_have_value("2/4, -4/2")
            expect(page.locator("#polynomial")).to_have_value(polynomial)
            result = self.submit()["result"]
            self.assertEqual((result["scope"], result["point"], result["value"]), ("point", jacobian["point"], 2))
            results.append(result)
        point.fill("0, 0")
        with page.expect_response(self.url + "/api/algebra") as api:
            page.locator("#run").click()
        failure = json.loads(api.value.body())
        self.assertEqual(failure["status"], "error")
        expect(page.locator("#error")).to_contain_text(failure["error"]["message"])
        expect(page.locator("#result")).to_be_hidden()

        choose("local-quotient-basis")
        page.locator("#ideal-source").select_option("A")
        expect(point).to_have_value("1/2, -2")
        result = self.submit()["result"]
        self.assertEqual(result["basis"], jacobian["basis"])
        results.append(result)
        # The derivatives of f are global polynomials, so one may localize them elsewhere.
        point.fill("1/2, 0")
        result = self.submit()["result"]
        self.assertEqual((result["point"], result["dimension"]), (["1/2", "0"], 0))
        results.append(result)
        choose("quotient-basis")
        expect(point).to_be_hidden()
        result = self.submit()["result"]
        self.assertEqual((result["scope"], result["dimension"]), ("global", 2))
        self.assertNotIn("point", result)
        results.append(result)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("ideal(A.generators,", script)
        self.assertEqual([row["result"] for row in self.replay(script)], results)

    def test_implicit_origin_explicit_origin_and_local_basis_source(self):
        page = self.page
        choose = page.locator("#operation").select_option
        point = page.locator("#local-point")
        results = []
        choose("local-quotient-basis")
        self.rows("#generators", "#add-generator", ["x^2*(x-1/2)", "y"])
        result = self.submit()["result"]
        self.assertEqual((result["scope"], result["dimension"]), ("origin", 2))
        self.assertNotIn("point", result)
        results.append(result)
        point.fill("0/2, -0")
        result = self.submit()["result"]
        self.assertEqual((result["scope"], result["point"], result["dimension"]), ("point", ["0", "0"], 2))
        results.append(result)
        point.fill("1/2, 0")
        choose("local-standard-basis")
        results.append(self.submit("B")["result"])
        choose("local-quotient-basis")
        point.fill("")
        page.locator("#ideal-source").select_option("B")
        expect(point).to_have_value("1/2, 0")
        result = self.submit()["result"]
        self.assertEqual((result["scope"], result["point"], result["basis"]), ("point", ["1/2", "0"], ["1"]))
        results.append(result)
        point.fill("0, 0")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("was computed at (1/2, 0)")
        page.locator("#ideal-source").select_option("")
        page.locator("#variables").fill("")
        self.rows("#generators", "#add-generator", [])
        point.fill("()")
        result = self.submit()["result"]
        self.assertEqual((result["scope"], result["point"], result["basis"]), ("point", [], ["1"]))
        results.append(result)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("local_quotient_basis(B,", script)
        self.assertEqual([row["result"] for row in self.replay(script)], results)

    @unittest.skipUnless(os.getenv("ATLAS_GROEBNER", "native") == "native",
                         "requires the native sparse polynomial representation")
    def test_large_milnor_number_preserves_all_digits(self):
        page = self.page
        degree = 2**64 + 17
        page.locator("#variables").fill("x")
        page.locator("#operation").select_option("milnor-number")
        page.locator("#polynomial").fill(f"x^{degree}")
        record = self.submit("M")
        self.assertEqual(record["result"]["value"], degree - 1)
        expect(page.locator("#result-value")).to_have_text(str(degree - 1))
        self.assertEqual(json.loads(self.download("#export"))["result"]["value"], degree - 1)
        page.locator("#session-replay summary").click()
        replayed, = self.replay(self.download("#download-session"))
        self.assertEqual(replayed["result"], record["result"])
