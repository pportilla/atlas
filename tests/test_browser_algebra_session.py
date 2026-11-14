"""Named ideals, bases and reductions through the browser and their terminal replay."""

import json
from pathlib import Path
import subprocess
import tempfile

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT


class AlgebraSessionBrowserTests(AlgebraBrowserCase):
    def named(self, operation, name, *, source=None, polynomial=None, generators=None, variables=("x", "y")):
        page = self.page
        page.locator("#operation").select_option(operation)
        page.locator("#result-name").fill(name)
        if source:
            page.locator("#ideal-source").select_option(source)
        elif generators is not None:
            self.rows("#generators", "#add-generator", generators)
        if polynomial:
            page.locator("#polynomial").fill(polynomial)
        with page.expect_response(self.url + "/api/algebra") as api:
            page.locator("#run").click()
        response = api.value
        self.assertEqual(response.request.method, "POST")
        record = response.json()
        self.assertEqual(record["status"], "ok", record)
        expect(page.locator("#result-ring")).to_have_text(f"ℚ[{', '.join(variables)}] · lex")
        if name:
            expect(page.locator("#session-items").get_by_text(name, exact=True)).to_be_visible()
        self.assertEqual(record["result"]["ring"], {"field": "QQ", "variables": list(variables), "order": "lex"})
        return response.request.post_data_json, record

    def export_and_run(self):
        page = self.page
        if not page.locator("#session-replay").evaluate("details => details.open"):
            page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertEqual(script, page.locator("#session-script").text_content())
        page.locator("#copy-session").click()
        expect(page.locator("#session-status")).to_have_text("Script copied.")
        self.assertEqual(page.evaluate("navigator.clipboard.readText()"), script)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "session.atlas"
            file.write_text(script, encoding="utf-8")
            replay = subprocess.run(["./atlas", "run", str(file)], cwd=ROOT, capture_output=True,
                                    text=True, timeout=65)
            self.assertEqual(replay.returncode, 0, replay.stderr)
        return script, [json.loads(line) for line in replay.stdout.splitlines()]

    def test_two_reductions_reuse_the_named_basis_and_export_the_session(self):
        page = self.page
        page.locator("#compute").select_option("cpu")
        self.named("ideal", "I", generators=["x^2-1", "y-x"])
        _, basis = self.named("groebner", "G", source="I")
        self.assertEqual(basis["result"]["generators"], ["x-y", "y^2-1"])
        ref = basis["basis_ref"]
        self.assertTrue(ref)

        page.locator("#operation").select_option("normal-form")
        page.locator("#variables").fill("y, x")
        incompatible = page.locator('#ideal-source option[value="G"]:enabled')
        self.assertEqual(incompatible.count(), 0)
        page.locator("#variables").fill("x, y")
        expect(page.locator('#ideal-source option[value="G"]')).to_be_enabled()

        remainders = []
        for name, polynomial, expected in (("A", "x^2+y/2", "1/2*y+1"), ("B", "x*y+2/3", "5/3")):
            request, record = self.named("normal-form", name, source="G", polynomial=polynomial)
            self.assertEqual(request["operands"], {"basis_ref": ref, "polynomial": polynomial})
            stages = [s for s in record["execution"]["stages"] if s["operation"] == "groebner_basis"]
            self.assertEqual([s["known"] for s in stages], [len(basis["result"]["generators"])])
            downloaded = json.loads(self.download("#export"))
            self.assertEqual(downloaded, record)
            self.assertEqual(record["result"]["polynomial"], expected)
            remainders.append(record["result"])
            expect(page.locator("#result-value")).to_have_text(expected)
            page.locator("#copy-command").click()
            expect(page.locator("#copy-status")).to_have_text("Command copied.")
            command = page.evaluate("navigator.clipboard.readText()")
            repeated = subprocess.run(["sh", "-c", command], cwd=ROOT, capture_output=True,
                                      text=True, check=True, timeout=65)
            self.assertEqual(json.loads(repeated.stdout)["result"], downloaded["result"])

        page.locator("#polynomial").fill("undeclared")
        page.locator("#result-name").fill("Failed")
        with page.expect_response(self.url + "/api/algebra") as api:
            page.locator("#run").click()
        self.assertEqual(api.value.status, 400)
        expect(page.locator("#error")).to_contain_text(api.value.json()["error"]["message"])
        expect(page.locator("#session-items")).not_to_contain_text("Failed")

        script, records = self.export_and_run()
        self.assertNotIn("Failed", script)
        self.assertEqual(script.count("coordinates("), 1)
        self.assertRegex(script, r"I\s*=\s*ideal\(")
        self.assertEqual(script.count("groebner("), 1)
        self.assertRegex(script, r"G\s*=\s*groebner\(I\b")
        for name in ("A", "B"):
            self.assertRegex(script, rf"{name}\s*=\s*normal_form\([^\n]*,\s*G\b")
        self.assertEqual([r["result"] for r in records if r["operation"] == "groebner"], [basis["result"]])
        self.assertEqual([r["result"] for r in records if r["operation"] == "normal-form"], remainders)

    def test_literal_uses_the_elimination_ideals_coordinates(self):
        # An independently declared QQ[y] and the projection from QQ[x,y]
        # have identical ring text and distinct coordinate identities.
        self.page.locator("#variables").fill("y")
        self.named("ideal", "K", generators=["y-2"], variables=("y",))
        self.page.locator("#variables").fill("x, y")
        self.named("ideal", "I", generators=["x^2-1", "y-x"])
        self.page.locator("#operation").select_option("eliminate")
        self.page.get_by_label("y", exact=True).uncheck()
        self.page.get_by_label("x", exact=True).check()
        _, projected = self.named("eliminate", "E", source="I", variables=("y",))
        self.assertEqual(projected["result"]["generators"], ["y^2-1"])
        self.page.locator("#variables").fill("y")
        _, remainder = self.named("normal-form", "A", source="E", polynomial="y^2", variables=("y",))
        self.assertEqual(remainder["result"]["polynomial"], "1")
        script, records = self.export_and_run()
        self.assertIn("E.coordinates", script)
        self.assertEqual(records[-1]["result"], remainder["result"])

    def test_lazy_session_export_reassigns_generated_names_on_later_collisions(self):
        page = self.page
        # Count generation itself, since an empty hidden preview could still
        # mask eager work on every success.
        page.evaluate("""() => {
            const generate = AlgebraSession.prototype.script;
            window.fullScriptBuilds = 0;
            AlgebraSession.prototype.script = function (...args) {
                window.fullScriptBuilds++;
                return generate.apply(this, args);
            };
        }""")
        page.locator("#variables").fill("gcd")
        records = []

        def reduce(name, expression):
            _, record = self.named("normal-form", name, generators=["0"], polynomial=expression, variables=("gcd",))
            records.append(record["result"])

        reduce("", "2*gcd+1")
        reduce("", "gcd^2")
        self.assertEqual(page.evaluate("window.fullScriptBuilds"), 0)
        expect(page.locator("#session-script")).to_be_empty()

        summary = page.locator("#session-replay summary")
        summary.click()
        expect(page.locator("#session-script")).to_contain_text("normal_form(")
        self.assertEqual(page.evaluate("window.fullScriptBuilds"), 1)
        summary.click()
        summary.click()
        self.download("#download-session")
        self.assertEqual(page.evaluate("window.fullScriptBuilds"), 1)
        summary.click()

        reduce("answer1", "(gcd+1)*(gcd+2)")
        reduce("coord1", "(gcd+1)*(gcd+3)")
        page.locator("#operation").select_option("gcd")
        page.locator("#polynomial-source").select_option("answer1")
        page.locator("#other-polynomial-source").select_option("coord1")
        _, divisor = self.named("gcd", "", variables=("gcd",))
        self.assertEqual(divisor["result"]["polynomial"], "gcd+1")
        records.append(divisor["result"])
        self.assertEqual(page.evaluate("window.fullScriptBuilds"), 1)

        summary.click()
        expect(page.locator("#session-script")).to_contain_text("gcd(answer1, coord1,")
        self.assertEqual(page.evaluate("window.fullScriptBuilds"), 2)
        page.locator("#polynomial-source").select_option("")
        reduce("tail", "gcd^3+7")
        expect(page.locator("#session-script")).to_contain_text("tail =")
        self.assertEqual(page.evaluate("window.fullScriptBuilds"), 3)
        _, replayed = self.export_and_run()
        self.assertEqual(page.evaluate("window.fullScriptBuilds"), 3)
        self.assertEqual([r["result"] for r in replayed], records)
