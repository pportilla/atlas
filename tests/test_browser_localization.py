"""Guided localization, reusable fields and export of a session using a named prime."""
import json
import shlex

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.test_cli_session import run_script
from tests.test_point_standard_interface import shell


class LocalizationBrowserTests(AlgebraBrowserCase):
    def compute(self, name=""):
        self.page.locator("#result-name").fill(name)
        with self.page.expect_response(self.url + "/api/algebra") as pending:
            self.page.locator("#run").click()
        record = pending.value.json()
        self.assertEqual(record["status"], "ok", record)
        expect(self.page.locator("#result")).to_be_visible()
        return record["result"]

    def test_named_prime_fraction_and_session_export(self):
        page = self.page
        select = page.locator("#operation").select_option
        select("ideal")
        self.rows("#generators", "#add-generator", ["x"])
        self.compute("P")
        select("localization")
        self.rows("#generators", "#add-generator", ["x*y"])
        page.locator("#prime-source").select_option("P")
        page.locator("#fraction-numerator").fill("x")
        page.locator("#fraction-denominator").fill("y")
        result = self.compute("localize")  # Export also escapes names reserved by the session.
        expect(page.locator("#result-value")).to_contain_text("Contracted ideal")
        expect(page.locator("#result-value")).to_contain_text("Unit in the local ring: no")
        self.assertEqual(result["prime"], result["generators"])
        self.assertEqual(result["element"], {"numerator": "x", "denominator": "y", "unit": False, "member": True})
        self.assertEqual(json.loads(self.download("#export"))["result"], result)
        command = shell(shlex.split(page.locator("#command").text_content()))
        self.assertEqual(command.returncode, 0, command.stderr)
        self.assertEqual(json.loads(command.stdout)["result"], result)
        # Reuse all four structured fields in the same local ring.
        page.locator("#ideal-source").select_option("localize.contraction")
        page.locator("#prime-source").select_option("localize.prime")
        page.locator("#numerator-source").select_option("localize.element.denominator")
        page.locator("#denominator-source").select_option("localize.element.denominator")
        unit = self.compute()
        self.assertTrue(unit["element"]["unit"])
        self.assertFalse(unit["element"]["member"])
        select("membership")
        expect(page.locator("#prime-field")).to_be_hidden()
        page.locator("#polynomial-source").select_option("localize.element.numerator")
        member = self.compute()
        self.assertTrue(member["value"])
        page.locator("#session-replay summary").click()
        replay = run_script(self.download("#download-session"))
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual([json.loads(line)["result"] for line in replay.stdout.splitlines()], [result, unit, member])

    def test_zero_prime_and_invalid_denominator(self):
        page = self.page
        page.locator("#operation").select_option("localization")
        self.rows("#prime-generators", "#add-prime", [])
        self.rows("#generators", "#add-generator", [])
        page.locator("#fraction-denominator").fill("x")
        result = self.compute()
        self.assertEqual(result["prime"], [])
        self.assertEqual(result["generators"], [])
        self.assertTrue(result["element"]["unit"])
        self.rows("#prime-generators", "#add-prime", ["x"])
        page.locator("#fraction-numerator").fill("x")
        with page.expect_response(self.url + "/api/algebra") as pending:
            page.locator("#run").click()
        self.assertEqual(pending.value.status, 400)
        expect(page.locator("#error")).to_contain_text(pending.value.json()["error"]["message"])
