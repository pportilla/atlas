"""Compute both powers and reuse the resulting ideals in Chromium and exported scripts."""
import json

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.test_cli_session import run_script


class IdealPowerBrowserTests(AlgebraBrowserCase):
    def compute(self, name=""):
        self.page.locator("#result-name").fill(name)
        with self.page.expect_response(self.url + "/api/algebra") as pending:
            self.page.locator("#run").click()
        record = pending.value.json()
        self.assertEqual(record["status"], "ok", record)
        expect(self.page.locator("#result")).to_be_visible()
        return record["result"]

    def test_compare_and_replay_both_powers(self):
        page = self.page
        select = page.locator("#operation").select_option
        page.locator("#variables").fill("x, y, z")
        select("ideal")
        self.rows("#generators", "#add-generator", ["x*y", "x*z", "y*z"])
        self.compute("I")
        results = []
        for op, name in (("ideal-power", "ordinary"), ("symbolic-power", "symbolic_power")):
            select(op)
            page.locator("#ideal-source").select_option("I")
            page.locator("#exponent").fill("0002")
            result = self.compute(name)
            results.append(result)
            self.assertEqual(json.loads(self.download("#export"))["result"], result)
        select("membership")
        expect(page.locator("#exponent")).to_be_hidden()
        page.locator("#polynomial").fill("x*y*z")
        for name, expected in (("ordinary", False), ("symbolic_power", True)):
            page.locator("#ideal-source").select_option(name)
            result = self.compute()
            self.assertIs(result["value"], expected)
            results.append(result)
        page.locator("#session-replay summary").click()
        replay = run_script(self.download("#download-session"))
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual([json.loads(line)["result"] for line in replay.stdout.splitlines()], results)

    def test_wide_exponent_and_nonradical_input(self):
        page = self.page
        page.locator("#operation").select_option("symbolic-power")
        self.rows("#generators", "#add-generator", ["x"])
        n = str(2**64+11)
        page.locator("#exponent").fill(n)
        self.assertEqual(self.compute()["generators"], ["x^" + n])
        self.rows("#generators", "#add-generator", ["x^2"])
        page.locator("#exponent").fill("1")
        with page.expect_response(self.url + "/api/algebra") as pending:
            page.locator("#run").click()
        self.assertEqual(pending.value.status, 400)
        expect(page.locator("#error")).to_contain_text("radical")
