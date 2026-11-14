"""Ideal radicals in the guided Algebra view and exported Atlas sessions."""
import json
import shlex
import subprocess

from tests.algebra_browser import AlgebraBrowserCase, ROOT, expect
from tests.test_cli_session import run_script


class RadicalBrowserTests(AlgebraBrowserCase):
    def compute(self, name=""):
        self.page.locator("#result-name").fill(name)
        with self.page.expect_response(self.url + "/api/algebra") as pending:
            self.page.locator("#run").click()
        record = pending.value.json()
        self.assertEqual(record["status"], "ok", record)
        expect(self.page.locator("#result")).to_be_visible()
        return record["result"]

    def test_named_ideal_basis_and_radical_replay(self):
        page = self.page
        select = page.locator("#operation").select_option
        source = page.locator("#ideal-source")
        select("ideal")
        self.rows("#generators", "#add-generator", ["x*(x-1)", "x*(y-1)^2"])
        self.compute("I")
        select("groebner")
        source.select_option("I")
        basis = self.compute("B")
        select("radical")
        expect(page.locator("#local-point")).to_be_hidden()
        expect(page.locator("#polynomial")).to_be_hidden()
        source.select_option("B")
        # A saved name can coincide with the operation; export aliases it.
        result = self.compute("radical")
        expect(page.locator("#result-title")).to_contain_text("Ideal radical")
        self.assertEqual(result["type"], "ideal")
        self.assertEqual(result["ring"], basis["ring"])
        self.assertEqual(set(result["generators"]), {"x^2-x", "x*y-x"})
        expect(page.get_by_role("list", name="Generators").get_by_role("listitem")).to_have_count(2)
        self.assertEqual(json.loads(self.download("#export"))["result"], result)
        command = shlex.split(page.locator("#command").text_content())
        cli = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=65)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertEqual(json.loads(cli.stdout)["result"], result)

        select("membership")
        source.select_option("radical")
        page.locator("#polynomial").fill("x*(y-1)")
        member = self.compute()
        self.assertTrue(member["value"])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("radical(B,", script)
        replay = run_script(script)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual([json.loads(row)["result"] for row in replay.stdout.splitlines()], [basis, result, member])

    def test_zero_and_unit_in_the_scalar_ring(self):
        page = self.page
        page.locator("#operation").select_option("radical")
        page.locator("#variables").fill("")
        for generators, expected in (([], []), (["2/3"], ["1"])):
            with self.subTest(generators=generators):
                self.rows("#generators", "#add-generator", generators)
                result = self.compute()
                self.assertEqual(result["generators"], expected)
                expect(page.locator("#result-value")).to_have_text("1" if expected else "(0)")
