"""Inspect, reuse and replay primary components and associated primes in the browser."""
import json
import shlex

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import run_script, shell


class PrimaryDecompositionBrowserTests(AlgebraBrowserCase):
    def test_inspect_reuse_and_replay(self):
        page = self.page
        select = page.locator("#operation").select_option
        source = page.locator("#ideal-source")
        select("groebner")
        self.rows("#generators", "#add-generator", ["x^2", "x*y"])
        basis = self.compute("B")
        select("primary-decomposition")
        expect(page.locator("#local-point")).to_be_hidden()
        source.select_option("B")
        name = "primary_decomposition"  # Export must alias a name shared with the operation.
        decomposition = self.compute(name)
        expect(page.locator("#result-title")).to_contain_text("Primary decomposition")
        table = page.locator("#result-value table")
        expect(table.get_by_role("columnheader")).to_have_text(["Primary component", "Associated prime"])
        expect(table.locator("tbody tr")).to_have_count(2)
        self.assertEqual(decomposition["ring"], basis["ring"])
        self.assertEqual(json.loads(self.download("#export"))["result"], decomposition)
        cli = shell(shlex.split(page.locator("#command").text_content()))
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertEqual(json.loads(cli.stdout)["result"], decomposition)

        select("intersection")
        source.select_option(name + ".components[0].primary")
        page.locator("#second-ideal-source").select_option(name + ".components[1].primary")
        intersection = self.compute("J")
        self.assertEqual(intersection["generators"], basis["generators"])

        select("membership")
        source.select_option(name + ".components[0].prime")
        page.locator("#polynomial").fill("x")
        member = self.compute()
        self.assertTrue(member["value"])
        page.locator("#order").select_option("grevlex")
        expect(source.locator("option:checked")).to_have_js_property("disabled", True)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        for field in (".components[0].primary", ".components[1].primary", ".components[0].prime"):
            self.assertIn(field, script)
        replay = run_script(script)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual([json.loads(row)["result"] for row in replay.stdout.splitlines()],
                         [basis, decomposition, intersection, member])

    def test_zero_and_unit_in_the_scalar_ring(self):
        page = self.page
        page.locator("#operation").select_option("primary-decomposition")
        page.locator("#variables").fill("")
        self.rows("#generators", "#add-generator", [])
        zero = self.compute("zero")
        self.assertEqual(zero["components"], [{"primary": [], "prime": []}])
        expect(page.locator("#result-value td")).to_have_text(["⟨0⟩", "⟨0⟩"])
        self.rows("#generators", "#add-generator", ["1"])
        unit = self.compute()
        self.assertEqual(unit["components"], [])
        expect(page.locator("#result-value")).to_have_text("The unit ideal has no primary components.")
