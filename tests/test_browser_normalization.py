"""Use a named normalization and replay its source and target coordinate maps."""
import json
import shlex

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import run_script, shell


class NormalizationBrowserTests(AlgebraBrowserCase):
    def test_normalization_map_and_named_component_replay(self):
        page = self.page
        select = page.locator("#operation").select_option
        variables = page.locator("#variables")
        ideal = page.locator("#ideal-source")
        select("normalization")
        expect(page.locator("#local-point")).to_be_hidden()
        self.rows("#generators", "#add-generator", ["y^2-x^3"])
        name = "normalization"  # A session object may share the operation name.
        result = self.compute(name)
        expect(page.locator("#result-title")).to_contain_text("Normalization")
        view = page.locator("#result-value")
        expect(view).to_contain_text("Normalized algebra")
        expect(view.get_by_role("columnheader")).to_have_text(
            ["Input coordinate", "Image in the normalization", "Normalized coordinate", "Fraction on this component"])
        self.assertEqual(json.loads(self.download("#export"))["result"], result)
        replay = shell(shlex.split(page.locator("#command").text_content()))
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["result"], result)

        component = result["components"][0]
        prefix = name + ".components[0]"
        select("membership")
        variables.fill(", ".join(component["ring"]["variables"]))
        ideal.select_option(prefix + ".relations")
        page.locator("#polynomial").fill("0")
        in_relations = self.compute("check")
        self.assertTrue(in_relations["value"])
        page.locator("#polynomial-source").select_option(prefix + ".images[0]")
        image_test = self.compute()
        self.assertFalse(image_test["value"])

        variables.fill(", ".join(result["ring"]["variables"]))
        ideal.select_option(prefix + ".prime")
        page.locator("#polynomial-source").select_option(prefix + ".fractions[0].denominator")
        fraction_test = self.compute()
        self.assertFalse(fraction_test["value"])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        replay = run_script(script)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual([json.loads(line)["result"] for line in replay.stdout.splitlines()],
                         [result, in_relations, image_test, fraction_test])

    def test_reports_reduction_and_empty_normalization(self):
        self.page.locator("#operation").select_option("normalization")
        value = self.page.locator("#result-value")
        self.rows("#generators", "#add-generator", ["x^2"])
        self.assertFalse(self.compute()["input_reduced"])
        expect(value).to_contain_text("nilpotents")
        self.rows("#generators", "#add-generator", ["1"])
        self.assertEqual(self.compute()["components"], [])
        expect(value).to_have_text("The zero algebra has no components.")
