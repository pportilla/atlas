"""Explicit exhaustiveness in the two local browser forms."""
import json
import os
import subprocess

from engine.app import MAX_REQUEST
from tests.algebra_browser import AlgebraBrowserCase, ROOT, expect


class ExhaustiveBrowserTests(AlgebraBrowserCase):
    def test_large_algebra_request(self):
        coefficient = "9" * (MAX_REQUEST + 1)
        self.page.locator("#operation").select_option("ideal")
        self.rows("#generators", "#add-generator", [coefficient + "*x"])
        self.page.locator("#result-name").fill("I")
        self.page.get_by_label("Exhaustive mode").check()
        with self.page.expect_response(self.url + "/api/algebra/exhaustive") as api:
            self.page.locator("#run").click()
        self.assertEqual(api.value.status, 200)
        expect(self.page.locator("#result-value")).to_contain_text(coefficient)

    def test_exhaustive_resolution_download_keeps_numeric_multiplicity(self):
        env = dict(os.environ, ATLAS_GROEBNER="native")
        env.pop("PYTHONINTMAXSTRDIGITS", None)
        with subprocess.Popen([str(ROOT / "atlas"), "serve", "--port", "0"],
                              stdout=subprocess.PIPE, text=True, env=env) as server:
            try:
                url = server.stdout.readline().strip().split()[-1]
                self.page.goto(url)
                n = "9" * 4501
                self.page.locator("#polynomial").fill("x^" + n + "*y")
                self.page.locator("#variables").fill("x,y")
                self.page.locator("#zeta").uncheck()
                self.page.get_by_label("Exhaustive search", exact=True).check()
                with self.page.expect_response(url + "/api/searches") as started:
                    self.page.locator("#run").click()
                self.assertEqual(started.value.status, 202)
                expect(self.page.locator("#result")).to_be_visible(timeout=15000)
                record = json.loads(self.download("#export"), parse_int=str)
                self.assertEqual(record["multiplicity"], "1" + "0" * len(n))
                self.assertIn("search=exhaustive", self.page.url)
            finally:
                server.terminate()
