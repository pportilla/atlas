"""Exact exhaustive resolution downloads in the browser."""
import json
import os
import subprocess

from tests.browser_support import BrowserCase, expect
from tests.cli_support import ROOT


class ExhaustiveBrowserTests(BrowserCase):
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
