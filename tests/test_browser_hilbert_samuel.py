"""Hilbert–Samuel values at rational points through Chromium and exported scripts."""

import json
import math
import os
import shlex
import subprocess

from engine.polynomial import integer_from_decimal
from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT


class HilbertSamuelBrowserTests(AlgebraBrowserCase):
    def submit(self, name="", compute="cpu"):
        page = self.page
        page.locator("#result-name").fill(name)
        page.locator("#compute").select_option(compute)
        endpoint = "/api/algebra/exhaustive" if page.locator("#exhaustive").is_checked() else "/api/algebra"
        with page.expect_response(self.url + endpoint, timeout=65000) as pending:
            page.locator("#run").click()
        response = pending.value
        record = json.loads(response.body(), parse_int=integer_from_decimal)
        self.assertEqual(record["status"], "ok", record)
        expect(page.locator("#result")).to_be_visible()
        self.assertEqual(record["execution"]["requested"], compute)
        return json.loads(response.request.post_data, parse_int=integer_from_decimal), record

    def replay(self, script=None, command=None, env=None):
        run = subprocess.run(command or ["./atlas", "run", "-"], cwd=ROOT, input=script,
                             capture_output=True, text=True, timeout=65, env=env)
        self.assertEqual(run.returncode, 0, run.stderr)
        if command:
            return [json.loads(run.stdout, parse_int=integer_from_decimal)["result"]]
        return [json.loads(line, parse_int=integer_from_decimal)["result"] for line in run.stdout.splitlines()]

    def test_regular_plane_preserves_degree_order_duplicates_and_large_values(self):
        page = self.page
        page.locator("#operation").select_option("hilbert-samuel-function")
        self.rows("#generators", "#add-generator", [])
        n = 2**80 + 1
        page.locator("#degrees").fill(f"{n}, 000, 2, 2")
        request, record = self.submit("H")
        degrees = [n, 0, 2, 2]
        self.assertEqual(request["operands"]["degrees"], degrees)
        result = record["result"]
        self.assertEqual(result["degrees"], degrees)
        self.assertEqual(result["values"], [math.comb(d + 2, 2) for d in degrees])
        self.assertEqual(result["graded_values"], [d + 1 for d in degrees])
        self.assertEqual((result["dimension"], result["multiplicity"]), (2, 1))
        expect(page.get_by_role("columnheader", name="Cumulative length", exact=False)).to_be_visible()
        expect(page.get_by_role("columnheader", name="Graded dimension", exact=False)).to_be_visible()
        for i, d in enumerate(degrees):
            cells = page.locator("#result-value tbody tr").nth(i).get_by_role("cell")
            expect(cells).to_have_text([str(d), str(math.comb(d + 2, 2)), str(d + 1)])
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [result])
        self.assertEqual(self.replay(command=shlex.split(page.locator("#command").text_content())), [result])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn(f"degrees=[{n}, 0, 2, 2]", script)
        self.assertEqual(self.replay(script), [result])

    def test_exhaustive_session_replays_a_4501_digit_degree_under_a_digit_limit(self):
        page = self.page
        page.locator("#operation").select_option("ideal")
        self.rows("#generators", "#add-generator", ["y"])
        self.submit("Line")
        page.locator("#operation").select_option("hilbert-samuel-function")
        page.locator("#ideal-source").select_option("Line")
        digits = "9" * 4501
        length = "1" + "0" * len(digits)
        page.locator("#degrees").fill("2, " + digits)
        page.locator("#exhaustive").check()
        request, record = self.submit("H")
        n = 10**4501 - 1
        result = record["result"]
        self.assertEqual(request["operands"]["degrees"], [2, n])
        self.assertEqual((result["degrees"], result["values"], result["graded_values"]), ([2, n], [3, n+1], [1, 1]))
        expect(page.locator("#result-value tbody tr").nth(1).get_by_role("cell")).to_have_text([digits, length, "1"])
        self.assertEqual(json.loads(self.download("#export"), parse_int=integer_from_decimal)["result"], result)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("hilbert_samuel_function(Line,", script)
        self.assertEqual(self.replay(script, env={**os.environ, "PYTHONINTMAXSTRDIGITS": "640"}), [result])

    def test_saved_local_basis_defaults_empty_degrees_and_multiplicity(self):
        page = self.page
        choose = page.locator("#operation").select_option
        choose("local-standard-basis")
        self.rows("#generators", "#add-generator", ["(x-1/2)^2"])
        page.locator("#local-point").fill("1/2, 0")
        _, record = self.submit("B")
        results = [record["result"]]
        choose("hilbert-samuel-function")
        page.locator("#local-point").fill("")
        page.locator("#ideal-source").select_option("B")
        expect(page.locator("#local-point")).to_have_value("1/2, 0")
        request, record = self.submit()
        self.assertNotIn("degrees", request["operands"])
        result = record["result"]
        self.assertEqual(result["degrees"], list(range(6)))
        self.assertEqual(result["values"], [1, 3, 5, 7, 9, 11])
        self.assertEqual(result["graded_values"], [1, 2, 2, 2, 2, 2])
        self.assertEqual((result["point"], result["dimension"], result["multiplicity"]), (["1/2", "0"], 1, 2))
        expect(page.locator("#result-ring")).to_contain_text("at (x = 1/2, y = 0)")
        results.append(result)

        page.locator("#degrees").fill("1,-2")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("nonnegative integer")
        page.locator("#degrees").fill("[]")
        request, record = self.submit()
        self.assertEqual(request["operands"]["degrees"], [])
        self.assertEqual(record["result"]["values"], [])
        expect(page.locator("#result-value")).to_contain_text("No degrees requested.")
        results.append(record["result"])

        choose("hilbert-samuel-multiplicity")
        expect(page.locator("#degrees")).to_be_hidden()
        page.locator("#local-point").fill("1, 0")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("was computed at (1/2, 0)")
        page.locator("#local-point").fill("2/4, 0")
        request, record = self.submit("M", "gpu")
        self.assertNotIn("degrees", request["operands"])
        self.assertEqual((record["result"]["value"], record["result"]["dimension"]), (2, 1))
        expect(page.locator("#execution")).to_contain_text(record["execution"]["device"], ignore_case=True)
        results.append(record["result"])

        page.locator("#ideal-source").select_option("")
        page.locator("#local-point").fill("0,0")
        # The generator x-1/2 has nonzero value at the origin and is a local unit.
        _, record = self.submit()
        self.assertEqual((record["result"]["value"], record["result"]["dimension"]), (0, -1))
        expect(page.locator("#result-value")).to_contain_text("Krull dimension-1")
        results.append(record["result"])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("hilbert_samuel_function(B,", script)
        self.assertIn("hilbert_samuel_multiplicity(B,", script)
        self.assertEqual(self.replay(script), results)
