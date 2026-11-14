"""Live Algebra round trips. After installing Playwright and its Chromium:

python3 -m unittest discover -s tests -p test_browser_algebra.py -v
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from tests.algebra_browser import AlgebraBrowserCase, ROOT, expect


class AlgebraBrowserTests(AlgebraBrowserCase):
    def submit(self, operation, generators, polynomial=None, other=None, eliminate=None, compute="cpu"):
        self.page.locator("#operation").select_option(operation)
        self.page.locator("#compute").select_option(compute)
        self.rows("#generators", "#add-generator", generators)
        if polynomial is not None:
            self.page.locator("#polynomial").fill(polynomial)
        if other is not None:
            self.rows("#second-generators", "#add-second-generator", other)
        if eliminate is not None:
            self.page.get_by_label(eliminate, exact=True).check()
        with self.page.expect_response(self.url + "/api/algebra", timeout=65000) as api:
            self.page.locator("#run").click()
        response = api.value
        self.assertEqual(response.request.method, "POST")
        return response.json()

    def check_replay(self, response):
        record = json.loads(self.download("#export"))
        self.assertEqual(record, response)
        page = self.page
        page.locator("#copy-command").click()
        expect(page.locator("#copy-status")).to_have_text("Command copied.")
        command = page.evaluate("navigator.clipboard.readText()")
        self.assertEqual(command, page.locator("#command").text_content())
        page.locator("#copy-script").click()
        expect(page.locator("#copy-status")).to_have_text("Script copied.")
        script = page.evaluate("navigator.clipboard.readText()")
        self.assertEqual(script, page.locator("#script").text_content())
        downloaded_script = self.download("#download-script")
        self.assertEqual(downloaded_script.rstrip(), script.rstrip())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.atlas"
            path.write_text(downloaded_script, encoding="utf-8")
            for args in (["sh", "-c", command], ["./atlas", "run", str(path)]):
                run = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=65)
                self.assertEqual(run.returncode, 0, run.stderr or run.stdout)
                repeated = json.loads(run.stdout)
                self.assertEqual(repeated["status"], record["status"])
                self.assertEqual(repeated["operation"], record["operation"])
                self.assertEqual(repeated["result"], record["result"])
                self.assertEqual(repeated["execution"]["requested"], record["execution"]["requested"])
        return record["result"]

    def test_operations_and_exports(self):
        ideal = ["x^2-1", "y-x"]
        for operation, generators, operands, key, value in (
            ("normal-form", ideal, {"polynomial": "x^2+y/2"}, "polynomial", "1/2*y+1"),
            ("quotient", ["x^2*y"], {"other": ["x"]}, "generators", ["x*y"]),
            ("saturation", ["x^2*y"], {"polynomial": "x"}, "generators", ["y"]),
            ("groebner", ideal, {}, "generators", ["x-y", "y^2-1"]),
            ("eliminate", ideal, {"eliminate": "x"}, "generators", ["y^2-1"]),
            ("intersection", ["x"], {"other": ["y"]}, "generators", ["x*y"]),
            ("dimension", ["0"], {}, "value", 2),
            ("membership", ideal, {"polynomial": "y^2-1"}, "value", True),
        ):
            with self.subTest(operation=operation):
                response = self.submit(operation, generators, **operands)
                self.assertEqual(response["status"], "ok", response)
                result = self.check_replay(response)
                self.assertEqual(result[key], value)
                variables = ["y"] if operation == "eliminate" else ["x", "y"]
                self.assertEqual(result["ring"], {"field": "QQ", "variables": variables, "order": "lex"})
                expect(self.page.locator("#result-ring")).to_have_text(f"ℚ[{', '.join(variables)}] · lex")
                items = value if isinstance(value, list) else [value]
                for item in items:
                    expect(self.page.locator("#result-value")).to_contain_text(str(item), ignore_case=True)
                expect(self.page.locator("#execution")).to_contain_text("CPU", ignore_case=True)

    def test_zero_false_and_failure(self):
        ideal = ["x^2-1", "y-x"]
        response = self.submit("normal-form", ideal, polynomial="y^2-1", compute="auto")
        self.assertEqual(response["result"]["polynomial"], "0")
        expect(self.page.locator("#result-value")).to_contain_text("0")
        response = self.submit("membership", ideal, polynomial="y")
        self.assertIs(response["result"]["value"], False)
        expect(self.page.locator("#result-value")).to_contain_text("false", ignore_case=True)
        for polynomial, code in (("z", "invalid_request"), ("(" * 1100 + "x" + ")" * 1100, "computation_limit")):
            with self.subTest(code=code):
                response = self.submit("membership", ideal, polynomial=polynomial)
                self.assertEqual(response["status"], "error")
                self.assertEqual(response["error"]["code"], code)
                self.assertNotIn("result", response)
                expect(self.page.locator("#error")).to_contain_text(response["error"]["message"])
                expect(self.page.locator("#result")).to_be_hidden()
                self.assertEqual(json.loads(self.download("#export")), response)

    def test_gpu_receipt(self):
        from engine.gpu_devices import available_devices
        # A large input coefficient forces modular reconstruction and replay.
        self.page.locator("#order").select_option("grevlex")
        rendered = self.page.locator("#execution")

        def run_on(backend, index):
            with patch.dict(os.environ, RESOLUTION_GPU_BACKEND=backend, RESOLUTION_GPU_INDEX=str(index)):
                response = self.submit("normal-form", [f"x^2+{2**260+1}*y", "x*y+1"],
                                       polynomial="x*y", compute="gpu")
            self.assertEqual(response["status"], "ok", response)
            self.assertEqual(response["result"]["polynomial"], "-1")
            execution = response["execution"]
            self.assertEqual(execution["requested"], "gpu")
            expect(rendered).to_contain_text(execution["device"], ignore_case=True)
            return execution

        devices = available_devices()
        if devices:
            device = devices[0]
            execution = run_on(device["backend"], device["index"])
            stages = [stage for stage in execution["stages"] if stage["backend"] == device["backend"]
                      and stage["operation"] == "groebner_replay"]
            self.assertTrue(stages, execution)
            stage = stages[0]
            self.assertGreater(stage["primes"], 0)
            self.assertEqual(stage["device"]["name"], device["name"])
            expect(rendered).to_contain_text(device["name"])
            self.page.locator("#stage-summary").click()
            expect(self.page.locator("#stages")).to_contain_text(f"primes {stage['primes']}")
        missing = max((d["index"] for d in devices if d["backend"] == "opencl"), default=-1) + 1
        fallback = run_on("opencl", missing)
        self.assertEqual(fallback["backend"], "cpu")
        self.assertTrue(fallback["fallback_reasons"], fallback)
        expect(rendered).to_contain_text(fallback["fallback_reasons"][0])
