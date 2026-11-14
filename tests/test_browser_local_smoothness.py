"""Local dimensions and smoothness at rational points through Chromium and replay."""

import json
import os
import shlex
import subprocess

from engine.app import MAX_REQUEST
from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT
from tests.large_coupled_ideal import COUPLED_SQUARES, EXPONENT, POINT, VARIABLES, coupled_squares


class LocalSmoothnessBrowserTests(AlgebraBrowserCase):
    def submit(self, name="", compute="cpu", *, status="ok"):
        page = self.page
        page.locator("#result-name").fill(name)
        page.locator("#compute").select_option(compute)
        endpoint = "/api/algebra/exhaustive" if page.locator("#exhaustive").is_checked() else "/api/algebra"
        with page.expect_response(self.url + endpoint, timeout=65_000) as pending:
            page.locator("#run").click()
        response = pending.value
        record = json.loads(response.body())
        self.assertEqual(record["status"], status, record)
        if status == "ok":
            expect(page.locator("#result")).to_be_visible()
            self.assertEqual(record["execution"]["requested"], compute)
        return record

    def replay(self, script=None, command=None):
        run = subprocess.run(command or ["./atlas", "run", "-"], cwd=ROOT, input=script,
                             capture_output=True, text=True, timeout=65)
        self.assertEqual(run.returncode, 0, run.stderr)
        records = [run.stdout] if command else run.stdout.splitlines()
        return [json.loads(record)["result"] for record in records]

    def test_wide_exponents_saved_ideal_and_session(self):
        page = self.page
        choose = page.locator("#operation").select_option
        page.locator("#variables").fill(", ".join(VARIABLES))
        choose("ideal")
        self.rows("#generators", "#add-generator", COUPLED_SQUARES)
        self.submit("I")
        choose("local-dimension")
        page.locator("#ideal-source").select_option("I")
        point = page.locator("#local-point")
        point.fill(", ".join(POINT))
        dimension = self.submit("d")["result"]
        self.assertEqual((dimension["point"], dimension["value"]), (POINT, 0))
        expect(page.locator("#result-value")).to_have_text("0")
        choose("local-smoothness")
        record = self.submit("s", "gpu")
        singular = record["result"]
        self.assertIs(singular["value"], False)
        self.assertEqual((singular["local_dimension"], singular["embedding_dimension"], singular["jacobian_rank"]),
                         (0, 3, 0))
        expect(page.locator("#result-value")).to_contain_text("Singular at this point")
        if os.getenv("ATLAS_TEST_ALGEBRA_GPU") == "1":
            stages = record["execution"]["stages"]
            self.assertTrue(any(s["operation"] == "groebner_replay" and s["backend"] in ("opencl", "triton")
                                and s["primes"] > 0 and s["device"] for s in stages), stages)
            expect(page.locator("#execution")).to_contain_text(record["execution"]["device"], ignore_case=True)
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [singular])
        self.assertEqual(self.replay(command=shlex.split(page.locator("#command").text_content())), [singular])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn(str(EXPONENT), script)
        self.assertIn("local_dimension(I,", script)
        self.assertIn("local_smoothness(I,", script)
        self.assertEqual(self.replay(script), [dimension, singular])
        # At x=-1 this odd exponent is distinguishable from a rounded even one.
        point.fill("-1, 0, 0")
        failed = self.submit(status="error")
        self.assertIn("point must lie on the scheme", failed["error"]["message"].lower())

    def test_exhaustive_large_coupled_request_and_script(self):
        page = self.page
        page.locator("#variables").fill(", ".join(VARIABLES))
        page.locator("#operation").select_option("local-smoothness")
        exponent = "9" * (MAX_REQUEST // 2 + 1)
        self.rows("#generators", "#add-generator", coupled_squares(exponent))
        page.locator("#local-point").fill(", ".join(POINT))
        rejected = self.submit(status="error")
        self.assertIn("exhaustive", rejected["error"]["message"].lower())
        page.get_by_label("Exhaustive mode").check()
        record = self.submit()
        result = record["result"]
        self.assertIs(result["value"], False)
        self.assertEqual((result["local_dimension"], result["embedding_dimension"], result["jacobian_rank"]),
                         (0, 3, 0))
        script = self.download("#download-script")
        self.assertIn(exponent, script)
        self.assertEqual(self.replay(script), [result])

    def test_translated_cusp_saved_point_and_cone_vertex(self):
        page = self.page
        choose = page.locator("#operation").select_option
        point = page.locator("#local-point")
        source = page.locator("#ideal-source")
        display = page.locator("#result-value")
        choose("local-standard-basis")
        self.rows("#generators", "#add-generator", ["(y+2)^2-(x-1/2)^3"])
        point.fill("1/2, -2")
        results = [self.submit("B")["result"]]
        choose("local-dimension")
        point.fill("")
        source.select_option("B")
        expect(point).to_have_value("1/2, -2")
        dimension = self.submit("d")["result"]
        self.assertEqual((dimension["type"], dimension["value"], dimension["local_dimension"]), ("integer", 1, 1))
        expect(display).to_have_text("1")
        expect(page.locator("#result-ring")).to_contain_text("at (x = 1/2, y = -2)")
        results.append(dimension)
        choose("local-smoothness")
        point.fill("3/2, -1")
        page.locator("#result-name").fill("")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("was computed at (1/2, -2)")
        point.fill("2/4, -4/2")
        record = self.submit("S", "gpu")
        singular = record["result"]
        self.assertIs(singular["value"], False)
        self.assertEqual((singular["local_dimension"], singular["embedding_dimension"], singular["jacobian_rank"]), (1, 2, 0))
        expect(display).to_contain_text("Singular at this point")
        expect(display).to_contain_text("Local dimension1")
        expect(display).to_contain_text("Embedding dimension2")
        expect(display).to_contain_text("Jacobian rank0")
        expect(page.locator("#execution")).to_contain_text(record["execution"]["device"], ignore_case=True)
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [singular])
        self.assertEqual(self.replay(command=shlex.split(page.locator("#command").text_content())), [singular])
        results.append(singular)

        choose("tangent-cone")
        results.append(self.submit("C")["result"])
        choose("local-smoothness")
        source.select_option("C")
        expect(point).to_have_value("")
        vertex = self.submit()["result"]
        self.assertEqual(vertex["scope"], "origin")
        self.assertNotIn("point", vertex)
        self.assertIs(vertex["value"], False)
        results.append(vertex)
        choose("local-dimension")
        point.fill("1, 1")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("all tangent coordinates zero")
        point.fill("0, 0")
        vertex_dimension = self.submit()["result"]
        self.assertEqual((vertex_dimension["point"], vertex_dimension["value"]), (["0", "0"], 1))
        results.append(vertex_dimension)
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("local_dimension(B,", script)
        self.assertIn("local_smoothness(B,", script)
        self.assertIn('local_smoothness(C, compute="cpu")', script)
        self.assertIn("local_dimension(C,", script)
        self.assertEqual(self.replay(script), results)

    def test_dimension_changes_across_components_and_smooth_point(self):
        page = self.page
        choose = page.locator("#operation").select_option
        choose("local-dimension")
        # This scheme is the disjoint union of the line x=0 and the point (1,0).
        self.rows("#generators", "#add-generator", ["x*(x-1)", "x*y"])
        line = self.submit()["result"]
        self.assertEqual((line["scope"], line["value"]), ("origin", 1))
        page.locator("#local-point").fill("1,0")
        isolated = self.submit()["result"]
        self.assertEqual((isolated["scope"], isolated["value"]), ("point", 0))
        expect(page.locator("#result-value")).to_have_text("0")
        choose("local-smoothness")
        smooth = self.submit()["result"]
        self.assertIs(smooth["value"], True)
        self.assertEqual((smooth["local_dimension"], smooth["embedding_dimension"], smooth["jacobian_rank"]), (0, 0, 2))
        expect(page.locator("#result-value")).to_contain_text("Smooth at this point")
        page.locator("#local-point").fill("1,1")
        for op in ("local-smoothness", "local-dimension"):
            choose(op)
            failure = self.submit(status="error")
            self.assertIn("point must lie on the scheme", failure["error"]["message"].lower())
            expect(page.locator("#error")).to_contain_text(failure["error"]["message"])
            expect(page.locator("#result")).to_be_hidden()
            self.assertEqual(json.loads(self.download("#export")), failure)
