"""Guided tangent spaces, exact direction vectors, and replay of point-owned inputs."""

import json
import shlex
import subprocess

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import ROOT


class ZariskiTangentBrowserTests(AlgebraBrowserCase):
    def submit(self, name="", compute="cpu", *, status="ok"):
        page = self.page
        page.locator("#result-name").fill(name)
        page.locator("#compute").select_option(compute)
        with page.expect_response(self.url + "/api/algebra", timeout=65000) as pending:
            page.locator("#run").click()
        response = pending.value
        record = json.loads(response.body())
        self.assertEqual(record["status"], status, record)
        if status == "ok":
            expect(page.locator("#result")).to_be_visible()
            self.assertEqual(record["execution"]["requested"], compute)
        return response.request.post_data_json["operands"], record

    def replay(self, script=None, command=None):
        run = subprocess.run(command or ["./atlas", "run", "-"], cwd=ROOT, input=script,
                             capture_output=True, text=True, timeout=65)
        self.assertEqual(run.returncode, 0, run.stderr)
        records = [run.stdout] if command else run.stdout.splitlines()
        return [json.loads(record)["result"] for record in records]

    def test_rational_directions_local_basis_and_cone_vertex(self):
        page = self.page
        choose = page.locator("#operation").select_option
        point = page.locator("#local-point")
        source = page.locator("#ideal-source")
        variables = ["z", "x"]
        page.locator("#variables").fill(", ".join(variables))
        choose("local-standard-basis")
        point.fill("1/2, -2")
        # The tangent line at p has equation dz + c/7 dx = 0.
        c = 2**90 + 1
        self.rows("#generators", "#add-generator", [f"7*(z-1/2)+{c}*(x+2)+(x+2)^2"])
        _, basis = self.submit("B")
        results = [basis["result"]]

        choose("zariski-tangent-space")
        point.fill("")
        source.select_option("B")
        expect(point).to_have_value("1/2, -2")
        _, record = self.submit("V", "gpu")
        result = record["result"]
        self.assertEqual((result["dimension"], result["ambient_dimension"], result["jacobian_rank"]), (1, 2, 1))
        self.assertEqual((result["coordinate_system"], result["point"]), ("tangent", ["1/2", "-2"]))
        self.assertEqual(result["basis"], [[f"-{c}/7", "1"]])
        self.assertEqual(result["equations"], [f"z+{c}/7*x"])
        expect(page.locator("#result-value")).to_contain_text(result["equations"][0] + " = 0")
        expect(page.locator("#result-value th")).to_have_text(variables)
        expect(page.locator("#result-value td")).to_have_text(result["basis"][0])
        expect(page.locator("#result-ring")).to_contain_text("at (z = 1/2, x = -2)")
        self.assertEqual(json.loads(self.download("#export")), record)
        self.assertEqual(self.replay(self.download("#download-script")), [result])
        self.assertEqual(self.replay(command=shlex.split(page.locator("#command").text_content())), [result])
        results.append(result)

        choose("embedding-dimension")
        point.fill("0,0")
        page.locator("#result-name").fill("")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("was computed at (1/2, -2)")
        point.fill("2/4, -4/2")
        _, number = self.submit("e")
        self.assertEqual((number["result"]["value"], number["result"]["jacobian_rank"]), (1, 1))
        expect(page.locator("#result-value")).to_contain_text("Embedding dimension1")
        results.append(number["result"])

        choose("tangent-cone")
        _, cone = self.submit("C")
        results.append(cone["result"])
        choose("zariski-tangent-space")
        source.select_option("C")
        expect(point).to_have_value("")
        operands, vertex = self.submit()
        self.assertNotIn("point", operands)
        self.assertEqual((vertex["result"]["scope"], vertex["result"]["basis"]), ("origin", result["basis"]))
        results.append(vertex["result"])
        choose("embedding-dimension")
        point.fill("1/2,-2")
        page.locator("#run").click()
        expect(page.locator("#error")).to_contain_text("all tangent coordinates zero")
        point.fill("0,0")
        _, explicit = self.submit()
        self.assertEqual((explicit["result"]["point"], explicit["result"]["value"]), (["0", "0"], 1))
        results.append(explicit["result"])
        page.locator("#session-replay summary").click()
        script = self.download("#download-session")
        self.assertIn("zariski_tangent_space(B,", script)
        self.assertIn('zariski_tangent_space(C, compute="cpu")', script)
        self.assertIn("embedding_dimension(C,", script)
        self.assertEqual(self.replay(script), results)

    def test_zero_space_full_ambient_space_and_point_off_scheme(self):
        page = self.page
        choose = page.locator("#operation").select_option
        choose("zariski-tangent-space")
        self.rows("#generators", "#add-generator", ["x", "y"])
        _, point = self.submit()
        self.assertEqual((point["result"]["dimension"], point["result"]["basis"]), (0, []))
        expect(page.locator("#result-value")).to_contain_text("The tangent space is {0}.")
        expect(page.locator("#result-value")).to_contain_text("empty basis")
        # The cusp has zero first derivatives at its singular point.
        self.rows("#generators", "#add-generator", ["y^2-x^3"])
        _, cusp = self.submit()
        self.assertEqual((cusp["result"]["jacobian_rank"], cusp["result"]["equations"]), (0, []))
        self.assertEqual(cusp["result"]["basis"], [["1", "0"], ["0", "1"]])
        expect(page.locator("#result-value")).to_contain_text("full ambient space")
        page.locator("#local-point").fill("0, 1")
        for op in ("zariski-tangent-space", "embedding-dimension"):
            choose(op)
            _, error = self.submit(status="error")
            self.assertIn("point must lie on the scheme", error["error"]["message"].lower())
            expect(page.locator("#error")).to_contain_text(error["error"]["message"])
            expect(page.locator("#result")).to_be_hidden()
            self.assertEqual(json.loads(self.download("#export")), error)
