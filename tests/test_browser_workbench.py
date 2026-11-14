"""The problem previews follow the inputs and stay distinct from computed records."""

import json
from urllib.parse import urlencode

from tests.algebra_browser import AlgebraBrowserCase, expect


class WorkbenchBrowserTests(AlgebraBrowserCase):
    def setUp(self):
        context = self.browser.new_context()
        self.addCleanup(context.close)
        self.page = context.new_page()

    def test_resolution_examples_graph_and_reopened_record(self):
        page = self.page
        calculations = []
        page.on("request", lambda request: calculations.append(request.url) if request.method == "POST" else None)
        page.goto(self.url)
        polynomial = page.locator("#polynomial")
        expect(page.locator("#plate-curve")).to_be_visible()
        page.locator('[data-example="node"]').press("Enter")
        expect(polynomial).to_have_value("y^2-x^2")
        polynomial.fill("x*y")
        expect(page.locator("#plate-curve")).to_be_hidden()
        expect(page.locator("#plate-empty")).to_be_visible()
        page.locator('[data-example="cusp"]').click()
        expect(polynomial).to_have_value("y^2-x^3")
        self.assertEqual(calculations, [])

        with page.expect_response(self.url + "/api/resolve") as response:
            page.locator("#run").click()
        record = response.value.json()
        self.assertEqual(record["status"], "resolved")
        graph = page.locator("#panel-complex .vertex")
        expect(graph).to_have_count(len(record["dual_complex"]["vertices"]))
        graph.first.press("Enter")
        expect(graph.first).to_have_attribute("aria-pressed", "true")
        page.locator("#panel-charts summary").first.click()
        expect(page.locator("#panel-charts .filter-note")).to_be_visible()
        page.locator("#panel-charts .filter-note").get_by_role("button", name="Show all").click()
        expect(graph.first).to_have_attribute("aria-pressed", "false")
        polynomial.fill("x*y")
        expect(page.locator("#stale")).to_be_visible()
        self.assertEqual(json.loads(self.download("#export")), record)

        link = urlencode({"f": "y^2-x^3", "vars": "x,y", "mode": "intrinsic", "zeta": "0"})
        with page.expect_response(self.url + "/api/resolve") as reopened:
            page.goto(self.url + "/#" + link)
        self.assertEqual(reopened.value.json()["mode"], "intrinsic")
        expect(page.locator("#result-mode")).to_contain_text("intrinsic", ignore_case=True)
        page.set_viewport_size({"width": 320, "height": 780})
        record_url = page.url
        page.locator('a[href="#output"]').press("Enter")
        expect(page.locator("#output")).to_be_focused()
        self.assertEqual(page.url, record_url)
        self.assertEqual(len(calculations), 2)

    def test_algebra_preview_updates_without_replacing_last_answer(self):
        page = self.page
        page.set_viewport_size({"width": 320, "height": 780})
        page.goto(self.url)
        page.get_by_role("link", name="Algebra").press("Enter")
        equations = page.locator("#problem-equations")
        expect(equations).to_contain_text("x^2+y/2")
        expect(equations).to_contain_text("x^2-1")
        expect(page.locator("#problem-ring")).to_contain_text("ℚ[x, y]")
        expect(page.locator("#result")).to_be_hidden()
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 320)
        with page.expect_response(self.url + "/api/algebra") as response:
            page.locator("#run").click()
        answer = response.value.json()["result"]["polynomial"]
        expect(page.locator("#result-value")).to_contain_text(answer)
        page.locator("#operation").select_option("saturation")
        page.locator("#polynomial").fill("y^3")
        expect(equations).to_contain_text("y^3")
        expect(page.locator("#problem-operation")).to_have_text("Saturation")
        expect(page.locator("#result-value")).to_contain_text(answer)
        expect(page.locator("#stale")).to_be_visible()
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 320)
