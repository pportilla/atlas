"""Resolution history, progress and follow-up computations at the web entry point."""
import json
from urllib.parse import urlencode

from tests.browser_support import BrowserCase, expect


class WorkbenchBrowserTests(BrowserCase):
    def test_cusp_history_followups_and_shared_link(self):
        page = self.page
        searches = []
        page.on("request", lambda req: searches.append(req) if
                req.method == "POST" and req.url.endswith("/api/searches") else None)
        page.goto(self.url)
        expect(page.get_by_role("heading", name="Resolution", exact=True)).to_be_visible()
        expect(page.get_by_role("link", name="Algebra")).to_have_count(0)
        self.assertEqual(page.request.get(self.url + "/algebra.html").status, 404)
        logo = page.locator("#workspace-logo img")
        expect(logo).to_have_attribute("alt", "Atlas")
        expect(logo).to_be_visible()
        expect(page.locator(".curve-drawing")).to_have_count(0)
        page.locator('[data-example="node"]').press("Enter")
        expect(page.locator("#polynomial")).to_have_value("y^2-x^2")
        page.locator("#polynomial").fill("x*y")
        expect(logo).to_be_visible()
        page.locator('[data-example="cusp"]').click()
        page.locator('.info-popover summary').press("Enter")
        expect(page.locator(".info-popover p")).to_be_visible()
        page.locator('.info-popover summary').press("Enter")
        self.assertFalse(searches)

        for name in ("Monodromy zeta function", "Topological zeta function"):
            page.get_by_label(name, exact=True).uncheck()
        page.locator("#run").click()
        runs = page.locator(".resolution-run")
        expect(runs).to_have_count(1)
        expect(runs.first).to_have_attribute("data-state", "completed")
        expect(logo).to_be_hidden()
        expect(page.locator("#result summary").filter(has_text="Certificate")).to_have_count(0)
        check = page.get_by_role("heading", name="Topological monodromy conjecture")
        expect(check).to_have_count(0)
        graph = page.locator("#panel-complex .vertex")
        expect(graph).to_have_count(3)
        graph.first.press("Enter")
        page.locator("#panel-charts summary").first.click()
        expect(page.locator("#panel-charts .filter-note")).to_be_visible()
        page.get_by_role("button", name="Show all", exact=True).click()
        expect(graph.first).to_have_attribute("aria-pressed", "false")
        initial = json.loads(self.download("#export"))
        self.assertNotIn("monodromy_zeta", initial)
        self.assertNotIn("topological_zeta", initial)

        for key in ("monodromy", "topological"):
            with page.expect_response(lambda res: res.url.endswith("/zetas")) as added:
                page.get_by_role("button", name="Compute " + key + " zeta function").click()
            self.assertEqual(added.value.json()["result"][key + "_zeta"]["status"], "computed")
            if key == "monodromy":
                expect(check).to_have_count(0)
        expect(check).to_be_visible()
        expect(page.locator(".monodromy-verdict")).to_contain_text("Verified for this germ.")
        panel = page.locator("#panel-zeta")
        expect(panel.get_by_text("Poles: −1, −5/6", exact=True)).to_be_visible()
        formulas = panel.locator("details").filter(has=page.get_by_text("Formula", exact=True))
        expect(formulas).to_have_count(2)
        formulas.first.locator("summary").click()
        expect(formulas.first.locator("table")).to_be_visible()
        self.assertNotRegex(formulas.first.inner_text(), r"(?i)resolution engine|source|method")
        panel.get_by_text("Eigenvalue witnesses", exact=True).click()
        expect(panel.get_by_role("row").filter(has_text="−5/6")).to_contain_text("Monodromy zeta at the origin")
        updated = json.loads(self.download("#export"))
        self.assertEqual(updated["charts"], initial["charts"])
        self.assertEqual(updated["dual_complex"], initial["dual_complex"])
        self.assertEqual(len(searches), 1)

        for label in ("Compute intrinsic resolution", "View intrinsic resolution"):
            expect(runs.get_by_role("button", name=label)).to_have_count(0)
        page.get_by_label("Intrinsic", exact=True).check()
        page.locator("#run").click()
        expect(runs).to_have_count(2)
        expect(runs.first.locator(".run-mode")).to_have_text("intrinsic")
        expect(runs.first).to_have_attribute("data-state", "completed")
        for label in ("Compute embedded resolution", "View embedded resolution"):
            expect(runs.get_by_role("button", name=label)).to_have_count(0)
        expect(runs.last.locator(".run-toggle")).to_have_attribute("aria-expanded", "false")
        runs.last.locator(".run-toggle").click()
        self.assertEqual(json.loads(self.download("#export")), updated)
        page.reload()
        expect(runs).to_have_count(2)
        expect(runs.last.locator(".run-toggle")).to_have_attribute("aria-expanded", "true")
        self.assertEqual(len(searches), 2)

        link = urlencode({"f": "x*y", "vars": "x,y", "mode": "intrinsic", "zeta": "0"})
        page.goto(self.url + "/#" + link)
        expect(runs).to_have_count(3)
        expect(runs.first).to_have_attribute("data-state", "completed")
        page.set_viewport_size({"width": 320, "height": 780})
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 320)
        address = page.url
        page.locator('a[href="#output"]').press("Enter")
        expect(page.locator("#output")).to_be_focused()
        self.assertEqual(page.url, address)

    def test_expand_a_running_job_while_another_is_queued(self):
        page = self.page
        jobs = {}
        def handle(route):
            if route.request.method == "GET":
                return route.fulfill(json={"jobs": []})
            ident = str(len(jobs) + 1)
            jobs[ident] = {"id": ident, "request": route.request.post_data_json,
                           "state": "running" if ident == "1" else "queued", "elapsed_ms": 1000}
            route.fulfill(status=202, json=jobs[ident])

        page.route("**/api/searches", handle)
        page.route("**/api/searches/*", lambda route: route.fulfill(json=jobs[route.request.url.split("/")[-1]]))
        page.goto(self.url)
        page.locator("#run").click()
        runs = page.locator(".resolution-run")
        expect(runs.first).to_have_attribute("data-state", "running")
        expect(page.locator("#workspace-logo")).to_be_visible()
        page.locator('[data-example="node"]').click()
        page.locator("#run").click()
        expect(runs.first).to_have_attribute("data-state", "queued")
        expect(runs.last.locator(".run-body")).to_be_hidden()
        runs.last.locator(".run-toggle").click()
        jobs["1"]["progress"] = {"stage": "Searching for the next centre", "blowups": 0}
        expect(runs.last.locator(".run-blowups")).to_have_text("0 blowups so far")
        jobs["1"].update(progress={"stage": "Checking the new charts", "blowups": 2}, elapsed_ms=3000)
        expect(runs.last.locator(".run-stage")).to_have_text("Checking the new charts")
        expect(runs.last.locator(".run-blowups")).to_have_text("2 blowups so far")
        expect(runs.last.locator(".run-elapsed")).to_have_text("Elapsed 3 s")
        expect(runs.first.locator(".run-body")).to_be_hidden()
