"""Exercise the Zoo through the same catalogue, native search, and record URLs as a visitor."""
import json
from fractions import Fraction

from tests.browser_support import BrowserCase, expect


class ZooBrowserTests(BrowserCase):
    def test_equation_order_across_pages_and_dimension_filter(self):
        page = self.page
        page.goto(self.url + "/zoo/")
        entries = page.request.get(self.url + "/zoo/zoo-data/index.json").json()["entries"]
        width = max(len(entry["variables"]) for entry in entries)

        def equation_key(entry):
            terms = []
            for term in entry["terms"]:
                powers = tuple(int(power) for power in term["exponents"])
                terms.append((powers + (0,) * (width - len(powers)), Fraction(term["coefficient"])))
            return sorted(terms, reverse=True), entry["id"]

        ordered = sorted(entries, key=equation_key)
        rows = page.locator("#zoo-list a")
        expect(rows.first).to_be_visible()
        page_size = rows.count()

        def expect_ids(expected):
            ids = rows.evaluate_all("links => links.map(link => link.dataset.id)")
            self.assertEqual(ids, [entry["id"] for entry in expected])

        expect_ids(ordered[:page_size])
        page.get_by_role("button", name="Next", exact=True).click()
        expect_ids(ordered[page_size:2 * page_size])
        page.get_by_role("button", name="Plane curves").click()
        expect_ids([entry for entry in ordered if entry["dimension"] == 1][:page_size])

    def test_record_without_a_name(self):
        page = self.page

        def unnamed_record(route):
            record = route.fetch().json()
            record["name"] = ""
            route.fulfill(json=record)

        page.route("**/records/threefold-node.json", unnamed_record)
        page.goto(self.url + "/zoo/?record=threefold-node")
        detail = page.locator("#zoo-detail")
        formula = detail.locator("h2").get_by_role("math")
        expect(formula).to_have_attribute("aria-label", "x*y - z*w = 0")
        expect(detail.locator(".zoo-record-name")).to_be_hidden()
        expect(detail.get_by_role("list", name="Labels")).to_be_visible()
        row = page.locator('#zoo-list a[aria-current]')
        self.assertEqual(row.inner_text(), row.get_by_role("math").inner_text())
        self.assertEqual(row.get_by_role("math").get_attribute("aria-label"), formula.get_attribute("aria-label"))
        page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=self.url)
        detail.get_by_role("button", name="Copy Python").click()
        expect(detail.get_by_role("status")).to_have_text("Python copied.")
        self.assertEqual(page.evaluate("navigator.clipboard.readText()"), "x*y - z*w")

    def test_monomial_help_on_hover_click_and_keyboard(self):
        page = self.page
        page.goto(self.url + "/zoo/")
        trigger = page.get_by_label("Monomial search help")
        help_text = page.locator("#zoo-search-help")
        query = page.get_by_role("searchbox", name="Search by monomials")
        expect(help_text).to_be_hidden()
        bounds = query.bounding_box()
        trigger.hover()
        expect(help_text).to_be_visible()
        help_text.hover()
        expect(help_text).to_be_visible()
        query.hover()
        expect(help_text).to_be_hidden()
        trigger.click()
        query.hover()
        expect(help_text).to_be_visible()
        self.assertEqual(query.bounding_box(), bounds)
        query.click()
        expect(help_text).to_be_hidden()
        query.press("Shift+Tab")
        expect(trigger).to_be_focused()
        expect(help_text).to_be_visible()
        trigger.press("Escape")
        expect(help_text).to_be_hidden()
        trigger.press("Enter")
        expect(help_text).to_be_visible()
        trigger.press("Tab")
        expect(query).to_be_focused()
        expect(help_text).to_be_hidden()

    def test_monomial_help_on_touchscreen(self):
        context = self.browser.new_context(has_touch=True, is_mobile=True, viewport={"width": 320, "height": 700})
        self.addCleanup(context.close)
        page = context.new_page()
        page.goto(self.url + "/zoo/")
        trigger = page.get_by_label("Monomial search help")
        help_text = page.locator("#zoo-search-help")
        trigger.tap()
        expect(help_text).to_be_visible()
        bounds = help_text.bounding_box()
        self.assertGreaterEqual(bounds["x"], 0)
        self.assertLessEqual(bounds["x"] + bounds["width"], 320)
        trigger.tap()
        expect(help_text).to_be_hidden()

    def test_search_browse_and_read_record(self):
        page = self.page
        page.goto(self.url)
        page.get_by_role("link", name="Resolution Zoo").click()
        expect(page.locator("#zoo-detail h2").get_by_role("math")).to_be_visible()
        dimension = page.get_by_role("group", name="Dimension")
        expect(dimension.locator("button")).to_have_count(6)
        dimension.get_by_role("button", name="Surfaces").click()
        query = page.get_by_label("Search by monomials")
        query.fill("1/2*u*v^3, -w^4")
        expect(page.locator("#zoo-list a")).to_have_count(1)
        expect(page.locator("#zoo-list a")).to_have_attribute("data-id", "mixed-quartic")
        page.locator("#zoo-label").select_option("Non-isolated")
        expect(page.locator("#zoo-feedback")).to_contain_text("No records match")
        page.get_by_role("button", name="Clear filters").click()
        dimension.get_by_role("button", name="Surfaces").click()
        query.fill("-u*v^2, w^2")
        row = page.locator('#zoo-list a[data-id="whitney-umbrella"]')
        row.click()
        detail = page.locator("#zoo-detail")
        equation = detail.locator("h2").get_by_role("math")
        expect(equation).to_have_attribute("aria-label", "x^2 - y^2*z = 0")
        expect(detail.locator(".zoo-record-name")).to_contain_text("Whitney umbrella")
        expect(detail.locator(".zoo-record-engine")).to_contain_text("binomial-coordinate-blowups")
        expect(detail.locator(".zoo-invariants")).to_contain_text("∞")
        detail.get_by_text("Monodromy zeta function", exact=True).click()
        detail.get_by_text("Nearby strata (2)", exact=True).click()
        locus = detail.locator(".zoo-nearby").filter(has=page.get_by_role("math", name="1", exact=True))
        expect(locus).to_contain_text("x = y = 0, z ≠ 0")
        detail.get_by_text("Topological zeta function", exact=True).click()
        expect(detail.get_by_role("math", name="(s + 2)/(2*(s + 1)^2)", exact=True)).to_be_visible()
        expect(detail.get_by_text("Computation details", exact=True)).to_have_count(0)
        expect(detail.get_by_text("Pole witnesses", exact=True)).to_be_visible()
        record = json.loads(self.download("#zoo-detail a[download]"))
        self.assertEqual(record["id"], "whitney-umbrella")
        self.assertEqual(record["embedded"]["status"], "resolved")
        page.reload()
        expect(equation).to_have_attribute("aria-label", "x^2 - y^2*z = 0")
        page.get_by_label("Search by monomials").fill("x^-1")
        expect(query).to_have_attribute("aria-invalid", "true")
        expect(page.locator("#zoo-feedback .error")).to_be_visible()
        dimension.get_by_role("button", name="Threefolds").click()
        expect(page.locator("#zoo-feedback .error")).to_be_visible()
        page.get_by_role("button", name="Clear filters").click()
        expect(query).to_have_value("")

    def test_dimension_collections_on_mobile(self):
        page = self.page
        page.set_viewport_size({"width": 375, "height": 800})
        page.goto(self.url + "/zoo/")
        for name in ("Plane curves", "Surfaces", "Threefolds", "Fourfolds", "Higher dimensions"):
            page.get_by_role("button", name=name).click()
            row = page.locator("#zoo-list a").first
            ident = row.get_attribute("data-id")
            row.click()
            expect(page.locator("#zoo-detail a[download]")).to_have_attribute("download", ident + ".json")
            self.assertEqual(page.evaluate("document.activeElement.id"), "zoo-detail")
            self.assertFalse(page.evaluate("document.body.scrollWidth > innerWidth"))

    def test_delayed_hosted_queries_and_record_downloads_cannot_replace_a_new_selection(self):
        page = self.page
        page.add_init_script("""
            const fetchRecord = window.fetch;
            window.fetch = async (url, options) => {
              if (String(url).endsWith("/a2.json")) {
                await new Promise(resolve => { window.finishRecord = resolve; });
                return fetchRecord(url);
              }
              return fetchRecord(url, options);
            };
            window.atlasZooSearch = async (query, entries) => {
              if (query === "") return {ids: entries.map(entry => entry.id)};
              if (query === "u^2") {
                await new Promise(resolve => { window.finishQuery = resolve; });
                return {ids: ["a1"]};
              }
              return {ids: ["a3"]};
            };
        """)
        page.goto(self.url + "/zoo/?record=a2")
        page.wait_for_function("typeof window.finishRecord === 'function'")
        query = page.get_by_label("Search by monomials")
        query.fill("u^2")
        page.wait_for_function("typeof window.finishQuery === 'function'")
        query.fill("v^4")
        download = page.locator("#zoo-detail a[download]")
        expect(download).to_have_attribute("download", "a3.json")
        with page.expect_response(lambda r: r.url.endswith("/records/a2.json")):
            page.evaluate("() => { window.finishRecord(); window.finishQuery(); }")
        expect(page.locator("#zoo-list a")).to_have_attribute("data-id", "a3")
        expect(download).to_have_attribute("download", "a3.json")
