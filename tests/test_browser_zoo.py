"""Exercise the Zoo through the same catalogue, native search, and record URLs as a visitor."""
import json

from tests.browser_support import BrowserCase, expect


class ZooBrowserTests(BrowserCase):
    def test_search_browse_and_read_record(self):
        page = self.page
        page.goto(self.url)
        page.get_by_role("link", name="Resolution Zoo").click()
        expect(page.locator("#zoo-detail h2")).to_contain_text("A1")
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
        expect(detail.locator("h2")).to_have_text("Whitney umbrella")
        expect(detail.locator(".zoo-invariants")).to_contain_text("∞")
        detail.get_by_text("Monodromy zeta function", exact=True).click()
        detail.get_by_text("Nearby strata (2)", exact=True).click()
        locus = detail.locator(".zoo-nearby").filter(has=page.get_by_role("math", name="1", exact=True))
        expect(locus).to_contain_text("x = y = 0, z ≠ 0")
        detail.get_by_text("Topological zeta function", exact=True).click()
        expect(detail.get_by_role("math", name="(s + 2)/(2*(s + 1)^2)", exact=True)).to_be_visible()
        record = json.loads(self.download("#zoo-detail a[download]"))
        self.assertEqual(record["id"], "whitney-umbrella")
        self.assertEqual(record["embedded"]["status"], "resolved")
        page.reload()
        expect(detail.locator("h2")).to_have_text("Whitney umbrella")
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
        for name, ident in (("Plane curves", "plane-cusp"), ("Surfaces", "a2"),
                            ("Threefolds", "threefold-cubic"), ("Fourfolds", "fourfold-node"),
                            ("Higher dimensions", "fivefold-cubic")):
            page.get_by_role("button", name=name).click()
            page.locator(f'#zoo-list a[data-id="{ident}"]').click()
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
            window.atlasZooSearch = async query => {
              if (query === "u^2") {
                await new Promise(resolve => { window.finishQuery = resolve; });
                return {ids: ["a1"]};
              }
              return {ids: ["a3"]};
            };
        """)
        page.goto(self.url + "/zoo/")
        page.locator('#zoo-list a[data-id="a2"]').click()
        page.wait_for_function("typeof window.finishRecord === 'function'")
        query = page.get_by_label("Search by monomials")
        query.fill("u^2")
        page.wait_for_function("typeof window.finishQuery === 'function'")
        query.fill("v^4")
        expect(page.locator("#zoo-detail h2")).to_contain_text("A3")
        with page.expect_response(lambda r: r.url.endswith("/records/a2.json")):
            page.evaluate("() => { window.finishRecord(); window.finishQuery(); }")
        expect(page.locator("#zoo-list a")).to_have_attribute("data-id", "a3")
        expect(page.locator("#zoo-detail h2")).to_contain_text("A3")
