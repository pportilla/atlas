"""Resolution methods and center coordinates on actual web results."""
from tests.browser_support import BrowserCase, expect


class ResolutionStepsBrowserTests(BrowserCase):
    def test_point_ideal_and_toric_constructions(self):
        page = self.page
        page.goto(self.url)
        page.locator("#zeta").uncheck()
        page.locator("#topological").uncheck()

        def steps(polynomial, variables):
            page.locator("#polynomial").fill(polynomial)
            page.locator("#variables").fill(variables)
            page.locator("#run").click()
            expect(page.locator(".resolution-run").first).to_have_attribute("data-state", "completed", timeout=30000)
            panel = page.locator("#panel-steps")
            panel.locator("summary").first.click()
            expect(panel.get_by_role("columnheader")).to_have_text(["Step", "Method", "Center", "Details"])
            return panel.locator("tbody tr")

        cusp = steps("y^2-x^3", "x,y")
        expect(cusp).to_have_count(3)
        expect(cusp.locator("td:nth-child(2)")).to_have_text(["Point blow-up"] * 3)
        expect(cusp.first.locator("td").nth(2)).to_have_text("(x, y) = (0, 0) in the input chart")
        expect(cusp.last.locator("td").nth(2)).to_have_text("(u, v) = (0, 0) in B2.v")
        expect(cusp.last.locator("td").nth(3)).to_contain_text("Creates E3")
        expect(cusp.last.locator("td").nth(3)).to_contain_text("Total multiplicity 6")

        ideal = steps("(x+y+z)^2-z*(x-y)^2", "x,y,z")
        expect(ideal).to_have_count(1)
        expect(ideal.locator("td").nth(1)).to_have_text("Blow-up of an ideal")
        expect(ideal.locator("td").nth(2)).to_contain_text("= 0")

        toric = steps("x^2+y^3+z^5", "x,y,z")
        expect(toric).to_have_count(1)
        expect(toric.locator("td").nth(1)).to_have_text("Toric modification")
        expect(toric.locator("td").nth(2)).to_have_text("Over the origin")
        expect(toric.locator("td").nth(3)).to_contain_text("Weight (15, 10, 6)")
