"""Cell incidences in the picture and data views of a resolution."""
from tests.browser_support import BrowserCase, expect


class ComplexBrowserTests(BrowserCase):
    def test_cell_dimensions_intersections_and_face_ids(self):
        # Each of the four divisors has dimension 3. It represents a 0-cell
        # of the dual complex. Two distinct edges have the same vertex set.
        vertices = [{"id": name, "dimension": 3} for name in ("A", "B", "C", "D")]
        cells = [{"id": "AB.1", "vertices": ["A", "B"]},
                 {"id": "AB.2", "dimension": 1, "vertices": ["A", "B"]},
                 {"id": "ABC", "dimension": 2, "vertices": ["A", "B", "C"], "faces": ["AB.1", "BC", "AC"]},
                 {"id": "ABCD", "dimension": 3, "vertices": ["A", "B", "C", "D"],
                  "faces": ["BCD", "ACD", "ABD", "ABC"]}]
        request = {"polynomial": "y^2-x^3", "variables": ["x", "y"], "mode": "embedded"}
        result = {**request, "status": "resolved", "charts": [], "steps": [],
                  "dual_complex": {"status": "partial", "reason": "Additional attachments are unverified.",
                                   "vertices": vertices, "cells": cells}}
        job = {"id": "cell-example", "state": "completed", "request": request, "result": result}
        self.page.route("**/api/searches", lambda route: route.fulfill(
            status=202 if route.request.method == "POST" else 200,
            json=job if route.request.method == "POST" else {"jobs": []}))
        self.page.route("**/api/searches/cell-example", lambda route: route.fulfill(json=job))
        self.page.goto(self.url)
        self.page.get_by_role("button", name="Resolve", exact=True).click()

        panel = self.page.locator("#panel-complex")
        expect(panel).to_contain_text(result["dual_complex"]["reason"])
        expect(panel.locator(".graph")).to_be_visible()
        graph_vertex = panel.locator(".vertex").first
        graph_vertex.press("Enter")
        panel.get_by_role("button", name="Data", exact=True).click()
        expect(panel.locator(".graph")).to_be_hidden()
        expect(panel.locator("tr[data-id=A] button")).to_have_attribute("aria-pressed", "true")
        expect(panel.get_by_role("heading", name="0-dimensional cells · vertices")).to_be_visible()
        for dimension in (1, 2, 3):
            expect(panel.get_by_role("heading", name=f"{dimension}-dimensional cells")).to_be_visible()
        for cell in cells:
            row = panel.get_by_role("row").filter(has=self.page.get_by_role("cell", name=cell["id"], exact=True))
            expect(row).to_contain_text(" ∩ ".join(cell["vertices"]))
            if "faces" in cell:
                expect(row).to_contain_text(", ".join(cell["faces"]))

        edges = panel.locator(".complex-data > details").filter(
            has=self.page.get_by_role("heading", name="1-dimensional cells", exact=True))
        edges.locator("summary").first.press("Enter")
        expect(edges.locator("table")).to_be_hidden()
        expect(panel.locator("tr[data-id=A] button")).to_be_visible()
        edges.locator("summary").first.press("Enter")
        expect(edges.locator("table")).to_be_visible()
        panel.get_by_role("button", name="Picture").click()
        expect(graph_vertex).to_have_attribute("aria-pressed", "true")
        expect(panel.locator(".complex-data")).to_be_hidden()
