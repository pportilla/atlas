"""Compute in named quotient rings in Chromium and replay the session."""
import json

from tests.algebra_browser import AlgebraBrowserCase, expect
from tests.cli_support import run_script


class QuotientRingBrowserTests(AlgebraBrowserCase):
    def test_declare_reuse_and_replay(self):
        page = self.page
        page.locator("#operation").select_option("quotient-ring")
        self.rows("#generators", "#add-generator", ["x*y-1"])
        results = [self.compute("A")]
        action = page.locator("#quotient-action").select_option
        action("inverse")
        page.locator("#ideal-source").select_option("A")
        page.locator("#polynomial").fill("x")
        results.append(self.compute("b"))
        self.assertEqual(results[-1]["representative"], "y")
        action("power")
        page.locator("#polynomial-source").select_option("b")
        page.locator("#exponent").fill("-0002")
        results.append(self.compute("c"))
        self.assertEqual(results[-1]["representative"], "x^2")
        action("equal")
        page.locator("#polynomial-source").select_option("c")
        page.locator("#other-polynomial").fill("x^2")
        results.append(self.compute("eq"))
        self.assertTrue(results[-1]["value"])
        self.assertEqual(json.loads(self.download("#export"))["result"], results[-1])
        action("declare")
        page.locator("#ideal-source").select_option("")
        results.append(self.compute("B"))
        action("add")
        page.locator("#ideal-source").select_option("B")
        self.assertTrue(page.locator('#polynomial-source option[value="b"]').evaluate("option => option.disabled"))
        page.locator("#session-replay summary").click()
        replay = run_script(self.download("#download-session"))
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual([json.loads(line)["result"] for line in replay.stdout.splitlines()], results)

    def test_saved_ring_survives_expiration_of_the_server_basis(self):
        page = self.page
        page.locator("#operation").select_option("quotient-ring")
        self.rows("#generators", "#add-generator", ["x*y-1"])
        self.compute("A")
        page.locator("#quotient-action").select_option("inverse")
        page.locator("#ideal-source").select_option("A")
        page.locator("#polynomial").fill("x")
        requests = []

        def expire_once(route):
            operands = route.request.post_data_json["operands"]
            requests.append(operands)
            if len(requests) == 1:
                route.fulfill(status=410, json={"status": "error", "error": {"code": "stale_basis_ref"}})
            else:
                route.continue_()

        page.route("**/api/algebra", expire_once)
        page.locator("#result-name").fill("b")
        page.locator("#run").click()
        expect(page.locator("#result")).to_be_visible()
        self.assertEqual(json.loads(self.download("#export"))["result"]["representative"], "y")
        page.locator("#quotient-action").select_option("multiply")
        page.locator("#polynomial-source").select_option("b")
        page.locator("#other-polynomial").fill("x")
        self.assertEqual(self.compute()["representative"], "1")
        self.assertIn("basis_ref", requests[0])
        self.assertEqual(requests[1]["ideal"], ["x*y-1"])
        self.assertNotEqual(requests[0]["basis_ref"], requests[2]["basis_ref"])

    def test_raw_quotient_element_parent(self):
        page = self.page
        page.locator("#operation").select_option("quotient-ring")
        page.locator("#quotient-action").select_option("reduce")
        self.rows("#generators", "#add-generator", ["x^2"])
        page.locator("#polynomial").fill("x+1")
        results = [self.compute("e")]
        page.locator("#quotient-action").select_option("inverse")
        page.locator("#ideal-source").select_option("e.parent")
        page.locator("#polynomial-source").select_option("e")
        results.append(self.compute())
        self.assertEqual(results[-1]["representative"], "-x+1")
        page.locator("#session-replay summary").click()
        replay = run_script(self.download("#download-session"))
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual([json.loads(line)["result"] for line in replay.stdout.splitlines()], results)
