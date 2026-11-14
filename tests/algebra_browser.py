"""Real browser and isolated local Atlas server for the guided Algebra tests."""

import threading
import unittest

from engine.app import AtlasServer
from tests.test_http import QuietHandler
from tests.test_cli_session import ROOT

try:
    from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright
except ImportError:
    expect = sync_playwright = None


@unittest.skipUnless(sync_playwright, "requires Playwright and Chromium")
class AlgebraBrowserCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server = AtlasServer(("127.0.0.1", 0), QuietHandler)
        cls.url = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cls.addClassCleanup(thread.join)
        cls.addClassCleanup(server.server_close)
        cls.addClassCleanup(server.shutdown)
        playwright = sync_playwright().start()
        cls.addClassCleanup(playwright.stop)
        try:
            cls.browser = playwright.chromium.launch()
        except PlaywrightError as exc:
            if "Executable doesn't exist at " in str(exc):
                raise unittest.SkipTest("Chromium is missing; run playwright install chromium") from exc
            raise
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        context = self.browser.new_context(permissions=["clipboard-read", "clipboard-write"])
        self.addCleanup(context.close)
        self.page = context.new_page()
        self.page.goto(self.url)
        self.page.get_by_role("link", name="Algebra").click()
        self.page.locator("#variables").fill("x, y")
        self.page.locator("#order").select_option("lex")

    def rows(self, group, add, values):
        rows = self.page.locator(group + " input")
        while rows.count() < len(values):
            self.page.locator(add).click()
        while rows.count() > len(values):
            self.page.locator(group + " button").last.click()
        for row, value in zip(rows.all(), values):
            row.fill(value)

    def download(self, selector):
        with self.page.expect_download() as download:
            self.page.locator(selector).click()
        return download.value.path().read_text(encoding="utf-8")
