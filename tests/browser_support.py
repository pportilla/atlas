"""Exercise the Resolution workspace through Chromium and a local Atlas server."""

import threading
import unittest

from engine.app import AtlasServer
from tests.http_support import QuietHandler

try:
    from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright
except ImportError:
    expect = sync_playwright = None


@unittest.skipUnless(sync_playwright, "requires Playwright and Chromium")
class BrowserCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        server = AtlasServer(("127.0.0.1", 0), QuietHandler)
        self.url = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        context = self.browser.new_context()
        self.addCleanup(context.close)
        self.page = context.new_page()

    def download(self, selector):
        with self.page.expect_download() as download:
            self.page.locator(selector).click()
        return download.value.path().read_text(encoding="utf-8")
