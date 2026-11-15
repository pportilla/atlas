from html import unescape
import re
import threading
import unittest
from textwrap import dedent
from urllib.request import urlopen

from engine.app import AtlasServer, WEB, WEB_FONTS
from engine.web_brand import fill_brand, lettered_map_html, wordmark_html
from engine.welcome import WORDMARK, lettered_map
from tests.http_support import QuietHandler

TAG = re.compile(r"<[^>]+>")


class WebIdentityTests(unittest.TestCase):
    def test_brand_html_draws_the_terminal_rows(self):
        self.assertEqual(unescape(TAG.sub("", wordmark_html())).split("\n"), list(WORDMARK))
        plain = dedent("\n".join(row.rstrip() for row in lettered_map(color=False)))
        self.assertEqual(unescape(TAG.sub("", lettered_map_html())), plain)
        self.assertIn('<span class="t-paper">', lettered_map_html())
        self.assertIn('<span class="t-brand">', lettered_map_html())
        # The map carries every row of the wordmark, one letter to a panel.
        self.assertIn("/_/  |_|", plain)

    def test_every_role_the_html_uses_has_a_style(self):
        styles = (WEB / "identity.css").read_text()
        for role in set(re.findall(r't-(\w+)', lettered_map_html() + wordmark_html())):
            self.assertIn(f".t-{role}", styles)

    def test_server_fills_the_page_and_serves_the_identity(self):
        server = AtlasServer(("127.0.0.1", 0), QuietHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/", timeout=5) as response:
                page = response.read().decode()
            self.assertNotIn("<!--atlas:", page)
            self.assertIn(wordmark_html(), page)
            self.assertIn(lettered_map_html(), page)
            with urlopen(base + "/identity.css", timeout=5) as response:
                self.assertIn("--font-mono", response.read().decode())
            for font in WEB_FONTS:
                with self.subTest(font=font), urlopen(base + font, timeout=5) as response:
                    self.assertEqual(response.headers["Content-Type"], "font/woff2")
                    self.assertGreater(len(response.read()), 1000)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_fill_brand_leaves_other_pages_alone(self):
        self.assertEqual(fill_brand("<p>plain</p>"), "<p>plain</p>")


if __name__ == "__main__":
    unittest.main()
