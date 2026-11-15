from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import posixpath
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from engine.session import new_session

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("atlas_docs_build", ROOT / "docs" / "site" / "build.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Dataclasses look their module up while the file runs.
    spec.loader.exec_module(module)
    return module


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href"):
            self.hrefs.append(attrs["href"])


@unittest.skipUnless(importlib.util.find_spec("markdown_it"), "The docs build needs markdown-it-py")
class DocsSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name)
        cls.pages = load_builder().build(cls.out)
        cls.parsed = {}
        for path in cls.out.rglob("*.html"):
            links = Links()
            links.feed(path.read_text(encoding="utf-8"))
            cls.parsed[path.relative_to(cls.out).as_posix()] = links

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_markdown_pages_follow_the_readme_links(self):
        for page in ("index.html", "docs/sessions.html", "examples/algebra-api.html", "native/KERNEL.html",
                     "benchmarks/index.html", "reference/functions.html", "genindex.html"):
            self.assertIn(page, self.parsed)
        self.assertEqual(self.pages[0].out, "index.html")

    def test_internal_links_reach_a_page_and_an_anchor(self):
        for page, links in self.parsed.items():
            for href in links.hrefs:
                parts = urlsplit(href)
                if parts.scheme or href.startswith("//"):
                    continue
                target = posixpath.normpath(posixpath.join(posixpath.dirname(page), unquote(parts.path))) \
                    if parts.path else page
                with self.subTest(page=page, href=href):
                    self.assertIn(target, self.parsed)
                    if parts.fragment:
                        self.assertIn(unquote(parts.fragment), self.parsed[target].ids)

    def test_reference_holds_every_session_function_card(self):
        session = new_session(None)
        names = [name for name, value in session.builtins.items() if callable(value)]
        self.assertTrue(set(names) <= self.parsed["reference/functions.html"].ids)
        html = (self.out / "reference/functions.html").read_text(encoding="utf-8")
        self.assertIn("Compute the dimension over QQ of the Jacobian algebra", html)
        entry = html.split('<section class="api-entry" id="polynomial">', 1)[1].split("</section>", 1)[0]
        self.assertIn('href="../docs/sessions.html#choose-the-ring-of-a-polynomial"', entry)

    def test_inline_calls_link_to_the_reference_and_back(self):
        sessions = self.parsed["docs/sessions.html"].hrefs
        self.assertIn("../reference/functions.html#recompute_dual_complex", sessions)
        html = (self.out / "reference/functions.html").read_text(encoding="utf-8")
        entry = html.split('id="recompute_dual_complex"', 1)[1].split("</section>", 1)[0]
        # The entry links back to the section of the guide that mentions it.
        self.assertIn('href="../docs/sessions.html#', entry)

    def test_repository_files_link_to_their_source(self):
        hrefs = self.parsed["docs/lean-verification-roadmap.html"].hrefs
        self.assertTrue(any(h.endswith("/blob/main/native/groebner.cpp") for h in hrefs))

    def test_maths_and_transcripts(self):
        kernel = (self.out / "native/KERNEL.html").read_text(encoding="utf-8")
        self.assertIn('<span class="math">', kernel)
        self.assertIn("katex.min.js", kernel)
        sessions = (self.out / "docs/sessions.html").read_text(encoding="utf-8")
        self.assertIn('<span class="line in"><span class="t-brand">atlas</span>', sessions)
        self.assertIn('<span class="t-function">coordinates</span>', sessions)

    def test_search_index_and_static_files(self):
        index = (self.out / "_static/search-index.js").read_text(encoding="utf-8")
        self.assertTrue(index.startswith("window.ATLAS_SEARCH = ["))
        self.assertIn('"title":"milnor_number"', index)
        for name in ("identity.css", "docs.css", "docs.js", "fonts/source-code-pro.woff2"):
            self.assertTrue((self.out / "_static" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
