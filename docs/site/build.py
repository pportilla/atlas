"""Build the Atlas documentation site from the repository's Markdown.

The manual is organized like SageMath's: README.md is the front page and lists
the documents in DOCUMENTS (a tour, the tutorial, constructions, thematic
tutorials, the reference manual, the installation guide and the developer
guide). Each document opens on a landing page that links its chapters.

Some pages are completed by the builder, as Sphinx's autodoc completes Sage's
reference manual. A subject page of the reference manual receives the cards of
its session functions, combining help cards with the implementations and test
cases from catalog.py. The function index and the visual identity are
completed the same way. Source views show the files from the same checkout,
including uncommitted changes.

Run from the repository root:

    python3 docs/site/build.py [--out build/docs]

The build needs markdown-it-py. Maths renders with KaTeX in the browser.
"""
import argparse
from collections import deque
from dataclasses import dataclass, field
from html import escape
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import sys
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
SITE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from markdown_it import MarkdownIt
except ModuleNotFoundError as error:
    if error.name != "markdown_it":
        raise
    raise SystemExit("Install the documentation dependencies with "
                     f"{sys.executable} -m pip install -r requirements-docs.txt") from None

from docs.site.catalog import ALIASES, ENTRIES, TOPICS  # noqa: E402
from docs.site.function_reference import FUNCTIONS, reference_for  # noqa: E402
from docs.site.sources import SourceFiles, browsable  # noqa: E402
from engine.app import make_parser, shell_command_names  # noqa: E402
from engine.cli_highlight import assigned_names, command_pieces, expression_pieces  # noqa: E402
from engine.help_card import REPOSITORY_URL  # noqa: E402
from engine.session import new_session  # noqa: E402
from engine.session_function_help import function_documentation, function_help  # noqa: E402
from engine.web_brand import lettered_map_html, wordmark_html  # noqa: E402
from engine.welcome import atlas_version  # noqa: E402

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs"

TUTORIAL = "docs/tutorial/index.md"
CONSTRUCTIONS = "docs/constructions.md"
REFERENCE = "docs/reference/index.md"
FUNCTION_INDEX = "docs/reference/genindex.md"
IDENTITY = "docs/developer/identity.md"

# The values returned by the resolution functions have a page of their own
# beside the functions that create them.
RESOLUTION_VALUES = "docs/reference/resolution-values.md"
REFERENCE_TOPICS = tuple(page for topic in TOPICS for page in (
    (topic, RESOLUTION_VALUES) if topic == "docs/reference/resolution.md" else (topic,)))

# The front page lists these documents in the same order. Each document is a
# landing page and its chapters, like a Sphinx toctree.
DOCUMENTS = (
    ("Tutorials", (
        ("docs/tour.md", ()),
        (TUTORIAL, tuple(f"docs/tutorial/{name}.md" for name in (
            "prompt-and-scripts", "rings", "polynomials-and-ideals", "quotient-rings",
            "local-algebra", "local-geometry", "resolving-a-cusp"))),
        (CONSTRUCTIONS, ()),
        ("docs/thematic/index.md", tuple(f"docs/thematic/{name}.md" for name in (
            "zeta-functions", "resolution-records", "resolution-pipeline", "weighted-resolution"))),
    )),
    ("Reference", (
        (REFERENCE, (*REFERENCE_TOPICS, "docs/reference/saved-values.md", "docs/reference/command-line.md",
                     "docs/reference/http-and-compute.md", "docs/reference/algebra-requests.md",
                     "native/KERNEL.md", FUNCTION_INDEX)),
    )),
    ("Installation and development", (
        ("docs/installation.md", ()),
        ("docs/developer/index.md", ("benchmarks/README.md",
                                     "docs/developer/development-roadmap.md",
                                     "docs/developer/lean-verification-roadmap.md", IDENTITY)),
    )),
)
# These chapters gather the notes in their folder that their pages link to.
SECTION_FOLDERS = {
    "native/KERNEL.md": ("native/",),
    "benchmarks/README.md": ("benchmarks/", "docs/developer/perf-log-"),
}
TOPNAV = (("Tutorial", TUTORIAL), ("Constructions", CONSTRUCTIONS), ("Reference", REFERENCE),
          ("Index", FUNCTION_INDEX))
# A function's card links to the passages in these guides that use it.
GUIDES = ("docs/tour.md", CONSTRUCTIONS, "docs/tutorial/", "docs/thematic/")


@dataclass(eq=False)
class Page:
    source: str             # repository path of the Markdown, or "" for generated pages
    out: str                # site path of the HTML
    title: str = ""
    children: list = field(default_factory=list)
    headings: list = field(default_factory=list)   # (level, id, text)
    sections: list = field(default_factory=list)   # search entries
    body: str = ""
    head: set = field(default_factory=set)         # "math", "mermaid"
    lede: bool = False

    @property
    def root(self):
        return "../" * (len(PurePosixPath(self.out).parts) - 1)


# Markdown --------------------------------------------------------------------

def _math_inline(state, silent):
    """Read $...$ and $$...$$ without Markdown escapes, as TeX writers expect."""
    src, start = state.src, state.pos
    if src[start] != "$":
        return False
    delim = "$$" if src.startswith("$$", start) else "$"
    begin = start + len(delim)
    if begin >= state.posMax or (delim == "$" and src[begin].isspace()):
        return False
    end = begin
    while True:
        end = src.find(delim, end)
        if end == -1 or end >= state.posMax:
            return False
        if src[end - 1] == "\\":
            end += 1
            continue
        if delim == "$" and (src[end - 1].isspace() or src[end + 1:end + 2].isdigit()):
            end += 1
            continue
        break
    if not silent:
        token = state.push("math_display" if delim == "$$" else "math_inline", "", 0)
        token.content = src[begin:end].strip()
    state.pos = end + len(delim)
    return True


def markdown():
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    md.inline.ruler.before("escape", "math", _math_inline)
    return md


def slugify(text, used):
    """GitHub's heading anchors, so links written for GitHub keep working."""
    slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
    base, n = slug, 0
    while slug in used:
        n += 1
        slug = f"{base}-{n}"
    used.add(slug)
    return slug


def _inline_text(token):
    words = {"text": None, "code_inline": None, "math_inline": None, "softbreak": " ", "hardbreak": " "}
    return "".join(child.content if words[child.type] is None else words[child.type]
                   for child in token.children or () if child.type in words)


def _is_external(href):
    return bool(urlsplit(href).scheme) or href.startswith("//")


def out_path(source):
    """Site path of a Markdown source. Documents under docs/ sit at the top of the site."""
    path = PurePosixPath(source)
    if path.parts[0] == "docs":
        path = PurePosixPath(*path.parts[1:])
    return str(path.with_name("index.html") if path.name in ("README.md", "index.md") else path.with_suffix(".html"))


FUNCTION_PAGES = {name: out_path(source) for source, names in TOPICS.items() for name in names}


def function_url(name):
    """Site path of the reference card for a session function or one of its aliases."""
    name = ALIASES.get(name, name)
    return f"{FUNCTION_PAGES[name]}#{name}"


def repository_url():
    try:
        url = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True,
                                      stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return REPOSITORY_URL
    url = re.sub(r"^git@github\.com:", "https://github.com/", url)
    remote = urlsplit(url)
    if remote.scheme in ("http", "https") and remote.hostname:
        # The published manual needs only the public web address of the checkout.
        host = remote.netloc.rsplit("@", 1)[-1]
        return urlunsplit((remote.scheme, host, remote.path.rstrip("/").removesuffix(".git"), "", ""))
    if remote.scheme == "ssh" and remote.hostname == "github.com":
        return "https://github.com" + remote.path.removesuffix(".git")
    return REPOSITORY_URL


# Highlighting ----------------------------------------------------------------

def _spans(pieces, function_link=None):
    parts = []
    for text, role in pieces:
        span = f'<span class="t-{role}">{escape(text)}</span>' if role else escape(text)
        if function_link and role == "function":
            span = f'<a class="code-ref" href="{function_link(text)}">{span}</a>'
        parts.append(span)
    return "".join(parts)


class Highlighter:
    """Colour code the way the Atlas prompt colours it, role for role."""

    def __init__(self, functions, commands):
        self.functions, self.commands = functions, commands

    def _input(self, text, known):
        first = text.split(maxsplit=1)[0] if text.strip() else ""
        if first in self.commands and not text.lstrip().startswith(first + "("):
            return command_pieces(text, commands=self.commands, topics=self.functions)
        known |= assigned_names(text)
        return expression_pieces(text, names=known, functions=self.functions)

    def block(self, code, lang, function_link=None):
        lines = code.rstrip("\n").split("\n")
        known = set()
        out = []
        if any(line.startswith("atlas>") for line in lines):
            for line in lines:
                if line.startswith("atlas>"):
                    text = line[6:].removeprefix(" ")
                    out.append('<span class="line in"><span class="t-brand">atlas</span><span class="t-muted">&gt;</span> '
                               + _spans(self._input(text, known), function_link) + "</span>")
                elif line.startswith("... "):
                    out.append('<span class="line in"><span class="t-muted">...</span> '
                               + _spans(expression_pieces(line[4:], names=known, functions=self.functions), function_link) + "</span>")
                else:
                    pieces = expression_pieces(line, names=known, functions=self.functions, verdicts=True)
                    out.append('<span class="line out">' + _spans(pieces) + "</span>")
        elif lang in ("atlas", "python", "py"):
            out = ['<span class="line in">' + _spans(self._input(line, known), function_link) + "</span>" for line in lines]
        elif lang in ("sh", "bash", "shell", "console"):
            continued = False
            for line in lines:
                text = line.removeprefix("$ ")
                head = set() if continued or not text.strip() else {text.split()[0]}
                pieces = command_pieces(text, commands=head)
                out.append('<span class="line in">' + _spans(pieces) + "</span>")
                continued = line.rstrip().endswith("\\")
        elif lang == "json":
            out = ['<span class="line">' + _spans(expression_pieces(line)) + "</span>" for line in lines]
        else:
            out = ['<span class="line">' + escape(line) + "</span>" for line in lines]
        return "\n".join(out)


# Rendering -------------------------------------------------------------------

class Renderer:
    def __init__(self, functions, highlighter, source_url):
        self.md = markdown()
        self.pages = {}
        self.pending = deque()
        self.functions = functions
        self.highlighter = highlighter
        self.source_url = source_url
        self.sources = SourceFiles(ROOT)
        self.mentions = {}  # function name -> [(page, anchor, section title)]
        self.examples = {}  # first worked example of a function on each page
        rules = self.md.renderer.rules
        rules["math_inline"] = lambda tokens, i, *_: f'<span class="math">{escape(tokens[i].content)}</span>'
        rules["math_display"] = lambda tokens, i, *_: (
            f'<span class="math math-display">{escape(tokens[i].content)}</span>')
        rules["fence"] = self._fence
        rules["code_block"] = self._fence
        rules["code_inline"] = self._code_inline
        rules["table_open"] = lambda *_: '<div class="table-wrap"><table>\n'
        rules["table_close"] = lambda *_: "</table></div>\n"
        rules["heading_close"] = self._heading_close

    def guide(self, source):
        """Queue each linked guide once, in the order readers encounter it."""
        if source not in self.pages:
            page = Page(source, out_path(source))
            self.pages[source] = page
            self.pending.append(page)
        return self.pages[source]

    def render_guides(self):
        self.guide("README.md")
        while self.pending:
            self.page(self.pending.popleft())
        return self.pages

    def _fence(self, tokens, i, options, env):
        token = tokens[i]
        lang = (token.info or "").strip().partition(" ")[0].lower()
        if lang == "mermaid":
            env["page"].head.add("mermaid")
            return f'<pre class="mermaid">{escape(token.content)}</pre>\n'
        page = env["page"]

        def function_link(spelling):
            name = ALIASES.get(spelling, spelling)
            if "section" in token.meta:
                self.examples.setdefault(name, {}).setdefault(page, token.meta["section"])
            return page.root + function_url(name)

        code = self.highlighter.block(token.content, lang, function_link)
        label = f' data-lang="{escape(lang)}"' if lang and lang != "text" else ""
        return f'<pre class="code"{label}><code>{code}</code></pre>\n'

    def _code_inline(self, tokens, i, options, env):
        token = tokens[i]
        code = f"<code>{escape(token.content)}</code>"
        name = token.meta.get("function")
        if name:
            return f'<a class="api-ref" href="{env["page"].root}{function_url(name)}">{code}</a>'
        return code

    def _heading_close(self, tokens, i, options, env):
        anchor = tokens[i].meta.get("anchor")
        link = f'<a class="headerlink" href="#{anchor}" aria-label="Link to this section">¶</a>' if anchor else ""
        return f"{link}</{tokens[i].tag}>\n"

    def function_in(self, code):
        """Find a registered session function called or named in an inline code span."""
        for call in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\(", code):
            if call[1] in self.functions:
                return ALIASES.get(call[1], call[1])
        bare = code.strip()
        return ALIASES.get(bare, bare) if bare in self.functions else None

    def link(self, page, href, section_id, section_title):
        """Point a link written for GitHub at the matching page or source file."""
        if _is_external(href):
            return href, "external"
        if href.startswith("#"):
            return href, ""
        path, _, anchor = href.partition("#")
        target = posixpath.normpath(posixpath.join(posixpath.dirname(page.source), unquote(path)))
        suffix = f"#{anchor}" if anchor else ""
        if not (ROOT / target).exists():
            raise ValueError(f"{page.source} links to a missing file: {href}")
        if target.endswith(".md"):
            relative = posixpath.relpath(self.guide(target).out, posixpath.dirname(page.out) or ".")
            return quote(relative) + suffix, ""
        if browsable(target):
            source = self.sources.get(target)
            url = page.out + ("#" + section_id if section_id else "")
            source.readers[url] = section_title or page.title
            return page.root + quote(source.out) + suffix, "source"
        kind = "tree" if (ROOT / target).is_dir() else "blob"
        return f"{self.source_url}/{kind}/main/{quote(target)}{suffix}", "source"

    def _prepare(self, page, tokens):
        """Give headings their anchors, links their targets and code its references."""
        used = set()
        section_id, section_title = "", page.title
        text = []

        def close_section():
            if section_title:
                body = " ".join(" ".join(text).split())
                page.sections.append({"page": page.title, "title": section_title,
                                      "url": page.out + (f"#{section_id}" if section_id else ""),
                                      "text": body[:1600]})
            text.clear()

        for i, token in enumerate(tokens):
            if token.type == "heading_open":
                inline = tokens[i + 1]
                title = _inline_text(inline).strip()
                anchor = slugify(title, used)
                token.attrSet("id", anchor)
                tokens[i + 2].meta["anchor"] = anchor
                level = int(token.tag[1])
                if level == 1 and not page.title:
                    page.title = title
                    section_title = title
                    continue
                page.headings.append((level, anchor, title))
                if level <= 3:
                    close_section()
                    section_id, section_title = anchor, title
            elif token.type == "inline":
                text.append(_inline_text(token))
                inside_link = False
                for child in token.children or ():
                    if child.type == "link_open":
                        inside_link = True
                        href, kind = self.link(page, child.attrGet("href") or "", section_id, section_title)
                        child.attrSet("href", href)
                        if kind:
                            child.attrSet("class", kind)
                    elif child.type == "link_close":
                        inside_link = False
                    elif child.type == "code_inline" and not inside_link:
                        name = self.function_in(child.content)
                        if name:
                            child.meta["function"] = name
                            entry = (page, section_id, section_title)
                            if entry not in self.mentions.setdefault(name, []):
                                self.mentions[name].append(entry)
                    elif child.type in ("math_inline", "math_display"):
                        page.head.add("math")
            elif token.type in ("fence", "code_block"):
                token.meta["section"] = (section_id, section_title)
                text.append(token.content)
        close_section()

    def page(self, page):
        env = {"page": page}
        source = (ROOT / page.source).read_text(encoding="utf-8")
        tokens = self.md.parse(source, env)
        self._prepare(page, tokens)
        if page.out == "index.html" and tokens and tokens[0].type == "heading_open" and tokens[0].tag == "h1":
            # The lettered map stands in for the home page's title.
            tokens = tokens[3:]
            page.lede = True
        page.body = self.md.renderer.render(tokens, self.md.options, env)
        if page.lede:
            page.body = page.body.replace("<p>", '<p class="lede">', 1)
            page.body = (f'<div class="hero"><pre class="ascii-art" aria-hidden="true">{lettered_map_html()}</pre>'
                         f'<h1 class="visually-hidden">{escape(page.title)}</h1></div>\n' + page.body)

    def prose(self, text, page):
        """Render a short piece of Markdown, such as the purpose on a help card."""
        env = {"page": page}
        tokens = self.md.parse(text, env)
        for token in tokens:
            if token.type == "inline":
                inside_link = False
                for child in token.children or ():
                    if child.type == "link_open":
                        # Card text links to guides as the page's own Markdown does.
                        inside_link = True
                        href, kind = self.link(page, child.attrGet("href") or "", None, None)
                        child.attrSet("href", href)
                        if kind:
                            child.attrSet("class", kind)
                    elif child.type == "link_close":
                        inside_link = False
                    elif child.type == "code_inline" and not inside_link:
                        name = self.function_in(child.content)
                        if name:
                            child.meta["function"] = name
                    elif child.type.startswith("math"):
                        page.head.add("math")
        return self.md.renderer.render(tokens, self.md.options, env)


# Generated pages -------------------------------------------------------------

_CODE_WORD = re.compile(r"(?<![\w.`])(?!e\.g|i\.e)([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z_]\w*\([^()\s]*\))(?![\w`])")


def _card(text):
    """Split a help card into its signature, purpose and example lines."""
    lines = text.split("\n")
    signature = lines[0]
    try:
        cut = lines.index("Example")
    except ValueError:
        cut = len(lines)
    purpose = "\n".join(lines[1:cut]).strip()
    example = [line[2:] for line in lines[cut + 1:] if line.startswith("  ")]
    return signature, purpose, example


def implementation_panel(renderer, page, name):
    entry = ENTRIES[name]
    lists = []
    for title, references in (("Implementation", entry.sources), ("Test cases", entry.tests)):
        links = []
        for ref in references:
            source = renderer.sources.get(ref.path)
            source.readers[function_url(name)] = name
            href = page.root + source.out + source.fragment(ref)
            links.append(f'<li><a href="{href}">{escape(ref.label)}</a> '
                         f'<code>{escape(PurePosixPath(ref.path).name)}</code></li>')
        lists.append(f'<p class="api-label">{title}</p><ul>{"".join(links)}</ul>')
    return ('<details class="api-sources"><summary>Implementation and tests</summary>'
            + "".join(lists) + '</details>')


def check_catalog(session):
    """Require one reference card for every session function and its aliases."""
    names = {name for name, value in session.builtins.items() if callable(value)}
    documented = ENTRIES.keys() | ALIASES.keys()
    if documented != names:
        raise ValueError(f"Update the function catalog for {sorted(documented ^ names)}")
    if FUNCTIONS.keys() != ENTRIES.keys():
        raise ValueError(f"Update the input and return descriptions for {sorted(FUNCTIONS.keys() ^ ENTRIES.keys())}")
    for alias, name in ALIASES.items():
        if session.builtins[alias] is not session.builtins[name]:
            raise ValueError(f"{alias} no longer refers to {name}")


def function_card(renderer, page, session, highlighter, name):
    """Render a function's signature, purpose, inputs, returns, example, guides and sources."""
    signature, purpose, example = _card(function_help(name, session.builtins))
    inputs, returns = reference_for(name, session.builtins[name])
    params_html = _spans([(text, "field" if role is None and text.isidentifier() else role)
                          for text, role in expression_pieces(signature[len(name):])])
    parts = [f'<section class="api-entry" id="{name}">']
    aliases = [alias for alias, canonical in ALIASES.items() if canonical == name]
    # Preserve links and bookmarks under earlier spellings.
    parts.extend(f'<span id="{alias}"></span>' for alias in aliases)
    parts.append(f'<h2 class="api-signature"><code><span class="api-name">{name}</span>{params_html}</code>'
                 f'<a class="headerlink" href="#{name}" aria-label="Link to {name}">¶</a></h2>')
    if aliases:
        parts.append('<p>Also available as ' + ', '.join(f'<code>{a}</code>' for a in aliases) + '.</p>')
    parts.append(renderer.prose(_CODE_WORD.sub(r"`\1`", purpose), page))
    parts.append('<h3 class="api-label">Inputs and options</h3>')
    if inputs:
        parts.append('<dl class="api-parameters">')
        for parameter, description in inputs:
            parts.append(f'<dt><code>{escape(parameter)}</code></dt>'
                         f'<dd>{renderer.prose(description, page)}</dd>')
        parts.append('</dl>')
    else:
        parts.append('<p>This function takes no arguments.</p>')
    parts.append('<h3 class="api-label">Returns</h3>')
    parts.append(renderer.prose(returns, page))
    if example:
        parts.append('<h3 class="api-label">Example</h3>')
        transcript = []
        for line in example:
            command, marker, output = line.partition("  # ")
            transcript.append("atlas> " + command)
            if marker:
                transcript.append(output)
        lines = "\n".join(transcript)
        parts.append(f'<pre class="code"><code>{highlighter.block(lines, "text")}</code></pre>')

    source, section = function_documentation(name, session.builtins).split("#", 1)
    guide = renderer.pages[source]
    guide_url = page.root + guide.out + "#" + section
    guide_title = next((title for _, anchor, title in guide.headings if anchor == section), guide.title)
    guide_links = [f'<a href="{escape(guide_url)}">{escape(guide_title)}</a>']
    mentions = renderer.mentions.get(name, [])
    linked_pages = {p for p, _, _ in mentions}
    examples = [(p, *section) for p, section in renderer.examples.get(name, {}).items() if p not in linked_pages]
    for p, section_id, title in [*mentions, *examples]:
        if p.source.startswith(GUIDES):
            url = page.root + p.out + ("#" + section_id if section_id else "")
            if url != guide_url:
                guide_links.append(f'<a href="{url}">{escape(title)}</a>')
    parts.append('<p class="api-mentions"><span>In the guides</span> '
                 + ' · '.join(guide_links) + '</p>')
    parts.append(implementation_panel(renderer, page, name))
    parts.append('</section>')
    entry = {"page": page.title, "title": name, "url": function_url(name),
             "text": " ".join([signature, *aliases, purpose, returns, *(description for _, description in inputs)]),
             "kind": "function", "aliases": aliases}
    return "\n".join(parts), entry


def complete_subject(renderer, page, names, session, highlighter):
    """Append the cards of a subject's session functions to its page of the reference manual."""
    parts = ['<p class="topic-functions">' + ' · '.join(f'<a href="#{name}">{name}</a>' for name in names) + '</p>']
    for name in names:
        card, entry = function_card(renderer, page, session, highlighter, name)
        parts.append(card)
        page.headings.append((2, name, name))
        page.sections.append(entry)
    page.body += "\n".join(parts)


def _xterm(code):
    """The sRGB colour of an xterm 256-colour index."""
    if code < 16:
        return ["#000000", "#cd0000", "#00cd00", "#cdcd00", "#0000ee", "#cd00cd", "#00cdcd", "#e5e5e5",
                "#7f7f7f", "#ff0000", "#00ff00", "#ffff00", "#5c5cff", "#ff00ff", "#00ffff", "#ffffff"][code]
    if code < 232:
        steps = (0, 95, 135, 175, 215, 255)
        code -= 16
        return "#{:02x}{:02x}{:02x}".format(steps[code // 36], steps[code // 6 % 6], steps[code % 6])
    level = 8 + (code - 232) * 10
    return f"#{level:02x}{level:02x}{level:02x}"


def complete_identity(page):
    """Append the shared faces, palette and terminal roles, read from their sources."""
    from engine.cli_style import _ROLES

    css = (ROOT / "web" / "identity.css").read_text(encoding="utf-8")
    tokens = re.findall(r"--([\w-]+):\s*light-dark\((#[0-9a-f]{6}),\s*(#[0-9a-f]{6})\);(?:\s*/\*\s*(.*?)\s*\*/)?", css)
    rows = "".join(
        f'<tr><td><code>--{name}</code></td>'
        f'<td><span class="swatch" style="background:{light}"></span><code>{light}</code></td>'
        f'<td><span class="swatch" style="background:{dark}"></span><code>{dark}</code></td>'
        f'<td>{escape(note)}</td></tr>' for name, light, dark, note in tokens)
    source = (ROOT / "engine" / "cli_style.py").read_text(encoding="utf-8")
    notes = dict(re.findall(r'"(\w+)": \([^)]*\),\s*#\s*(.*)', source))
    roles = []
    for role, (_, rich) in _ROLES.items():
        codes = rich.split(";")
        colour = _xterm(int(codes[codes.index("5") + 1])) if "38" in codes else ""
        weight = "font-weight:700;" if "1" in codes[:1] else ""
        style = (f"color:{colour};" if colour else "") + weight + ("font-style:italic;" if "3" in codes[:2] else "")
        sample = f'<span style="{style}">{escape(role)}</span>'
        roles.append(f"<tr><td><code>{role}</code></td><td><code>{rich}</code></td>"
                     f'<td class="term dark">{sample}</td><td class="term light">{sample}</td>'
                     f"<td>{escape(notes.get(role, ''))}</td></tr>")
    page.body += f"""<div class="hero hero-small"><pre class="ascii-art" aria-hidden="true">{lettered_map_html()}</pre></div>
<h2 id="faces">Faces</h2>
<p>The monospace carries the wordmark, headings and code, as in a terminal. Text runs in its humanist companion.
Mathematics is set in STIX.</p>
<div class="specimens">
<p class="specimen" style="font-family:var(--font-mono);font-weight:650">Source Code Pro <span>Resolve a hypersurface germ</span></p>
<p class="specimen" style="font-family:var(--font-text)">Source Sans 3 <span>Atlas records each chart, divisor and blow-up.</span></p>
<p class="specimen" style="font-family:var(--font-math);font-style:italic">STIX Two Text <span>&#950;(T) = (1 &#8722; T<sup>6</sup>)/((1 &#8722; T<sup>2</sup>)(1 &#8722; T<sup>3</sup>))</span></p>
</div>
<h2 id="colours">Colours</h2>
<p>Each token has a light value for paper and a dark value for the terminal. Code blocks always use the dark values.</p>
<div class="table-wrap"><table class="palette"><thead><tr><th>Token</th><th>Light</th><th>Dark</th><th>Role</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h2 id="terminal-roles">Terminal roles</h2>
<p>The prompt colours each word by what it names. The web and the docs give the same roles the classes
<code>t-ROLE</code>.</p>
<div class="table-wrap"><table class="palette"><thead><tr><th>Role</th><th>256-colour code</th><th>On dark</th><th>On light</th><th>Use</th></tr></thead>
<tbody>{"".join(roles)}</tbody></table></div>
"""
    headings = [(2, "faces", "Faces"), (2, "colours", "Colours"), (2, "terminal-roles", "Terminal roles")]
    page.headings.extend(headings)
    page.sections.extend({"page": page.title, "title": title, "url": f"{page.out}#{anchor}", "text": title}
                         for _, anchor, title in headings)


def complete_index(page, functions):
    """Append an alphabetical list of the session functions and their aliases."""
    groups = {}
    names = {name: entry for entry in functions for name in (entry["title"], *entry["aliases"])}
    for name, entry in sorted(names.items()):
        groups.setdefault(name[0].upper(), []).append((name, entry))
    parts = ['<nav class="letter-bar" aria-label="Functions by letter">'
             + "".join(f'<a href="#letter-{letter}">{letter}</a>' for letter in groups) + '</nav>']
    for letter, entries in groups.items():
        parts.append(f'<h2 id="letter-{letter}">{letter}</h2><ul class="index-list">')
        for name, entry in entries:
            detail = entry["page"] if name == entry["title"] else f'Alias of {entry["title"]}'
            parts.append(f'<li><a href="{page.root}{entry["url"]}"><code>{escape(name)}</code></a> '
                         f'<span class="index-where">{escape(detail)}</span></li>')
        parts.append('</ul>')
    page.body += "\n".join(parts)


def lede(page):
    """Set the first paragraph of a landing page as its lede."""
    if 'class="lede"' not in page.body:
        page.body = page.body.replace("<p>", '<p class="lede">', 1)


# Layout ----------------------------------------------------------------------


def walk(page):
    yield page
    for child in page.children:
        yield from walk(child)


def navigation(pages):
    """Attach chapters to their documents and return the sidebar groups of landing pages."""
    for chapter, folders in SECTION_FOLDERS.items():
        parent = pages[chapter]
        parent.children = [p for path, p in pages.items() if path.startswith(folders) and p is not parent]
    groups = []
    for title, documents in DOCUMENTS:
        landings = []
        for landing, chapters in documents:
            missing = [path for path in (landing, *chapters) if path not in pages]
            if missing:
                raise ValueError(f"No page links to {missing}; add them to their document's landing page")
            pages[landing].children = [pages[path] for path in chapters]
            landings.append(pages[landing])
        groups.append((title, landings))
    return groups


def sidebar(page, home, groups):
    """List every document, with the chapters of the current document open."""
    def item(p):
        current = ' aria-current="page"' if p is page else ""
        link = f'<a href="{page.root}{p.out}"{current}>{escape(p.title)}</a>'
        children = ""
        if p.children and page in walk(p):
            children = '<ul>' + "".join(item(child) for child in p.children) + '</ul>'
        return f"<li>{link}{children}</li>"

    return (f'<div class="nav-group"><ul>{item(home)}</ul></div>'
            + "".join(f'<div class="nav-group"><h2>{escape(title)}</h2><ul>'
                      + "".join(item(p) for p in members) + '</ul></div>'
                      for title, members in groups))


def topnav(page, pages, documents):
    """Link the tutorial, constructions, reference and index, marking the current one."""
    targets = [(label, pages[source]) for label, source in TOPNAV]
    exact = any(target is page for _, target in targets)
    return "".join(
        f'<a href="{page.root}{target.out}"'
        + (' aria-current="true"' if (target is page if exact else documents.get(page) is target) else "")
        + f">{label}</a>" for label, target in targets)


def search_kind(page):
    source = page.source
    for prefix, kind in (("native/", "Algorithm"), ("benchmarks/", "Benchmark"), ("docs/developer/perf-log-", "Benchmark"),
                         ("docs/reference/", "Reference"),
                         ("docs/developer/", "Developer guide"), (CONSTRUCTIONS, "Construction"),
                         ("docs/installation.md", "Installation")):
        if source.startswith(prefix):
            return kind
    return "Tutorial"


def local_toc(page):
    rows = [(level, anchor, title) for level, anchor, title in page.headings if level in (2, 3)]
    if len(rows) < 2:
        return ""
    links = "".join(f'<li><a class="level-{level}" href="#{anchor}">{escape(title)}</a></li>'
                    for level, anchor, title in rows)
    return f"<h2>On this page</h2><ul>{links}</ul>"


def pager(page, order):
    i = order.index(page)
    parts = []
    if i > 0:
        p = order[i - 1]
        parts.append(f'<a class="prev" href="{page.root}{p.out}"><span class="label">Previous</span>{escape(p.title)}</a>')
    if i + 1 < len(order):
        p = order[i + 1]
        parts.append(f'<a class="next" href="{page.root}{p.out}"><span class="label">Next</span>{escape(p.title)}</a>')
    return f'<nav class="pager" aria-label="Pages">{"".join(parts)}</nav>' if parts else ""


def head_extra(page):
    tags = []
    if "math" in page.head:
        tags.append(f'<link rel="stylesheet" href="{KATEX}katex.min.css">'
                    f'<script src="{KATEX}katex.min.js" defer></script>')
    if "mermaid" in page.head:
        tags.append(f'<script type="module">import mermaid from "{MERMAID}";'
                    'mermaid.initialize({startOnLoad: true, theme: "neutral", fontFamily: "Source Sans 3"});</script>')
    return "\n".join(tags)


def render(template, values):
    return re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], template)


def build(out):
    session = new_session(None)
    check_catalog(session)
    functions = {name for name, value in session.builtins.items() if callable(value)}
    commands = shell_command_names(make_parser()) | {"help"}
    highlighter = Highlighter(functions, commands)
    renderer = Renderer(functions, highlighter, repository_url())
    pages = renderer.render_guides()
    home = pages["README.md"]
    home.title = "Documentation"
    groups = navigation(pages)
    for source, names in TOPICS.items():
        complete_subject(renderer, pages[source], names, session, highlighter)
    complete_index(pages[FUNCTION_INDEX], [entry for source in TOPICS for entry in pages[source].sections
                                           if entry.get("kind") == "function"])
    complete_identity(pages[IDENTITY])
    documents = {}
    for _, landings in groups:
        for landing in landings:
            lede(landing)
            documents.update((p, landing) for p in walk(landing))
    for completed in (*TOPICS, FUNCTION_INDEX, IDENTITY):
        lede(pages[completed])
    order = [home, *documents]
    for p in pages.values():
        if p not in documents and p is not home:
            raise ValueError(f"Missing documentation navigation for {p.source}")
    source_pages = []
    for source in renderer.sources.files.values():
        p = Page(source.path, source.out, source.path)
        p.body = source.body(p.root)
        source_pages.append(p)

    template = (SITE / "template.html").read_text(encoding="utf-8")
    version = escape(atlas_version())
    source_url = renderer.source_url
    out.mkdir(parents=True, exist_ok=True)
    all_pages = [*order, *source_pages]
    for page in all_pages:
        source = (f'<a class="source" href="{escape(source_url)}/blob/main/{quote(page.source)}">'
                  f'{escape(page.source)}</a>' if page.source else "")
        footer = f"Atlas {version}." + (f" Source file · {source}" if page.source else "")
        toc = local_toc(page)
        content = page.body
        if toc:
            content = f'<details class="toc-inline"><summary>On this page</summary>{toc}</details>\n' + content
        html = render(template, {
            "title": "Atlas documentation" if page is home else f"{escape(page.title)} · Atlas documentation",
            "root": page.root, "head": head_extra(page), "wordmark": wordmark_html(),
            "version": version, "topnav": topnav(page, pages, documents), "sidebar": sidebar(page, home, groups),
            "content": content, "toc": toc, "pager": pager(page, order) if page in order else "", "footer": footer,
        })
        target = out / page.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")

    static = out / "_static"
    (static / "fonts").mkdir(parents=True, exist_ok=True)
    for name in ("identity.css", "atlas-mark.svg"):
        shutil.copy2(ROOT / "web" / name, static / name)
    for font in (ROOT / "web" / "fonts").iterdir():
        shutil.copy2(font, static / "fonts" / font.name)
    for name in ("docs.css", "docs.js"):
        shutil.copy2(SITE / name, static / name)
    entries = []
    for page in order:
        entries.extend({"kind": search_kind(page), **section} for section in page.sections)
    (static / "search-index.js").write_text(
        "window.ATLAS_SEARCH = " + json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    return all_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "docs", help="output directory")
    args = parser.parse_args()
    pages = build(args.out)
    source_views = sum(page.out.startswith("source/") for page in pages)
    print(f"Wrote {len(pages) - source_views} documentation pages and {source_views} source views to {args.out}")


if __name__ == "__main__":
    main()
