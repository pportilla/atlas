"""Build the Atlas documentation site from the repository's Markdown.

The site has three sources and copies none of them:

* Markdown. README.md is the home page. Every Markdown file it links to,
  directly or through another page, becomes a page, in the order the links
  appear. Links between Markdown files become links between pages, and links
  to other repository files point at the source on GitHub.
* Code. The function reference holds the cards that `help NAME` prints at
  the prompt, read from the shared session. Inline code such as
  `milnor_number(f)` in any page links to its entry, and each entry links back
  to the sections that mention it.
* HTML. docs/site/template.html lays out every page. web/identity.css,
  shared with the web workspace, supplies the faces and colours.

Run from the repository root:

    python3 docs/site/build.py [--out build/docs]

The build needs markdown-it-py. Maths renders with KaTeX in the browser.
"""
import argparse
from dataclasses import dataclass, field
from html import escape
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
SITE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from markdown_it import MarkdownIt  # noqa: E402

from engine.app import make_parser, shell_command_names  # noqa: E402
from engine.cli_highlight import assigned_names, command_pieces, expression_pieces  # noqa: E402
from engine.session import new_session  # noqa: E402
from engine.session_help import function_help  # noqa: E402
from engine.web_brand import lettered_map_html, wordmark_html  # noqa: E402
from engine.welcome import atlas_version  # noqa: E402

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs"
REFERENCE = "reference/functions.html"
INDEX = "genindex.html"


@dataclass
class Page:
    source: str             # repository path of the Markdown, or "" for generated pages
    out: str                # site path of the HTML
    title: str = ""
    parent: "Page | None" = None
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
    if begin >= state.posMax or src[begin].isspace():
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
        token.content = src[begin:end]
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
    path = PurePosixPath(source)
    return str(path.with_name("index.html") if path.name == "README.md" else path.with_suffix(".html"))


def repository_url():
    try:
        url = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True,
                                      stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    url = re.sub(r"^git@github\.com:", "https://github.com/", url)
    return url.removesuffix(".git")


def discover(md):
    """Pages in link order, starting from README.md, each under the page that first links it."""
    pages = {"README.md": Page("README.md", "index.html")}
    queue = ["README.md"]
    while queue:
        page = pages[queue.pop(0)]
        for token in md.parse((ROOT / page.source).read_text(encoding="utf-8")):
            for child in token.children or ():
                if child.type != "link_open":
                    continue
                href = child.attrGet("href") or ""
                if _is_external(href) or href.startswith("#"):
                    continue
                target = posixpath.normpath(posixpath.join(posixpath.dirname(page.source), href.split("#")[0]))
                if target.endswith(".md") and target not in pages and (ROOT / target).is_file():
                    pages[target] = Page(target, out_path(target), parent=page)
                    page.children.append(pages[target])
                    queue.append(target)
    return pages


# Highlighting ----------------------------------------------------------------

def _spans(pieces):
    return "".join(f'<span class="t-{role}">{escape(text)}</span>' if role else escape(text)
                   for text, role in pieces)


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

    def block(self, code, lang):
        lines = code.rstrip("\n").split("\n")
        known = set()
        out = []
        if any(line.startswith("atlas>") for line in lines):
            for line in lines:
                if line.startswith("atlas>"):
                    text = line[6:].removeprefix(" ")
                    out.append('<span class="line in"><span class="t-brand">atlas</span><span class="t-muted">&gt;</span> '
                               + _spans(self._input(text, known)) + "</span>")
                elif line.startswith("... "):
                    out.append('<span class="line in"><span class="t-muted">...</span> '
                               + _spans(expression_pieces(line[4:], names=known, functions=self.functions)) + "</span>")
                else:
                    pieces = expression_pieces(line, names=known, functions=self.functions, verdicts=True)
                    out.append('<span class="line out">' + _spans(pieces) + "</span>")
        elif lang in ("atlas", "python", "py", "singular"):
            out = ['<span class="line in">' + _spans(self._input(line, known)) + "</span>" for line in lines]
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
    def __init__(self, pages, functions, highlighter, source_url):
        self.md = markdown()
        self.pages = pages
        self.functions = functions
        self.highlighter = highlighter
        self.source_url = source_url
        self.mentions = {}  # function name -> [(page, anchor, section title)]
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

    def _fence(self, tokens, i, options, env):
        token = tokens[i]
        lang = (token.info or "").split(maxsplit=1)[0].lower()
        if lang == "mermaid":
            env["page"].head.add("mermaid")
            return f'<pre class="mermaid">{escape(token.content)}</pre>\n'
        code = self.highlighter.block(token.content, lang)
        label = f' data-lang="{escape(lang)}"' if lang and lang != "text" else ""
        return f'<pre class="code"{label}><code>{code}</code></pre>\n'

    def _code_inline(self, tokens, i, options, env):
        token = tokens[i]
        code = f"<code>{escape(token.content)}</code>"
        name = token.meta.get("function")
        if name:
            return f'<a class="api-ref" href="{env["page"].root}{REFERENCE}#{name}">{code}</a>'
        return code

    def _heading_close(self, tokens, i, options, env):
        anchor = tokens[i].meta.get("anchor")
        link = f'<a class="headerlink" href="#{anchor}" aria-label="Link to this section">¶</a>' if anchor else ""
        return f"{link}</{tokens[i].tag}>\n"

    def function_in(self, code):
        """The session function an inline code span calls, or names when it has an underscore."""
        for call in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\(", code):
            if call[1] in self.functions:
                return call[1]
        bare = code.strip()
        return bare if bare in self.functions and "_" in bare else None

    def link(self, page, href):
        """Point a link written for GitHub at the matching page or source file."""
        if _is_external(href) or href.startswith("#") or href.startswith("mailto:"):
            return href, "external" if _is_external(href) else ""
        path, _, anchor = href.partition("#")
        target = posixpath.normpath(posixpath.join(posixpath.dirname(page.source), path))
        suffix = f"#{anchor}" if anchor else ""
        if target in self.pages:
            relative = posixpath.relpath(self.pages[target].out, posixpath.dirname(page.out) or ".")
            return relative + suffix, ""
        if self.source_url:
            kind = "tree" if (ROOT / target).is_dir() else "blob"
            return f"{self.source_url}/{kind}/main/{target}{suffix}", "source"
        return href, "source"

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
                for child in token.children or ():
                    if child.type == "link_open":
                        href, kind = self.link(page, child.attrGet("href") or "")
                        child.attrSet("href", href)
                        if kind:
                            child.attrSet("class", kind)
                    elif child.type == "code_inline":
                        name = self.function_in(child.content)
                        if name:
                            child.meta["function"] = name
                            entry = (page, section_id, section_title)
                            if entry not in self.mentions.setdefault(name, []):
                                self.mentions[name].append(entry)
                    elif child.type in ("math_inline", "math_display"):
                        page.head.add("math")
            elif token.type in ("fence", "code_block"):
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
                for child in token.children or ():
                    if child.type == "code_inline":
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


def reference_page(renderer, session, highlighter):
    page = Page("", REFERENCE, title="Session functions")
    names = sorted(name for name, value in session.builtins.items() if callable(value))
    entries, rows = [], []
    for name in names:
        signature, purpose, example = _card(function_help(name, session.builtins))
        params = signature[len(name):]
        params_html = _spans([(text, "field" if role is None and text.isidentifier() else role)
                              for text, role in expression_pieces(params)])
        # Mark dotted fields and calls in the card's prose as code, so they read and link as code.
        purpose_md = _CODE_WORD.sub(r"`\1`", purpose)
        summary = purpose.split("\n\n")[0].split(". ")[0].rstrip(".") + "."
        rows.append(f'<tr><td><a class="api-ref" href="#{name}"><code>{name}</code></a></td>'
                    f"<td>{escape(summary)}</td></tr>")
        lines = "\n".join("atlas> " + line for line in example)
        mentions = renderer.mentions.get(name, [])
        links = ", ".join(
            f'<a href="{page.root}{p.out}{"#" + anchor if anchor else ""}">{escape(title)}</a>'
            + (f' <span class="api-page">({escape(p.title)})</span>' if title != p.title else "")
            for p, anchor, title in mentions)
        entries.append(
            f'<section class="api-entry" id="{name}">\n'
            f'<h2 class="api-signature"><code><span class="api-name">{name}</span>{params_html}</code>'
            f'<a class="headerlink" href="#{name}" aria-label="Link to {name}">¶</a></h2>\n'
            f'{renderer.prose(purpose_md, page)}'
            f'<p class="api-label">Example</p>\n'
            f'<pre class="code"><code>{highlighter.block(lines, "text")}</code></pre>\n'
            + (f'<p class="api-mentions"><span>Used in</span> {links}</p>\n' if links else "")
            + "</section>")
        page.headings.append((2, name, name))
        page.sections.append({"page": page.title, "title": name, "url": f"{REFERENCE}#{name}",
                              "text": f"{signature} {purpose}", "kind": "function"})
    letters = sorted({name[0].upper() for name in names})
    page.body = (
        "<h1>Session functions</h1>\n"
        '<p class="lede">These functions work at the <code>atlas&gt;</code> prompt and in scripts run with '
        "<code>./atlas run</code>. Each entry shows the card that <code>help NAME</code> prints, "
        "followed by the guide sections that use the function.</p>\n"
        '<nav class="letter-bar" aria-label="Functions by letter">'
        + "".join(f'<a href="#{next(n for n in names if n[0].upper() == letter)}">{letter}</a>' for letter in letters)
        + "</nav>\n"
        '<div class="table-wrap"><table class="api-summary"><thead><tr><th>Function</th><th>Purpose</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>\n' + "\n".join(entries))
    page.headings = []  # Fifty entries make a poor local table of contents; the summary table serves.
    return page


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


def identity_page():
    """Show the shared palette, faces and terminal roles, read from their sources."""
    from engine.cli_style import _ROLES

    page = Page("", "reference/identity.html", title="Visual identity")
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
    page.body = f"""<h1>Visual identity</h1>
<p class="lede">The terminal, the web workspace and these pages share one identity. This page reads it from
<code>web/identity.css</code> and <code>engine/cli_style.py</code>, so it shows what ships.</p>
<div class="hero hero-small"><pre class="ascii-art" aria-hidden="true">{lettered_map_html()}</pre></div>
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
    page.headings = [(2, "faces", "Faces"), (2, "colours", "Colours"), (2, "terminal-roles", "Terminal roles")]
    page.sections = [{"page": page.title, "title": title, "url": f"{page.out}#{anchor}", "text": title}
                     for _, anchor, title in page.headings]
    return page


def index_page(pages):
    page = Page("", INDEX, title="Index")
    entries = []
    for p in pages:
        if p.out == INDEX:
            continue
        entries.append((p.title, p.out, "page", ""))
        for level, anchor, title in p.headings:
            entries.append((title, f"{p.out}#{anchor}", "section", p.title))
        if p.out == REFERENCE:
            for section in p.sections:
                entries.append((section["title"], section["url"], "function", ""))
    def sort_key(title):
        # Numbered sections such as "7.3 Certification" file under their words.
        return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).lstrip("`:./").lower()

    entries.sort(key=lambda e: (sort_key(e[0]), e[3]))
    groups = {}
    for entry in entries:
        key = sort_key(entry[0])[:1].upper()
        groups.setdefault(key if key.isalpha() else "#", []).append(entry)
    parts = ['<h1>Index</h1>\n<p class="lede">Every page, section and session function, in alphabetical order.</p>',
             '<nav class="letter-bar" aria-label="Index by letter">'
             + "".join(f'<a href="#letter-{k}">{k}</a>' for k in groups) + "</nav>"]
    for key, items in groups.items():
        parts.append(f'<h2 id="letter-{key}">{key}</h2>\n<ul class="index-list">')
        for title, url, kind, where in items:
            label = f"<code>{escape(title)}</code>" if kind == "function" else escape(title)
            context = f' <span class="index-where">{escape(where)}</span>' if where else ""
            tag = {"page": "page", "function": "function"}.get(kind, "")
            badge = f' <span class="index-kind">{tag}</span>' if tag else ""
            parts.append(f'<li><a href="{url}">{label}</a>{badge}{context}</li>')
        parts.append("</ul>")
    page.body = "\n".join(parts)
    return page


# Layout ----------------------------------------------------------------------

def sidebar(page, home, guide, reference):
    def item(p, depth=0):
        current = ' aria-current="page"' if p is page else ""
        link = f'<a href="{page.root}{p.out}"{current} class="depth-{depth}">{escape(p.title)}</a>'
        open_branch = p is page or any(c is page or page in c.children for c in p.children)
        kids = "".join(item(c, depth + 1) for c in p.children) if open_branch and p.children else ""
        return f"<li>{link}{f'<ul>{kids}</ul>' if kids else ''}</li>"

    return (f'<div class="nav-group"><h2>Guide</h2><ul>{item(home)}'
            + "".join(item(p) for p in guide) + "</ul></div>"
            + '<div class="nav-group"><h2>Reference</h2><ul>' + "".join(item(p) for p in reference) + "</ul></div>")


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
    md = markdown()
    pages = discover(md)
    session = new_session(None)
    functions = {name for name in session.builtins}
    commands = shell_command_names(make_parser()) | {"help"}
    highlighter = Highlighter(functions, commands)
    renderer = Renderer(pages, functions, highlighter, repository_url())
    for page in pages.values():
        renderer.page(page)
    home = pages["README.md"]
    home.children = []  # Pages linked from the home page form the guide.
    home.title = "Overview"
    guide = [p for p in pages.values() if p.parent is home]
    reference = reference_page(renderer, session, highlighter)
    order = [home]
    for p in guide:
        stack = [p]
        while stack:
            node = stack.pop(0)
            order.append(node)
            stack[0:0] = node.children
    identity = identity_page()
    index = index_page([*order, reference, identity])
    reference_pages = [reference, identity, index]
    order += reference_pages

    template = (SITE / "template.html").read_text(encoding="utf-8")
    version = escape(atlas_version())
    source_url = renderer.source_url
    out.mkdir(parents=True, exist_ok=True)
    for page in order:
        section = home if page not in reference_pages else page
        topnav = "".join(
            f'<a href="{page.root}{p.out}"' + (' aria-current="true"' if p is section else "") + f">{label}</a>"
            for label, p in (("Guide", home), ("Functions", reference), ("Index", index)))
        source = (f'<a class="source" href="{source_url}/blob/main/{page.source}">{escape(page.source)}</a>'
                  if page.source and source_url else escape(page.source))
        footer = (f"Atlas {version}. "
                  + (f"Generated from {source} by <code>docs/site/build.py</code>." if page.source else
                     "Generated from the Atlas session by <code>docs/site/build.py</code>."))
        toc = local_toc(page)
        content = page.body
        if toc:
            content = f'<details class="toc-inline"><summary>On this page</summary>{toc}</details>\n' + content
        html = render(template, {
            "title": "Atlas documentation" if page is home else f"{escape(page.title)} · Atlas documentation",
            "root": page.root, "head": head_extra(page), "wordmark": wordmark_html(),
            "version": version, "topnav": topnav, "sidebar": sidebar(page, home, guide, reference_pages),
            "content": content, "toc": toc, "pager": pager(page, order), "footer": footer,
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
    entries = [section for page in order for section in page.sections]
    (static / "search-index.js").write_text(
        "window.ATLAS_SEARCH = " + json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "docs", help="output directory")
    args = parser.parse_args()
    pages = build(args.out)
    print(f"Wrote {len(pages)} pages to {args.out}")


if __name__ == "__main__":
    main()
