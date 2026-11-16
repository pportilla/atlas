"""Source views for the exact checkout used to build the documentation."""

from dataclasses import dataclass, field
from html import escape
from pathlib import PurePosixPath
import re


def browsable(path):
    folder, _, filename = path.partition("/")
    return folder in {"native", "engine", "tests", "examples"} and PurePosixPath(filename).suffix in {
        ".cpp", ".hpp", ".h", ".py", ".atlas",
    }


@dataclass
class SourceFile:
    path: str
    lines: list[str]
    readers: dict = field(default_factory=dict)  # site URL -> guide section or function

    @property
    def out(self):
        return f"source/{self.path}.html"

    def fragment(self, link):
        if link.symbol:
            definition = re.compile(rf"^\s*def {re.escape(link.symbol)}\s*\(")
            matches = [n for n, line in enumerate(self.lines, 1) if definition.match(line)]
        elif link.needle:
            matches = [n for n, line in enumerate(self.lines, 1) if link.needle in line]
        else:
            return ""
        if len(matches) != 1:
            raise ValueError(f"{self.path}: expected one source line for {link.symbol or link.needle!r}")
        return f"#L{matches[0]}"

    def body(self, root):
        backlinks = " · ".join(f'<a href="{root}{escape(url)}">{escape(label)}</a>'
                               for url, label in self.readers.items())
        rows = "\n".join(
            f'<span class="source-line" id="L{n}"><a href="#L{n}" aria-label="Line {n}">{n}</a>'
            f'{escape(line)}</span>' for n, line in enumerate(self.lines, 1))
        return (f'<h1 class="source-title">{escape(self.path)}</h1>\n'
                '<details class="source-readers"><summary>Related documentation</summary>'
                f'<p>{backlinks}</p></details>\n'
                f'<pre class="source-code"><code>{rows}</code></pre>')


class SourceFiles:
    def __init__(self, root):
        self.root = root
        self.files = {}

    def get(self, path):
        if path not in self.files:
            lines = (self.root / path).read_text(encoding="utf-8").splitlines()
            self.files[path] = SourceFile(path, lines)
        return self.files[path]
