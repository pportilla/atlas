"""Replayable mathematical statements entered at the Atlas prompt."""

import ast
from pathlib import Path
from tempfile import NamedTemporaryFile


class SessionTranscript:
    def __init__(self):
        self.statements = []

    def entries(self, source, count):
        if count <= 1:
            return (source,)
        # The evaluator expands ^ to ** for precedence. Read statement spans
        # from the original source so a later error on this line does not
        # displace the end of any earlier successful statement.
        return tuple(ast.get_source_segment(source, s) for s in ast.parse(source).body)

    def save(self, filename):
        path = Path(filename)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                    prefix=".atlas-session-", delete=False) as stream:
                temporary = Path(stream.name)
                for statement in self.statements:
                    stream.write(statement + "\n")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
