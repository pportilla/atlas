"""Run the worked guides as written and compare their published outputs."""
import re
import unittest

from tests.cli_support import ROOT, run_prompt, run_script


def expected_output(section, source):
    # Point-by-point examples give the printed columns in a table, with the
    # input point in the first column. Other examples annotate each print.
    table = re.search(r"^\| Point in loop order \|.*?(?=\n\n|\Z)", section, re.MULTILINE | re.DOTALL)
    if table:
        return [" ".join(cell.strip() for cell in row.strip("|").split("|")[1:])
                for row in table.group().splitlines()[2:]]
    return [line.split("#", 1)[1].strip() for line in source.splitlines() if line.lstrip().startswith("print(")]


class DocumentedExamplesTests(unittest.TestCase):
    def test_prompt_transcripts(self):
        guides = ("README.md", "examples/algebra-api.md", "docs/sessions.md", "docs/resolving-a-cusp.md",
                  "docs/resolution-records.md", "docs/weighted-resolution.md")
        for guide in guides:
            contents = (ROOT / guide).read_text(encoding="utf-8")
            # The ring naming example explicitly starts a fresh session.
            for section in contents.split("\n### Name, select and delete rings\n"):
                blocks = re.findall(r"^```text\n(atlas> .*?)^```$", section, re.MULTILINE | re.DOTALL)
                with self.subTest(guide=guide, section=section.splitlines()[0]):
                    self.assertTrue(blocks)
                    commands = []
                    expected = []
                    for statement in "".join(blocks).split("atlas> ")[1:]:
                        command, _, response = statement.partition("\n")
                        commands.append(command)
                        expected.append(response)
                    status, output, error = run_prompt("\n".join([*commands, "quit"]))
                    self.assertEqual((status, error), (0, ""))
                    self.assertEqual(output.split("atlas> ")[1:-1], expected)

    def test_worked_guides(self):
        for guide in ("examples/polynomial-ideals.md", "examples/quotient-algebras.md",
                      "examples/local-algebra.md", "examples/local-geometry.md", "docs/value-methods.md"):
            contents = (ROOT / guide).read_text(encoding="utf-8")
            for section in contents.split("\n## ")[1:]:
                heading, _, body = section.partition("\n")
                with self.subTest(guide=guide, section=heading):
                    blocks = re.findall(r"^```atlas\n(.*?)^```$", body, re.MULTILINE | re.DOTALL)
                    self.assertTrue(blocks)
                    source = "\n".join(blocks)
                    expected = expected_output(body, source)
                    self.assertTrue(expected)

                    script = run_script(source)
                    self.assertEqual(script.returncode, 0, script.stderr)
                    self.assertEqual(script.stdout.splitlines(), expected)
                    status, output, error = run_prompt(source + "\nquit\n")
                    self.assertEqual((status, error), (0, ""))
                    lines = output.replace("atlas> ", "").replace("... ", "").strip().splitlines()
                    self.assertEqual(lines, expected)
