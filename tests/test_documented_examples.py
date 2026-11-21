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
        guides = ("docs/tour.md", "docs/tutorial/prompt-and-scripts.md", "docs/tutorial/rings.md",
                  "docs/tutorial/resolving-a-cusp.md", "docs/thematic/resolution-records.md",
                  "docs/thematic/weighted-resolution.md")
        for guide in guides:
            contents = (ROOT / guide).read_text(encoding="utf-8")
            # The ring naming example explicitly starts a fresh session.
            for section in contents.split("\n## Name, select and delete rings\n"):
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
        for guide in ("docs/tutorial/polynomials-and-ideals.md", "docs/tutorial/quotient-rings.md",
                      "docs/tutorial/local-algebra.md", "docs/tutorial/local-geometry.md", "docs/reference/saved-values.md"):
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

    def test_constructions(self):
        # Each construction is a complete script, so each one also runs in a fresh session.
        contents = (ROOT / "docs/constructions.md").read_text(encoding="utf-8")
        recipes = re.findall(r"^### (.*?)\n(.*?)(?=^##|\Z)", contents, re.MULTILINE | re.DOTALL)
        self.assertGreater(sum("```atlas" in body for _, body in recipes), 25)
        for heading, body in recipes:
            for source in re.findall(r"^```atlas\n(.*?)^```$", body, re.MULTILINE | re.DOTALL):
                with self.subTest(construction=heading):
                    expected = expected_output(body, source)
                    self.assertTrue(expected)
                    script = run_script(source)
                    self.assertEqual(script.returncode, 0, script.stderr)
                    self.assertEqual(script.stdout.splitlines(), expected)
                    status, output, error = run_prompt(source + "\nquit\n")
                    self.assertEqual((status, error), (0, ""))
                    self.assertEqual(output.replace("atlas> ", "").strip().splitlines(), expected)
