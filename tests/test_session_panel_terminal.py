import re
import sys
import unittest

from tests.cli_support import CliTerminalCase, PROMPT


SAVE = b"\x1b7"
RESTORE = b"\x1b8"
ANSI = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")


@unittest.skipUnless(sys.platform.startswith("linux"), "Uses a Linux pseudo-terminal.")
class SessionPanelTerminalTests(CliTerminalCase):
    def command(self, command):
        self.send(command.encode() + b"\n")
        return self.read_until(PROMPT)

    def panel_update(self, output):
        # The welcome also saves the cursor to animate below the pinned rows.
        start = output.index(SAVE + b"\x1b[1;1H\x1b[2K")
        end = output.index(RESTORE, start + len(SAVE))
        update = output[start + len(SAVE):end]
        self.assertIn(b"\x1b[2;1H\x1b[2K", update)
        return ANSI.sub(b" ", update).decode()

    def check_exit(self, keys):
        self.send(keys)
        output = self.read_until(None)
        self.assertEqual(self.child.wait(timeout=5), 0, output)
        # DECSTBM resets to the whole screen with no arguments or explicit bounds.
        self.assertRegex(output, rb"\x1b\[(?:1;32)?r")
        if b"\x1b[?1049h" in self.startup:
            self.assertIn(b"\x1b[?1049l", output)

    def test_bindings_stay_pinned_while_commands_scroll(self):
        self.startup = startup = self.read_until(PROMPT)
        self.assertIn(b"Welcome to Atlas", startup)
        self.assertGreater(startup.index(b"Welcome to Atlas"), startup.rfind(b"\x1b[2J"))
        self.assertRegex(startup, rb"\x1b\[3;(?:32)?r")
        self.assertIn("Rings", self.panel_update(startup))
        self.assertIn("Variables", self.panel_update(startup))

        rings = self.panel_update(self.command('x, y = coordinates("x", "y")'))
        self.assertIn("QQ[x, y]", rings)
        variables = self.panel_update(self.command("f = y^2 - x^3"))
        self.assertIn("QQ[x, y]", variables)
        names = variables.split("Variables", 1)[1]
        for name in ("x", "y", "f"):
            self.assertRegex(names, rf"\b{name}\b")

        self.command("f = 7")
        expression = self.command("f + 1")
        self.assertRegex(expression, rb"\b8\r?\n")
        self.assertNotIn(SAVE, expression)
        workspace = self.command(":workspace")
        self.assertNotIn(SAVE, workspace)
        workspace = ANSI.sub(b" ", workspace)
        self.assertIn(b"QQ[x, y]", workspace)
        self.assertRegex(workspace, rb"\bf\b")
        self.assertRegex(workspace, rb"\b7\b")
        self.check_exit(b"quit\n")

    def test_eof_restores_normal_scrolling(self):
        self.startup = self.read_until(PROMPT)
        self.assertRegex(self.startup, rb"\x1b\[3;(?:32)?r")
        self.check_exit(b"\x04")
