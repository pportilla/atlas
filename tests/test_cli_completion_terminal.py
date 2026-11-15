"""Readline discovery and startup input exercised through a real terminal."""
from importlib.util import find_spec
import sys
import unittest

from tests.cli_support import CliTerminalCase, PROMPT


@unittest.skipUnless(sys.platform.startswith("linux") and find_spec("readline"),
                     "Requires a Linux pseudo-terminal with readline.")
class CliCompletionTerminalTests(CliTerminalCase):
    def test_type_edit_and_submit_while_map_is_still_unfolding(self):
        # The first fold contains |::|. Start timing only once it is drawn.
        self.read_until(b"|::|")
        self.send(b"print(83")
        self.read_until(b"print(83", timeout=1.5)
        # Leave the line half-written through a repaint before editing it.
        self.read_until(b"|  :  :  :  :  |", timeout=1.5)
        self.send(b"\x7f*7)\n")
        self.read_until(b"56\r\n", timeout=1.5)
        self.send(b"quit\n")
        self.assertEqual(self.child.wait(timeout=5), 0)

    def test_tab_during_animation_keeps_the_edit_buffer(self):
        self.read_until(b"|::|")
        self.send(b"local_\t")
        choices = self.read_until(b"local_standard_basis", timeout=1.5)
        # A short list fits below the banner, so the map keeps unfolding.
        self.assertNotRegex(choices, rb"\x1b\[5;(?:32)?r")
        self.read_until(b"\x1b7\x1b[6;1H", timeout=1.5)
        self.send(b'standard_basis()\n')
        self.read_until(b"help local_standard_basis", timeout=1.5)
        self.send(b"quit\n")
        self.assertEqual(self.child.wait(timeout=5), 0)

    def test_a_long_completion_list_settles_the_map_first(self):
        self.read_until(b"|::|")
        self.send(b"\t")
        choices = self.read_until(b"zariski_tangent_space", timeout=3)
        # Restore all rows below the pinned header before listing every name.
        self.assertRegex(choices, rb"(?s)\x1b\[5;(?:32)?r.*zariski_tangent_space")
        self.send(b"quit\n")
        self.assertEqual(self.child.wait(timeout=5), 0)

    def test_tab_lists_resolution_members_and_completes_live_names(self):
        self.read_until(PROMPT)
        self.send(b'x, y = coordinates("x", "y")\n')
        self.read_until(PROMPT)
        self.send(b"R = resolve(y^2 - x^3)\n")
        self.read_until(PROMPT)

        self.send(b"R.\t")
        choices = self.read_until(PROMPT)
        for member in (b"branches", b"charts", b"divisors", b"chart", b"divisor", b"status"):
            self.assertRegex(choices, rb"\b" + member + rb"\b")
        self.assertNotIn(b"_record", choices)
        self.send(b"sta\t\n")
        self.assertIn(b"resolved\r\n", self.read_until(PROMPT))
        self.send(b"help R.bran\t\n")
        self.assertIn(b"R.branches", self.read_until(PROMPT))
        self.send(b"cusp_sample = 72\n")
        self.read_until(PROMPT)
        self.send(b"print(cusp_sa\t)\n")
        self.assertIn(b"72\r\n", self.read_until(PROMPT))

        self.send(b'coordina\t"omega")\n')
        self.assertIn(b"omega\r\n", self.read_until(PROMPT))
        self.send(b"quit\n")
        self.assertEqual(self.child.wait(timeout=5), 0)
