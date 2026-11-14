import errno
import os
from pathlib import Path
import re
import select
import signal
import struct
import subprocess
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROMPT = b"atlas> "
SAVE = b"\x1b7"
RESTORE = b"\x1b8"
ANSI = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")


@unittest.skipUnless(sys.platform.startswith("linux"), "Uses a Linux pseudo-terminal.")
class SessionPanelTerminalTests(unittest.TestCase):
    def setUp(self):
        import fcntl
        import termios

        self.master, slave = os.openpty()
        self.addCleanup(os.close, self.master)
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 110, 0, 0))
            env = dict(os.environ, TERM="xterm", NO_COLOR="1")
            env.pop("COLUMNS", None)
            env.pop("LINES", None)
            self.child = subprocess.Popen([sys.executable, str(ROOT / "atlas")], cwd=ROOT,
                                          stdin=slave, stdout=slave, stderr=slave,
                                          start_new_session=True, env=env)
            self.addCleanup(self.stop_child)
        finally:
            os.close(slave)

    def stop_child(self):
        if self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.child.wait(timeout=5)

    def read_output(self, *, to_eof=False):
        output = bytearray()
        deadline = time.monotonic() + 10
        while to_eof or PROMPT not in output:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.master], [], [], remaining)[0]:
                self.fail(f"Timed out reading Atlas: {bytes(output)!r}")
            try:
                chunk = os.read(self.master, 8192)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                chunk = b""  # The last slave descriptor was closed.
            if not chunk:
                self.assertTrue(to_eof, f"Atlas exited before the prompt: {bytes(output)!r}")
                break
            output.extend(chunk)
        return bytes(output)

    def command(self, command):
        os.write(self.master, command.encode() + b"\n")
        return self.read_output()

    def panel_update(self, output):
        start = output.find(SAVE)
        end = output.find(RESTORE, start + len(SAVE))
        self.assertGreaterEqual(start, 0, output)
        self.assertGreater(end, start, output)
        return ANSI.sub(b" ", output[start + len(SAVE):end]).decode()

    def check_exit(self, keys):
        os.write(self.master, keys)
        output = self.read_output(to_eof=True)
        self.assertEqual(self.child.wait(timeout=5), 0, output)
        # DECSTBM resets to the whole screen with no arguments or explicit bounds.
        self.assertRegex(output, rb"\x1b\[(?:1;32)?r")
        if b"\x1b[?1049h" in self.startup:
            self.assertIn(b"\x1b[?1049l", output)

    def test_bindings_stay_pinned_while_commands_scroll(self):
        self.startup = startup = self.read_output()
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
        self.startup = self.read_output()
        self.assertRegex(self.startup, rb"\x1b\[3;(?:32)?r")
        self.check_exit(b"\x04")
