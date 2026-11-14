"""Shared helpers for Atlas scripts, interactive prompts, and CLI replay."""

import errno
import io
import os
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PROMPT = b"atlas> "


class Terminal(io.StringIO):
    def isatty(self):
        return True


def run_script(source):
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "curve.atlas"
        script.write_text(dedent(source).lstrip(), encoding="utf-8")
        return subprocess.run(
            [str(ROOT / "atlas"), "run", str(script)],
            capture_output=True, text=True, timeout=15,
        )


def run_prompt(source):
    from engine.app import main

    stdout, stderr = Terminal(), io.StringIO()
    with (
        patch.object(sys, "argv", ["atlas"]),
        patch.object(sys, "stdin", Terminal(dedent(source).lstrip())),
        patch.object(sys, "stdout", stdout),
        patch.object(sys, "stderr", stderr),
        patch("engine.welcome.print_welcome"),
    ):
        status = main()
    return status, stdout.getvalue(), stderr.getvalue()


def shell(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)


class CliTerminalCase(unittest.TestCase):
    """Interact with the real Atlas process through a fixed-size Linux terminal."""

    def setUp(self):
        import fcntl
        import termios

        self.pending = bytearray()
        self.master, slave = os.openpty()
        self.addCleanup(os.close, self.master)
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 110, 0, 0))
            env = dict(os.environ, TERM="xterm", NO_COLOR="1", INPUTRC="/dev/null")
            env.pop("COLUMNS", None)
            env.pop("LINES", None)
            self.child = subprocess.Popen(
                [sys.executable, str(ROOT / "atlas")], cwd=ROOT, env=env,
                stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
            )
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

    def send(self, keys):
        os.write(self.master, keys)

    def read_until(self, marker, timeout=10):
        """Consume through marker or, for None, through terminal EOF."""
        deadline = time.monotonic() + timeout
        while marker is None or marker not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.master], [], [], remaining)[0]:
                self.fail(f"Timed out waiting for {marker!r}. Terminal output: {bytes(self.pending)!r}")
            try:
                chunk = os.read(self.master, 8192)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                chunk = b""  # The last slave descriptor was closed.
            if not chunk:
                self.assertIsNone(marker, f"Atlas exited early: {bytes(self.pending)!r}")
                break
            self.pending.extend(chunk)
        end = len(self.pending) if marker is None else self.pending.index(marker) + len(marker)
        data = bytes(self.pending[:end])
        del self.pending[:end]
        return data
