import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "native" / "build" / "resolve_supervisor"


@unittest.skipUnless(SUPERVISOR.is_file(), "Requires the native Linux resolve supervisor")
class ResolveSupervisor(unittest.TestCase):
    def test_deadline_interrupt_and_worker_exit_reap_detached_children(self):
        with tempfile.TemporaryDirectory() as dirname:
            folder = Path(dirname)
            (folder / "engine").mkdir()
            entry = folder / "atlas"
            entry.touch()
            # The fixture issues the same start/done notices as the real CLI.
            # Fork a child and grandchild in separate sessions so killing just
            # the original process group would leave both tasks behind.
            (folder / "engine" / "supervised_cli.py").write_text("""
import json, os, pathlib, signal, sys, time
control, _, action, filename = sys.argv[1:]
if control in ("timeout", "stopped"):
    print(control)
    sys.exit(3 if control == "timeout" else 2)
fd = int(control)
os.set_inheritable(fd, False)
os.write(fd, b'Q' + (250).to_bytes(8, sys.byteorder) if action == "timeout" else b'E')
if os.fork() == 0:
    os.setsid()
    grand = os.fork()
    if grand == 0:
        os.setsid()
        time.sleep(5)
        os._exit(0)
    pathlib.Path(filename).write_text(json.dumps([os.getpid(), grand]))
    time.sleep(5)
    os._exit(0)
while not pathlib.Path(filename).exists():
    time.sleep(.002)
if action == "crash":
    os._exit(1)
if action == "done":
    os.write(fd,b'D')
    print('done')
else:
    time.sleep(5)
""")
            for action, expected, sent in (("timeout", 3, None), ("crash", 2, None),
                                           ("done", 0, None), ("interrupt", 130, signal.SIGINT),
                                           ("terminate", 143, signal.SIGTERM)):
                with self.subTest(action=action):
                    pidfile = folder / action
                    with subprocess.Popen([SUPERVISOR, entry, "resolve", action, pidfile],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                        if sent:
                            deadline = time.monotonic() + 4
                            while (not pidfile.exists() or not pidfile.stat().st_size) and time.monotonic() < deadline:
                                time.sleep(.005)
                            self.assertTrue(pidfile.exists())
                            proc.send_signal(sent)
                        stdout, stderr = proc.communicate(timeout=5)
                    self.assertEqual(proc.returncode, expected)
                    self.assertEqual(stdout.strip(), {"timeout":"timeout", "crash":"stopped", "done":"done"}.get(action, ""))
                    self.assertEqual(stderr, "Resolution interrupted.\n" if sent == signal.SIGINT else "")
                    self.assertFalse(any(Path(f"/proc/{pid}").exists() for pid in json.loads(pidfile.read_text())))
