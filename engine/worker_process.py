"""Run one isolated Atlas computation and reap its process group on interruption."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def _kill_group(child):
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class WorkerGroup:
    """Keep HTTP computations attached to the lifetime of their server."""
    def __init__(self):
        self._lock = threading.Lock()
        self._children = set()
        self._closed = False

    def start(self, *args, **options):
        with self._lock:
            if self._closed:
                raise RuntimeError("The computation server is shutting down.")
            child = subprocess.Popen(*args, **options)
            self._children.add(child)
            return child

    def discard(self, child):
        with self._lock:
            self._children.discard(child)

    def close(self):
        with self._lock:
            self._closed = True
            children = tuple(self._children)
            for child in children:
                if child.poll() is None:
                    _kill_group(child)
        for child in children:
            child.wait()


def run_worker(module_args, data, timeout, *, owner=None):
    root = Path(__file__).resolve().parents[1]
    payload = json.dumps(data)
    start = owner.start if owner is not None else subprocess.Popen
    child = start([sys.executable, "-m", *module_args], cwd=root,
                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                  text=True, start_new_session=True)
    try:
        out, _ = child.communicate(payload, timeout=timeout)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        _kill_group(child)
        child.communicate()
        raise
    finally:
        if owner is not None:
            owner.discard(child)
    if child.returncode:
        raise RuntimeError("Computation worker stopped unexpectedly.")
    try:
        status, record = json.loads(out)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Computation worker returned an invalid response.") from exc
    if type(status) is not int or not isinstance(record, dict):
        raise RuntimeError("Computation worker returned an invalid response.")
    return status, record
