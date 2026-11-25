"""Run one isolated Atlas computation and reap its process group on interruption."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .exact_json import result_json
from . import fork_server
from .polynomial import integer_from_decimal


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
            child = fork_server.popen(*args, **options)
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


def _exchange(command, payload, root, deadline, owner, env=None):
    if deadline is not None and time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired(command, 0)
    start = owner.start if owner is not None else fork_server.popen
    child = start(command, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                  text=True, start_new_session=True, env=env)
    try:
        with child:
            try:
                timeout = None if deadline is None else max(0, deadline - time.monotonic())
                out, _ = child.communicate(payload, timeout=timeout)
            except (KeyboardInterrupt, subprocess.TimeoutExpired, UnicodeEncodeError):
                _kill_group(child)
                raise
    finally:
        if owner is not None:
            owner.discard(child)
    return child.returncode, out


def run_worker(module_args, data, timeout, *, owner=None, native_command=None, env=None):
    """An optional native worker can decline and share its deadline with the general worker.

    `env` is the environment of the general worker (this process's own when None)."""
    root = Path(__file__).resolve().parents[1]
    deadline = None if timeout is None else time.monotonic() + timeout
    payload = result_json(data)
    command = [sys.executable, "-m", *module_args]
    if native_command is not None:
        try:
            code, out = _exchange(native_command, payload, root, deadline, owner)
        except (FileNotFoundError, PermissionError):
            code = 75
        # Old native executables return 2 for an unknown invocation. A current
        # one returns 75 for the general parser or device dispatcher.
        if code in (2, 75):
            code, out = _exchange(command, payload, root, deadline, owner, env)
    else:
        code, out = _exchange(command, payload, root, deadline, owner, env)
    if code:
        raise RuntimeError("Computation worker stopped unexpectedly.")
    try:
        # Worker output can contain computed dimensions with thousands of digits.
        status, record = json.loads(out, parse_int=integer_from_decimal)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Computation worker returned an invalid response.") from exc
    if type(status) is not int or not isinstance(record, dict):
        raise RuntimeError("Computation worker returned an invalid response.")
    return status, record
